#!/usr/bin/env python3
"""Painel de acompanhamento das execuções de ``scripts/training/run_seeds.sh``.

Roda no host (fora do container), só com a biblioteca padrão, e se atualiza sozinho:

    python3 scripts/training/monitor.py                 # execução mais recente em ~/runs
    python3 scripts/training/monitor.py ~/runs/<nome>   # uma execução específica
    python3 scripts/training/monitor.py --once          # um retrato só, sem atualizar

Lê o ``progress.tsv`` e os logs que o ``run_seeds.sh`` grava, e os JSONs do ``evaluate.py``.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRATCH = {"A-RAW", "B"}
LABELS = {
    "A-RAW": "do zero, sem aug",
    "B": "do zero, com aug",
    "C-RAW": "TL, sem aug",
    "D": "TL, com aug",
}
# Duração (min) de cada treino no protocolo completo, medida nos runs originais da RTX 4090.
TRAIN_MINUTES_FULL = {"A-RAW": 44, "B": 44, "C-RAW": 16, "D": 28}
EVAL_MINUTES = 2.0
STALE_MINUTES = 10

RE_TRAIN = re.compile(
    r"Epoch\(train\)\s+\[(\d+)\]\[\s*(\d+)/(\d+)\]\s+lr: (\S+)\s+eta: (\S+).*?"
    r"memory: (\d+)\s+loss: ([\d.]+).*?acc_pose: ([\d.]+)"
)
RE_VAL = re.compile(r"Epoch\(val\)\s+\[(\d+)\]\[\s*\d+/\d+\]\s+PCK: ([\d.]+)\s+PDJ: ([\d.]+)")
RE_PHASE_SINGLE = re.compile(r"\[Cenário \S+\] Fase única — (\d+) épocas")
RE_PHASE_TL = re.compile(r"\[Cenário \S+\] Fase (\d) — (.+?)(?: \[ATIVADA\])?$")
RE_PHASE_EPOCHS = re.compile(r"Épocas: (\d+)")
RE_PHASE_RESULT = re.compile(r"\[Fase (\d)\] Best checkpoint: .*= ([\d.]+)")
RE_SELECTION = re.compile(r"\[Seleção\] .*→ (fase\d)")

USE_COLOR = sys.stdout.isatty()


def c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def bold(t: str) -> str:
    return c(t, "1")


def dim(t: str) -> str:
    return c(t, "2")


def green(t: str) -> str:
    return c(t, "32")


def yellow(t: str) -> str:
    return c(t, "33")


def red(t: str) -> str:
    return c(t, "31")


def cyan(t: str) -> str:
    return c(t, "36")


def bar(frac: float, width: int = 30) -> str:
    frac = max(0.0, min(1.0, frac))
    full = int(round(frac * width))
    return green("█" * full) + dim("░" * (width - full))


def fmt_dur(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def read_params(run: Path) -> dict:
    params = {}
    f = run / "params.env"
    if f.exists():
        for line in f.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                params[k.strip()] = v.strip()
    return params


def read_progress(run: Path) -> list[dict]:
    f = run / "progress.tsv"
    if not f.exists():
        return []
    lines = f.read_text().splitlines()
    if not lines:
        return []
    header = lines[0].split("\t")
    return [dict(zip(header, ln.split("\t"))) for ln in lines[1:] if ln.strip()]


def step_states(progress: list[dict]) -> dict:
    """(seed, scenario, step) -> {'status', 'seconds', 'log', 'started'}."""
    states: dict = {}
    for row in progress:
        if len(row) < 7:  # linha incompleta (escrita em andamento)
            continue
        key = (row["seed"], row["scenario"], row["step"])
        st = states.setdefault(key, {})
        st["status"] = row["status"]
        st["log"] = row["log"]
        if row["status"] == "start":
            st["started"] = datetime.fromisoformat(row["time"])
        else:
            st["seconds"] = float(row["seconds"] or 0)
    return states


def expected_minutes(scen: str, step: str, params: dict, states: dict) -> float:
    observed = [
        st["seconds"] / 60 for (_, sc, sp), st in states.items()
        if sc == scen and sp == step and st["status"] == "done"
    ]
    if observed:
        return statistics.mean(observed)
    if step != "train":
        return EVAL_MINUTES
    if scen in SCRATCH:
        epochs, full = int(params.get("EPOCHS_SCRATCH", 150)), 150
    else:
        epochs = sum(int(params.get(k, d)) for k, d in (("F1", 45), ("F2", 60), ("F3", 45)))
        full = 150
    return TRAIN_MINUTES_FULL.get(scen, 30) * epochs / full


def parse_train_log(log: Path) -> dict:
    """Estado do treino a partir do log do train.py/MMEngine."""
    info = {"phase": None, "phase_desc": "", "phase_epochs": None, "train": None,
            "val": [], "phase_results": {}, "selected": None, "last_line": ""}
    if not log.exists():
        return info
    text = log.read_text(errors="replace")
    for line in text.splitlines():
        if m := RE_PHASE_SINGLE.search(line):
            info.update(phase="única", phase_desc="treino em fase única",
                        phase_epochs=int(m.group(1)), val=[])
        elif m := RE_PHASE_TL.search(line):
            info.update(phase=m.group(1), phase_desc=m.group(2), val=[])
        elif (m := RE_PHASE_EPOCHS.search(line)) and info["phase"] not in (None, "única"):
            info["phase_epochs"] = int(m.group(1))
        elif m := RE_TRAIN.search(line):
            info["train"] = {
                "epoch": int(m.group(1)), "iter": int(m.group(2)), "iters": int(m.group(3)),
                "lr": float(m.group(4)), "eta": m.group(5), "mem_mb": int(m.group(6)),
                "loss": float(m.group(7)), "acc": float(m.group(8)),
            }
        elif m := RE_VAL.search(line):
            info["val"].append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
        elif m := RE_PHASE_RESULT.search(line):
            info["phase_results"][m.group(1)] = float(m.group(2))
        elif "[Fase 3] PULADA" in line:
            info["phase_results"]["3"] = None
        elif m := RE_SELECTION.search(line):
            info["selected"] = m.group(1)
        if line.strip():
            info["last_line"] = line.strip()
    return info


def resolve_log(run: Path, logged: str) -> Path:
    """O run_seeds.sh grava o caminho visto no container (runs/<nome>/...); aqui vira host."""
    path = Path(logged)
    try:
        return run / path.relative_to(Path("runs") / run.name)
    except ValueError:
        return path


def load_results(run: Path) -> dict:
    """{scenario: {seed: {'val': {...}, 'train': {...}}}} a partir dos JSONs do evaluate.py."""
    results: dict = {}
    for f in sorted(run.glob("seed_*/tables/finetuned_cenario_*_*.json")):
        m = re.match(r"finetuned_cenario_(.+)_(val|train)\.json", f.name)
        if not m:
            continue
        seed = f.parent.parent.name.removeprefix("seed_")
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        results.setdefault(m.group(1), {}).setdefault(seed, {})[m.group(2)] = data
    return results


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def summarize(results: dict, scenarios: list[str]) -> dict:
    summary = {}
    for scen in scenarios:
        per_seed = results.get(scen, {})
        vals = [r["val"] for r in per_seed.values() if "val" in r]
        gaps = [100 * (r["train"]["pck_02"] - r["val"]["pck_02"])
                for r in per_seed.values() if "val" in r and "train" in r]
        entry = {"n": len(vals)}
        for key in ("pck_02", "pdj_05", "oks", "oks_coco"):
            xs = [100 * v[key] for v in vals if key in v]
            entry[key] = dict(zip(("mean", "std"), mean_std(xs)))
        entry["mpjpe_2d_px"] = dict(zip(("mean", "std"),
                                        mean_std([v["mpjpe_2d_px"] for v in vals])))
        entry["gap_pp"] = dict(zip(("mean", "std"), mean_std(gaps)))
        entry["pck_02_per_seed"] = {s: 100 * r["val"]["pck_02"]
                                    for s, r in per_seed.items() if "val" in r}
        summary[scen] = entry
    return summary


def gpu_status() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,"
             "temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()[0]
        util, used, total, temp, power = (x.strip() for x in out.split(","))
        return (f"GPU {util:>3}% · {float(used)/1024:.1f}/{float(total)/1024:.1f} GB · "
                f"{temp}°C · {float(power):.0f} W")
    except (OSError, IndexError, ValueError, subprocess.SubprocessError):
        return None


def container_running() -> bool | None:
    try:
        out = subprocess.run(
            ["docker", "ps", "--filter", "ancestor=football-finetuning", "--format", "{{.ID}}"],
            capture_output=True, text=True, timeout=5,
        )
        return bool(out.stdout.strip()) if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def fmt_ms(entry: dict, digits: int = 1) -> str:
    mean, std = entry["mean"], entry["std"]
    if mean != mean:  # NaN
        return "—"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def render(run: Path, system: bool) -> str:
    params = read_params(run)
    progress = read_progress(run)
    states = step_states(progress)
    seeds = params.get("SEEDS", "").split()
    scenarios = params.get("SCENARIOS", "").split()
    steps = ["train", "eval_val", "eval_train"]
    plan = [(sd, sc, sp) for sd in seeds for sc in scenarios for sp in steps]
    now = datetime.now().astimezone()
    out: list[str] = []

    started = params.get("STARTED")
    finished = params.get("FINISHED")
    elapsed = ""
    if started:
        t0 = datetime.fromisoformat(started)
        t1 = datetime.fromisoformat(finished) if finished else now
        elapsed = fmt_dur((t1 - t0).total_seconds())

    out.append(bold(f"═══ {run.name} ═══") + dim(f"   commit {params.get('GIT_COMMIT', '?')[:7]}"))
    out.append(
        f"sementes: {' '.join(seeds)}  ·  cenários: {' '.join(scenarios)}  ·  "
        f"épocas: do zero {params.get('EPOCHS_SCRATCH')}, TL {params.get('F1')}/"
        f"{params.get('F2')}/{params.get('F3')}  ·  batch {params.get('BATCH')}"
    )

    # ── progresso geral ─────────────────────────────────────────────────────
    done = [k for k in plan if states.get(k, {}).get("status") == "done"]
    failed = [k for k in plan if states.get(k, {}).get("status") == "fail"]
    running = [k for k in plan if states.get(k, {}).get("status") == "start"]
    total_min = sum(expected_minutes(sc, sp, params, states) for _, sc, sp in plan) or 1
    done_min = sum(expected_minutes(sc, sp, params, states) for _, sc, sp in done + failed)
    remaining = total_min - done_min
    if running:
        st = states[running[0]]
        remaining -= min((now - st["started"]).total_seconds() / 60,
                         expected_minutes(running[0][1], running[0][2], params, states))
    frac = 1 - remaining / total_min
    status = green("CONCLUÍDO") if finished else (yellow("RODANDO") if running else dim("parado"))
    out.append("")
    out.append(
        f"{bold('PROGRESSO')}  {bar(frac)} {frac*100:5.1f}%   {status}   "
        f"decorrido {elapsed or '—'}"
        + ("" if finished else f"   · restante ~{fmt_dur(remaining * 60)}")
    )
    out.append(
        f"  etapas: {len(done)}/{len(plan)} concluídas"
        + (red(f"  ·  {len(failed)} com falha") if failed else "")
    )

    if system:
        g = gpu_status()
        ctr = container_running()
        line = "  " + (g or dim("GPU: nvidia-smi indisponível"))
        if ctr is not None:
            line += "   ·   container " + (green("ativo") if ctr else dim("parado"))
        out.append(line)

    # ── etapa em andamento ──────────────────────────────────────────────────
    if running:
        seed, scen, step = running[0]
        st = states[running[0]]
        log = resolve_log(run, st["log"])
        out.append("")
        out.append(
            bold("EM ANDAMENTO") + f"  seed {seed} · {cyan(scen)} ({LABELS.get(scen, '')}) · "
            f"{step} · {fmt_dur((now - st['started']).total_seconds())}"
        )
        if log.exists():
            age = (time.time() - log.stat().st_mtime) / 60
            if age > STALE_MINUTES:
                out.append(red(f"  ⚠ log sem atualização há {age:.0f} min"
                               " — o processo pode ter travado"))
        if step == "train":
            info = parse_train_log(log)
            if info["phase"]:
                n_phases = "1" if info["phase"] == "única" else "3"
                head = ("fase única" if info["phase"] == "única"
                        else f"fase {info['phase']}/{n_phases} ({info['phase_desc']})")
                tr = info["train"]
                if tr:
                    total_ep = info["phase_epochs"] or tr["epoch"]
                    out.append(
                        f"  {head} · época {tr['epoch']}/{total_ep} "
                        f"{bar(tr['epoch'] / total_ep, 20)} · iter {tr['iter']}/{tr['iters']}"
                    )
                    out.append(
                        f"  loss {tr['loss']:.4f} · acc_pose {tr['acc']:.3f} · lr {tr['lr']:.1e}"
                        f" · memória {tr['mem_mb']/1024:.1f} GB · ETA da fase {tr['eta']}"
                    )
                else:
                    out.append(f"  {head} · preparando (carregando modelo e dados)…")
                if info["val"]:
                    best = max(info["val"], key=lambda v: v[1])
                    recent = "  ".join(f"ep{e} {p*100:.1f}%" for e, p, _ in info["val"][-6:])
                    out.append(f"  PCK@0.2 val por época: {recent}")
                    out.append(f"  melhor da fase: {green(f'{best[1]*100:.2f}%')} (época {best[0]})"
                               f" · PDJ {best[2]*100:.1f}%")
            if info["phase_results"]:
                parts = []
                for ph in sorted(info["phase_results"]):
                    v = info["phase_results"][ph]
                    parts.append(f"F{ph} " + (dim("pulada") if v is None else f"{v*100:.2f}%"))
                sel = f" → escolhida: {info['selected']}" if info["selected"] else ""
                out.append(f"  fases concluídas (PCK estrito): {' · '.join(parts)}{sel}")
            if info["last_line"]:
                out.append(dim(f"  último log: {info['last_line'][:110]}"))
        elif log.exists():
            out.append(dim(f"  último log: {log.read_text(errors='replace').strip()[-110:]}"))

    # ── falhas ──────────────────────────────────────────────────────────────
    for key in failed:
        st = states[key]
        out.append(red(f"  ✗ seed {key[0]} · {key[1]} · {key[2]} falhou — log: {st['log']}"))

    # ── resultados por semente ──────────────────────────────────────────────
    results = load_results(run)
    out.append("")
    out.append(bold("RESULTADOS POR SEMENTE")
               + dim("   PCK@0.2 val (%)  ·  entre parênteses: gap treino−val (pp)"))
    out.append("  " + "seed".ljust(7) + "".join(sc.ljust(17) for sc in scenarios))
    for seed in seeds:
        cells = []
        for scen in scenarios:
            r = results.get(scen, {}).get(seed, {})
            state = states.get((seed, scen, "train"), {}).get("status")
            if "val" in r:
                v = r["val"]["pck_02"] * 100
                gap = ""
                if "train" in r:
                    gap = f" ({(r['train']['pck_02'] - r['val']['pck_02']) * 100:+.1f})"
                cells.append(f"{v:5.1f}{gap}".ljust(17))
            elif state == "start":
                cells.append(yellow("rodando".ljust(17)))
            elif state == "fail":
                cells.append(red("falhou".ljust(17)))
            else:
                cells.append(dim("·".ljust(17)))
        out.append("  " + seed.ljust(7) + "".join(cells))

    # ── média ± desvio ──────────────────────────────────────────────────────
    summary = summarize(results, scenarios)
    out.append("")
    out.append(bold("MÉDIA ± DESVIO PADRÃO") + dim("   (val; gap = treino − val)"))
    out.append("  " + "cenário".ljust(9) + "n".rjust(3) + "  " + "PCK@0.2".ljust(14)
               + "PDJ@0.5".ljust(14) + "OKS".ljust(14) + "OKS COCO".ljust(14)
               + "MPJPE px".ljust(14) + "gap pp")
    for scen in scenarios:
        e = summary[scen]
        out.append(
            "  " + scen.ljust(9) + str(e["n"]).rjust(3) + "  " + fmt_ms(e["pck_02"]).ljust(14)
            + fmt_ms(e["pdj_05"]).ljust(14) + fmt_ms(e["oks"]).ljust(14)
            + fmt_ms(e["oks_coco"]).ljust(14) + fmt_ms(e["mpjpe_2d_px"], 2).ljust(14)
            + fmt_ms(e["gap_pp"])
        )

    out.append("")
    out.append(dim(f"atualizado {now:%H:%M:%S} · Ctrl+C para sair (o treino continua rodando)"))
    return "\n".join(out)


def _nan_to_none(obj):
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, float) and obj != obj:
        return None
    return obj


def latest_run(root: Path) -> Path | None:
    runs = [p for p in root.glob("*") if (p / "progress.tsv").exists()]
    return max(runs, key=lambda p: (p / "progress.tsv").stat().st_mtime) if runs else None


def main() -> None:
    p = argparse.ArgumentParser(description="Painel das execuções de run_seeds.sh")
    p.add_argument("run", nargs="?", type=Path, default=None,
                   help="pasta da execução (padrão: a mais recente em ~/runs)")
    p.add_argument("--root", type=Path, default=Path.home() / "runs")
    p.add_argument("--interval", type=float, default=10.0, help="segundos entre atualizações")
    p.add_argument("--once", action="store_true", help="mostra uma vez e sai")
    p.add_argument("--no-system", action="store_true", help="não consulta GPU nem Docker")
    p.add_argument("--summary-json", type=Path, default=None,
                   help="também grava média/desvio por cenário neste JSON")
    args = p.parse_args()

    run = args.run or latest_run(args.root)
    if run is None or not (run / "progress.tsv").exists():
        sys.exit(f"nenhuma execução encontrada em {args.run or args.root}")

    if args.summary_json:
        scenarios = read_params(run).get("SCENARIOS", "").split()
        summary = summarize(load_results(run), scenarios)
        args.summary_json.write_text(json.dumps(_nan_to_none(summary), indent=2))

    if args.once:
        print(render(run, system=not args.no_system))
        return
    try:
        while True:
            frame = render(run, system=not args.no_system)
            sys.stdout.write("\033[H\033[2J" + frame + "\n")
            sys.stdout.flush()
            if "FINISHED" in read_params(run):
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
