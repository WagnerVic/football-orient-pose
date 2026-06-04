#!/usr/bin/env python
"""Orquestrador de fine-tuning RTMPose-X nos 4 cenários da matriz 2×2.

Cenários A/B (from scratch): uma única fase de treinamento.
Cenários C/D (transfer learning): 3 fases sequenciais de progressive unfreezing.

Uso:
    # Cenário A — from scratch, sem augmentation
    python scripts/train.py --cenario A [--epochs 50]

    # Cenário C — TL, sem augmentation, 3 fases
    python scripts/train.py --cenario C [--epochs-fase1 15] [--epochs-fase2 20] [--epochs-fase3 15]

    # Controlar threshold para ativar fase 3 (delta PCK mínimo)
    python scripts/train.py --cenario C --delta-pck 0.05

Saída:
    results/checkpoints/cenario_<X>/best_PCK.pth   — melhor checkpoint final
    results/checkpoints/cenario_<X>/fase*/         — checkpoints intermediários (TL)
"""

from __future__ import annotations

from mmpose.utils import register_all_modules
register_all_modules()  # registra TopdownPoseEstimator, RTMCCHead, CSPNeXt, etc.
import argparse
import glob
import os
import shutil
import sys
from pathlib import Path

# Garante que o src/ do projeto está no path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

# O MMEngine Runner usa GPU automaticamente quando disponível; não há override
# por-run simples (daí o --device foi removido). Este device serve só para o
# eval de gating da fase 3.
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tuning RTMPose-X — Épico 1")
    p.add_argument(
        "--cenario", required=True, choices=["A", "B", "C", "D"],
        help="Cenário da matriz 2×2 (A=from scratch/sem aug, B=from scratch/aug, "
             "C=TL/sem aug, D=TL/aug)",
    )
    # Épocas para A/B
    p.add_argument("--epochs", type=int, default=50, help="Épocas totais (A/B)")
    # Épocas por fase para C/D
    p.add_argument("--epochs-fase1", type=int, default=15, help="Épocas fase 1 — head only (C/D)")
    p.add_argument("--epochs-fase2", type=int, default=20, help="Épocas fase 2 — unfreeze top (C/D)")
    p.add_argument("--epochs-fase3", type=int, default=15, help="Épocas fase 3 — condicional (C/D)")
    p.add_argument(
        "--delta-pck", type=float, default=0.05,
        help="Δ PCK mínimo (fase2-fase1) para ativar fase 3 (default: 0.05 = 5pp)",
    )
    p.add_argument(
        "--batch-size", type=int, default=None,
        help="Sobrescreve o batch_size do train/val dataloader (útil p/ GPUs pequenas/smoke test)",
    )
    p.add_argument(
        "--val-interval", type=int, default=None,
        help="Sobrescreve val_interval (épocas entre validações; útil p/ treinos curtos)",
    )
    p.add_argument(
        "--work-dir", default=None,
        help="Diretório raiz de checkpoints (default: results/checkpoints/cenario_<X>)",
    )
    return p.parse_args()


def _build_runner(config_path: str, overrides: dict):
    """Constrói um MMPose Runner com config + sobrescritas programáticas."""
    from mmengine.config import Config
    from mmengine.runner import Runner

    cfg = Config.fromfile(config_path)

    # Aplica sobrescritas (frozen_stages, load_from, max_epochs, optim_wrapper, work_dir)
    for key, value in overrides.items():
        if "." in key:
            parts = key.split(".")
            obj = cfg
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
        else:
            setattr(cfg, key, value)

    return Runner.from_cfg(cfg)


def _find_best_checkpoint(work_dir: str) -> str:
    """Retorna o path do melhor checkpoint salvo pelo CheckpointHook."""
    # CheckpointHook salva 'best_PCK_epoch_N.pth' e um symlink 'best_PCK.pth'
    # Tenta o symlink/arquivo canônico primeiro
    candidate = Path(work_dir) / "best_PCK.pth"
    if candidate.exists():
        return str(candidate.resolve())

    # Fallback: procura arquivos best_*.pth mais recentes
    matches = sorted(
        glob.glob(str(Path(work_dir) / "best_*.pth")),
        key=os.path.getmtime,
        reverse=True,
    )
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Nenhum checkpoint best_*.pth encontrado em {work_dir}. "
        "Verifique se o treinamento completou ao menos uma época de validação."
    )


def _scheduler_override(max_epochs: int) -> list[dict]:
    """param_scheduler proporcional ao nº de épocas da fase.

    Sem isto, o warmup (LinearLR) e o cosseno (CosineAnnealingLR) ficam presos
    nas épocas hardcoded do config (ex.: 5/50) e não acompanham ``--epochs`` nem
    as fases do TL (15/20/15) — o LR fica inconsistente entre cenários e
    contamina a comparação. Aqui o warmup é ~10% das épocas e o cosseno anela
    de ponta a ponta até ``eta_min``.
    """
    warmup = max(1, min(5, round(max_epochs * 0.1)))
    if warmup >= max_epochs:
        warmup = max(1, max_epochs - 1)
    return [
        dict(type="LinearLR", start_factor=1e-3, begin=0, end=warmup,
             by_epoch=True, convert_to_iter_based=True),
        dict(type="CosineAnnealingLR", eta_min=1e-6, begin=warmup, end=max_epochs,
             by_epoch=True, convert_to_iter_based=True),
    ]


def _strict_pck_on_val(checkpoint: str, config_path: str, data_dir: Path,
                       split_config: Path, device: str) -> float:
    """PCK@0.2 ESTRITO (``compute_pck`` do projeto) no val — a métrica do artigo.

    Usado na decisão da fase 3. Substitui o parsing do log, que devolvia a PCK
    LENIENTE do ``PCKAccuracy(norm_item='bbox')`` (escala ~0.9), incompatível
    com ``--delta-pck`` (pensado na escala estrita ~0.2).
    """
    from evaluate import run_evaluation
    res = run_evaluation(
        checkpoint_path=checkpoint, config_path=config_path, split="val",
        data_dir=data_dir, split_config=split_config,
        output_dir=Path("results/tables"), device=device,
    )
    return float(res["pck_02"])


def _run_from_scratch(
    cenario: str, work_dir: str, epochs: int,
    batch_size: int | None = None, val_interval: int | None = None,
) -> str:
    """Treina cenário A/B (from scratch, fase única). Retorna path do best checkpoint."""
    config_path = f"configs/cenario_{cenario.lower()}.py"
    print(f"\n{'='*60}")
    print(f"[Cenário {cenario}] From scratch — {epochs} épocas")
    print(f"  Config:   {config_path}")
    print(f"  work_dir: {work_dir}")
    print(f"{'='*60}")

    overrides = {
        "work_dir": work_dir,
        "train_cfg.max_epochs": epochs,
        "param_scheduler": _scheduler_override(epochs),
    }
    if batch_size is not None:
        overrides["train_dataloader.batch_size"] = batch_size
        overrides["val_dataloader.batch_size"] = batch_size
    if val_interval is not None:
        overrides["train_cfg.val_interval"] = val_interval
    runner = _build_runner(config_path, overrides)
    runner.train()

    best_ckpt = _find_best_checkpoint(work_dir)
    print(f"\n[Cenário {cenario}] Treinamento concluído. Best checkpoint: {best_ckpt}")
    return best_ckpt


def _build_tl_optim_override(phase: int) -> dict:
    """Retorna o optim_wrapper para cada fase do TL."""
    if phase == 1:
        # Backbone completamente congelado — só o head treina com LR alto
        return dict(
            type="OptimWrapper",
            optimizer=dict(type="AdamW", lr=1e-3, weight_decay=0.05),
        )
    if phase == 2:
        # Head: LR=1e-4, backbone (stages 2-3): LR=1e-5 (mult=0.1)
        return dict(
            type="OptimWrapper",
            optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.05),
            paramwise_cfg=dict(
                custom_keys={"backbone": dict(lr_mult=0.1)},
            ),
        )
    # phase == 3: Head: LR=1e-4, backbone (stage 1-3): LR=1e-6 (mult=0.01)
    return dict(
        type="OptimWrapper",
        optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.05),
        paramwise_cfg=dict(
            custom_keys={"backbone": dict(lr_mult=0.01)},
        ),
    )


def _run_transfer_learning(
    cenario: str,
    work_dir: str,
    epochs_fase1: int,
    epochs_fase2: int,
    epochs_fase3: int,
    delta_pck: float,
    batch_size: int | None = None,
    val_interval: int | None = None,
) -> str:
    """Treina cenário C/D com 3 fases de progressive unfreezing.

    Retorna o path do melhor checkpoint final.
    """
    config_path = f"configs/cenario_{cenario.lower()}.py"

    def _bs(overrides: dict) -> dict:
        if batch_size is not None:
            overrides["train_dataloader.batch_size"] = batch_size
            overrides["val_dataloader.batch_size"] = batch_size
        if val_interval is not None:
            overrides["train_cfg.val_interval"] = val_interval
        return overrides

    from mmengine.config import Config
    base_cfg = Config.fromfile(config_path)
    coco_checkpoint = getattr(base_cfg, "COCO_CHECKPOINT", "checkpoints/rtmpose-x_coco.pth")
    data_dir = Path(getattr(base_cfg, "data_root", "data/3dsp"))
    split_config = Path(getattr(base_cfg, "split_file", "configs/split.json"))

    # ── Fase 1: backbone completamente congelado, treina só o head ──────────
    work_dir_f1 = str(Path(work_dir) / "fase1")
    epochs_total_f1 = epochs_fase1

    print(f"\n{'='*60}")
    print(f"[Cenário {cenario}] Fase 1 — backbone congelado (frozen_stages=4)")
    print(f"  load_from: {coco_checkpoint}")
    print(f"  Épocas: {epochs_total_f1}  |  work_dir: {work_dir_f1}")
    print(f"{'='*60}")

    overrides_f1 = {
        "work_dir": work_dir_f1,
        "train_cfg.max_epochs": epochs_total_f1,
        "param_scheduler": _scheduler_override(epochs_total_f1),
        "model.backbone.frozen_stages": 4,
        "load_from": coco_checkpoint,
        "optim_wrapper": _build_tl_optim_override(1),
    }
    runner1 = _build_runner(config_path, _bs(overrides_f1))
    runner1.train()

    ckpt_f1 = _find_best_checkpoint(work_dir_f1)
    pck_f1 = _strict_pck_on_val(ckpt_f1, config_path, data_dir, split_config, _DEVICE)
    print(f"\n[Fase 1] Best checkpoint: {ckpt_f1}  |  PCK@0.2 (estrito) = {pck_f1:.4f}")

    # ── Fase 2: descongelar stages 2-3, LR diferenciado ─────────────────────
    work_dir_f2 = str(Path(work_dir) / "fase2")
    epochs_total_f2 = epochs_fase2

    print(f"\n{'='*60}")
    print(f"[Cenário {cenario}] Fase 2 — stages 0-1 congelados (frozen_stages=2)")
    print(f"  load_from: {ckpt_f1}")
    print(f"  Épocas: {epochs_total_f2}  |  work_dir: {work_dir_f2}")
    print(f"{'='*60}")

    overrides_f2 = {
        "work_dir": work_dir_f2,
        "train_cfg.max_epochs": epochs_total_f2,
        "param_scheduler": _scheduler_override(epochs_total_f2),
        "model.backbone.frozen_stages": 2,
        "load_from": ckpt_f1,
        "optim_wrapper": _build_tl_optim_override(2),
    }
    runner2 = _build_runner(config_path, _bs(overrides_f2))
    runner2.train()

    ckpt_f2 = _find_best_checkpoint(work_dir_f2)
    pck_f2 = _strict_pck_on_val(ckpt_f2, config_path, data_dir, split_config, _DEVICE)
    print(f"\n[Fase 2] Best checkpoint: {ckpt_f2}  |  PCK@0.2 (estrito) = {pck_f2:.4f}")

    delta = pck_f2 - pck_f1
    print(f"\n[Δ PCK fase2−fase1] = {delta:.4f} (threshold = {delta_pck})")

    # ── Fase 3 (condicional) ─────────────────────────────────────────────────
    final_ckpt = ckpt_f2
    if delta > delta_pck:
        work_dir_f3 = str(Path(work_dir) / "fase3")
        epochs_total_f3 = epochs_fase3

        print(f"\n{'='*60}")
        print(f"[Cenário {cenario}] Fase 3 — stage 0 congelado (frozen_stages=1) [ATIVADA]")
        print(f"  load_from: {ckpt_f2}")
        print(f"  Épocas: {epochs_total_f3}  |  work_dir: {work_dir_f3}")
        print(f"{'='*60}")

        overrides_f3 = {
            "work_dir": work_dir_f3,
            "train_cfg.max_epochs": epochs_total_f3,
            "param_scheduler": _scheduler_override(epochs_total_f3),
            "model.backbone.frozen_stages": 1,
            "load_from": ckpt_f2,
            "optim_wrapper": _build_tl_optim_override(3),
        }
        runner3 = _build_runner(config_path, _bs(overrides_f3))
        runner3.train()

        ckpt_f3 = _find_best_checkpoint(work_dir_f3)
        pck_f3 = _strict_pck_on_val(ckpt_f3, config_path, data_dir, split_config, _DEVICE)
        print(f"\n[Fase 3] Best checkpoint: {ckpt_f3}  |  PCK@0.2 (estrito) = {pck_f3:.4f}")
        final_ckpt = ckpt_f3
    else:
        print(f"[Fase 3] PULADA (Δ PCK {delta:.4f} ≤ {delta_pck})")

    # Copia o melhor checkpoint para o diretório raiz do cenário
    dest = str(Path(work_dir) / "best_PCK.pth")
    shutil.copy2(final_ckpt, dest)
    print(f"\n[Cenário {cenario}] Treinamento completo. Best checkpoint copiado para: {dest}")
    return dest


def main() -> None:
    args = _parse_args()
    cenario = args.cenario.upper()

    work_dir = args.work_dir or f"results/checkpoints/cenario_{cenario}"
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    if cenario in ("A", "B"):
        _run_from_scratch(
            cenario=cenario,
            work_dir=work_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            val_interval=args.val_interval,
        )
    else:  # C ou D
        _run_transfer_learning(
            cenario=cenario,
            work_dir=work_dir,
            epochs_fase1=args.epochs_fase1,
            epochs_fase2=args.epochs_fase2,
            epochs_fase3=args.epochs_fase3,
            delta_pck=args.delta_pck,
            batch_size=args.batch_size,
            val_interval=args.val_interval,
        )


if __name__ == "__main__":
    main()
