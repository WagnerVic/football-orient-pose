#!/usr/bin/env python
"""Smoke test end-to-end do pipeline de fine-tuning (Cenário A).

Valida, num subconjunto pequeno e em 1 época, que: dataset carrega, modelo
constrói, treino roda, validação calcula PCK e o CheckpointHook salva o best.
Não é treino real — é verificação de fiação.

    python scripts/training/smoke_cenario_a.py [--batch-size 2] [--n-train 32] [--n-val 16]
"""

from __future__ import annotations

from mmpose.utils import register_all_modules

register_all_modules()

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from train import _build_runner, _find_best_checkpoint  # reutiliza o caminho real


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--n-train", type=int, default=32)
    p.add_argument("--n-val", type=int, default=16)
    p.add_argument("--work-dir", default="results/smoke_cenario_a")
    args = p.parse_args()

    overrides = {
        "work_dir": args.work_dir,
        "train_cfg.max_epochs": 1,
        "train_cfg.val_interval": 1,
        "train_dataloader.batch_size": args.batch_size,
        "val_dataloader.batch_size": args.batch_size,
        "train_dataloader.dataset.indices": args.n_train,
        "val_dataloader.dataset.indices": args.n_val,
        # workers menores p/ subconjunto pequeno
        "train_dataloader.num_workers": 2,
        "val_dataloader.num_workers": 2,
    }

    runner = _build_runner("configs/cenario_a.py", overrides)
    runner.train()

    ckpt = _find_best_checkpoint(args.work_dir)
    print(f"\n[SMOKE OK] best checkpoint salvo: {ckpt}")


if __name__ == "__main__":
    main()
