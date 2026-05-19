"""Monitoramento de métricas por época para fine-tuning (EPIC 5)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float
    val_pdj: float | None = None
    val_pck: float | None = None
    val_oks: float | None = None


class TrainingMonitor:
    """Registra métricas por época e gera plots de loss e métricas."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.history: list[EpochRecord] = []

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_pdj: float | None = None,
        val_pck: float | None = None,
        val_oks: float | None = None,
    ) -> None:
        self.history.append(EpochRecord(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_pdj=val_pdj,
            val_pck=val_pck,
            val_oks=val_oks,
        ))

    def save(self, output_dir: str | Path) -> None:
        """Salva history.json e gera loss_curve.png e metrics_curve.png."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "history.json").write_text(
            json.dumps([asdict(r) for r in self.history], indent=2)
        )
        self._plot_loss_curve(out)
        self._plot_metrics_curve(out)

    def _plot_loss_curve(self, out: Path) -> None:
        import matplotlib.pyplot as plt

        epochs = [r.epoch for r in self.history]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, [r.train_loss for r in self.history], label="train")
        ax.plot(epochs, [r.val_loss for r in self.history], label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{self.model_name} — Loss")
        ax.legend()
        fig.savefig(out / "loss_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_metrics_curve(self, out: Path) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        for attr, label in [("val_pdj", "PDJ"), ("val_pck", "PCK"), ("val_oks", "OKS")]:
            pairs = [
                (r.epoch, getattr(r, attr))
                for r in self.history
                if getattr(r, attr) is not None
            ]
            if pairs:
                xs, ys = zip(*pairs)
                ax.plot(xs, ys, label=label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_title(f"{self.model_name} — Métricas por Época")
        ax.legend()
        fig.savefig(out / "metrics_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
