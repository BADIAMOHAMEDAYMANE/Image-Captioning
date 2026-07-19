"""Phase 4: plot training curves for every model that has a log.

Reads outputs/training_log_<model>.csv and produces:
  - outputs/training_curves.png   (one subplot per model: train vs val)
  - outputs/val_loss_comparison.png  (all models on one axis)

Usage:
    python src/plot_curves.py
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import cfg

MODELS = ["lstm", "attention", "transformer"]
COLORS = {"lstm": "tab:blue", "attention": "tab:orange",
          "transformer": "tab:green"}


def read_log(model):
    path = cfg.output_dir / f"training_log_{model}.csv"
    if not path.exists():
        return None
    epochs, train, val = [], [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            epochs.append(int(r["epoch"]))
            train.append(float(r["train_loss"]))
            val.append(float(r["val_loss"]))
    return epochs, train, val


def main():
    logs = {m: read_log(m) for m in MODELS}
    logs = {m: v for m, v in logs.items() if v}
    if not logs:
        print("no training logs found — train a model first")
        return

    # figure 1: per-model train/val subplots
    fig, axes = plt.subplots(1, len(logs), figsize=(5 * len(logs), 4),
                             squeeze=False)
    for ax, (m, (ep, tr, va)) in zip(axes[0], logs.items()):
        ax.plot(ep, tr, label="train", color=COLORS[m])
        ax.plot(ep, va, label="val", color=COLORS[m], linestyle="--")
        best = min(range(len(va)), key=lambda i: va[i])
        ax.scatter([ep[best]], [va[best]], color=COLORS[m], zorder=5,
                   label=f"best val {va[best]:.3f}")
        ax.set_title(m)
        ax.set_xlabel("epoch")
        ax.set_ylabel("cross-entropy / token")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Training curves (teacher-forced loss)")
    fig.tight_layout()
    out1 = cfg.output_dir / "training_curves.png"
    fig.savefig(out1, dpi=120)
    print(f"saved -> {out1}")

    # figure 2: validation loss, all models together
    fig2, ax = plt.subplots(figsize=(7, 4.5))
    for m, (ep, _, va) in logs.items():
        ax.plot(ep, va, label=m, color=COLORS[m])
    ax.set_title("Validation loss comparison")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val cross-entropy / token")
    ax.legend()
    ax.grid(alpha=0.3)
    fig2.tight_layout()
    out2 = cfg.output_dir / "val_loss_comparison.png"
    fig2.savefig(out2, dpi=120)
    print(f"saved -> {out2}")


if __name__ == "__main__":
    main()
