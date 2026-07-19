"""Phase 4: build the model comparison table.

Aggregates, for every trained model:
  - parameter count
  - best epoch + best val loss (from the checkpoint)
  - average seconds/epoch (from training_log_<model>.csv)
  - BLEU-1..4 (from eval_results_<model>.json, produced by evaluate.py)

Writes outputs/comparison.md (+ .json) and prints the table.

Usage:
    python src/compare_models.py
"""
import csv
import json

import torch

from config import cfg
from train import build_model
from vocabulary import Vocabulary

MODELS = ["lstm", "attention", "transformer"]


def collect(model_name, vocab):
    row = {"model": model_name}

    # parameters
    model = build_model(model_name, vocab)
    row["params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)

    # checkpoint
    ckpt_path = cfg.checkpoint_dir / f"{model_name}_best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        row["best_epoch"] = ckpt.get("epoch", "?")
        vl = ckpt.get("val_loss")
        row["val_loss"] = round(vl, 3) if vl is not None else None
    else:
        row["best_epoch"] = row["val_loss"] = None   # not trained yet

    # epoch time
    log_path = cfg.output_dir / f"training_log_{model_name}.csv"
    if log_path.exists():
        with open(log_path) as f:
            secs = [float(r["sec"]) for r in csv.DictReader(f)]
        row["sec_per_epoch"] = round(sum(secs) / len(secs), 1) if secs else None
    else:
        row["sec_per_epoch"] = None

    # BLEU
    eval_path = cfg.output_dir / f"eval_results_{model_name}.json"
    if eval_path.exists():
        with open(eval_path) as f:
            scores = json.load(f)
        for k in range(1, 5):
            row[f"BLEU-{k}"] = round(scores.get(f"BLEU-{k}", 0), 4)
    else:
        for k in range(1, 5):
            row[f"BLEU-{k}"] = None
    return row


def fmt(v):
    return "—" if v is None else str(v)


def main():
    vocab = Vocabulary.load(cfg.processed_dir / "vocab.json")
    rows = [collect(m, vocab) for m in MODELS]

    cols = ["model", "params_M", "best_epoch", "val_loss", "sec_per_epoch",
            "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]
    header = ["Model", "Params (M)", "Best epoch", "Val loss", "s/epoch",
              "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]

    lines = ["# Model comparison", "",
             "| " + " | ".join(header) + " |",
             "|" + "---|" * len(header)]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    lines += ["",
              "Missing values (—): train the model (`train.py --model X`) "
              "and/or run `evaluate.py --model X` first."]
    table = "\n".join(lines)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "comparison.md").write_text(table)
    with open(cfg.output_dir / "comparison.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(table)
    print(f"\nsaved -> {cfg.output_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
