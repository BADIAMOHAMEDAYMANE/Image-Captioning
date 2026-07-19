import argparse
import csv
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import cfg
from dataset import CaptionDataset, collate_fn
from models.decoder_attention import DecoderAttention
from models.decoder_lstm import DecoderLSTM
from models.decoder_transformer import DecoderTransformer

GRID_MODELS = ("attention", "transformer")   # need (49, 2048) region features


def build_model(name, vocab):
    if name == "attention":
        return DecoderAttention(cfg.feature_dim, cfg.embed_dim,
                                cfg.hidden_dim, cfg.attn_dim, len(vocab),
                                cfg.dropout, vocab.pad_id)
    if name == "transformer":
        return DecoderTransformer(cfg.feature_dim, cfg.tf_d_model,
                                  cfg.tf_heads, cfg.tf_layers, cfg.tf_ff_dim,
                                  len(vocab), cfg.tf_dropout, vocab.pad_id)
    return DecoderLSTM(cfg.feature_dim, cfg.embed_dim, cfg.hidden_dim,
                       len(vocab), cfg.dropout, vocab.pad_id)


def run_epoch(model, loader, criterion, vocab_size, device,
              optimizer=None, scaler=None):
    """One pass over `loader`. Trains if optimizer is given, else evaluates.
    Returns the average cross-entropy per token (comparable across models)."""
    training = optimizer is not None
    model.train(training)
    total_loss, total_tokens = 0.0, 0
    amp = scaler is not None

    with torch.set_grad_enabled(training):
        for feats, caps in loader:
            feats, caps = feats.to(device), caps.to(device)
            with torch.autocast(device_type="cuda", enabled=amp):
                out = model(feats, caps)
                if isinstance(out, tuple):                 # attention model
                    logits, alphas = out                   # alphas (B,T-1,R)
                    ce = criterion(logits.reshape(-1, vocab_size),
                                   caps[:, 1:].reshape(-1))
                    # doubly stochastic regularization ("Show, Attend and
                    # Tell"): encourage each region to receive ~1 of total
                    # attention over the caption -> model looks everywhere
                    reg = ((1.0 - alphas.sum(dim=1)) ** 2).mean()
                    loss = ce + cfg.attn_reg_lambda * reg
                else:                                      # baseline
                    logits = out                           # (B, T-1, V)
                    ce = criterion(logits.reshape(-1, vocab_size),
                                   caps[:, 1:].reshape(-1))
                    loss = ce
            if training:
                optimizer.zero_grad()
                if amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    optimizer.step()
            n_tokens = (caps[:, 1:] != 0).sum().item()         # non-pad targets
            total_loss += ce.item() * n_tokens
            total_tokens += n_tokens
    return total_loss / max(total_tokens, 1)


def main(model_name, epochs, batch_size, resume=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | model: {model_name}")
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)

    grid = model_name in GRID_MODELS       # region features needed?
    train_ds = CaptionDataset("train", grid=grid)
    val_ds = CaptionDataset("val", grid=grid)
    vocab = train_ds.vocab
    print(f"train samples: {len(train_ds)}  val: {len(val_ds)}  "
          f"vocab: {len(vocab)}")

    train_loader = DataLoader(train_ds, batch_size, shuffle=True,
                              collate_fn=collate_fn,
                              num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False,
                            collate_fn=collate_fn,
                            num_workers=cfg.num_workers)

    model = build_model(model_name, vocab).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"parameters: {n_params:.1f}M")
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=2)
    scaler = torch.amp.GradScaler() if device == "cuda" else None

    best_val, bad_epochs, start_epoch = float("inf"), 0, 1
    ckpt_path = cfg.checkpoint_dir / f"{model_name}_best.pt"
    if resume and ckpt_path.exists():                # continue training
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        best_val = ckpt.get("val_loss", float("inf"))
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"resumed from epoch {start_epoch - 1} "
              f"(val loss {best_val:.4f})")

    log_path = cfg.output_dir / f"training_log_{model_name}.csv"
    if not (resume and log_path.exists()):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss",
                                    "lr", "sec"])

    for epoch in range(start_epoch, start_epoch + epochs):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, criterion, len(vocab),
                               device, optimizer, scaler)
        val_loss = run_epoch(model, val_loader, criterion, len(vocab), device)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0
        print(f"epoch {epoch:3d} | train {train_loss:.4f} | "
              f"val {val_loss:.4f} | lr {lr:.2e} | {dt:.0f}s")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.4f}",
                                    f"{val_loss:.4f}", f"{lr:.2e}",
                                    f"{dt:.0f}"])

        if val_loss < best_val:                      # save BEST, not last
            best_val, bad_epochs = val_loss, 0
            # atomic save: never leaves a half-written checkpoint if the
            # process is killed mid-save
            tmp = ckpt_path.with_suffix(".tmp")
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "val_loss": val_loss, "epoch": epoch}, tmp)
            tmp.replace(ckpt_path)
            print("  ✓ new best checkpoint saved")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:           # early stopping
                print(f"early stopping (no improvement for "
                      f"{cfg.patience} epochs)")
                break

    print(f"done. best val loss: {best_val:.4f}  "
          f"(perplexity {torch.tensor(best_val).exp():.1f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["lstm", "attention", "transformer"],
                   default="lstm")
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--batch-size", type=int, default=cfg.batch_size)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--attn-reg", type=float, default=cfg.attn_reg_lambda)
    p.add_argument("--resume", action="store_true",
                   help="continue from the saved checkpoint")
    args = p.parse_args()
    cfg.lr = args.lr
    cfg.attn_reg_lambda = args.attn_reg
    main(args.model, args.epochs, args.batch_size, args.resume)
