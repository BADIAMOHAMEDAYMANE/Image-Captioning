import argparse
import math

import matplotlib
matplotlib.use("Agg")                    
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from config import cfg
from inference import load_model
from models.encoder import image_transform


@torch.no_grad()
def visualize(image_path, out_path, no_pretrained_encoder=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder, decoder, vocab = load_model(device, "attention",
                                         no_pretrained_encoder)

    img = Image.open(image_path).convert("RGB")
    feats = encoder(image_transform(img).unsqueeze(0).to(device),
                    return_grid=True)                      # (1, 49, 2048)
    ids, alphas = decoder.generate_greedy(feats, vocab.start_id,
                                          vocab.end_id, cfg.max_caption_len)
    words = [vocab.id2word[i] for i in ids]
    caption = " ".join(words)
    print(f"caption: {caption}")

    # display image (what the CNN actually saw: 224x224 center crop)
    disp = img.resize((224, 224))
    side = int(math.isqrt(cfg.n_regions))                  # 7

    n = len(words) + 1
    cols = min(5, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_1d(axes).ravel()

    axes[0].imshow(disp)
    axes[0].set_title("original", fontsize=11)
    for k, (word, alpha) in enumerate(zip(words, alphas), start=1):
        heat = alpha.reshape(side, side).numpy()
        heat = np.array(Image.fromarray(
            (heat / heat.max() * 255).astype(np.uint8)
        ).resize((224, 224), Image.BICUBIC)) / 255.0
        axes[k].imshow(disp)
        axes[k].imshow(heat, cmap="jet", alpha=0.5)
        axes[k].set_title(word, fontsize=12)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f'"{caption}"', fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--out", default=None)
    p.add_argument("--no-pretrained-encoder", action="store_true")
    args = p.parse_args()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    from pathlib import Path
    out = args.out or cfg.output_dir / f"attention_{Path(args.image).stem}.png"
    visualize(args.image, out, args.no_pretrained_encoder)
