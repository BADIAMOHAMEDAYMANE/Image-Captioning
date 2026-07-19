"""Phase 4: error analysis on the test split.

For one model:
  1. generates a caption for every test image (beam search)
  2. scores each against its 5 references (sentence BLEU-1)
  3. flags failure patterns:
       - repetition   (a bigram appears >= 2 times)
       - too_short    (< 4 words)
       - unk          (contains <unk>)
       - generic      (caption identical to >= 20% of other predictions)
  4. writes outputs/error_analysis_<model>.md with category counts,
     the K worst and K best examples.

Usage:
    python src/error_analysis.py --model attention [--limit 100] [--top 10]
"""
import argparse
import json
from collections import Counter

import torch
from PIL import Image

from config import cfg
from evaluate import corpus_bleu
from inference import load_model
from models.encoder import image_transform
from vocabulary import clean_caption


def sentence_bleu1(cand_tokens, refs_tokens):
    return corpus_bleu([cand_tokens], [refs_tokens], max_n=1)["BLEU-1"]


def flags_for(caption, all_preds):
    words = caption.split()
    flags = []
    bigrams = Counter(zip(words, words[1:]))
    if bigrams and bigrams.most_common(1)[0][1] >= 2:
        flags.append("repetition")
    if len(words) < 4:
        flags.append("too_short")
    if "<unk>" in caption:
        flags.append("unk")
    if all_preds.count(caption) >= max(2, int(0.2 * len(all_preds))):
        flags.append("generic")
    return flags


@torch.no_grad()
def main(model_name, limit, top, no_pretrained_encoder):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder, decoder, vocab = load_model(device, model_name,
                                         no_pretrained_encoder)
    grid = model_name in ("attention", "transformer")
    is_attn = model_name == "attention"

    with open(cfg.processed_dir / "captions.json") as f:
        captions = json.load(f)
    with open(cfg.processed_dir / "splits.json") as f:
        test_ids = json.load(f)["test"]
    if limit:
        test_ids = test_ids[:limit]

    results = []
    for k, img_id in enumerate(test_ids):
        img = Image.open(cfg.raw_dir / "images" / img_id).convert("RGB")
        feats = encoder(image_transform(img).unsqueeze(0).to(device),
                        return_grid=grid)
        out = decoder.generate_beam(feats, vocab.start_id, vocab.end_id,
                                    cfg.beam_size, cfg.max_caption_len)
        ids = out[0] if is_attn else out
        hyp = vocab.decode(ids)
        refs = [clean_caption(c) for c in captions[img_id]]
        score = sentence_bleu1(hyp.split(), [r.split() for r in refs])
        results.append({"image": img_id, "pred": hyp, "refs": refs,
                        "bleu1": round(score, 4)})
        print(f"\r{k + 1}/{len(test_ids)}", end="", flush=True)
    print()

    all_preds = [r["pred"] for r in results]
    for r in results:
        r["flags"] = flags_for(r["pred"], all_preds)

    # ---- report ----
    counts = Counter(f for r in results for f in r["flags"])
    mean_bleu = sum(r["bleu1"] for r in results) / len(results)
    by_score = sorted(results, key=lambda r: r["bleu1"])

    lines = [f"# Error analysis — {model_name}", "",
             f"test images: {len(results)}   "
             f"mean sentence BLEU-1: {mean_bleu:.4f}", "",
             "## Failure categories", ""]
    for name, n in counts.most_common():
        lines.append(f"- **{name}**: {n}/{len(results)} "
                     f"({100 * n / len(results):.0f}%)")
    if not counts:
        lines.append("- none detected")

    def block(title, items):
        out = ["", f"## {title}", ""]
        for r in items:
            out += [f"**{r['image']}** — BLEU-1 {r['bleu1']}"
                    + (f"  `{','.join(r['flags'])}`" if r["flags"] else ""),
                    f"- pred: {r['pred']}",
                    f"- ref:  {r['refs'][0]}", ""]
        return out

    lines += block(f"{top} worst captions", by_score[:top])
    lines += block(f"{top} best captions", by_score[-top:][::-1])

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.output_dir / f"error_analysis_{model_name}.md"
    out_path.write_text("\n".join(lines))
    print(f"mean BLEU-1 {mean_bleu:.4f} | flags: {dict(counts)}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["lstm", "attention", "transformer"],
                   default="lstm")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--no-pretrained-encoder", action="store_true")
    args = p.parse_args()
    main(args.model, args.limit, args.top, args.no_pretrained_encoder)
