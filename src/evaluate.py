"""Step 5: evaluate the model on the test split.

Computes corpus BLEU-1..4 (self-contained implementation, no extra deps).
For METEOR/CIDEr/SPICE later, install pycocoevalcap and plug in here.

Usage:
    python src/evaluate.py [--model attention] [--greedy] [--limit 100]
"""
import argparse
import json
import math
from collections import Counter

import torch

from config import cfg
from inference import load_model
from models.encoder import image_transform
from PIL import Image
from vocabulary import clean_caption


# ---------------- corpus BLEU (standard, with brevity penalty) --------------
def ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(candidates, references_list, max_n=4):
    """candidates: [[tok,...]]; references_list: [[[tok,...], x5], ...]"""
    weights = [1.0 / max_n] * max_n
    p_num = [0] * max_n
    p_den = [0] * max_n
    cand_len, ref_len = 0, 0
    for cand, refs in zip(candidates, references_list):
        cand_len += len(cand)
        # closest reference length (standard BLEU rule)
        ref_len += min((abs(len(r) - len(cand)), len(r)) for r in refs)[1]
        for n in range(1, max_n + 1):
            cand_ng = ngrams(cand, n)
            max_ref = Counter()
            for r in refs:
                for ng, c in ngrams(r, n).items():
                    max_ref[ng] = max(max_ref[ng], c)
            p_num[n - 1] += sum(min(c, max_ref[ng])
                                for ng, c in cand_ng.items())
            p_den[n - 1] += max(sum(cand_ng.values()), 1)
    bp = 1.0 if cand_len > ref_len else math.exp(1 - ref_len / max(cand_len, 1))
    scores = {}
    for k in range(1, max_n + 1):
        precisions = [p_num[i] / p_den[i] if p_den[i] else 0 for i in range(k)]
        if min(precisions) > 0:
            s = math.exp(sum(math.log(p) for p in precisions) / k)
        else:
            s = 0.0
        scores[f"BLEU-{k}"] = bp * s
    return scores


# ---------------- evaluation loop ----------------
@torch.no_grad()
def main(model_name, greedy, limit, no_pretrained_encoder):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder, decoder, vocab = load_model(device, model_name,
                                         no_pretrained_encoder)
    is_attn = model_name == "attention"
    grid = model_name in ("attention", "transformer")

    with open(cfg.processed_dir / "captions.json") as f:
        captions = json.load(f)
    with open(cfg.processed_dir / "splits.json") as f:
        test_ids = json.load(f)["test"]
    if limit:
        test_ids = test_ids[:limit]

    cands, refs_list, samples = [], [], []
    for k, img_id in enumerate(test_ids):
        img = Image.open(cfg.raw_dir / "images" / img_id).convert("RGB")
        feats = encoder(image_transform(img).unsqueeze(0).to(device),
                        return_grid=grid)
        if greedy:
            out = decoder.generate_greedy(feats, vocab.start_id, vocab.end_id,
                                          cfg.max_caption_len)
        else:
            out = decoder.generate_beam(feats, vocab.start_id, vocab.end_id,
                                        cfg.beam_size, cfg.max_caption_len)
        ids = out[0] if is_attn else out       # attention also returns alphas
        hyp = vocab.decode(ids)
        cands.append(hyp.split())
        refs_list.append([clean_caption(c).split() for c in captions[img_id]])
        if k < 5:
            samples.append((img_id, hyp, captions[img_id][0]))
        print(f"\r{k + 1}/{len(test_ids)}", end="", flush=True)
    print()

    scores = corpus_bleu(cands, refs_list)
    for name, val in scores.items():
        print(f"{name}: {val:.4f}")
    print("\nsample predictions:")
    for img_id, hyp, ref in samples:
        print(f"  {img_id}\n    pred: {hyp}\n    ref:  {ref}")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.output_dir / f"eval_results_{model_name}.json", "w") as f:
        json.dump(scores, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["lstm", "attention", "transformer"],
                   default="lstm")
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="eval on first N only")
    p.add_argument("--no-pretrained-encoder", action="store_true")
    args = p.parse_args()
    main(args.model, args.greedy, args.limit, args.no_pretrained_encoder)
