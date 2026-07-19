"""PyTorch Dataset over (precomputed feature, encoded caption) pairs.

One sample = one (image, caption) pair, so each image appears ~5x per epoch.
"""
import json

import torch
from torch.utils.data import Dataset

from config import cfg
from vocabulary import Vocabulary


class CaptionDataset(Dataset):
    def __init__(self, split: str, grid: bool = False):
        """grid=True loads (49, 2048) region features (attention model);
        grid=False loads the (2048,) global vector (baseline)."""
        self.suffix = ".grid.pt" if grid else ".pt"
        with open(cfg.processed_dir / "captions.json") as f:
            captions = json.load(f)
        with open(cfg.processed_dir / "splits.json") as f:
            image_ids = json.load(f)[split]
        self.vocab = Vocabulary.load(cfg.processed_dir / "vocab.json")
        # flatten: [(image_id, encoded_caption_ids), ...]
        self.samples = [
            (img, self.vocab.encode(c, cfg.max_caption_len))
            for img in image_ids for c in captions[img]
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img_id, cap_ids = self.samples[i]
        feats = torch.load(cfg.features_dir / (img_id + self.suffix),
                           weights_only=True)
        return feats, torch.tensor(cap_ids, dtype=torch.long)


def collate_fn(batch):
    """Pad captions to the longest in the batch. pad_id is 0 by construction."""
    feats, caps = zip(*batch)
    feats = torch.stack(feats)
    max_len = max(len(c) for c in caps)
    padded = torch.zeros(len(caps), max_len, dtype=torch.long)  # 0 == <pad>
    for i, c in enumerate(caps):
        padded[i, : len(c)] = c
    return feats, padded
