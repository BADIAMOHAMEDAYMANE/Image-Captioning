import argparse
import json

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config import cfg
from models.encoder import EncoderCNN, image_transform


class ImageOnlyDataset(Dataset):
    def __init__(self, image_ids):
        self.image_ids = image_ids

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, i):
        img_id = self.image_ids[i]
        img = Image.open(cfg.raw_dir / "images" / img_id).convert("RGB")
        return img_id, image_transform(img)


def main(grid: bool, no_pretrained: bool):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.features_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg.processed_dir / "captions.json") as f:
        image_ids = sorted(json.load(f))

    encoder = EncoderCNN(pretrained=not no_pretrained).to(device)
    loader = DataLoader(ImageOnlyDataset(image_ids), batch_size=64,
                        num_workers=cfg.num_workers)

    done = 0
    for ids, images in loader:
        feats = encoder(images.to(device), return_grid=grid).cpu()
        for img_id, f in zip(ids, feats):
            suffix = ".grid.pt" if grid else ".pt"
            torch.save(f.clone(), cfg.features_dir / (img_id + suffix))
        done += len(ids)
        print(f"\r{done}/{len(image_ids)}", end="", flush=True)
    print(f"\nsaved features for {done} images -> {cfg.features_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--grid", action="store_true",
                   help="save (49,2048) region features (needed for attention)")
    p.add_argument("--no-pretrained", action="store_true",
                   help="random-init encoder (testing only, no download)")
    args = p.parse_args()
    main(args.grid, args.no_pretrained)
