import csv
import json
import random
from collections import defaultdict

from config import cfg
from vocabulary import Vocabulary, clean_caption


def load_raw_captions(path):
    """Parse captions.txt -> {image_id: [raw captions]}. Handles the Kaggle
    CSV format (image,caption) and the classic TSV format (image#N\tcaption)."""
    caps = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        first = f.readline()
        f.seek(0)
        if "\t" in first:  # classic format: "1000268201.jpg#0\tA child ..."
            for line in f:
                if not line.strip():
                    continue
                img, cap = line.split("\t", 1)
                caps[img.split("#")[0]].append(cap.strip())
        else:  # Kaggle CSV format with header "image,caption"
            reader = csv.reader(f)
            header = next(reader)
            assert header[0].lower() == "image", f"unexpected header: {header}"
            for row in reader:
                if len(row) >= 2:
                    caps[row[0]].append(",".join(row[1:]).strip())
    return dict(caps)


def main():
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_captions(cfg.raw_dir / "captions.txt")
    print(f"images with captions: {len(raw)}")

    # keep only images that actually exist on disk
    img_dir = cfg.raw_dir / "images"
    raw = {k: v for k, v in raw.items() if (img_dir / k).exists()}
    print(f"images found on disk:  {len(raw)}")

    # clean
    captions = {img: [clean_caption(c) for c in caps if clean_caption(c)]
                for img, caps in raw.items()}
    captions = {k: v for k, v in captions.items() if v}

    # split by IMAGE (never let captions of one image leak across splits)
    images = sorted(captions)
    random.Random(cfg.seed).shuffle(images)
    n = len(images)
    n_test = int(n * cfg.test_fraction)
    n_val = int(n * cfg.val_fraction)
    splits = {
        "test": images[:n_test],
        "val": images[n_test:n_test + n_val],
        "train": images[n_test + n_val:],
    }
    print({k: len(v) for k, v in splits.items()})

    # vocabulary from TRAIN captions only (no leakage)
    train_caps = [c for img in splits["train"] for c in captions[img]]
    vocab = Vocabulary.build(train_caps, cfg.min_word_freq)
    print(f"vocabulary size: {len(vocab)}")

    with open(cfg.processed_dir / "captions.json", "w") as f:
        json.dump(captions, f)
    with open(cfg.processed_dir / "splits.json", "w") as f:
        json.dump(splits, f)
    vocab.save(cfg.processed_dir / "vocab.json")
    print("saved captions.json, splits.json, vocab.json")


if __name__ == "__main__":
    main()
