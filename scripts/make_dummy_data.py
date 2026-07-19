"""Generate a tiny fake dataset to verify the whole pipeline end-to-end.

Creates 40 images of colored shapes with matching captions, so the model can
actually LEARN something in a few epochs (color/shape words correlate with
pixels). Not meant to produce a good model — only to prove the code works.
"""
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "data" / "raw" / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {"red": (220, 40, 40), "blue": (40, 80, 220),
          "green": (40, 180, 80), "yellow": (230, 210, 50)}
SHAPES = ["circle", "square"]

rng = random.Random(0)
lines = ["image,caption"]

n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
for i in range(n):
    color = rng.choice(list(COLORS))
    shape = rng.choice(SHAPES)
    img = Image.new("RGB", (256, 256), (240, 240, 240))
    d = ImageDraw.Draw(img)
    box = [64, 64, 192, 192]
    if shape == "circle":
        d.ellipse(box, fill=COLORS[color])
    else:
        d.rectangle(box, fill=COLORS[color])
    name = f"dummy_{i:03d}.jpg"
    img.save(IMG_DIR / name)
    # 5 caption variants per image, like real datasets
    caps = [
        f"a {color} {shape} on a white background",
        f"there is a big {color} {shape}",
        f"a large {color} {shape} in the center",
        f"one {color} {shape} is shown",
        f"the picture shows a {color} {shape}",
    ]
    for c in caps:
        lines.append(f"{name},{c}")

(ROOT / "data" / "raw" / "captions.txt").write_text("\n".join(lines))
print(f"created {n} dummy images + captions.txt")
