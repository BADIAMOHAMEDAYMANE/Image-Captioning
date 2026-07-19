"""Central configuration. Every hyperparameter lives here."""
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # --- paths ---
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"            # images/ + captions.txt
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    features_dir: Path = PROJECT_ROOT / "data" / "processed" / "features"
    checkpoint_dir: Path = PROJECT_ROOT / "checkpoints"
    output_dir: Path = PROJECT_ROOT / "outputs"

    # --- vocabulary ---
    min_word_freq: int = 5          # keep words appearing >= this many times
    max_caption_len: int = 35       # truncate longer captions (incl. <start>/<end>)

    # --- data split (by image, reproducible) ---
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int = 42

    # --- encoder ---
    encoder_name: str = "resnet50"  # torchvision model
    feature_dim: int = 2048         # ResNet50 global vector size

    # --- decoder (LSTM baseline) ---
    embed_dim: int = 256
    hidden_dim: int = 512
    dropout: float = 0.3

    # --- attention decoder (phase 2) ---
    attn_dim: int = 512             # attention MLP size
    attn_reg_lambda: float = 1.0    # doubly stochastic regularization weight
    n_regions: int = 49             # 7x7 CNN grid

    # --- transformer decoder (phase 3) ---
    tf_d_model: int = 512
    tf_heads: int = 8
    tf_layers: int = 3
    tf_ff_dim: int = 2048
    tf_dropout: float = 0.1

    # --- training ---
    batch_size: int = 64
    epochs: int = 30
    lr: float = 3e-4
    grad_clip: float = 5.0
    patience: int = 5               # early stopping (epochs without val improvement)
    num_workers: int = 2

    # --- inference ---
    beam_size: int = 3


cfg = Config()
