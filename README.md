# Image Captioning — CNN+LSTM → +Attention → Transformer

A from-scratch image captioning system in PyTorch (see
`image_captioning_guide.md` for the full theory). Three decoders share one
pipeline, one training script and one evaluation protocol:

- **Phase 1 (done):** frozen ResNet50 encoder + LSTM decoder, teacher
  forcing, greedy + beam search, BLEU evaluation.
- **Phase 2 (done):** Bahdanau (soft) attention over 7×7 region features
  ("Show, Attend and Tell"), doubly stochastic regularization, attention
  heatmap visualization, `--resume` training.
- **Phase 3 (done):** Transformer decoder (`nn.TransformerDecoder`, 3
  layers, 8 heads) with sinusoidal positional encoding, causal + padding
  masks, cross-attention to image regions. Parallel training over time steps.
- **Phase 4 (done):** comparison table (`compare_models.py`), training
  curves (`plot_curves.py`), error analysis with failure categories
  (`error_analysis.py`).
- **Phase 5 (done):** Streamlit demo (`app.py`) and presentation deck
  (`presentation/presentation.pptx`).

## Project structure

```
image-captioning/
├── data/
│   ├── raw/                 # put Flickr8k here: images/ + captions.txt
│   └── processed/           # generated: vocab, splits, precomputed features
├── src/
│   ├── config.py            # ALL hyperparameters
│   ├── vocabulary.py        # cleaning, word<->id, encode/decode
│   ├── prepare_data.py      # step 1: parse, clean, split, build vocab
│   ├── extract_features.py  # step 2: precompute ResNet50 features
│   ├── dataset.py           # Dataset + padding collate_fn
│   ├── models/
│   │   ├── encoder.py       # frozen ResNet50 (global + grid features)
│   │   ├── decoder_lstm.py  # phase 1: LSTM decoder
│   │   ├── decoder_attention.py  # phase 2: Bahdanau attention + LSTMCell
│   │   └── decoder_transformer.py # phase 3: Transformer decoder
│   ├── train.py             # step 3: training loop (--model, --resume)
│   ├── inference.py         # step 4: caption any image (--model)
│   ├── evaluate.py          # step 5: BLEU-1..4 on test split (--model)
│   ├── visualize_attention.py    # phase 2: heatmap per generated word
│   ├── compare_models.py    # phase 4: comparison table (all models)
│   ├── plot_curves.py       # phase 4: training curves PNG
│   └── error_analysis.py    # phase 4: best/worst captions + failure flags
├── scripts/make_dummy_data.py   # tiny fake dataset for pipeline testing
├── app.py                   # phase 5: Streamlit demo (upload -> caption)
├── presentation/            # phase 5: slides (.pptx) + generator script
├── checkpoints/             # best model saved here
└── outputs/                 # training logs, eval results, figures
```

## Setup (your GPU machine)

```bash
pip install torch torchvision Pillow   # CUDA build: see pytorch.org/get-started
```

## Quick pipeline test (no dataset needed, ~2 min)

```bash
python scripts/make_dummy_data.py 60
cd src
python prepare_data.py
python extract_features.py --no-pretrained
python extract_features.py --grid --no-pretrained
python train.py --epochs 8 --batch-size 32
python train.py --model attention --epochs 30 --batch-size 32
python evaluate.py --no-pretrained-encoder
python visualize_attention.py ../data/raw/images/dummy_000.jpg --no-pretrained-encoder
```

✅ Already verified end-to-end (CPU): loss 4.4 → 0.63 in 8 epochs, inference
and BLEU all working. (`--no-pretrained` uses a random encoder to avoid
weight downloads — only for plumbing tests, captions will be poor.)

## Real training on Flickr8k

1. **Get the dataset** (easiest: Kaggle — `kaggle datasets download adityajn105/flickr8k`)
   and place it as:

```
data/raw/images/*.jpg        (8,091 images)
data/raw/captions.txt        (CSV "image,caption" — Kaggle format;
                              the classic TSV "img.jpg#0<TAB>caption" also works)
```

2. **Run the pipeline:**

```bash
cd src
python prepare_data.py            # vocab ~3k words, 80/10/10 split by image
python extract_features.py        # global features (phase 1), run ONCE
python extract_features.py --grid # region features 49x2048 (phase 2), ~3 GB

# --- phase 1: baseline ---
python train.py --model lstm
python evaluate.py --model lstm
python inference.py path/to/photo.jpg --model lstm

# --- phase 2: attention ---
python train.py --model attention            # add --resume to continue a run
python evaluate.py --model attention
python inference.py photo.jpg --model attention
python visualize_attention.py photo.jpg     # heatmap per word -> outputs/

# --- phase 3: transformer ---
python train.py --model transformer
python evaluate.py --model transformer
python inference.py photo.jpg --model transformer

# --- phase 4: comparison & analysis (after training + evaluating) ---
python compare_models.py        # table -> outputs/comparison.md
python plot_curves.py           # curves -> outputs/*.png
python error_analysis.py --model transformer   # report -> outputs/*.md

# --- phase 5: demo (from the project root) ---
pip install streamlit
streamlit run app.py            # upload photo -> caption (+ heatmaps)
```

Expected on Flickr8k: baseline **BLEU-4 ≈ 0.17–0.21**; attention
**≈ 0.21–0.25** with interpretable heatmaps; transformer **≈ 0.22–0.27**.
Training logs land in `outputs/training_log_<model>.csv` — plot train/val
curves for your report. Notes: the attention model trains slower per epoch
(a Python loop over time steps) and needs more epochs before beating the
baseline; the transformer trains in parallel over time steps (fast epochs)
but its beam search re-runs the decoder per prefix (slower inference —
fine at these lengths).

## Design decisions worth knowing

- **Features are precomputed** (`extract_features.py`) because the encoder is
  frozen — epochs are ~10× faster since the CNN never runs during training.
- **Loss ignores `<pad>`** (`ignore_index=0`) and teacher forcing uses
  `caption[:-1]` → predict `caption[1:]` — the two classic bugs, avoided.
- **Vocabulary is built from train captions only** (no leakage), words with
  frequency < 5 → `<unk>`.
- **Best checkpoint is saved, not last** — early stopping with patience 5.
- **Beam search is length-normalized** (α=0.7) so it doesn't prefer short captions.
- **Checkpoints are written atomically** (temp file + rename) and include the
  optimizer state, so `--resume` continues cleanly after any interruption.
- **Attention context is projected to embed_dim** before entering the LSTM —
  a raw 2048-d context drowns out the 256-d word embedding (scale mismatch).
- **Doubly stochastic regularization** (`attn_reg_lambda`) pushes the model
  to spread attention over the whole image across the caption.
- **Image memory is LayerNorm-ed** in the transformer — large-scale CNN
  features would otherwise saturate the cross-attention softmax (we measured
  this: without LayerNorm the model plateaus at unigram level).

## Tuning knobs (src/config.py)

| Param | Default | Notes |
|---|---|---|
| `embed_dim` / `hidden_dim` | 256 / 512 | bigger = better up to overfitting |
| `min_word_freq` | 5 | lower → bigger vocab, more `<unk>`-free |
| `batch_size` | 64 | raise if GPU allows |
| `lr` | 3e-4 | ReduceLROnPlateau halves it automatically |
| `beam_size` | 3 | 3–5 is the sweet spot |

## Presentation

`presentation/presentation.pptx` — 14 slides (French): problem, pipeline,
the three architectures, training, metrics, comparison, error analysis,
demo, lessons learned. Figures come from the test pipeline; regenerate them
on Flickr8k (`plot_curves.py`, `visualize_attention.py`) and update slides
7, 9 and 11 with real results. `make_slides.js` regenerates the deck
(`NODE_PATH=<npm modules> node make_slides.js`).

## Suggested demo flow (defense day)

1. Slides 1–12 (~10 min), 2. `streamlit run app.py` — caption a photo taken
in the room, toggle the heatmaps, 3. Slide 14 (lessons) and questions.
