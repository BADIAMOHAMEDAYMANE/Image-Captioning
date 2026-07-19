"""Phase 5: Streamlit demo — upload a photo, get a caption (+ heatmaps).

Run from the project root:
    streamlit run app.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))          # import src/*

from config import cfg                                    # noqa: E402
from inference import load_model                          # noqa: E402
from models.decoder_attention import DecoderAttention     # noqa: E402
from models.encoder import image_transform                # noqa: E402

st.set_page_config(page_title="Image Captioning", page_icon="🖼️",
                   layout="centered")
st.title("🖼️ Image Captioning")
st.caption("CNN + LSTM · +Attention · Transformer — projet from scratch")

# ---------------- sidebar ----------------
available = [m for m in ["lstm", "attention", "transformer"]
             if (cfg.checkpoint_dir / f"{m}_best.pt").exists()]
if not available:
    st.error("Aucun checkpoint trouvé dans checkpoints/ — entraînez un "
             "modèle d'abord (`python src/train.py`).")
    st.stop()

with st.sidebar:
    st.header("Réglages")
    model_name = st.selectbox("Modèle", available,
                              index=len(available) - 1)
    use_beam = st.toggle("Beam search", value=True)
    beam_size = st.slider("Largeur du beam", 2, 8, cfg.beam_size,
                          disabled=not use_beam)
    test_mode = st.toggle("Mode test (encodeur aléatoire)", value=False,
                          help="Uniquement pour vérifier le pipeline sans "
                               "télécharger les poids ImageNet.")

device = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource(show_spinner="Chargement du modèle…")
def get_model(name: str, test: bool):
    return load_model(device, name, no_pretrained_encoder=test)


encoder, decoder, vocab = get_model(model_name, test_mode)

# ---------------- main ----------------
file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])
if file is None:
    st.info("Uploadez une photo pour générer sa description.")
    st.stop()

img = Image.open(file).convert("RGB")
st.image(img, use_container_width=True)

is_attn = isinstance(decoder, DecoderAttention)
grid = model_name in ("attention", "transformer")

with st.spinner("Génération…"):
    feats = encoder(image_transform(img).unsqueeze(0).to(device),
                    return_grid=grid)
    with torch.no_grad():
        if use_beam:
            out = decoder.generate_beam(feats, vocab.start_id, vocab.end_id,
                                        beam_size, cfg.max_caption_len)
        else:
            out = decoder.generate_greedy(feats, vocab.start_id,
                                          vocab.end_id, cfg.max_caption_len)
    ids, alphas = out if is_attn else (out, None)
    caption = vocab.decode(ids)

st.success(f"**« {caption} »**")
st.caption(f"modèle : {model_name} · "
           f"{'beam ' + str(beam_size) if use_beam else 'greedy'} · {device}")

# ---------------- attention heatmaps ----------------
if is_attn and alphas and st.toggle("Voir où le modèle regarde", value=True):
    words = caption.split()
    side = int(math.isqrt(cfg.n_regions))                  # 7
    disp = np.asarray(img.resize((224, 224))).astype(float) / 255.0

    cols_per_row = 4
    for start in range(0, len(words), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, (word, alpha) in zip(
                cols, list(zip(words, alphas))[start:start + cols_per_row]):
            heat = alpha.reshape(side, side).numpy()
            heat = np.array(Image.fromarray(
                (heat / heat.max() * 255).astype(np.uint8)
            ).resize((224, 224), Image.BICUBIC)) / 255.0
            # red overlay proportional to attention
            overlay = disp * 0.55 + np.stack(
                [heat, heat * 0.2, heat * 0.1], axis=-1) * 0.45
            col.image(np.clip(overlay, 0, 1), caption=word,
                      use_container_width=True)
