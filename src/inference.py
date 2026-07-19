
import argparse

import torch
from PIL import Image

from config import cfg
from models.decoder_attention import DecoderAttention
from models.decoder_lstm import DecoderLSTM
from models.decoder_transformer import DecoderTransformer
from models.encoder import EncoderCNN, image_transform
from vocabulary import Vocabulary

GRID_MODELS = (DecoderAttention, DecoderTransformer)


def load_model(device, model_name="lstm", no_pretrained_encoder=False):
    vocab = Vocabulary.load(cfg.processed_dir / "vocab.json")
    if model_name == "attention":
        decoder = DecoderAttention(cfg.feature_dim, cfg.embed_dim,
                                   cfg.hidden_dim, cfg.attn_dim, len(vocab),
                                   cfg.dropout, vocab.pad_id).to(device)
    elif model_name == "transformer":
        decoder = DecoderTransformer(cfg.feature_dim, cfg.tf_d_model,
                                     cfg.tf_heads, cfg.tf_layers,
                                     cfg.tf_ff_dim, len(vocab),
                                     cfg.tf_dropout, vocab.pad_id).to(device)
    else:
        decoder = DecoderLSTM(cfg.feature_dim, cfg.embed_dim, cfg.hidden_dim,
                              len(vocab), cfg.dropout, vocab.pad_id).to(device)
    ckpt = torch.load(cfg.checkpoint_dir / f"{model_name}_best.pt",
                      map_location=device, weights_only=True)
    decoder.load_state_dict(ckpt["model"])
    decoder.eval()
    encoder = EncoderCNN(pretrained=not no_pretrained_encoder).to(device)
    return encoder, decoder, vocab


@torch.no_grad()
def caption_image(image_path, encoder, decoder, vocab, device,
                  beam_size=3, greedy=False):
    """Returns (caption str, alphas or None).
    alphas: list of (49,) attention weights, one per generated word
    (attention model only)."""
    is_attn = isinstance(decoder, DecoderAttention)
    grid = isinstance(decoder, GRID_MODELS)
    img = Image.open(image_path).convert("RGB")
    feats = encoder(image_transform(img).unsqueeze(0).to(device),
                    return_grid=grid)   # (1,2048) or (1,49,2048)
    if greedy:
        out = decoder.generate_greedy(feats, vocab.start_id, vocab.end_id,
                                      cfg.max_caption_len)
    else:
        out = decoder.generate_beam(feats, vocab.start_id, vocab.end_id,
                                    beam_size, cfg.max_caption_len)
    ids, alphas = out if is_attn else (out, None)
    return vocab.decode(ids), alphas


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--model", choices=["lstm", "attention", "transformer"],
                   default="lstm")
    p.add_argument("--beam", type=int, default=cfg.beam_size)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--no-pretrained-encoder", action="store_true",
                   help="testing only")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder, decoder, vocab = load_model(device, args.model,
                                         args.no_pretrained_encoder)
    caption, _ = caption_image(args.image, encoder, decoder, vocab, device,
                               args.beam, args.greedy)
    print(caption)
