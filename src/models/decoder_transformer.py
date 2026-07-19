"""Phase 3 decoder: Transformer decoder over CNN region features.

Architecture (see guide, section 7):
  - image memory: 49 region vectors projected 2048 -> d_model
  - token embeddings + sinusoidal positional encoding
  - N decoder layers, each: masked self-attention (words), cross-attention
    (words -> image regions), feed-forward
  - causal mask: position t cannot see positions > t (no cheating)

Training is PARALLEL over time steps (one forward for the whole caption),
unlike the LSTM decoders which loop step by step.
"""
import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Classic sin/cos positional encoding from 'Attention Is All You Need'."""

    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))       # (1, max_len, d)

    def forward(self, x):                                  # x (B, T, d)
        return x + self.pe[:, : x.size(1)]


class DecoderTransformer(nn.Module):
    def __init__(self, feature_dim, d_model, n_heads, n_layers, ff_dim,
                 vocab_size, dropout=0.1, pad_id=0, max_len=100):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        # image -> memory, with LayerNorm: CNN features can have a large
        # scale, which would saturate cross-attention softmaxes
        self.feat_proj = nn.Sequential(nn.Linear(feature_dim, d_model),
                                       nn.LayerNorm(d_model))
        layer = nn.TransformerDecoderLayer(
            d_model, n_heads, ff_dim, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, n_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, vocab_size)

    def _decode(self, tokens, memory):
        """tokens (B, T) -> logits (B, T, V). memory (B, R, d_model)."""
        T = tokens.size(1)
        x = self.embed(tokens) * math.sqrt(self.d_model)   # scale (standard)
        x = self.dropout(self.pos_enc(x))
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool,
                                       device=tokens.device), diagonal=1)
        pad_mask = tokens == self.pad_id                   # (B, T)
        out = self.decoder(x, memory, tgt_mask=causal,
                           tgt_key_padding_mask=pad_mask)
        return self.fc(out)

    def forward(self, features, captions):
        """features (B, R, 2048); captions (B, T) full sequences.
        Returns logits (B, T-1, V) predicting captions[:, 1:]."""
        memory = self.feat_proj(features)                  # (B, R, d_model)
        return self._decode(captions[:, :-1], memory)

    # ---------------- inference ----------------
    @torch.no_grad()
    def generate_greedy(self, features, start_id, end_id, max_len=35):
        """features (1, R, 2048) -> list of token ids (no specials)."""
        memory = self.feat_proj(features)
        tokens = torch.tensor([[start_id]], device=features.device)
        for _ in range(max_len):
            logits = self._decode(tokens, memory)          # (1, T, V)
            nxt = logits[0, -1].argmax().item()
            if nxt == end_id:
                break
            tokens = torch.cat(
                [tokens, torch.tensor([[nxt]], device=features.device)], dim=1)
        return tokens[0, 1:].tolist()

    @torch.no_grad()
    def generate_beam(self, features, start_id, end_id, beam_size=3,
                      max_len=35, length_norm=0.7):
        """Beam search (re-runs the decoder on each candidate prefix —
        simple and correct; caching K/V would be the optimized version)."""
        device = features.device
        memory = self.feat_proj(features)
        beams = [(0.0, [start_id], False)]                 # (logp, ids, done)
        for _ in range(max_len):
            candidates = []
            for logp, ids, done in beams:
                if done:
                    candidates.append((logp, ids, True))
                    continue
                tokens = torch.tensor([ids], device=device)
                logits = self._decode(tokens, memory)
                log_probs = torch.log_softmax(logits[0, -1], dim=-1)
                top_lp, top_ids = log_probs.topk(beam_size)
                for lp, tid in zip(top_lp.tolist(), top_ids.tolist()):
                    candidates.append(
                        (logp + lp, ids + [tid], tid == end_id))
            candidates.sort(
                key=lambda b: b[0] / (len(b[1]) ** length_norm), reverse=True)
            beams = candidates[:beam_size]
            if all(b[2] for b in beams):
                break
        best = beams[0][1]
        return [t for t in best[1:] if t != end_id]
