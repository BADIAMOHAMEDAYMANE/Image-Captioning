"""Phase 2 decoder: LSTM + Bahdanau (additive, soft) attention.
"Show, Attend and Tell" (Xu et al., 2015).

Differences vs the baseline decoder:
  - input is the GRID of region features (49, 2048), not one global vector
  - at every step, attention picks WHERE to look -> context vector c_t
  - LSTM input = [word embedding ; c_t]
  - returns attention weights (alphas) for visualization + regularization

Teacher forcing convention is the same:
    inputs  = caption[:, :-1]   targets = caption[:, 1:]
"""
import torch
import torch.nn as nn


class BahdanauAttention(nn.Module):
    """score(a_i, h) = v^T tanh(W_a a_i + W_h h)  -> softmax over 49 regions."""

    def __init__(self, feature_dim, hidden_dim, attn_dim):
        super().__init__()
        self.W_a = nn.Linear(feature_dim, attn_dim)   # project regions
        self.W_h = nn.Linear(hidden_dim, attn_dim)    # project decoder state
        self.v = nn.Linear(attn_dim, 1)

    def forward(self, features, h):
        """features (B, R, F); h (B, H) -> context (B, F), alpha (B, R)"""
        scores = self.v(torch.tanh(
            self.W_a(features) + self.W_h(h).unsqueeze(1))).squeeze(2)  # (B,R)
        alpha = torch.softmax(scores, dim=1)                            # (B,R)
        context = (features * alpha.unsqueeze(2)).sum(dim=1)            # (B,F)
        return context, alpha


class DecoderAttention(nn.Module):
    def __init__(self, feature_dim, embed_dim, hidden_dim, attn_dim,
                 vocab_size, dropout=0.3, pad_id=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.attention = BahdanauAttention(feature_dim, hidden_dim, attn_dim)
        # project the 2048-d context down to embed_dim BEFORE feeding the
        # LSTM: keeps word signal and image signal at comparable scales
        # (a raw 2048-d context drowns out the 256-d word embedding)
        self.ctx_proj = nn.Linear(feature_dim, embed_dim)
        # manual cell (not nn.LSTM): the input changes at every step
        self.lstm_cell = nn.LSTMCell(embed_dim * 2, hidden_dim)
        self.init_h = nn.Linear(feature_dim, hidden_dim)
        self.init_c = nn.Linear(feature_dim, hidden_dim)
        self.f_beta = nn.Linear(hidden_dim, feature_dim)  # attention gate
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def _init_state(self, features):
        mean_feat = features.mean(dim=1)                  # (B, F)
        return (torch.tanh(self.init_h(mean_feat)),
                torch.tanh(self.init_c(mean_feat)))

    def _step(self, token_emb, features, h, c):
        """One decoding step. Returns logits, new state, attention weights."""
        context, alpha = self.attention(features, h)
        gate = torch.sigmoid(self.f_beta(h))              # how much image now?
        context = self.ctx_proj(gate * context)           # (B, embed_dim)
        h, c = self.lstm_cell(torch.cat([token_emb, context], dim=1), (h, c))
        logits = self.fc(self.dropout(h))
        return logits, h, c, alpha

    def forward(self, features, captions):
        """features (B, R, F); captions (B, T).
        Returns logits (B, T-1, V) and alphas (B, T-1, R)."""
        B, T = captions.shape
        h, c = self._init_state(features)
        emb = self.embed(captions)                        # (B, T, E)
        logits_all, alphas_all = [], []
        for t in range(T - 1):                            # teacher forcing
            logits, h, c, alpha = self._step(emb[:, t], features, h, c)
            logits_all.append(logits)
            alphas_all.append(alpha)
        return torch.stack(logits_all, dim=1), torch.stack(alphas_all, dim=1)

    # ---------------- inference ----------------
    @torch.no_grad()
    def generate_greedy(self, features, start_id, end_id, max_len=35):
        """features (1, R, F) -> (ids, alphas list) for visualization."""
        h, c = self._init_state(features)
        token = torch.tensor([start_id], device=features.device)
        out_ids, out_alphas = [], []
        for _ in range(max_len):
            logits, h, c, alpha = self._step(self.embed(token), features, h, c)
            token = logits.argmax(-1)                     # (1,)
            if token.item() == end_id:
                break
            out_ids.append(token.item())
            out_alphas.append(alpha.squeeze(0).cpu())
        return out_ids, out_alphas

    @torch.no_grad()
    def generate_beam(self, features, start_id, end_id, beam_size=3,
                      max_len=35, length_norm=0.7):
        """Beam search. Returns (ids, alphas list) of the best sequence."""
        device = features.device
        h, c = self._init_state(features)
        # beam: (logp_sum, [ids], h, c, [alphas], finished)
        beams = [(0.0, [start_id], h.squeeze(0), c.squeeze(0), [], False)]
        for _ in range(max_len):
            candidates = []
            for logp, ids, bh, bc, alphas, done in beams:
                if done:
                    candidates.append((logp, ids, bh, bc, alphas, True))
                    continue
                token = torch.tensor([ids[-1]], device=device)
                logits, nh, nc, alpha = self._step(
                    self.embed(token), features, bh.unsqueeze(0),
                    bc.unsqueeze(0))
                log_probs = torch.log_softmax(logits[0], dim=-1)
                top_lp, top_ids = log_probs.topk(beam_size)
                for lp, tid in zip(top_lp.tolist(), top_ids.tolist()):
                    candidates.append((logp + lp, ids + [tid],
                                       nh.squeeze(0), nc.squeeze(0),
                                       alphas + [alpha.squeeze(0).cpu()],
                                       tid == end_id))
            candidates.sort(
                key=lambda b: b[0] / (len(b[1]) ** length_norm), reverse=True)
            beams = candidates[:beam_size]
            if all(b[5] for b in beams):
                break
        best = beams[0]
        ids = [t for t in best[1][1:] if t != end_id]
        alphas = best[4][: len(ids)]
        return ids, alphas
