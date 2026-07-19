"""Baseline decoder: image global vector initializes an LSTM that generates
the caption word by word ("Show and Tell" style).

Teacher forcing convention:
    inputs  = caption[:, :-1]   (<start> w1 ... wN)
    targets = caption[:, 1:]    (w1 ... wN <end>)
"""
import torch
import torch.nn as nn


class DecoderLSTM(nn.Module):
    def __init__(self, feature_dim, embed_dim, hidden_dim, vocab_size,
                 dropout=0.3, pad_id=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.init_h = nn.Linear(feature_dim, hidden_dim)  # image -> h0
        self.init_c = nn.Linear(feature_dim, hidden_dim)  # image -> c0
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def _init_state(self, features):
        h0 = torch.tanh(self.init_h(features)).unsqueeze(0)  # (1, B, H)
        c0 = torch.tanh(self.init_c(features)).unsqueeze(0)
        return h0, c0

    def forward(self, features, captions):
        """features (B, feature_dim); captions (B, T) full sequences.
        Returns logits (B, T-1, vocab) predicting captions[:, 1:]."""
        state = self._init_state(features)
        emb = self.embed(captions[:, :-1])                   # (B, T-1, E)
        out, _ = self.lstm(emb, state)
        return self.fc(self.dropout(out))

    # ---------------- inference ----------------
    @torch.no_grad()
    def generate_greedy(self, features, start_id, end_id, max_len=35):
        """features (1, feature_dim) -> list of token ids (no <start>)."""
        state = self._init_state(features)
        token = torch.tensor([[start_id]], device=features.device)
        out_ids = []
        for _ in range(max_len):
            emb = self.embed(token)                          # (1, 1, E)
            out, state = self.lstm(emb, state)
            token = self.fc(out).argmax(-1)                  # (1, 1)
            if token.item() == end_id:
                break
            out_ids.append(token.item())
        return out_ids

    @torch.no_grad()
    def generate_beam(self, features, start_id, end_id, beam_size=3,
                      max_len=35, length_norm=0.7):
        """Beam search. Returns best sequence of token ids (no specials)."""
        device = features.device
        state = self._init_state(features)
        # each beam: (log_prob_sum, [ids], (h, c), finished)
        beams = [(0.0, [start_id], state, False)]
        for _ in range(max_len):
            candidates = []
            for logp, ids, st, done in beams:
                if done:
                    candidates.append((logp, ids, st, True))
                    continue
                emb = self.embed(torch.tensor([[ids[-1]]], device=device))
                out, new_st = self.lstm(emb, st)
                log_probs = torch.log_softmax(self.fc(out)[0, -1], dim=-1)
                top_lp, top_ids = log_probs.topk(beam_size)
                for lp, tid in zip(top_lp.tolist(), top_ids.tolist()):
                    candidates.append(
                        (logp + lp, ids + [tid], new_st, tid == end_id))
            # keep the best `beam_size`, scores normalized by length
            candidates.sort(
                key=lambda b: b[0] / (len(b[1]) ** length_norm), reverse=True)
            beams = candidates[:beam_size]
            if all(b[3] for b in beams):
                break
        best = beams[0][1]
        return [t for t in best[1:] if t != end_id]          # strip specials
