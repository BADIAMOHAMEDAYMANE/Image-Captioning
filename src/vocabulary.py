import json
import re
from collections import Counter

PAD, START, END, UNK = "<pad>", "<start>", "<end>", "<unk>"
SPECIALS = [PAD, START, END, UNK]  # ids 0,1,2,3


def clean_caption(text: str) -> str:
    """Lowercase, strip punctuation/digits, collapse spaces."""
    text = text.lower()
    text = re.sub(r"[^a-z ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # drop single-letter noise tokens except 'a'
    return " ".join(w for w in text.split() if len(w) > 1 or w == "a")


class Vocabulary:
    def __init__(self, word2id=None):
        self.word2id = word2id or {}
        self.id2word = {i: w for w, i in self.word2id.items()}

    # ---------- construction ----------
    @classmethod
    def build(cls, captions, min_freq: int):
        """captions: iterable of cleaned caption strings."""
        counter = Counter(w for c in captions for w in c.split())
        words = [w for w, n in counter.most_common() if n >= min_freq]
        word2id = {w: i for i, w in enumerate(SPECIALS + words)}
        return cls(word2id)

    # ---------- persistence ----------
    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.word2id, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))

    # ---------- encoding / decoding ----------
    def encode(self, caption: str, max_len: int):
        """cleaned caption -> [<start>, w1, ..., wN, <end>] ids, truncated."""
        ids = [self.word2id.get(w, self.word2id[UNK]) for w in caption.split()]
        ids = ids[: max_len - 2]
        return [self.word2id[START]] + ids + [self.word2id[END]]

    def decode(self, ids) -> str:
        """ids -> sentence, stopping at <end>, skipping specials."""
        words = []
        for i in ids:
            w = self.id2word.get(int(i), UNK)
            if w == END:
                break
            if w not in (PAD, START):
                words.append(w)
        return " ".join(words)

    def __len__(self):
        return len(self.word2id)

    @property
    def pad_id(self):
        return self.word2id[PAD]

    @property
    def start_id(self):
        return self.word2id[START]

    @property
    def end_id(self):
        return self.word2id[END]
