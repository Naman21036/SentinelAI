import json
import re
import string
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_LENGTH = 100
VOCAB_SIZE = 20_000
EMBED_DIM = 128
LSTM_HIDDEN = 128
ATTN_HEADS = 4
DROPOUT = 0.3


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[" + re.escape(string.punctuation) + "]", " ", text)
    text = re.sub(r"\d+", "", text)
    return re.sub(r"\s+", " ", text).strip()


class WordTokenizer:
    PAD = 0
    OOV = 1

    def __init__(self, num_words: int = VOCAB_SIZE):
        self.num_words = num_words
        self.word2idx: dict[str, int] = {"<PAD>": 0, "<OOV>": 1}

    def fit(self, texts: list[str]) -> None:
        counter: Counter = Counter()
        for t in texts:
            counter.update(t.split())
        for word, _ in counter.most_common(self.num_words - 2):
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)

    def encode(self, text: str, max_len: int = MAX_LENGTH) -> list[int]:
        tokens = text.split()[:max_len]
        ids = [self.word2idx.get(tok, self.OOV) for tok in tokens]
        ids += [self.PAD] * (max_len - len(ids))
        return ids

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"word2idx": self.word2idx, "num_words": self.num_words}, f)

    @classmethod
    def load(cls, path: str) -> "WordTokenizer":
        with open(path) as f:
            d = json.load(f)
        tok = cls(d["num_words"])
        tok.word2idx = {k: int(v) for k, v in d["word2idx"].items()}
        return tok


class HateSpeechClassifier(nn.Module):

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = EMBED_DIM,
        lstm_hidden: int = LSTM_HIDDEN,
        attn_heads: int = ATTN_HEADS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embed_dim, lstm_hidden, batch_first=True, bidirectional=True)
        lstm_out = lstm_hidden * 2
        self.attention = nn.MultiheadAttention(lstm_out, attn_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(lstm_out)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_out, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.dropout(self.embedding(x))
        lstm_out, _ = self.bilstm(emb)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        x = self.norm(lstm_out + attn_out)
        x = x.mean(dim=1)
        x = F.relu(self.fc1(self.dropout(x)))
        return self.fc2(x).squeeze(-1)
