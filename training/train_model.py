import json
import os

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from services.classifier import (
    ATTN_HEADS,
    DROPOUT,
    EMBED_DIM,
    LSTM_HIDDEN,
    MAX_LENGTH,
    VOCAB_SIZE,
    HateSpeechClassifier,
    WordTokenizer,
    clean_text,
)

BATCH_SIZE = 64
EPOCHS = 15
LR = 1e-3
PATIENCE = 4

RAW_DATA = "data/raw_data.csv"
IMBALANCED_DATA = "data/imbalanced_data.csv"
ARTIFACTS_DIR = "artifacts/model"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pt")
VOCAB_PATH = os.path.join(ARTIFACTS_DIR, "vocab.json")
CONFIG_PATH = os.path.join(ARTIFACTS_DIR, "config.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TweetDataset(Dataset):
    def __init__(self, texts: list, labels: list, tokenizer: WordTokenizer):
        self.seqs = [torch.tensor(tokenizer.encode(t), dtype=torch.long) for t in texts]
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.seqs[idx], self.labels[idx]


def load_data() -> pd.DataFrame:
    dfs = []

    raw = pd.read_csv(RAW_DATA)
    raw.drop(
        columns=["Unnamed: 0", "count", "hate_speech", "offensive_language", "neither"],
        errors="ignore",
        inplace=True,
    )
    raw["label"] = raw["class"].apply(lambda x: 1 if x in [0, 1] else 0)
    dfs.append(raw[["tweet", "label"]].dropna())

    imb = pd.read_csv(IMBALANCED_DATA)
    imb.drop(columns=["id"], errors="ignore", inplace=True)
    dfs.append(imb[["tweet", "label"]].dropna())

    df = pd.concat(dfs, ignore_index=True).dropna()
    df["tweet"] = df["tweet"].apply(clean_text)
    return df[df["tweet"].str.len() > 0].reset_index(drop=True)


def train_epoch(model, loader, optimizer, criterion) -> float:
    model.train()
    total_loss = 0.0
    for seqs, labels in loader:
        seqs, labels = seqs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(seqs), labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader) -> tuple[float, float]:
    model.eval()
    preds, trues = [], []
    for seqs, labels in loader:
        probs = torch.sigmoid(model(seqs.to(DEVICE))).cpu().numpy()
        preds.extend((probs > 0.5).astype(int))
        trues.extend(labels.numpy().astype(int))
    return accuracy_score(trues, preds), f1_score(trues, preds)


def main() -> None:
    print(f"Device: {DEVICE}\n")

    print("Loading data...")
    df = load_data()
    print(f"Total samples : {len(df)}")
    print(f"Label counts  :\n{df['label'].value_counts().to_string()}\n")

    X, y = df["tweet"].to_numpy(), df["label"].to_numpy().astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Building vocabulary...")
    tokenizer = WordTokenizer(VOCAB_SIZE)
    tokenizer.fit(X_train)
    print(f"Vocabulary size: {len(tokenizer.word2idx)}\n")

    train_ds = TweetDataset(X_train, y_train, tokenizer)
    test_ds = TweetDataset(X_test, y_test, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=0)

    model = HateSpeechClassifier(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        lstm_hidden=LSTM_HIDDEN,
        attn_heads=ATTN_HEADS,
        dropout=DROPOUT,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}\n")

    pos_weight = torch.tensor(
        [(y_train == 0).sum() / max((y_train == 1).sum(), 1)], dtype=torch.float32
    ).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    best_f1 = 0.0
    best_state: dict = {}
    no_improve = 0

    print("Training...\n")
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion)
        acc, f1 = evaluate(model, test_loader)
        scheduler.step(f1)

        marker = " [best]" if f1 > best_f1 else ""
        print(f"Epoch {epoch:02d}/{EPOCHS}  loss={loss:.4f}  acc={acc:.4f}  f1={f1:.4f}{marker}")

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODEL_PATH)
    tokenizer.save(VOCAB_PATH)

    with open(CONFIG_PATH, "w") as f:
        json.dump(
            {
                "vocab_size": VOCAB_SIZE,
                "embed_dim": EMBED_DIM,
                "lstm_hidden": LSTM_HIDDEN,
                "attn_heads": ATTN_HEADS,
                "dropout": DROPOUT,
                "max_length": MAX_LENGTH,
            },
            f,
            indent=2,
        )

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for seqs, labels in test_loader:
            probs = torch.sigmoid(model(seqs.to(DEVICE))).cpu().numpy()
            all_preds.extend((probs > 0.5).astype(int))
            all_labels.extend(labels.numpy().astype(int))

    print(f"\nBest F1: {best_f1:.4f}")
    print(classification_report(all_labels, all_preds, target_names=["Safe", "Hate/Toxic"]))
    print(f"Model      -> {MODEL_PATH}")
    print(f"Vocabulary -> {VOCAB_PATH}")
    print(f"Config     -> {CONFIG_PATH}")


if __name__ == "__main__":
    main()
