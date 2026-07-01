import json
import os

import torch

from services.classifier import HateSpeechClassifier, WordTokenizer, clean_text

ARTIFACTS_DIR = os.path.join("artifacts", "model")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pt")
VOCAB_PATH = os.path.join(ARTIFACTS_DIR, "vocab.json")
CONFIG_PATH = os.path.join(ARTIFACTS_DIR, "config.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model: HateSpeechClassifier | None = None
_tokenizer: WordTokenizer | None = None
_max_length: int = 100


def _load() -> None:
    global _model, _tokenizer, _max_length

    if _model is not None:
        return

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. "
            "Run: python -m training.train_model"
        )

    with open(CONFIG_PATH) as f:
        cfg = json.load(f)

    _max_length = cfg["max_length"]
    _tokenizer = WordTokenizer.load(VOCAB_PATH)

    _model = HateSpeechClassifier(
        vocab_size=cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        lstm_hidden=cfg["lstm_hidden"],
        attn_heads=cfg["attn_heads"],
        dropout=0.0,
    ).to(DEVICE)

    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    _model.load_state_dict(state)
    _model.eval()


def predict_text(text: str) -> dict:
    _load()

    clean = clean_text(text)
    ids = _tokenizer.encode(clean, _max_length)
    tensor = torch.tensor([ids], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        logit = _model(tensor).item()

    prob = float(torch.sigmoid(torch.tensor(logit)))
    toxic_probability = prob
    safe_probability = 1.0 - prob
    is_hate = prob > 0.5

    return {
        "prediction": "Hate / Abusive" if is_hate else "No Hate",
        "safe_probability": safe_probability,
        "toxic_probability": toxic_probability,
    }
