import os
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

MODEL_PATH = os.path.join(
    "artifacts",
    "03_04_2026_09_49_32",
    "ModelEvaluationArtifacts",
    "best_Model"
)

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


def predict_text(text: str):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]

    p0 = probs[0].item()
    p1 = probs[1].item()

    # Determine which is larger
    if p0 > p1:
        prediction = "No Hate"
        safe_probability = p0
        toxic_probability = p1
        confidence = p0
    else:
        prediction = "Hate / Abusive"
        safe_probability = p0
        toxic_probability = p1
        confidence = p1

    return {
        "prediction": prediction,
        "confidence": confidence,
        "safe_probability": safe_probability,
        "toxic_probability": toxic_probability
    }