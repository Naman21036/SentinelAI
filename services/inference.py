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


def predict_text(text):

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

    probs = torch.softmax(outputs.logits, dim=1)

    prediction = torch.argmax(probs).item()
    confidence = probs[0][prediction].item()

    if confidence>0.5 or prediction == 1:
        result = "Hate / Abusive"
    else:
        result = "No Hate"

    return result, round(confidence, 3)