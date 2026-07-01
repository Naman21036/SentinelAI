# SentinelAI

**AI Powered Hate Speech Detection using BiLSTM + Attention**

**Live Demo → [sentinelai-egh5.onrender.com](https://sentinelai-egh5.onrender.com)**

SentinelAI is a deep learning system that detects hate speech and toxic content in real time. The project uses a custom **Bidirectional LSTM with Multi-Head Self-Attention** model — trained from scratch on 56,000+ samples — integrated with a **FastAPI web application** and a modern dark-theme interface.

---

## Overview

Online platforms generate massive amounts of user generated content every day. Detecting abusive or hateful language automatically is critical for moderation and safe communication.

SentinelAI provides an AI powered system that:

- Detects hate speech, toxicity, and offensive language
- Uses a custom BiLSTM + Attention deep learning model (no pretrained weights, no API calls)
- Provides real time predictions with a toxicity score and severity classification
- Delivers results through a professional dark-theme web interface with animated visualizations
- Can be deployed on any container platform (Docker ready)

---

## Model

The project uses a **custom BiLSTM + Multi-Head Self-Attention** model trained from scratch — no HuggingFace, no pretrained weights, no external API.

| Property | Value |
|---|---|
| Architecture | Embedding → BiLSTM → Multi-Head Attention → LayerNorm → FC |
| Parameters | ~3.1M |
| Vocabulary | 20,000 words (custom WordTokenizer, serialized to JSON) |
| Training data | 56,701 samples (raw_data.csv + imbalanced_data.csv) |
| Loss | BCEWithLogitsLoss with pos_weight (handles class imbalance) |
| Optimizer | AdamW + ReduceLROnPlateau scheduler |
| Accuracy | 95% |
| F1 Score | 0.93 |
| Classes | 0 → No Hate, 1 → Hate / Abusive |

Model artifacts (`model.pt`, `vocab.json`, `config.json`) are committed to the repository — zero downloads at runtime.

---

## Features

- Real time hate speech and toxicity detection
- Custom BiLSTM + Multi-Head Self-Attention (trained from scratch)
- Toxicity score with animated count-up display
- Green → Yellow → Red spectrum bar with position marker
- Severity classification: CLEAR / LOW / MEDIUM / HIGH / CRITICAL
- Animated neural network canvas background
- Gradient shield logo and SVG favicon
- Two-column result layout (score + breakdown)
- Lazy model loading on first request
- Docker ready, deployed on Render

---

## Tech Stack

### Machine Learning
- Python 3.11
- PyTorch (CPU)
- Custom WordTokenizer (word-level, JSON serialized)
- Scikit-learn (train/test split, metrics)
- Pandas / NumPy

### Backend
- FastAPI
- Uvicorn / Gunicorn
- Jinja2 (server-side rendering)

### Frontend
- HTML / CSS / JavaScript
- Inter + JetBrains Mono (Google Fonts)
- Canvas-based animated neural network background
- No CSS framework (custom design system)

---

## Project Structure

```
SentinelAI/
│
├── app.py                        # FastAPI application entry point
│
├── routers/
│   └── predict.py                # GET / and POST /predict routes
│
├── services/
│   ├── classifier.py             # BiLSTM model + WordTokenizer definitions
│   └── inference.py              # Lazy model loading + predict_text()
│
├── training/
│   └── train_model.py            # Standalone training script
│
├── templates/
│   ├── base.html                 # Layout: topbar, canvas, fonts
│   └── index.html                # Input form + result card
│
├── static/
│   ├── css/styles.css
│   ├── js/script.js
│   └── favicon.svg
│
├── artifacts/
│   └── model/
│       ├── model.pt              # Trained model weights (12 MB)
│       ├── vocab.json            # WordTokenizer vocabulary
│       └── config.json           # Model hyperparameters
│
├── data/
│   ├── raw_data.csv
│   └── imbalanced_data.csv
│
├── Dockerfile
├── render.yaml
├── requirements.txt
└── runtime.txt
```

---

## Installation

```bash
git clone https://github.com/Naman21036/SentinelAI.git
cd SentinelAI
```

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / Mac
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

---

## Running the Application

```bash
python -m uvicorn app:app --reload --port 8000
```

Open in browser: `http://127.0.0.1:8000`

---

## Training the Model

If you want to retrain from scratch:

```bash
python -m training.train_model
```

This reads `data/raw_data.csv` and `data/imbalanced_data.csv`, trains the BiLSTM model, and saves artifacts to `artifacts/model/`.

---

## Deployment

The app is Docker ready. It uses the CPU-only PyTorch wheel to keep the image size small.

```bash
docker build -t sentinelai .
docker run -p 8000:8000 sentinelai
```

**Render:** Connect the GitHub repo. Render uses `render.yaml` to detect Docker environment automatically. Set `PYTHON_VERSION=3.11.9` as an environment variable if using the native Python runtime.

---

## Example

Input
```
You are a useless idiot
```

Output
```
Threat Detected · HIGH
Toxicity Score: 82.4%
Safe: 17.6%  |  Toxic: 82.4%
```

---

## Use Cases

- Content moderation systems
- Social media platforms
- Online communities
- Gaming chat monitoring
- Comment filtering pipelines

---

## Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## License

This project is licensed under the MIT License.

---

## Author

**Naman Gupta**  
BIT Mesra  
AI Enginnering Enthusiast
