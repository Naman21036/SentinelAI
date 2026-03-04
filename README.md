# SentinelAI

**AI Powered Hate Speech Detection using DistilBERT**

SentinelAI is a transformer based Natural Language Processing system that detects hate speech and abusive language in real time.
The project uses a fine tuned **DistilBERT model** integrated with a **FastAPI web application** to deliver fast and reliable predictions through a modern web interface.

---

## Overview

Online platforms generate massive amounts of user generated content every day. Detecting abusive or hateful language automatically is critical for moderation and safe communication.

SentinelAI provides an AI powered system that:

• Detects toxic or abusive text
• Uses transformer based deep learning
• Provides real time predictions via web interface
• Can be integrated into moderation pipelines

---

## Features

Real time hate speech detection
Transformer based NLP model (DistilBERT)
FastAPI backend for high performance inference
Interactive web interface
Confidence score visualization
Modular ML pipeline architecture

---

## Tech Stack

### Machine Learning

Python
PyTorch
Transformers (HuggingFace)
DistilBERT

### Backend

FastAPI
Uvicorn

### Frontend

HTML
TailwindCSS
JavaScript

### Data Processing

Pandas
NLTK
Scikit Learn

---

## Project Architecture

```
SentinelAI
│
├── app.py
│
├── routers
│   └── predict.py
│
├── services
│   └── inference.py
│
├── templates
│   ├── base.html
│   └── index.html
│
├── static
│   ├── css
│   └── js
│
├── artifacts
│   └── trained models
│
└── notebooks
```

---

## Model

The project uses a **DistilBERT transformer model** fine tuned for hate speech classification.

Model characteristics:

Architecture: DistilBERT
Task: Text Classification
Classes:

0 → Non Hate
1 → Hate / Abusive

---

## Installation

Clone the repository

```
git clone https://github.com/yourusername/SentinelAI.git
cd SentinelAI
```

Create virtual environment

```
python -m venv venv
```

Activate environment

Windows

```
venv\Scripts\activate
```

Linux / Mac

```
source venv/bin/activate
```

Install dependencies

```
pip install -r requirements.txt
```

---

## Running the Application

Start the FastAPI server

```
uvicorn app:app --reload
```

Open in browser

```
http://127.0.0.1:8000
```

---

## Example

Input

```
You are a useless idiot
```

Output

```
Prediction: Hate / Abusive
Confidence: 0.93
```

---

## Use Cases

Content moderation systems
Social media platforms
Online communities
Gaming chat monitoring
Comment filtering systems

---

## Future Improvements

Real time streaming inference
Multilingual hate speech detection
API integration for external applications
Model explainability using SHAP
Deployment with Docker and Kubernetes

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

Naman Gupta
BIT Mesra

AI and Quantitative Finance Enthusiast
