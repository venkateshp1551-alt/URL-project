# PhishGuard AI

PhishGuard AI uses a hybrid phishing detection architecture consisting of
curated trusted official-domain verification followed by Random Forest
machine-learning analysis for unknown domains.

The trusted-domain list is a curated list of well-known official domains and
is not a universal whitelist.

## Project structure

- `dataset/` — labeled URL data used for training and evaluation.
- `model/` — saved model, feature order, evaluation metadata, and plots.
- `backend/` — trusted-domain verification, URL-text feature extraction, and local model prediction.
- `frontend/` — responsive HTML/CSS/JavaScript user interface.
- `train_model.py` — extracts URL-text features, trains/evaluates the Random Forest, and saves artifacts.
- `backend/predictor.py` — loads the saved model and predicts from URL text without visiting URLs.
- `backend/trusted_domains.py` — selected trusted-domain entries; unknown domains continue to machine learning.
- `evaluate_model.py` — future entry point for measuring model performance.
- `requirements.txt` — Python package dependencies.
- `HACKATHON_PRESENTATION.md` — outline for the project presentation.
- `JUDGES_QA.md` — preparation notes for likely judge questions.

## Setup

Create and activate a Python virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

## Train the model

From the project directory, run:

```bash
python train_model.py
```

The script uses only URL strings and extracted features; it does not visit the
URLs. It reproducibly splits the data with `random_state=42`, evaluates the
model on the held-out test split, and writes the model, feature order,
metadata, confusion-matrix plot, and feature-importance plot to `model/`.
Reported metrics describe this dataset and split; they are not a guarantee of
performance on future URLs.

## Test URL prediction

From the project directory, run:

```bash
python backend/predictor.py
```

The predictor only parses the supplied URL text and extracts features locally.
Its risk score and model confidence are indicators, not guarantees.

## Run the Flask API

From the project directory, run:

```bash
python -m flask --app backend.app run --host 127.0.0.1 --port 5001
```

The API listens locally at `http://127.0.0.1:5001`. The Flask app serves the
existing frontend at `/`, its assets at their file paths, and predictions at
`POST /api/predict` (also available locally at `POST /predict`; JSON body:
`{"url": "https://example.com"}`). The frontend uses the same-origin
`/api/predict` path, so it works locally and after deployment without a
localhost production URL. Predictions use URL text only; URLs are not visited
or persisted.

## Deploy to Vercel

Deploy the project root (`PhishGuard-AI`) as the Vercel project root. The
`pyproject.toml` entrypoint runs `backend.app:app`; `vercel.json` includes the
frontend files and model artifacts in that Python function. Flask serves the
frontend at `/` and handles API requests at `/api/predict`.
