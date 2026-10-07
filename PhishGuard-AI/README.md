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

The API listens locally at `http://127.0.0.1:5001`. It provides
`GET /` and `POST /predict` (JSON body: `{"url": "https://example.com"}`).
Predictions are computed from URL text only; the API does not visit submitted
URLs or persist them.

## Run the frontend

From the project directory, start a static file server:

```bash
python -m http.server 8080 --bind 127.0.0.1 --directory frontend
```

Then open `http://127.0.0.1:8080`. The frontend sends prediction requests to
`http://127.0.0.1:5001/predict`; make sure the compatible PhishGuard API is
listening on that port.
