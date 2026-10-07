"""Simple Flask API for PhishGuard AI predictions."""

from pathlib import Path
import sys

from flask import Flask, jsonify, request
from flask_cors import CORS

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.predictor import predict_url

app = Flask(__name__)
CORS(app)


@app.get("/")
def index():
    return jsonify({"status": "PhishGuard AI API is running"})


@app.post("/predict")
def predict():
    data = request.get_json()
    result = predict_url(data["url"])
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
