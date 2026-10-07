"""Simple Flask API for PhishGuard AI predictions."""

from pathlib import Path
import sys

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.predictor import predict_url

app = Flask(__name__)
CORS(app)
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"


@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.post("/predict")
@app.post("/api/predict")
def predict():
    data = request.get_json()
    result = predict_url(data["url"])
    return jsonify(result)


@app.get("/<path:asset_path>")
def frontend_asset(asset_path):
    return send_from_directory(FRONTEND_DIR, asset_path)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
