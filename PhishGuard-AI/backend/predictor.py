"""Predict URL risk from URL text and saved PhishGuard AI model artifacts."""

import json
import math
import pickle
from pathlib import Path
import sys
from typing import Any
from urllib.parse import urlsplit

import joblib
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.feature_extraction import FEATURE_NAMES, extract_features
from backend.trusted_domains import TRUSTED_DOMAINS

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "model" / "phishing_model.pkl"
FEATURE_NAMES_PATH = BASE_DIR / "model" / "feature_names.json"

_KEYWORD_FEATURES = (
    ("login", "contains_login"),
    ("verify", "contains_verify"),
    ("account", "contains_account"),
    ("secure", "contains_secure"),
    ("update", "contains_update"),
    ("password", "contains_password"),
)


class PredictionError(Exception):
    """A safe, user-readable prediction or model-loading error."""


def _validate_url_text(url: str) -> str:
    if not isinstance(url, str):
        raise PredictionError("URL must be provided as text.")

    value = url.strip()
    if not value:
        raise PredictionError("URL cannot be empty.")
    if any(character.isspace() for character in value):
        raise PredictionError("URL contains whitespace and is not valid.")

    try:
        parsed = urlsplit(value)
        if parsed.scheme:
            if parsed.scheme.lower() not in {"http", "https"}:
                raise PredictionError("Only HTTP or HTTPS URL text is supported.")
        else:
            candidate = value if value.startswith("//") else f"//{value}"
            parsed = urlsplit(candidate)
        hostname = parsed.hostname
        parsed.port  # Validate port syntax; accessing it can raise ValueError.
    except ValueError as error:
        raise PredictionError("URL is not valid.") from error

    if not hostname:
        raise PredictionError("URL must contain a valid hostname.")
    return value


def _normalized_hostname(url: str) -> str:
    parsed = urlsplit(url)
    if not parsed.scheme:
        parsed = urlsplit(url if url.startswith("//") else f"//{url}")
    hostname = (parsed.hostname or "").lower().rstrip(".")
    if hostname.startswith("www."):
        hostname = hostname[4:]
    return hostname


def _is_trusted_domain(hostname: str) -> bool:
    return any(
        hostname == domain or hostname.endswith(f".{domain}")
        for domain in TRUSTED_DOMAINS
    )


class URLPredictor:
    """Load the trained model and predict using the saved, fixed feature order."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        feature_names_path: Path = FEATURE_NAMES_PATH,
    ) -> None:
        if not model_path.is_file():
            raise PredictionError(f"Trained model file was not found: {model_path}")
        if not feature_names_path.is_file():
            raise PredictionError(
                f"Feature names file was not found: {feature_names_path}"
            )

        try:
            saved_feature_names = json.loads(
                feature_names_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise PredictionError("Feature names file could not be read.") from error

        if not isinstance(saved_feature_names, list) or not all(
            isinstance(name, str) for name in saved_feature_names
        ):
            raise PredictionError("Feature names file must contain a JSON list of names.")
        if saved_feature_names != list(FEATURE_NAMES):
            raise PredictionError(
                "Saved feature order does not match the URL feature extractor."
            )

        try:
            model = joblib.load(model_path)
        except (
            OSError,
            ValueError,
            TypeError,
            EOFError,
            ImportError,
            pickle.UnpicklingError,
        ) as error:
            raise PredictionError("Trained model file could not be loaded.") from error

        if not callable(getattr(model, "predict", None)) or not callable(
            getattr(model, "predict_proba", None)
        ):
            raise PredictionError("Trained model does not support prediction probabilities.")
        if getattr(model, "n_features_in_", None) != len(saved_feature_names):
            raise PredictionError(
                "Trained model input size does not match the saved feature names."
            )

        classes = getattr(model, "classes_", None)
        if classes is None or 0 not in classes or 1 not in classes:
            raise PredictionError("Trained model must support labels 0 and 1.")

        self._model: Any = model
        self._feature_names = tuple(saved_feature_names)
        self._classes = list(classes)

    def predict(self, url: str) -> dict[str, Any]:
        """Verify curated domains first, then use ML for unknown URL hosts."""
        url_text = _validate_url_text(url)
        hostname = _normalized_hostname(url_text)

        if _is_trusted_domain(hostname):
            try:
                features = extract_features(url_text)
            except (TypeError, ValueError) as error:
                raise PredictionError("URL features could not be extracted.") from error
            return {
                "prediction": "Legitimate",
                "risk_score": 0,
                "risk_level": "Low Risk",
                "confidence": 100,
                "confidence_description": (
                    "Trusted-domain verification result; not a guarantee."
                ),
                "risk_score_description": (
                    "Trusted-domain verification result; not a guarantee."
                ),
                "reasons": ["Recognized trusted official domain"],
                "features": features,
                "detection_method": "Trusted Official Domain Verification",
            }

        try:
            features = extract_features(url_text)
        except (TypeError, ValueError) as error:
            raise PredictionError("URL features could not be extracted.") from error

        if tuple(features) != self._feature_names:
            raise PredictionError(
                "Extracted feature order does not match the saved model feature order."
            )

        feature_vector = np.asarray(
            [[features[name] for name in self._feature_names]], dtype=np.float32
        )
        if not np.isfinite(feature_vector).all():
            raise PredictionError("Extracted URL features are not finite numbers.")

        try:
            predicted_label = int(self._model.predict(feature_vector)[0])
            probabilities = self._model.predict_proba(feature_vector)[0]
        except (ValueError, TypeError, AttributeError, IndexError) as error:
            raise PredictionError("The trained model could not process this URL.") from error

        phishing_index = self._classes.index(1)
        phishing_probability = float(probabilities[phishing_index])
        predicted_index = self._classes.index(predicted_label)
        model_confidence = float(probabilities[predicted_index])
        if not (
            math.isfinite(phishing_probability)
            and math.isfinite(model_confidence)
            and 0.0 <= phishing_probability <= 1.0
            and 0.0 <= model_confidence <= 1.0
        ):
            raise PredictionError("The model returned an invalid prediction probability.")

        risk_score = round(phishing_probability * 100)
        confidence = round(model_confidence * 100)
        if risk_score <= 30:
            risk_level = "Low Risk"
        elif risk_score <= 70:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"

        reasons: list[str] = []
        if features["url_length"] > 75:
            reasons.append("Unusually long URL")
        if features["has_ip_address"]:
            reasons.append("URL uses an IP address")
        if features["number_of_subdomains"] >= 2:
            reasons.append("Multiple subdomains detected")
        for keyword, feature_name in _KEYWORD_FEATURES:
            if features[feature_name]:
                reasons.append(f"Suspicious keyword detected: {keyword}")
        if features["number_of_at_symbols"]:
            reasons.append("URL contains @ symbol")
        if features["number_of_special_characters"] >= 10:
            reasons.append("Large number of special characters")
        if features["number_of_digits"] >= 5:
            reasons.append("Large number of digits")
        if not features["uses_https"]:
            reasons.append("HTTPS is not detected")
        if features["uses_url_shortener"]:
            reasons.append("URL shortening service detected")

        return {
            "prediction": "Phishing" if predicted_label == 1 else "Legitimate",
            "risk_score": risk_score,
            "risk_level": risk_level,
            "confidence": confidence,
            "confidence_description": (
                "Model confidence for its predicted class; not a guarantee."
            ),
            "risk_score_description": (
                "ML-derived phishing risk indicator; not a guaranteed probability."
            ),
            "reasons": reasons,
            "features": features,
            "detection_method": "Random Forest Machine Learning",
        }


_DEFAULT_PREDICTOR: URLPredictor | None = None


def predict_url(url: str) -> dict[str, Any]:
    """Predict a URL using the saved PhishGuard AI model."""
    global _DEFAULT_PREDICTOR
    if _DEFAULT_PREDICTOR is None:
        _DEFAULT_PREDICTOR = URLPredictor()
    return _DEFAULT_PREDICTOR.predict(url)


if __name__ == "__main__":
    import sys

    test_urls = (
        "https://www.google.com",
        "https://github.com",
        "http://example.com/login",
    )
    try:
        predictor = URLPredictor()
        print(f"Loaded model from: {MODEL_PATH}")
        print(f"Verified feature order: {len(predictor._feature_names)} features")
        for test_url in test_urls:
            print(f"\nURL text: {test_url}")
            print(json.dumps(predictor.predict(test_url), indent=2))
        print("\nNo URLs were visited or requested; only URL strings were analyzed.")
    except PredictionError as error:
        print(f"Prediction setup failed: {error}", file=sys.stderr)
        raise SystemExit(1) from None
