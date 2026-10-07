"""Train and evaluate the PhishGuard AI URL-feature Random Forest."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from backend.feature_extraction import FEATURE_NAMES, extract_features

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_ESTIMATORS = 100
POSITIVE_LABEL = 1
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "phishing_urls.csv"
MODEL_DIR = BASE_DIR / "model"


def load_clean_dataset() -> pd.DataFrame:
    """Load required dataset columns, remove incomplete/duplicate URLs and invalid labels."""
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    data = pd.read_csv(DATASET_PATH)
    missing_columns = {"url", "label"} - set(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    clean = data[["url", "label"]].dropna(subset=["url", "label"]).copy()
    clean["url"] = clean["url"].astype(str).str.strip()
    clean = clean.loc[clean["url"].ne("")]
    clean["label"] = pd.to_numeric(clean["label"], errors="coerce")
    clean = clean.loc[clean["label"].isin([0, 1])]
    clean = clean.drop_duplicates(subset="url", keep="first").reset_index(drop=True)
    if clean.empty or clean["label"].nunique() != 2:
        raise ValueError("Training requires non-empty data with both labels 0 and 1.")
    clean["label"] = clean["label"].astype("int64")
    return clean


def extract_dataset_features(urls: pd.Series) -> pd.DataFrame:
    """Extract every URL's features in the stable FEATURE_NAMES order."""
    rows: list[dict[str, int | float]] = []
    total = len(urls)
    print(f"Extracting {len(FEATURE_NAMES)} features from {total:,} URLs...")
    for index, url in enumerate(urls, start=1):
        rows.append(extract_features(url))
        if index % 25_000 == 0 or index == total:
            print(f"  Processed {index:,}/{total:,} URLs.", flush=True)
    return pd.DataFrame.from_records(rows, columns=list(FEATURE_NAMES))


def save_confusion_matrix(matrix: np.ndarray, destination: Path) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Legitimate (0)", "Phishing (1)"],
        yticklabels=["Legitimate (0)", "Phishing (1)"],
        xlabel="Predicted label",
        ylabel="True label",
        title="PhishGuard AI - Confusion Matrix",
    )
    axis.set_ylim(1.5, -0.5)
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def save_feature_importance(
    model: RandomForestClassifier, destination: Path
) -> pd.DataFrame:
    importance = pd.DataFrame(
        {"feature": FEATURE_NAMES, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False, kind="stable")
    top_features = importance.head(10).sort_values("importance", ascending=True)

    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(top_features["feature"], top_features["importance"], color="steelblue")
    axis.set(
        xlabel="Feature importance",
        title="PhishGuard AI - Top 10 Feature Importances",
    )
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return importance


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    clean = load_clean_dataset()
    print(f"Loaded and cleaned {len(clean):,} dataset rows.")

    feature_frame = extract_dataset_features(clean["url"])
    X = feature_frame.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=np.float32)
    y = clean["label"].to_numpy(dtype=np.int64)
    if not np.isfinite(X).all():
        raise ValueError("Extracted features contain NaN or infinite values.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(
        f"Training Random Forest ({N_ESTIMATORS} trees) with "
        f"{len(X_train):,} samples; testing with {len(X_test):,} samples..."
    )
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = float(accuracy_score(y_test, predictions))
    precision = float(
        precision_score(y_test, predictions, pos_label=POSITIVE_LABEL, zero_division=0)
    )
    recall = float(
        recall_score(y_test, predictions, pos_label=POSITIVE_LABEL, zero_division=0)
    )
    f1 = float(
        f1_score(y_test, predictions, pos_label=POSITIVE_LABEL, zero_division=0)
    )
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = (
        int(matrix[0, 0]),
        int(matrix[0, 1]),
        int(matrix[1, 0]),
        int(matrix[1, 1]),
    )

    model_path = MODEL_DIR / "phishing_model.pkl"
    feature_names_path = MODEL_DIR / "feature_names.json"
    metadata_path = MODEL_DIR / "model_metadata.json"
    confusion_path = MODEL_DIR / "confusion_matrix.png"
    importance_path = MODEL_DIR / "feature_importance.png"

    joblib.dump(model, model_path)
    feature_names_path.write_text(
        json.dumps(list(FEATURE_NAMES), indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "model_type": "RandomForestClassifier",
        "number_of_features": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "training_dataset_size": int(len(X_train)),
        "test_dataset_size": int(len(X_test)),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "random_state": RANDOM_STATE,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    save_confusion_matrix(matrix, confusion_path)
    importance = save_feature_importance(model, importance_path)

    required_files = (
        model_path,
        feature_names_path,
        metadata_path,
        confusion_path,
        importance_path,
    )
    missing_files = [str(path) for path in required_files if not path.is_file()]
    if missing_files:
        raise RuntimeError(f"Expected output files were not created: {missing_files}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Legitimate (0)", "Phishing (1)"],
            zero_division=0,
        )
    )
    print("Confusion matrix (rows=true, columns=predicted; labels=[0, 1]):")
    print(matrix)
    print(
        "True Positives: "
        f"{true_positive:,}; True Negatives: {true_negative:,}; "
        f"False Positives: {false_positive:,}; False Negatives: {false_negative:,}"
    )
    print("\nTop 10 important features:")
    print(importance.head(10).to_string(index=False))

    print("\nMODEL TRAINING COMPLETE")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Training samples: {len(X_train):,}")
    print(f"Testing samples: {len(X_test):,}")
    print(f"Number of features: {len(FEATURE_NAMES)}")
    print("Model: RandomForestClassifier")
    print(f"Accuracy: {accuracy:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1-score: {f1:.6f}")
    print("\nConfusion Matrix:")
    print(matrix)
    print("\nTop 10 Important Features:")
    print(importance.head(10).to_string(index=False))
    print("\nVerified output files:")
    for path in required_files:
        print(f"  {path} ({path.stat().st_size:,} bytes)")
    print(
        "\nOnly URL strings and extracted features were processed. "
        "No URLs were visited, requested, downloaded, or executed."
    )


if __name__ == "__main__":
    main()
