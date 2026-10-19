"""Evaluate the trained landmark Random Forest models on held-out test data.

For ASL and ISL separately this script:

    * Reloads the exact landmark CSV and reconstructs the SAME stratified
      train/validation/test split used during training (same ``random_state``).
    * Predicts the test split with the saved ``joblib`` model.
    * Saves a classification report, a confusion-matrix PNG/CSV, per-letter
      accuracy, and a JSON metrics summary.

Outputs are written under ``results/landmark/``.

Run from the project root::

    python scripts/evaluate_landmark_models.py
    python scripts/evaluate_landmark_models.py --dataset asl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless-safe backend for saving figures

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402

from recognition.landmark_extractor import FEATURE_COLUMNS  # noqa: E402
from utils.label_mapping import LETTERS  # noqa: E402

LANDMARKS_DIR = os.path.join(PROJECT_ROOT, "dataset", "landmarks")
MODEL_DIR = os.path.join(PROJECT_ROOT, "trained_model")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "landmark")

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.15

DATASETS = {
    "asl": {
        "csv": os.path.join(LANDMARKS_DIR, "asl_landmarks.csv"),
        "model": os.path.join(MODEL_DIR, "asl_landmark_rf.joblib"),
        "prefix": "asl",
    },
    "isl": {
        "csv": os.path.join(LANDMARKS_DIR, "isl_landmarks.csv"),
        "model": os.path.join(MODEL_DIR, "isl_landmark_rf.joblib"),
        "prefix": "isl",
    },
}


def build_test_split(csv_path: str):
    df = pd.read_csv(csv_path)
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE, shuffle=True
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=VAL_SIZE,
        stratify=y_train,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    return X_test, y_test


def save_confusion_matrix(cm: np.ndarray, classes: list[str], png_path: str, csv_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="viridis")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="w" if cm[i, j] > cm.max() / 2 else "k",
                fontsize=7,
            )
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    np.savetxt(csv_path, cm, delimiter=",", fmt="%d", header=",".join(classes), comments="")


def evaluate_dataset(dataset_name: str) -> None:
    cfg = DATASETS[dataset_name]
    prefix = cfg["prefix"]

    if not os.path.exists(cfg["model"]):
        raise SystemExit(f"Model not found: {cfg['model']}. Train it first.")

    print(f"\n[{prefix.upper()}] Loading model {cfg['model']}")
    clf = joblib.load(cfg["model"])

    print(f"[{prefix.upper()}] Building test split from {cfg['csv']}")
    X_test, y_test = build_test_split(cfg["csv"])

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Classification report.
    report = classification_report(y_test, y_pred, labels=LETTERS, zero_division=0)
    report_path = os.path.join(RESULTS_DIR, f"{prefix}_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report)

    # Confusion matrix.
    cm = confusion_matrix(y_test, y_pred, labels=LETTERS)
    save_confusion_matrix(
        cm,
        LETTERS,
        os.path.join(RESULTS_DIR, f"{prefix}_confusion_matrix.png"),
        os.path.join(RESULTS_DIR, f"{prefix}_confusion_matrix.csv"),
    )

    # Per-letter accuracy.
    per_letter = {}
    for i, letter in enumerate(LETTERS):
        mask = y_test == letter
        if mask.sum() == 0:
            per_letter[letter] = None
        else:
            per_letter[letter] = float(accuracy_score(y_test[mask], y_pred[mask]))

    # Aggregate metrics (weighted average, zero_division safe).
    metrics = {
        "dataset": dataset_name,
        "test_samples": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "per_letter_accuracy": per_letter,
        "classes": LETTERS,
    }
    metrics_path = os.path.join(RESULTS_DIR, f"{prefix}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"\n[{prefix.upper()}] Test accuracy : {metrics['accuracy']:.4f}")
    print(f"[{prefix.upper()}] Precision(macro): {metrics['precision_macro']:.4f}")
    print(f"[{prefix.upper()}] Recall(macro)   : {metrics['recall_macro']:.4f}")
    print(f"[{prefix.upper()}] F1(macro)       : {metrics['f1_macro']:.4f}")
    print(f"[{prefix.upper()}] Results saved to {RESULTS_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["asl", "isl", "all"],
        default="all",
        help="Which model to evaluate (default: all).",
    )
    args = parser.parse_args()

    names = ["asl", "isl"] if args.dataset == "all" else [args.dataset]
    for name in names:
        evaluate_dataset(name)


if __name__ == "__main__":
    main()
