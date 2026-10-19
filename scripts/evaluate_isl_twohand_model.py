"""Evaluate the NEW two-hand ISL Random Forest model.

The two-hand model was trained only on letters that had at least 2 valid
two-hand samples.  This evaluation reloads the same CSV and applies the same
letter inclusion rule, then reports:

    * accuracy (on the included letters)
    * macro precision / recall / F1
    * per-letter precision / recall / F1 and accuracy
    * confusion matrix (PNG + CSV)
    * a JSON metrics summary

Letters excluded during training are listed explicitly in the report.

Outputs go to ``results/landmark/isl_2hand_*``.

Run from the project root::

    python scripts/evaluate_isl_twohand_model.py
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")

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

from recognition.landmark_extractor import TWO_HAND_FEATURE_COLUMNS  # noqa: E402
from utils.label_mapping import LETTERS  # noqa: E402

LANDMARKS_DIR = os.path.join(PROJECT_ROOT, "dataset", "landmarks")
MODEL_DIR = os.path.join(PROJECT_ROOT, "trained_model")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "landmark")

CSV_PATH = os.path.join(LANDMARKS_DIR, "isl_landmarks_2hand.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "isl_landmark_rf_2hand.joblib")

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.15
MIN_SAMPLES_FOR_STRATIFY = 2


def build_included_split():
    df = pd.read_csv(CSV_PATH)
    counts = df["label"].value_counts().to_dict()
    included = [l for l in LETTERS if counts.get(l, 0) >= MIN_SAMPLES_FOR_STRATIFY]
    excluded = [l for l in LETTERS if counts.get(l, 0) < MIN_SAMPLES_FOR_STRATIFY]

    subset = df[df["label"].isin(included)]
    X = subset[TWO_HAND_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = subset["label"].to_numpy()

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
    return X_test, y_test, included, excluded


def save_confusion_matrix(cm, classes, png_path, csv_path):
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


def main() -> None:
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"Model not found: {MODEL_PATH}. Train it first.")

    print(f"\n[ISL 2-HAND] Loading model {MODEL_PATH}")
    clf = joblib.load(MODEL_PATH)

    print(f"[ISL 2-HAND] Building test split from {CSV_PATH}")
    X_test, y_test, included, excluded = build_included_split()

    y_pred = clf.predict(X_test)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Classification report (included letters only).
    report = classification_report(y_test, y_pred, labels=included, zero_division=0)
    report_path = os.path.join(RESULTS_DIR, "isl_2hand_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report)
        fh.write("\nExcluded letters (insufficient two-hand training samples): ")
        fh.write("".join(excluded))
        fh.write("\n")

    # Confusion matrix.
    cm = confusion_matrix(y_test, y_pred, labels=included)
    save_confusion_matrix(
        cm,
        included,
        os.path.join(RESULTS_DIR, "isl_2hand_confusion_matrix.png"),
        os.path.join(RESULTS_DIR, "isl_2hand_confusion_matrix.csv"),
    )

    # Per-letter metrics.
    per_letter_accuracy = {}
    per_letter_precision = {}
    per_letter_recall = {}
    per_letter_f1 = {}
    for letter in included:
        mask = y_test == letter
        if mask.sum() == 0:
            per_letter_accuracy[letter] = None
            per_letter_precision[letter] = None
            per_letter_recall[letter] = None
            per_letter_f1[letter] = None
        else:
            per_letter_accuracy[letter] = float(accuracy_score(y_test[mask], y_pred[mask]))

    metrics = {
        "dataset": "isl_2hand",
        "test_samples": int(len(y_test)),
        "included_letters": included,
        "excluded_letters": excluded,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "per_letter_accuracy": per_letter_accuracy,
    }
    metrics_path = os.path.join(RESULTS_DIR, "isl_2hand_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"\n[ISL 2-HAND] Test accuracy (included letters): {metrics['accuracy']:.4f}")
    print(f"[ISL 2-HAND] Precision(macro): {metrics['precision_macro']:.4f}")
    print(f"[ISL 2-HAND] Recall(macro)   : {metrics['recall_macro']:.4f}")
    print(f"[ISL 2-HAND] F1(macro)       : {metrics['f1_macro']:.4f}")
    print(f"[ISL 2-HAND] Excluded letters: {''.join(excluded)}")
    print(f"[ISL 2-HAND] Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
