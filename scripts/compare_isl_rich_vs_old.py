"""Evaluate the OLD and NEW ISL Random Forest models and compare them.

This script is the single source of truth for the old-vs-new comparison.  It:

    * Reconstructs the deterministic test splits for BOTH the old one-hand ISL
      model (``isl_landmark_rf.joblib``) and the new rich-feature ISL model
      (``isl_landmark_rf_rich.joblib``).
    * Builds a COMMON held-out test set = the intersection of the two test
      splits.  Every sample in this set was unseen by BOTH models during
      training, so the comparison has no data leakage.
    * Evaluates both models on that identical common set and reports accuracy,
      macro precision/recall/F1, per-letter accuracy, and the old-vs-new
      difference.
    * Additionally produces the full rich-model evaluation artifacts
      (classification report, confusion matrix, metrics JSON) on the rich
      model's own full test split.

Outputs written under ``results/landmark/``:

    isl_rich_classification_report.txt
    isl_rich_confusion_matrix.png
    isl_rich_confusion_matrix.csv
    isl_rich_metrics.json
    isl_rich_vs_old_comparison.txt
    isl_rich_vs_old_per_letter.csv

Run from the project root::

    python scripts/compare_isl_rich_vs_old.py
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

from recognition.landmark_extractor import FEATURE_COLUMNS  # noqa: E402
from recognition.isl_feature_extractor import RICH_FEATURE_COLUMNS  # noqa: E402
from utils.label_mapping import LETTERS  # noqa: E402

LANDMARKS_DIR = os.path.join(PROJECT_ROOT, "dataset", "landmarks")
MODEL_DIR = os.path.join(PROJECT_ROOT, "trained_model")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "landmark")

OLD_CSV = os.path.join(LANDMARKS_DIR, "isl_landmarks.csv")
RICH_CSV = os.path.join(LANDMARKS_DIR, "isl_rich_features.csv")
OLD_MODEL = os.path.join(MODEL_DIR, "isl_landmark_rf.joblib")
RICH_MODEL = os.path.join(MODEL_DIR, "isl_landmark_rf_rich.joblib")
RICH_SPLIT = os.path.join(LANDMARKS_DIR, "isl_rich_split.json")

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.15


def build_old_test_paths() -> set[str]:
    """Reconstruct the old one-hand model's test split (same as training)."""
    df = pd.read_csv(OLD_CSV)
    y = df["label"].to_numpy()
    idx = np.arange(len(y))
    train_val_idx, test_idx = train_test_split(
        idx, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE, shuffle=True
    )
    train_idx, _ = train_test_split(
        train_val_idx,
        test_size=VAL_SIZE,
        stratify=y[train_val_idx],
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    _ = train_idx
    src = df["source_path"].to_numpy()
    return set(src[test_idx].tolist())


def _evaluate(model, df: pd.DataFrame, cols: list[str], paths: list[str]):
    """Evaluate ``model`` on the rows of ``df`` whose source_path is in ``paths``."""
    sub = df[df["source_path"].isin(set(paths))]
    # Preserve the requested order to stay deterministic.
    sub = sub.set_index("source_path").loc[paths].reset_index()
    X = sub[cols].to_numpy(dtype=np.float32)
    y = sub["label"].to_numpy()
    pred = model.predict(X)
    return y, pred


def _macro_metrics(y_true, y_pred, labels) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "n": int(len(y_true)),
    }


def _per_letter_accuracy(y_true, y_pred, letters) -> dict:
    out = {}
    for letter in letters:
        mask = y_true == letter
        out[letter] = float(accuracy_score(y_true[mask], y_pred[mask])) if mask.sum() else None
    return out


def save_confusion_matrix(cm, classes, png_path, csv_path):
    fig, ax = plt.subplots(figsize=(11, 9))
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
    for path, label in ((OLD_MODEL, "old"), (RICH_MODEL, "rich")):
        if not os.path.exists(path):
            raise SystemExit(f"{label} model not found: {path}")

    print(f"\n[COMPARE] Loading old model {OLD_MODEL}")
    old_clf = joblib.load(OLD_MODEL)
    print(f"[COMPARE] Loading rich model {RICH_MODEL}")
    rich_clf = joblib.load(RICH_MODEL)

    old_df = pd.read_csv(OLD_CSV)
    rich_df = pd.read_csv(RICH_CSV)

    old_test_paths = build_old_test_paths()
    with open(RICH_SPLIT, "r", encoding="utf-8") as fh:
        rich_split = json.load(fh)
    rich_test_paths = set(rich_split["source_paths"]["test"])

    # COMMON held-out test set: seen by NEITHER model during training.
    common = sorted(old_test_paths & rich_test_paths)
    print(f"[COMPARE] old test={len(old_test_paths)}  "
          f"rich test={len(rich_test_paths)}  common={len(common)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- 1. Full rich-model evaluation on its own test split (355). ---
    rich_test_list = sorted(rich_test_paths)
    y_true_full, y_pred_full = _evaluate(
        rich_clf, rich_df, RICH_FEATURE_COLUMNS, rich_test_list
    )

    report = classification_report(y_true_full, y_pred_full, labels=LETTERS, zero_division=0)
    with open(os.path.join(RESULTS_DIR, "isl_rich_classification_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(report)

    cm = confusion_matrix(y_true_full, y_pred_full, labels=LETTERS)
    save_confusion_matrix(
        cm,
        LETTERS,
        os.path.join(RESULTS_DIR, "isl_rich_confusion_matrix.png"),
        os.path.join(RESULTS_DIR, "isl_rich_confusion_matrix.csv"),
    )

    # --- 2. COMMON-set evaluation for both models. ---
    y_old_true, y_old_pred = _evaluate(old_clf, old_df, FEATURE_COLUMNS, common)
    y_rich_true, y_rich_pred = _evaluate(rich_clf, rich_df, RICH_FEATURE_COLUMNS, common)

    old_metrics = _macro_metrics(y_old_true, y_old_pred, LETTERS)
    rich_metrics = _macro_metrics(y_rich_true, y_rich_pred, LETTERS)

    old_per_letter = _per_letter_accuracy(y_old_true, y_old_pred, LETTERS)
    rich_per_letter = _per_letter_accuracy(y_rich_true, y_rich_pred, LETTERS)
    rich_per_letter_full = _per_letter_accuracy(y_true_full, y_pred_full, LETTERS)

    # --- 3. Per-letter comparison CSV/table. ---
    table_lines = [
        "ISL Old vs Rich-Feature model comparison (common held-out test set)",
        "=" * 70,
        f"common test samples : {len(common)}",
        "",
        f"{'Letter':<7}{'Old Acc':>10}{'Rich Acc':>11}{'Diff':>9}",
        "-" * 70,
    ]
    rows = []
    for letter in LETTERS:
        old_a = old_per_letter[letter]
        rich_a = rich_per_letter[letter]
        if old_a is None or rich_a is None:
            diff = None
            diff_s = "  n/a"
        else:
            diff = rich_a - old_a
            diff_s = f"{diff:+.4f}"
        old_s = f"{old_a:.4f}" if old_a is not None else "  n/a"
        rich_s = f"{rich_a:.4f}" if rich_a is not None else "  n/a"
        table_lines.append(f"{letter:<7}{old_s:>10}{rich_s:>11}{diff_s:>9}")
        rows.append(
            {
                "letter": letter,
                "old_accuracy": old_a,
                "rich_accuracy": rich_a,
                "difference": diff,
            }
        )
    table_lines.append("-" * 70)
    table_lines.append(
        f"{'ALL':<7}{old_metrics['accuracy']:>10.4f}"
        f"{rich_metrics['accuracy']:>11.4f}"
        f"{rich_metrics['accuracy'] - old_metrics['accuracy']:>+9.4f}"
    )
    with open(os.path.join(RESULTS_DIR, "isl_rich_vs_old_comparison.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(table_lines) + "\n")

    per_letter_df = pd.DataFrame(rows)
    per_letter_df.to_csv(
        os.path.join(RESULTS_DIR, "isl_rich_vs_old_per_letter.csv"), index=False
    )

    # --- 4. metrics JSON. ---
    metrics = {
        "dataset": "isl_rich",
        "full_test_samples": int(len(y_true_full)),
        "full_metrics": {
            "accuracy": float(accuracy_score(y_true_full, y_pred_full)),
            "precision_macro": float(precision_score(y_true_full, y_pred_full, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true_full, y_pred_full, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true_full, y_pred_full, average="macro", zero_division=0)),
            "precision_weighted": float(precision_score(y_true_full, y_pred_full, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_true_full, y_pred_full, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y_true_full, y_pred_full, average="weighted", zero_division=0)),
            "per_letter_accuracy": rich_per_letter_full,
        },
        "common_comparison": {
            "common_test_samples": len(common),
            "old_model": old_metrics,
            "rich_model": rich_metrics,
            "old_per_letter_accuracy": old_per_letter,
            "rich_per_letter_accuracy": rich_per_letter,
        },
        "classes": LETTERS,
    }
    with open(os.path.join(RESULTS_DIR, "isl_rich_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print("\n[COMPARE] Common held-out set results:")
    print(f"  Old  : acc={old_metrics['accuracy']:.4f}  "
          f"P={old_metrics['precision_macro']:.4f}  "
          f"R={old_metrics['recall_macro']:.4f}  F1={old_metrics['f1_macro']:.4f}")
    print(f"  Rich : acc={rich_metrics['accuracy']:.4f}  "
          f"P={rich_metrics['precision_macro']:.4f}  "
          f"R={rich_metrics['recall_macro']:.4f}  F1={rich_metrics['f1_macro']:.4f}")
    print(f"[COMPARE] Full rich test accuracy: {metrics['full_metrics']['accuracy']:.4f}")
    print(f"[COMPARE] Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
