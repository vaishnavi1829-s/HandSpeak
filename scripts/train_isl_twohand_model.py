"""Train a NEW two-hand ISL Random Forest on ``isl_landmarks_2hand.csv``.

This is a separate experimental model.  It does NOT overwrite the existing
one-hand ISL model (``trained_model/isl_landmark_rf.joblib``) and does NOT
touch any CNN model.

The two-hand CSV is produced by ``scripts/extract_isl_twohand_landmarks.py`` and
contains ONLY images where BOTH a left and right hand were detected.  This
script is intentionally defensive about data quality:

* It reports per-letter two-hand sample counts before training.
* It only includes letters that have enough two-hand samples to support a
  stratified split (>= 2 samples).  Letters with 0 or 1 two-hand samples are
  reported as **excluded** rather than being silently filled with fabricated
  landmarks.
* A deterministic stratified train/validation/test split (``random_state=42``)
  is used, with ``class_weight="balanced"`` to reduce the impact of imbalanced
  letter counts.

Run from the project root::

    python scripts/train_isl_twohand_model.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from recognition.landmark_extractor import TWO_HAND_FEATURE_COLUMNS  # noqa: E402
from utils.label_mapping import LETTERS  # noqa: E402

LANDMARKS_DIR = os.path.join(PROJECT_ROOT, "dataset", "landmarks")
MODEL_DIR = os.path.join(PROJECT_ROOT, "trained_model")

CSV_PATH = os.path.join(LANDMARKS_DIR, "isl_landmarks_2hand.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "isl_landmark_rf_2hand.joblib")
SPLIT_PATH = os.path.join(LANDMARKS_DIR, "isl_landmark_2hand_split.json")

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.15
MIN_SAMPLES_FOR_STRATIFY = 2

RF_PARAMS = {
    "n_estimators": 400,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "class_weight": "balanced",
}


def main() -> None:
    if not os.path.exists(CSV_PATH):
        raise SystemExit(
            f"Two-hand CSV not found: {CSV_PATH}\n"
            f"Run: python scripts/extract_isl_twohand_landmarks.py"
        )

    df = pd.read_csv(CSV_PATH)
    print(f"\n[ISL 2-HAND] Loaded {CSV_PATH}")
    print(f"  rows           : {len(df)}")

    # Per-letter counts (for full transparency).
    counts = df["label"].value_counts().to_dict()
    included = []
    excluded = []
    for letter in LETTERS:
        n = counts.get(letter, 0)
        if n >= MIN_SAMPLES_FOR_STRATIFY:
            included.append(letter)
        else:
            excluded.append(letter)
        print(f"  {letter}: {n}")

    if not included:
        raise SystemExit(
            "No letters have enough two-hand samples to train a model. "
            "The existing ISL dataset appears to be predominantly single-hand."
        )

    print(f"\n[ISL 2-HAND] Included letters (>= {MIN_SAMPLES_FOR_STRATIFY} samples): "
          f"{''.join(included)}")
    if excluded:
        print(f"[ISL 2-HAND] Excluded letters (insufficient two-hand samples): "
              f"{''.join(excluded)}")

    subset = df[df["label"].isin(included)]
    X = subset[TWO_HAND_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = subset["label"].to_numpy()

    print(f"  training rows  : {len(subset)}")
    print(f"  feature dim    : {X.shape[1]}")

    # Stratified train/val/test.
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
    print(f"  train/val/test : {len(y_train)}/{len(y_val)}/{len(y_test)}")

    clf = RandomForestClassifier(**RF_PARAMS)
    print("\n[ISL 2-HAND] Training RandomForest...")
    clf.fit(X_train, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"  saved model -> {MODEL_PATH}")

    val_acc = float(clf.score(X_val, y_val))
    print(f"  validation accuracy (included letters only): {val_acc:.4f}")

    split_info = {
        "dataset": "isl_2hand",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "val_size": VAL_SIZE,
        "included_letters": included,
        "excluded_letters": excluded,
        "counts": {
            "train": int(len(y_train)),
            "val": int(len(y_val)),
            "test": int(len(y_test)),
        },
    }
    with open(SPLIT_PATH, "w", encoding="utf-8") as fh:
        json.dump(split_info, fh, indent=2)
    print(f"  split info   -> {SPLIT_PATH}")


if __name__ == "__main__":
    main()
