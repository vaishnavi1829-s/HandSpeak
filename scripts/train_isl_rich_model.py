"""Train a NEW ISL Random Forest on rich geometric features.

This is an experimental model.  It does NOT overwrite:

    trained_model/isl_landmark_rf.joblib        (legacy one-hand ISL)
    trained_model/isl_landmark_rf_2hand.joblib  (two-hand ISL)
    any CNN model

It loads ``dataset/landmarks/isl_rich_features.csv`` (produced by
``scripts/extract_isl_rich_features.py``) and trains a RandomForestClassifier
using the full 214-feature rich representation.

Data splitting
--------------
A deterministic stratified train/validation/test split is used
(``random_state=42``, ``test_size=0.2``, then ``val_size=0.15`` of the
remaining training data).  The split (including source image paths) is saved to
``dataset/landmarks/isl_rich_split.json`` so evaluation and the old-vs-new
comparison can reconstruct the exact same test set.  All A-Z letters are
present in the dataset, so no class is silently removed.

Run from the project root::

    python scripts/train_isl_rich_model.py
"""

from __future__ import annotations

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

from recognition.isl_feature_extractor import RICH_FEATURE_COLUMNS  # noqa: E402
from utils.label_mapping import LETTERS  # noqa: E402

LANDMARKS_DIR = os.path.join(PROJECT_ROOT, "dataset", "landmarks")
MODEL_DIR = os.path.join(PROJECT_ROOT, "trained_model")

CSV_PATH = os.path.join(LANDMARKS_DIR, "isl_rich_features.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "isl_landmark_rf_rich.joblib")
SPLIT_PATH = os.path.join(LANDMARKS_DIR, "isl_rich_split.json")

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.15

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


def _split_paths(df: pd.DataFrame, train_idx, val_idx, test_idx) -> dict:
    src = df["source_path"].to_numpy()
    return {
        "train": src[train_idx].tolist(),
        "val": src[val_idx].tolist(),
        "test": src[test_idx].tolist(),
    }


def main() -> None:
    if not os.path.exists(CSV_PATH):
        raise SystemExit(
            f"Rich feature CSV not found: {CSV_PATH}\n"
            f"Run: python scripts/extract_isl_rich_features.py"
        )

    df = pd.read_csv(CSV_PATH)
    print(f"\n[ISL RICH] Loaded {CSV_PATH}")
    print(f"  rows           : {len(df)}")
    print(f"  feature dim    : {len(RICH_FEATURE_COLUMNS)}")

    # Per-letter counts (transparency; no class is removed).
    counts = df["label"].value_counts().to_dict()
    for letter in LETTERS:
        print(f"  {letter}: {counts.get(letter, 0)}")

    X = df[RICH_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy()

    # Stratified train/test split, then train/val split (deterministic).
    idx = np.arange(len(y))
    train_val_idx, test_idx = train_test_split(
        idx,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=VAL_SIZE,
        stratify=y[train_val_idx],
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    print(f"  train/val/test : {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    clf = RandomForestClassifier(**RF_PARAMS)
    print("\n[ISL RICH] Training RandomForest...")
    clf.fit(X_train, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"  saved model -> {MODEL_PATH}")

    val_acc = float(clf.score(X_val, y_val))
    print(f"  validation accuracy: {val_acc:.4f}")

    split_info = {
        "dataset": "isl_rich",
        "csv": os.path.relpath(CSV_PATH, PROJECT_ROOT),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "val_size": VAL_SIZE,
        "feature_count": int(X.shape[1]),
        "counts": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "source_paths": _split_paths(df, train_idx, val_idx, test_idx),
    }
    with open(SPLIT_PATH, "w", encoding="utf-8") as fh:
        json.dump(split_info, fh, indent=2)
    print(f"  split info   -> {SPLIT_PATH}")


if __name__ == "__main__":
    main()
