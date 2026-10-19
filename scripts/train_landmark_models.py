"""Train Random Forest baseline models on MediaPipe hand-landmark features.

Produces two INDEPENDENT models (ASL and ISL are never mixed):

    trained_model/asl_landmark_rf.joblib
    trained_model/isl_landmark_rf.joblib

For each language the script:

    1. Loads ``dataset/landmarks/<lang>_landmarks.csv``.
    2. Splits into train / validation / test with stratified splitting so every
       letter is represented in every split.
    3. Trains a ``RandomForestClassifier`` (reproducible ``random_state``).
    4. Saves the model with ``joblib``.

The existing CNN models (``handspeak_cnn*.h5``) are never touched.

Run from the project root::

    python scripts/train_landmark_models.py
    python scripts/train_landmark_models.py --dataset asl
    python scripts/train_landmark_models.py --dataset isl
"""

from __future__ import annotations

import argparse
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

from recognition.landmark_extractor import FEATURE_COLUMNS  # noqa: E402
from utils.label_mapping import LETTERS  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LANDMARKS_DIR = os.path.join(PROJECT_ROOT, "dataset", "landmarks")
MODEL_DIR = os.path.join(PROJECT_ROOT, "trained_model")

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.15  # fraction of the remaining training data used for validation

# Reasonable, reproducible Random Forest configuration.
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

DATASETS = {
    "asl": {
        "csv": os.path.join(LANDMARKS_DIR, "asl_landmarks.csv"),
        "model": os.path.join(MODEL_DIR, "asl_landmark_rf.joblib"),
        "split": os.path.join(LANDMARKS_DIR, "asl_landmark_split.json"),
    },
    "isl": {
        "csv": os.path.join(LANDMARKS_DIR, "isl_landmarks.csv"),
        "model": os.path.join(MODEL_DIR, "isl_landmark_rf.joblib"),
        "split": os.path.join(LANDMARKS_DIR, "isl_landmark_split.json"),
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_features(csv_path: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load landmark features and labels from a CSV.

    Returns ``(X, y, df)`` where ``X`` is float32 ``(N, 63)`` and ``y`` is the
    string letter label ``(N,)``.
    """
    df = pd.read_csv(csv_path)
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy()
    return X, y, df


def stratified_split(X: np.ndarray, y: np.ndarray):
    """Split into train/validation/test, stratified on the letter label."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=VAL_SIZE,
        stratify=y_train,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def train_dataset(dataset_name: str) -> None:
    cfg = DATASETS[dataset_name]

    if not os.path.exists(cfg["csv"]):
        raise SystemExit(
            f"Landmark CSV not found: {cfg['csv']}\n"
            f"Run `python scripts/extract_landmarks.py --dataset {dataset_name}` first."
        )

    print(f"\n[{dataset_name.upper()}] Loading {cfg['csv']}")
    X, y, df = load_features(cfg["csv"])
    print(f"  samples        : {X.shape[0]}")
    print(f"  feature dim    : {X.shape[1]}")
    print(f"  classes        : {len(set(y.tolist()))}")
    print(f"  per-letter min : {pd.Series(y).value_counts().min()}")

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = stratified_split(X, y)
    print(f"  train/val/test : {len(y_train)}/{len(y_val)}/{len(y_test)}")

    clf = RandomForestClassifier(**RF_PARAMS)
    print(f"\n[{dataset_name.upper()}] Training RandomForest...")
    clf.fit(X_train, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, cfg["model"])
    print(f"  saved model -> {cfg['model']}")

    # Persist the exact split for transparent, reproducible evaluation.
    split_info = {
        "dataset": dataset_name,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "val_size": VAL_SIZE,
        "train_indices": None,  # indices are not stable across CSV re-loads,
        "note": "Split is deterministic given the same CSV ordering and random_state.",
        "counts": {
            "train": int(len(y_train)),
            "val": int(len(y_val)),
            "test": int(len(y_test)),
        },
    }
    with open(cfg["split"], "w", encoding="utf-8") as fh:
        import json

        json.dump(split_info, fh, indent=2)

    # Quick validation accuracy for immediate feedback.
    val_acc = float(clf.score(X_val, y_val))
    print(f"  validation accuracy: {val_acc:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["asl", "isl", "all"],
        default="all",
        help="Which model to train (default: all).",
    )
    args = parser.parse_args()

    names = ["asl", "isl"] if args.dataset == "all" else [args.dataset]
    for name in names:
        train_dataset(name)


if __name__ == "__main__":
    main()
