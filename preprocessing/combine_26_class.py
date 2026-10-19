"""Build the combined 26-class (A-Z) sign language dataset.

This module is the **preprocessing pipeline** that merges two data sources:

1. Sign Language MNIST  (24 classes, 28x28 grayscale, CSV format)
   - Raw labels 0-8 = A-I and 10-24 = K-Y (J=9 and Z=25 are absent).
2. J/Z ASL alphabet images (200x200 RGB JPEG, ``dataset/jz/J`` and
   ``dataset/jz/Z``) recovered from the classic ASL Alphabet dataset.

The output is a single 26-class dataset with **contiguous labels 0-25 (A-Z)**:

    - SLMNIST raw labels are kept as-is (0-8 = A-I, 10-24 = K-Y).
    - J images -> label 9
    - Z images -> label 25

This module only *reads* the dataset files. It never modifies the source
CSVs or the J/Z images. Processed arrays are written to ``output/``.

It does NOT train a model, launch Flask, open the camera, or run TTS.
"""

import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Ensure the project root is importable when this module is run directly.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.config import (
    PROJECT_ROOT,
    RAW_TRAIN_CSV,
    RAW_TEST_CSV,
    IMAGE_SIZE,
    CHANNELS,
    RANDOM_STATE,
    VALIDATION_SPLIT,
)

# ---------------------------------------------------------------------------
# Final dataset constants
# ---------------------------------------------------------------------------
NUM_CLASSES_26 = 26               # A-Z
J_LABEL = 9                       # J is the 10th letter (A=0)
Z_LABEL = 25                      # Z is the 26th letter (A=0)

J_DIR = os.path.join(PROJECT_ROOT, "dataset", "jz", "J")
Z_DIR = os.path.join(PROJECT_ROOT, "dataset", "jz", "Z")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# J/Z split: hold out this fraction of each letter for the test set.
JZ_TEST_SPLIT = 0.2

LETTERS = [chr(ord("A") + i) for i in range(NUM_CLASSES_26)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _natural_key(filename):
    """Return a sort key that orders ``J2`` before ``J10`` (natural order)."""
    digits = "".join(ch for ch in filename if ch.isdigit())
    return int(digits) if digits else 0


def _list_images(folder):
    """Return image paths in a folder, sorted in natural order."""
    names = sorted(os.listdir(folder), key=_natural_key)
    return [os.path.join(folder, n) for n in names]


def load_slmnist():
    """Load Sign Language MNIST CSVs into 28x28x1 float arrays (0-1).

    Returns
    -------
    (X_train, y_train, X_test, y_test) as float32 numpy arrays.
    """
    train_df = pd.read_csv(RAW_TRAIN_CSV)
    test_df = pd.read_csv(RAW_TEST_CSV)

    X_train = train_df.drop("label", axis=1).values.astype(np.float32)
    y_train = train_df["label"].values.astype(np.int64)
    X_test = test_df.drop("label", axis=1).values.astype(np.float32)
    y_test = test_df["label"].values.astype(np.int64)

    X_train = X_train.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS) / 255.0
    X_test = X_test.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS) / 255.0

    return X_train, y_train, X_test, y_test


def load_jz(folder, label):
    """Load and preprocess one J/Z letter folder.

    The J/Z ASL images use the **opposite polarity** from Sign Language MNIST:
    the hand is light and the background is dark (SL-MNIST has a dark hand on a
    light background). To keep a single, consistent 26-class format we invert
    each J/Z image so the hand is dark and the background is light, matching
    both SL-MNIST and the live webcam preprocessing path.

    Returns
    -------
    (X, y) where X has shape (n, 28, 28, 1) and y is filled with ``label``.
    """
    paths = _list_images(folder)
    if not paths:
        raise FileNotFoundError(f"No images found in {folder}")

    X = np.empty((len(paths), IMAGE_SIZE, IMAGE_SIZE, CHANNELS),
                 dtype=np.float32)
    for i, path in enumerate(paths):
        with Image.open(path) as img:
            # Grayscale + resize to match the SLMNIST 28x28 shape.
            gray = img.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
            arr = np.asarray(gray, dtype=np.float32) / 255.0
            # Normalize polarity: hand (foreground) -> dark, background -> light.
            # The J/Z source has a light hand on a dark background, so invert.
            X[i, :, :, 0] = 1.0 - arr

    y = np.full(len(paths), label, dtype=np.int64)
    return X, y


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def build_dataset():
    """Load all sources and return the combined 26-class dataset.

    Returns
    -------
    dict with keys:
        X_train, y_train (one-hot), X_val, y_val (one-hot),
        X_test, y_test (one-hot), and y_test_raw for reporting.
    """
    # 1. SLMNIST (24 classes).
    X_train_sl, y_train_sl, X_test_sl, y_test_sl = load_slmnist()

    # 2. J and Z images.
    X_j, y_j = load_jz(J_DIR, J_LABEL)
    X_z, y_z = load_jz(Z_DIR, Z_LABEL)

    # 3. Split J and Z into train/test (mirrors the SLMNIST split layout).
    X_j_train, X_j_test, y_j_train, y_j_test = train_test_split(
        X_j, y_j, test_size=JZ_TEST_SPLIT, random_state=RANDOM_STATE,
    )
    X_z_train, X_z_test, y_z_train, y_z_test = train_test_split(
        X_z, y_z, test_size=JZ_TEST_SPLIT, random_state=RANDOM_STATE,
    )

    # 4. Combine train and test sources.
    X_train_all = np.concatenate([X_train_sl, X_j_train, X_z_train], axis=0)
    y_train_all = np.concatenate([y_train_sl, y_j_train, y_z_train], axis=0)

    X_test = np.concatenate([X_test_sl, X_j_test, X_z_test], axis=0)
    y_test_raw = np.concatenate([y_test_sl, y_j_test, y_z_test], axis=0)

    # 5. Hold out a validation set from the combined training data.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all,
        y_train_all,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_STATE,
        stratify=y_train_all,
    )

    # 6. One-hot encode all labels (26 classes).
    y_train_oh = to_categorical(y_train, num_classes=NUM_CLASSES_26)
    y_val_oh = to_categorical(y_val, num_classes=NUM_CLASSES_26)
    y_test_oh = to_categorical(y_test_raw, num_classes=NUM_CLASSES_26)

    return {
        "X_train": X_train,
        "y_train": y_train_oh,
        "y_train_raw": y_train,
        "X_val": X_val,
        "y_val": y_val_oh,
        "y_val_raw": y_val,
        "X_test": X_test,
        "y_test": y_test_oh,
        "y_test_raw": y_test_raw,
    }


def report_class_distribution(split_name, y_raw):
    """Print per-letter counts for a raw-label array."""
    counts = Counter(int(v) for v in y_raw)
    print(f"\n  [{split_name}] {len(y_raw):>7} samples across "
          f"{len(counts)} classes")
    for i in range(NUM_CLASSES_26):
        print(f"    {LETTERS[i]} ({i:>2}): {counts.get(i, 0)}")


def save_outputs(data):
    """Persist the processed arrays to ``output/`` as .npy files."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = {
        "X_train.npy": data["X_train"],
        "y_train.npy": data["y_train"],
        "X_val.npy": data["X_val"],
        "y_val.npy": data["y_val"],
        "X_test.npy": data["X_test"],
        "y_test.npy": data["y_test"],
    }
    for name, arr in files.items():
        np.save(os.path.join(OUTPUT_DIR, name), arr)
        print(f"  saved {name}: {arr.shape} {arr.dtype}")


def main():
    print("=" * 60)
    print("HandSpeak - 26-class dataset preprocessing (A-Z)")
    print("=" * 60)

    data = build_dataset()

    print("\nShapes:")
    for key in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"):
        print(f"  {key:>10}: {data[key].shape} {data[key].dtype}")

    print("\nClass distribution (raw labels):")
    report_class_distribution("train", data["y_train_raw"])
    report_class_distribution("val", data["y_val_raw"])
    report_class_distribution("test", data["y_test_raw"])

    print("\nSaving processed arrays ...")
    save_outputs(data)

    print("\nDone. 26-class dataset is ready for training.")


if __name__ == "__main__":
    main()
