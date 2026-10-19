"""Train and evaluate a 26-class INDIAN SIGN LANGUAGE (ISL) CNN.

This script consumes the collected ISL webcam dataset under
``dataset/isl_webcam/<LETTER>/processed/*.png`` and:

    1. PREPROCESSING CHECK: verifies every sample is a readable 28x28 grayscale
       image and is compatible with the existing ``recognition.preprocessing``
       pipeline (which produced it).
    2. TRAIN/VALIDATION SPLIT: reproducible 80/20 per-class stratified split.
    3. DATA AUGMENTATION: mild, webcam-realistic OFFLINE augmentation applied to
       the TRAINING split only (small rotation / translation / zoom / brightness;
       no horizontal flip, which would change asymmetric ISL signs).
    4. TRAIN: builds a fresh 26-class CNN (same architecture as
       ``models/cnn_model.py``) using EarlyStopping, ReduceLROnPlateau, and
       ModelCheckpoint.
    5. SAVE: writes the model to ``trained_model/handspeak_cnn_isl.h5``.
    6. ACCURACY + CONFUSION MATRIX: evaluates on the held-out validation split
       and saves reports under ``reports/``.

The existing ASL model and the ASL webcam dataset are never modified.

Run from the project root::

    python scripts/train_isl_cnn.py
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless-safe backend for saving figures

import cv2
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  # noqa: E402
from keras.models import load_model  # noqa: E402
from keras.utils import to_categorical  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from models.cnn_model import build_cnn_model  # noqa: E402
from utils.config import (  # noqa: E402
    MODEL_DIR,
    MODEL_ISL_PATH,
    HISTORY_ISL_PATH,
    IMAGE_SIZE,
    CHANNELS,
    INPUT_SHAPE,
    NUM_CLASSES,
    RANDOM_STATE,
)
from utils.label_mapping import LETTERS  # noqa: E402

ISL_ROOT = os.path.join(PROJECT_ROOT, "dataset", "isl_webcam")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

CLASS_REPORT_ISL_PATH = os.path.join(REPORTS_DIR, "classification_report_isl.txt")
CONFUSION_CSV_ISL_PATH = os.path.join(REPORTS_DIR, "confusion_matrix_isl.csv")
CONFUSION_PNG_ISL_PATH = os.path.join(REPORTS_DIR, "confusion_matrix_isl.png")
PLOT_ISL_PATH = os.path.join(REPORTS_DIR, "training_accuracy_loss_isl.png")
PER_LETTER_REPORT_ISL_PATH = os.path.join(REPORTS_DIR, "per_letter_accuracy_isl.txt")

# Training hyperparameters.
LEARNING_RATE = 1e-3
L2 = 1e-4
EPOCHS = 60
PATIENCE = 10
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2
AUG_COPIES = 2  # number of augmented copies per training sample


def _png_files(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    return sorted(
        name
        for name in os.listdir(folder)
        if name.lower().endswith((".png", ".jpg", ".jpeg"))
    )


# ---------------------------------------------------------------------------
# Step 1: dataset verification + Step 2: preprocessing compatibility check
# ---------------------------------------------------------------------------
def load_isl_dataset():
    """Load all ISL processed 28x28 images and their A-Z labels.

    Blank/constant images and unreadable files are skipped (they are recorded so
    the operator can review them). Returns (X, y, counts, skipped).
    """
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    counts: dict[str, int] = {}
    skipped: list[str] = []

    for letter in LETTERS:
        label_index = ord(letter) - ord("A")
        folder = os.path.join(ISL_ROOT, letter, "processed")
        files = _png_files(folder)
        counts[letter] = len(files)

        for name in files:
            path = os.path.join(folder, name)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                skipped.append(path)
                counts[letter] -= 1
                continue
            if img.shape[0] != IMAGE_SIZE or img.shape[1] != IMAGE_SIZE:
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE),
                                 interpolation=cv2.INTER_AREA)
            if img.max() - img.min() <= 0:
                skipped.append(path)
                counts[letter] -= 1
                continue
            arr = img.astype(np.float32) / 255.0
            X_list.append(arr.reshape(IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
            y_list.append(label_index)

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    return X, y, counts, skipped


# ---------------------------------------------------------------------------
# Step 3: stratified train/validation split
# ---------------------------------------------------------------------------
def build_split(X: np.ndarray, y: np.ndarray):
    """Create a reproducible 80/20 per-class stratified split."""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=VALIDATION_SPLIT,
        stratify=y,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    return X_train, X_val, y_train, y_val


# ---------------------------------------------------------------------------
# Step 4: offline augmentation (training split only)
# ---------------------------------------------------------------------------
def augment_image(img: np.ndarray) -> np.ndarray:
    """Apply one mild, webcam-realistic augmentation to a 28x28 image (in [0,1]).

    No horizontal flip: it would change the identity of asymmetric ISL signs.
    """
    img_u8 = (img * 255.0).astype(np.uint8)
    h, w = img_u8.shape

    angle = np.random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img_u8 = cv2.warpAffine(img_u8, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    tx = int(np.random.uniform(-2, 2))
    ty = int(np.random.uniform(-2, 2))
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    img_u8 = cv2.warpAffine(img_u8, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    scale = np.random.uniform(0.9, 1.1)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
    img_u8 = cv2.warpAffine(img_u8, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    if np.random.rand() < 0.5:
        beta = np.random.uniform(-15, 15)
        img_u8 = np.clip(img_u8.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    return img_u8.astype(np.float32) / 255.0


def augment_training_split(X_train: np.ndarray, y_train: np.ndarray,
                           copies: int = AUG_COPIES):
    """Return original training samples plus ``copies`` augmented variants."""
    Xs = [X_train]
    ys = [y_train]
    n = len(X_train)
    for _ in range(copies):
        aug = np.empty_like(X_train)
        for i in range(n):
            aug[i, :, :, 0] = augment_image(X_train[i, :, :, 0])
        Xs.append(aug)
        ys.append(y_train)
    X_out = np.concatenate(Xs, axis=0)
    y_out = np.concatenate(ys, axis=0)

    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.permutation(len(X_out))
    return X_out[idx], y_out[idx]


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def save_history(history, path):
    serializable = {
        key: [float(v) for v in values] for key, values in history.history.items()
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"  saved training history: {path}")


def plot_history(history, path):
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, acc, "b-", label="Training Accuracy")
    axes[0].plot(epochs, val_acc, "r-", label="Validation Accuracy")
    axes[0].set_title("ISL Training: Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].plot(epochs, loss, "b-", label="Training Loss")
    axes[1].plot(epochs, val_loss, "r-", label="Validation Loss")
    axes[1].set_title("ISL Training: Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved accuracy/loss graph: {path}")


def save_confusion_matrix(y_true, y_pred, class_names, csv_path, png_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    np.savetxt(csv_path, cm, delimiter=",", fmt="%d",
               header=",".join(class_names), comments="")

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        title="Confusion Matrix (A-Z, ISL)",
        xlabel="Predicted Label",
        ylabel="True Label",
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=6)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  saved confusion matrix: {png_path} / {csv_path}")
    return cm


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  saved: {path}")


def per_letter_accuracy(y_true, y_pred, class_names):
    rows = []
    for i, letter in enumerate(class_names):
        mask = y_true == i
        support = int(mask.sum())
        correct = int((y_pred[mask] == i).sum()) if support else 0
        acc = correct / support if support else 0.0
        rows.append((letter, acc, support))
    return rows


def main():
    print("=" * 70)
    print("HandSpeak - ISL CNN Training (ONE 26-class A-Z model)")
    print("=" * 70)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 + 2: load + verify dataset / preprocessing compatibility.
    # ------------------------------------------------------------------
    print("\n[1] Loading ISL webcam dataset ...")
    X, y, counts, skipped = load_isl_dataset()
    total = int(X.shape[0])
    print(f"  Total usable samples: {total}")
    if skipped:
        print(f"  Skipped {len(skipped)} unreadable/blank files (see below):")
        for s in skipped[:20]:
            print(f"    - {s}")
    print("  Per-letter counts:")
    line = "  ".join(f"{l}:{counts[l]}" for l in LETTERS)
    print(f"    {line}")

    missing = [l for l in LETTERS if counts[l] < 10]
    if missing:
        print("  WARNING: under-represented classes (need >=10): "
              + "".join(missing))
    else:
        print("  OK: every class has >= 10 usable samples.")

    # ------------------------------------------------------------------
    # Step 3: stratified 80/20 split.
    # ------------------------------------------------------------------
    print("\n[2] Building 80/20 per-class stratified split ...")
    X_train, X_val, y_train_int, y_val_int = build_split(X, y)
    print(f"  Train: {X_train.shape[0]}  Val: {X_val.shape[0]}")

    train_counts = {l: 0 for l in LETTERS}
    val_counts = {l: 0 for l in LETTERS}
    for lab in y_train_int:
        train_counts[LETTERS[lab]] += 1
    for lab in y_val_int:
        val_counts[LETTERS[lab]] += 1
    print("  Train per letter: " + "  ".join(f"{l}:{train_counts[l]}" for l in LETTERS))
    print("  Val   per letter: " + "  ".join(f"{l}:{val_counts[l]}" for l in LETTERS))

    # ------------------------------------------------------------------
    # Step 4: augment training split only.
    # ------------------------------------------------------------------
    print(f"\n[3] Augmenting training split (offline, {AUG_COPIES} copies/sample) ...")
    X_train_aug, y_train_aug = augment_training_split(X_train, y_train_int,
                                                      copies=AUG_COPIES)
    print(f"  Augmented train: {X_train_aug.shape[0]} samples")

    y_train_cat = to_categorical(y_train_aug, num_classes=NUM_CLASSES)
    y_val_cat = to_categorical(y_val_int, num_classes=NUM_CLASSES)

    # ------------------------------------------------------------------
    # Step 5: build and train.
    # ------------------------------------------------------------------
    print("\n[4] Building ISL CNN (fresh 26-class model) ...")
    model = build_cnn_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES,
                            learning_rate=LEARNING_RATE, l2=L2)
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4,
                          min_lr=1e-6, verbose=1),
        ModelCheckpoint(
            filepath=MODEL_ISL_PATH,
            monitor="val_accuracy", mode="max",
            save_best_only=True, verbose=1,
        ),
    ]

    print(f"\nTraining up to {EPOCHS} epochs (LR={LEARNING_RATE}, batch={BATCH_SIZE}) ...")
    history = model.fit(
        X_train_aug, y_train_cat,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val_cat),
        callbacks=callbacks,
        verbose=1,
    )

    save_history(history, HISTORY_ISL_PATH)
    plot_history(history, PLOT_ISL_PATH)

    # Reload the best-checkpointed weights.
    model = load_model(MODEL_ISL_PATH)
    model.save(MODEL_ISL_PATH)
    print(f"  saved new ISL model: {MODEL_ISL_PATH}")

    # ------------------------------------------------------------------
    # Step 6: accuracy + confusion matrix.
    # ------------------------------------------------------------------
    print("\n[5] Evaluating ISL model on held-out validation split ...")
    probs = model.predict(X_val, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    val_acc = float(np.mean(y_pred == y_val_int))
    print(f"  Validation accuracy: {val_acc * 100:.4f}%")

    cm = save_confusion_matrix(
        y_val_int, y_pred, LETTERS,
        CONFUSION_CSV_ISL_PATH, CONFUSION_PNG_ISL_PATH,
    )

    report = classification_report(
        y_val_int, y_pred, labels=list(range(NUM_CLASSES)),
        target_names=LETTERS, digits=4,
    )
    write_text(CLASS_REPORT_ISL_PATH, report)

    per_rows = per_letter_accuracy(y_val_int, y_pred, LETTERS)
    per_lines = ["Per-letter validation accuracy (ISL model)",
                 "Letter  Accuracy   Support", "-" * 30]
    for letter, acc, support in per_rows:
        per_lines.append(f"{letter:<7} {acc * 100:>7.2f}%  {support:>7}")
    write_text(PER_LETTER_REPORT_ISL_PATH, "\n".join(per_lines))

    print("\n  Per-letter validation accuracy (ISL model):")
    for letter, acc, support in per_rows:
        print(f"    {letter}: {acc * 100:6.2f}%  (n={support})")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total ISL samples used        : {total}")
    print(f"  Train / validation            : {X_train.shape[0]} / {X_val.shape[0]}")
    print(f"  ISL validation accuracy       : {val_acc * 100:.4f}%")
    print(f"  ISL model saved to            : {MODEL_ISL_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()
