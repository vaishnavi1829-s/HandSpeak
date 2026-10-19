"""Fine-tune the existing 26-class CNN on the collected webcam dataset (A-Z).

This is the SINGLE improved model requested: it does NOT train 26 separate
models and it does NOT modify/retrain the original SL-MNIST model or dataset.

Pipeline (one 26-class model for A-Z):

    1. Load ``dataset/webcam/<LETTER>/processed/*.png`` (28x28 grayscale, the
       exact CNN input produced by ``recognition.preprocessing``).
    2. Build a reproducible 80/20 per-class stratified train/validation split.
    3. Apply mild, webcam-realistic OFFLINE augmentation to the TRAINING split
       only (small rotation / translation / zoom / brightness; no horizontal
       flip because it changes asymmetric ASL letters).
    4. Fine-tune the existing ``trained_model/handspeak_cnn_improved.h5`` model
       with a conservative learning rate (1e-4), EarlyStopping, and
       ReduceLROnPlateau.
    5. Save the new model to ``trained_model/handspeak_cnn_webcam.h5`` (the
       original models are never overwritten).
    6. Evaluate on the held-out webcam validation split: overall accuracy,
       per-letter accuracy, confusion matrix, top confusing pairs.
    7. Compare the new model against the previous improved model on the SAME
       validation split.

Run from the project root::

    python scripts/train_webcam_cnn.py
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

from utils.config import (  # noqa: E402
    PROJECT_ROOT as _ROOT,
    MODEL_DIR,
    MODEL_IMPROVED_PATH,
    IMAGE_SIZE,
    CHANNELS,
    NUM_CLASSES,
    RANDOM_STATE,
)
from utils.label_mapping import LETTERS  # noqa: E402

WEBCAM_ROOT = os.path.join(_ROOT, "dataset", "webcam")
REPORTS_DIR = os.path.join(_ROOT, "reports")

# New model / artifact paths (never overwrite existing model files).
MODEL_WEBCAM_PATH = os.path.join(MODEL_DIR, "handspeak_cnn_webcam.h5")
HISTORY_WEBCAM_PATH = os.path.join(MODEL_DIR, "training_history_webcam.json")
CLASS_REPORT_WEBCAM_PATH = os.path.join(REPORTS_DIR, "classification_report_webcam.txt")
CONFUSION_CSV_WEBCAM_PATH = os.path.join(REPORTS_DIR, "confusion_matrix_webcam.csv")
CONFUSION_PNG_WEBCAM_PATH = os.path.join(REPORTS_DIR, "confusion_matrix_webcam.png")
PLOT_WEBCAM_PATH = os.path.join(REPORTS_DIR, "training_accuracy_loss_webcam.png")
PER_LETTER_REPORT_PATH = os.path.join(REPORTS_DIR, "per_letter_accuracy_webcam.txt")
COMPARISON_PATH = os.path.join(REPORTS_DIR, "webcam_model_comparison.txt")

# Fine-tuning hyperparameters.
FINE_TUNE_LR = 1e-4
EPOCHS = 50
PATIENCE = 10
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.2
AUG_COPIES = 2  # number of augmented copies per training sample

# Difficult pairs to highlight in the confusion report.
DIFFICULT_PAIRS = [
    ("A", "Y"), ("D", "P"), ("E", "M"), ("S", "T"), ("U", "V"),
    ("H", "N"), ("G", "K"), ("C", "O"), ("R", "K"), ("Z", "O"),
]


def _natural_key(name: str) -> int:
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else 0


def _png_files(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    return sorted(
        name
        for name in os.listdir(folder)
        if name.lower().endswith((".png", ".jpg", ".jpeg"))
    )


def load_webcam_dataset() -> tuple[np.ndarray, np.ndarray, dict]:
    """Load all webcam processed 28x28 images and their A-Z labels.

    Returns
    -------
    (X, y, counts) where ``X`` is float32 (N, 28, 28, 1) in [0, 1], ``y`` is
    int (N,) with values 0-25, and ``counts`` maps letter -> number of samples.
    """
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    counts: dict[str, int] = {}

    for letter in LETTERS:
        label_index = ord(letter) - ord("A")
        folder = os.path.join(WEBCAM_ROOT, letter, "processed")
        files = _png_files(folder)
        counts[letter] = len(files)

        for name in files:
            path = os.path.join(folder, name)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape[0] != IMAGE_SIZE or img.shape[1] != IMAGE_SIZE:
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE),
                                 interpolation=cv2.INTER_AREA)
            arr = img.astype(np.float32) / 255.0
            X_list.append(arr.reshape(IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
            y_list.append(label_index)

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    return X, y, counts


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


def augment_image(img: np.ndarray) -> np.ndarray:
    """Apply one mild, webcam-realistic augmentation to a 28x28 image (in [0,1]).

    No horizontal flip: it would change the identity of asymmetric ASL letters.
    """
    img_u8 = (img * 255.0).astype(np.uint8)
    h, w = img_u8.shape

    # Small rotation.
    angle = np.random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img_u8 = cv2.warpAffine(img_u8, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Small translation.
    tx = int(np.random.uniform(-2, 2))
    ty = int(np.random.uniform(-2, 2))
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    img_u8 = cv2.warpAffine(img_u8, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Small zoom (scale + recenter).
    scale = np.random.uniform(0.9, 1.1)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
    img_u8 = cv2.warpAffine(img_u8, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Mild brightness / contrast variation.
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
    axes[0].set_title("Webcam Fine-tune: Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].plot(epochs, loss, "b-", label="Training Loss")
    axes[1].plot(epochs, val_loss, "r-", label="Validation Loss")
    axes[1].set_title("Webcam Fine-tune: Loss")
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
        title="Confusion Matrix (A-Z, Webcam Fine-tuned)",
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


def per_letter_accuracy(y_true, y_pred, class_names):
    """Return per-letter accuracy string and the list of (letter, acc, support)."""
    rows = []
    for i, letter in enumerate(class_names):
        mask = y_true == i
        support = int(mask.sum())
        correct = int((y_pred[mask] == i).sum()) if support else 0
        acc = correct / support if support else 0.0
        rows.append((letter, acc, support))
    return rows


def top_confusing_pairs(cm, class_names, top_k=10):
    """Return the top off-diagonal (true, predicted) confusion pairs."""
    pairs = []
    n = cm.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                pairs.append((class_names[i], class_names[j], int(cm[i, j])))
    pairs.sort(key=lambda t: t[2], reverse=True)
    return pairs[:top_k]


def evaluate_model(model, X_val, y_val_int, class_names, label):
    """Evaluate a model and print/return overall + per-letter accuracy."""
    probs = model.predict(X_val, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    acc = float(np.mean(y_pred == y_val_int))
    return y_pred, acc


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  saved: {path}")


def main():
    print("=" * 70)
    print("HandSpeak - Webcam CNN Fine-tune (ONE 26-class A-Z model)")
    print("=" * 70)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load webcam dataset.
    # ------------------------------------------------------------------
    print("\n[1] Loading webcam dataset ...")
    X, y, counts = load_webcam_dataset()
    total = int(X.shape[0])
    print(f"  Total samples: {total}")
    print("  Per-letter counts:")
    line = "  ".join(f"{l}:{counts[l]}" for l in LETTERS)
    print(f"    {line}")

    # ------------------------------------------------------------------
    # 2. Stratified 80/20 split.
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
    # 3. Augment training split only.
    # ------------------------------------------------------------------
    print(f"\n[3] Augmenting training split (offline, {AUG_COPIES} copies/sample) ...")
    X_train_aug, y_train_aug = augment_training_split(X_train, y_train_int,
                                                      copies=AUG_COPIES)
    print(f"  Augmented train: {X_train_aug.shape[0]} samples")

    y_train_cat = to_categorical(y_train_aug, num_classes=NUM_CLASSES)
    y_val_cat = to_categorical(y_val_int, num_classes=NUM_CLASSES)

    # ------------------------------------------------------------------
    # 4. Load base model and fine-tune.
    # ------------------------------------------------------------------
    print("\n[4] Loading base model for fine-tuning ...")
    print(f"  base model: {MODEL_IMPROVED_PATH}")
    if not os.path.exists(MODEL_IMPROVED_PATH):
        raise FileNotFoundError(f"Base model not found: {MODEL_IMPROVED_PATH}")

    model = load_model(MODEL_IMPROVED_PATH)
    print(f"  input {model.input_shape} -> output {model.output_shape}")

    # Recompile with the conservative fine-tune learning rate.
    from keras import optimizers
    model.compile(
        optimizer=optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4,
                          min_lr=1e-6, verbose=1),
        ModelCheckpoint(
            filepath=MODEL_WEBCAM_PATH,
            monitor="val_accuracy", mode="max",
            save_best_only=True, verbose=1,
        ),
    ]

    print(f"\nFine-tuning up to {EPOCHS} epochs (LR={FINE_TUNE_LR}, "
          f"batch={BATCH_SIZE}) ...")
    history = model.fit(
        X_train_aug, y_train_cat,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val_cat),
        callbacks=callbacks,
        verbose=1,
    )

    save_history(history, HISTORY_WEBCAM_PATH)
    plot_history(history, PLOT_WEBCAM_PATH)

    # Reload the best-checkpointed weights.
    model = load_model(MODEL_WEBCAM_PATH)
    model.save(MODEL_WEBCAM_PATH)
    print(f"  saved new model: {MODEL_WEBCAM_PATH}")

    # ------------------------------------------------------------------
    # 5. Evaluate new webcam model on the held-out validation split.
    # ------------------------------------------------------------------
    print("\n[5] Evaluating NEW webcam-fine-tuned model (validation split) ...")
    y_pred_new, new_val_acc = evaluate_model(model, X_val, y_val_int,
                                             LETTERS, "webcam")
    print(f"  Validation accuracy: {new_val_acc * 100:.4f}%")

    cm_new = save_confusion_matrix(
        y_val_int, y_pred_new, LETTERS,
        CONFUSION_CSV_WEBCAM_PATH, CONFUSION_PNG_WEBCAM_PATH,
    )

    report = classification_report(
        y_val_int, y_pred_new, labels=list(range(NUM_CLASSES)),
        target_names=LETTERS, digits=4,
    )
    write_text(CLASS_REPORT_WEBCAM_PATH, report)

    # Per-letter validation accuracy.
    per_rows = per_letter_accuracy(y_val_int, y_pred_new, LETTERS)
    per_lines = ["Per-letter validation accuracy (webcam fine-tuned model)",
                 "Letter  Accuracy   Support", "-" * 30]
    for letter, acc, support in per_rows:
        per_lines.append(f"{letter:<7} {acc * 100:>7.2f}%  {support:>7}")
    per_text = "\n".join(per_lines)
    write_text(PER_LETTER_REPORT_PATH, per_text)

    print("\n  Per-letter validation accuracy (new model):")
    for letter, acc, support in per_rows:
        print(f"    {letter}: {acc * 100:6.2f}%  (n={support})")

    top_pairs = top_confusing_pairs(cm_new, LETTERS, top_k=10)
    print("\n  Top confusing pairs (new model):")
    if top_pairs:
        for a, b, cnt in top_pairs:
            print(f"    {a} -> {b}: {cnt}")
    else:
        print("    None")

    # Highlight the special difficult pairs.
    print("\n  Special difficult-pair check (true -> predicted counts):")
    for a, b in DIFFICULT_PAIRS:
        ia, ib = ord(a) - ord("A"), ord(b) - ord("A")
        forward = int(cm_new[ia, ib])
        backward = int(cm_new[ib, ia])
        print(f"    {a}->{b}: {forward}    {b}->{a}: {backward}")

    # ------------------------------------------------------------------
    # 6. Compare against the previous improved model on the SAME split.
    # ------------------------------------------------------------------
    print("\n[6] Comparing previous improved model on the SAME validation split ...")
    base_model = load_model(MODEL_IMPROVED_PATH)
    y_pred_old, old_val_acc = evaluate_model(base_model, X_val, y_val_int,
                                             LETTERS, "improved")
    print(f"  Previous improved model validation accuracy: {old_val_acc * 100:.4f}%")

    cm_old = confusion_matrix(y_val_int, y_pred_old,
                              labels=list(range(NUM_CLASSES)))

    # Build comparison report.
    cmp_lines = [
        "Webcam-domain model comparison (same held-out webcam validation split)",
        "=" * 70,
        f"Validation samples       : {X_val.shape[0]}",
        f"Previous improved acc    : {old_val_acc * 100:.4f}%",
        f"New webcam-finetuned acc : {new_val_acc * 100:.4f}%",
        f"Absolute improvement     : {(new_val_acc - old_val_acc) * 100:+.4f} pp",
        "",
        "Per-letter comparison (previous -> new):",
        "Letter  Prev%     New%    Support",
        "-" * 40,
    ]
    old_rows = per_letter_accuracy(y_val_int, y_pred_old, LETTERS)
    new_rows = per_letter_accuracy(y_val_int, y_pred_new, LETTERS)
    for (letter, old_acc, sup), (_, new_acc, _) in zip(old_rows, new_rows):
        cmp_lines.append(
            f"{letter:<7} {old_acc * 100:>6.2f}%  {new_acc * 100:>6.2f}%  {sup:>7}"
        )
    cmp_lines.append("")
    cmp_lines.append("Previous improved model top confusions:")
    for a, b, cnt in top_confusing_pairs(cm_old, LETTERS, top_k=10):
        cmp_lines.append(f"    {a} -> {b}: {cnt}")
    cmp_lines.append("")
    cmp_lines.append("New webcam model top confusions:")
    for a, b, cnt in top_pairs:
        cmp_lines.append(f"    {a} -> {b}: {cnt}")
    cmp_text = "\n".join(cmp_lines)
    write_text(COMPARISON_PATH, cmp_text)

    # ------------------------------------------------------------------
    # 7. Summary.
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total webcam samples          : {total}")
    print(f"  Train / validation            : {X_train.shape[0]} / {X_val.shape[0]}")
    print(f"  Previous improved val acc     : {old_val_acc * 100:.4f}%")
    print(f"  New webcam-finetuned val acc  : {new_val_acc * 100:.4f}%")
    print(f"  Improvement                   : {(new_val_acc - old_val_acc) * 100:+.4f} pp")
    print(f"  New model saved to            : {MODEL_WEBCAM_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()
