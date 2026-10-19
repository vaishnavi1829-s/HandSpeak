"""Train and evaluate the improved HandSpeak 26-class (A-Z) CNN.

This script consumes the regenerated preprocessed dataset in ``output/``
(where J/Z polarity has been normalized to match SL-MNIST and webcam frames)
and:

    1. Builds the improved CNN defined in ``models/cnn_model.py`` (L2 regularized).
    2. Applies mild, realistic OFFLINE augmentation to the TRAINING split only
       (small rotation/translation/zoom/brightness; no horizontal flip).
    3. Applies softened inverse-frequency class weights (mild J/Z imbalance).
    4. Uses EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint callbacks.
    5. Saves the best model to ``trained_model/handspeak_cnn_improved.h5``.
    6. Evaluates on the untouched test split and saves reports/graphs.

The original ``handspeak_cnn.h5`` is never modified.

Run from the project root::

    python scripts/train_improved_cnn.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless-safe backend for saving figures

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  # noqa: E402

from models.cnn_model import build_cnn_model  # noqa: E402
from utils.config import (  # noqa: E402
    PROJECT_ROOT as _ROOT,
    MODEL_DIR,
    MODEL_IMPROVED_PATH,
    HISTORY_IMPROVED_PATH,
    CLASS_REPORT_IMPROVED_PATH,
    CONFUSION_MATRIX_IMPROVED_PATH,
    CONFUSION_CSV_IMPROVED_PATH,
    PLOT_IMPROVED_PATH,
    INPUT_SHAPE,
    BATCH_SIZE,
)
from utils.label_mapping import LETTERS  # noqa: E402

OUTPUT_DIR = os.path.join(_ROOT, "output")
REPORTS_DIR = os.path.join(_ROOT, "reports")

NUM_CLASSES = 26
EPOCHS = 40
LEARNING_RATE = 1e-3
L2 = 1e-4
PATIENCE = 8

X_TRAIN_PATH = os.path.join(OUTPUT_DIR, "X_train.npy")
Y_TRAIN_PATH = os.path.join(OUTPUT_DIR, "y_train.npy")
X_VAL_PATH = os.path.join(OUTPUT_DIR, "X_val.npy")
Y_VAL_PATH = os.path.join(OUTPUT_DIR, "y_val.npy")
X_TEST_PATH = os.path.join(OUTPUT_DIR, "X_test.npy")
Y_TEST_PATH = os.path.join(OUTPUT_DIR, "y_test.npy")


def load_data():
    """Load the regenerated preprocessed 26-class arrays."""
    print("Loading preprocessed 26-class dataset ...")
    X_train = np.load(X_TRAIN_PATH).astype(np.float32)
    y_train = np.load(Y_TRAIN_PATH).astype(np.float32)
    X_val = np.load(X_VAL_PATH).astype(np.float32)
    y_val = np.load(Y_VAL_PATH).astype(np.float32)
    X_test = np.load(X_TEST_PATH).astype(np.float32)
    y_test = np.load(Y_TEST_PATH).astype(np.float32)
    print(f"  X_train: {X_train.shape} {X_train.dtype}")
    print(f"  y_train: {y_train.shape} {y_train.dtype}")
    print(f"  X_val:   {X_val.shape} {X_val.dtype}")
    print(f"  X_test:  {X_test.shape} {X_test.dtype}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def compute_class_weights(y_train):
    """Compute softened inverse-frequency class weights (mild imbalance handling)."""
    y = np.argmax(y_train, axis=1)
    counts = np.bincount(y, minlength=NUM_CLASSES).astype(np.float32)
    counts[counts == 0] = 1.0
    total = counts.sum()
    weights = total / (NUM_CLASSES * counts)
    weights = np.power(weights, 0.5)  # soften so J/Z (~2x) aren't over-penalized
    return {i: float(w) for i, w in enumerate(weights)}


def augment_image(img):
    """Apply one mild, random augmentation to a single 28x28 grayscale image.

    The image is in [0, 1] float. We upscale, transform, and downscale to keep
    transforms smooth. No horizontal flip (would change asymmetric ASL signs).
    """
    img_u8 = (img * 255.0).astype(np.uint8)

    # Small rotation.
    angle = np.random.uniform(-8, 8)
    h, w = img_u8.shape
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

    # Small brightness/contrast variation.
    if np.random.rand() < 0.5:
        beta = np.random.uniform(-15, 15)
        img_u8 = np.clip(img_u8.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    return img_u8.astype(np.float32) / 255.0


def augment_training_split(X_train, y_train, copies_per_sample=1):
    """Return augmented training arrays (original + `copies_per_sample` variants)."""
    Xs = [X_train]
    ys = [y_train]
    n = len(X_train)
    for _ in range(copies_per_sample):
        aug = np.empty_like(X_train)
        for i in range(n):
            aug[i, :, :, 0] = augment_image(X_train[i, :, :, 0])
        Xs.append(aug)
        ys.append(y_train)
    X_out = np.concatenate(Xs, axis=0)
    y_out = np.concatenate(ys, axis=0)

    # Shuffle deterministically.
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X_out))
    return X_out[idx], y_out[idx]


def save_history_json(history):
    os.makedirs(MODEL_DIR, exist_ok=True)
    serializable = {
        key: [float(v) for v in values] for key, values in history.history.items()
    }
    with open(HISTORY_IMPROVED_PATH, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"  saved training history: {HISTORY_IMPROVED_PATH}")


def plot_training_history(history):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, acc, "b-", label="Training Accuracy")
    axes[0].plot(epochs, val_acc, "r-", label="Validation Accuracy")
    axes[0].set_title("Training and Validation Accuracy (Improved)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].plot(epochs, loss, "b-", label="Training Loss")
    axes[1].plot(epochs, val_loss, "r-", label="Validation Loss")
    axes[1].set_title("Training and Validation Loss (Improved)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(PLOT_IMPROVED_PATH, dpi=150)
    plt.close(fig)
    print(f"  saved accuracy/loss graph: {PLOT_IMPROVED_PATH}")


def save_confusion_matrix(y_true, y_pred, class_names):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        title="Confusion Matrix (A-Z, Improved)",
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
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=6,
            )

    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_IMPROVED_PATH, dpi=150)
    plt.close(fig)
    print(f"  saved confusion matrix: {CONFUSION_MATRIX_IMPROVED_PATH}")

    np.savetxt(
        CONFUSION_CSV_IMPROVED_PATH,
        cm,
        delimiter=",",
        fmt="%d",
        header=",".join(class_names),
        comments="",
    )
    print(f"  saved confusion matrix CSV: {CONFUSION_CSV_IMPROVED_PATH}")


def save_classification_report(y_true, y_pred, class_names):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report = classification_report(
        y_true, y_pred, labels=list(range(NUM_CLASSES)),
        target_names=class_names, digits=4,
    )
    with open(CLASS_REPORT_IMPROVED_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  saved classification report: {CLASS_REPORT_IMPROVED_PATH}")
    print("\nClassification Report (A-Z, Improved):\n")
    print(report)


def main():
    print("=" * 60)
    print("HandSpeak - Improved CNN Training (26 classes A-Z)")
    print("=" * 60)

    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    print("\nAugmenting training split (offline, mild) ...")
    X_train, y_train = augment_training_split(X_train, y_train, copies_per_sample=1)
    print(f"  augmented X_train: {X_train.shape}")

    print("\nBuilding improved CNN model (L2 regularized) ...")
    model = build_cnn_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES,
                            learning_rate=LEARNING_RATE, l2=L2)
    model.summary()

    class_weights = compute_class_weights(y_train)
    print("\nClass weights (softened inverse-frequency):")
    for i in range(NUM_CLASSES):
        print(f"  {LETTERS[i]}: {class_weights[i]:.3f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4,
                          min_lr=1e-6, verbose=1),
        ModelCheckpoint(
            filepath=MODEL_IMPROVED_PATH,
            monitor="val_accuracy", mode="max",
            save_best_only=True, verbose=1,
        ),
    ]

    print(f"\nTraining for up to {EPOCHS} epochs (batch size {BATCH_SIZE}) ...")
    history = model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    save_history_json(history)
    model.save(MODEL_IMPROVED_PATH)
    print(f"  saved improved model: {MODEL_IMPROVED_PATH}")

    print("\nEvaluating improved model on test set ...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_accuracy:.4f}")

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\nGenerating reports ...")
    plot_training_history(history)
    save_confusion_matrix(y_true, y_pred, LETTERS)
    save_classification_report(y_true, y_pred, LETTERS)

    print("\nDone. Improved CNN training, evaluation, and reporting complete.")


if __name__ == "__main__":
    main()
