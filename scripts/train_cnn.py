"""Train and evaluate the HandSpeak 26-class (A-Z) CNN.

This script consumes the preprocessed dataset already written to ``output/``
(X_train, X_val, X_test, y_train, y_val, y_test as .npy files) and:

    1. Builds the CNN defined in ``models/cnn_model.py``.
    2. Trains on the training split and validates on the validation split.
    3. Uses EarlyStopping and ModelCheckpoint callbacks.
    4. Saves the best model inside ``trained_model/``.
    5. Evaluates the best model on the test split (prints loss/accuracy).
    6. Saves training/validation accuracy and loss graphs to ``reports/``.
    7. Saves a confusion matrix and a classification report to ``reports/``.

It does NOT modify the dataset or preprocessing files, and it does not
download or replace any dataset. Run it from the project root:

    python scripts/train_cnn.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless-safe backend for saving figures

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Ensure the project root is on the path so that ``models`` and ``utils``
# packages can be imported when this script is run directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from keras.callbacks import EarlyStopping, ModelCheckpoint

from models.cnn_model import LETTERS, build_cnn_model
from utils.config import (
    PROJECT_ROOT,
    MODEL_DIR,
    MODEL_WEIGHTS_PATH,
    MODEL_FULL_PATH,
    HISTORY_PATH,
    CLASS_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    BATCH_SIZE,
    EPOCHS,
    INPUT_SHAPE,
)

# ---------------------------------------------------------------------------
# Paths to the preprocessed 26-class arrays
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

X_TRAIN_PATH = os.path.join(OUTPUT_DIR, "X_train.npy")
Y_TRAIN_PATH = os.path.join(OUTPUT_DIR, "y_train.npy")
X_VAL_PATH = os.path.join(OUTPUT_DIR, "X_val.npy")
Y_VAL_PATH = os.path.join(OUTPUT_DIR, "y_val.npy")
X_TEST_PATH = os.path.join(OUTPUT_DIR, "X_test.npy")
Y_TEST_PATH = os.path.join(OUTPUT_DIR, "y_test.npy")

NUM_CLASSES_26 = 26

# Paths for the training history graphs.
ACC_LOSS_PLOT_PATH = os.path.join(REPORTS_DIR, "training_accuracy_loss.png")
CONFUSION_CSV_PATH = os.path.join(REPORTS_DIR, "confusion_matrix.csv")


def load_data():
    """Load the preprocessed 26-class arrays from ``output/``."""
    print("Loading preprocessed 26-class dataset ...")
    X_train = np.load(X_TRAIN_PATH)
    y_train = np.load(Y_TRAIN_PATH)
    X_val = np.load(X_VAL_PATH)
    y_val = np.load(Y_VAL_PATH)
    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)

    print(f"  X_train: {X_train.shape} {X_train.dtype}")
    print(f"  y_train: {y_train.shape} {y_train.dtype}")
    print(f"  X_val:   {X_val.shape} {X_val.dtype}")
    print(f"  y_val:   {y_val.shape} {y_val.dtype}")
    print(f"  X_test:  {X_test.shape} {X_test.dtype}")
    print(f"  y_test:  {y_test.shape} {y_test.dtype}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def save_history_json(history):
    """Persist the training history to JSON inside ``trained_model/``."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    # Convert numpy floats to plain Python floats for JSON serialization.
    serializable = {
        key: [float(v) for v in values] for key, values in history.history.items()
    }
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"  saved training history: {HISTORY_PATH}")


def plot_training_history(history):
    """Save accuracy and loss curves (training vs validation)."""
    os.makedirs(REPORTS_DIR, exist_ok=True)

    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    epochs = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, acc, "b-", label="Training Accuracy")
    axes[0].plot(epochs, val_acc, "r-", label="Validation Accuracy")
    axes[0].set_title("Training and Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].plot(epochs, loss, "b-", label="Training Loss")
    axes[1].plot(epochs, val_loss, "r-", label="Validation Loss")
    axes[1].set_title("Training and Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(ACC_LOSS_PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"  saved accuracy/loss graph: {ACC_LOSS_PLOT_PATH}")


def save_confusion_matrix(y_true, y_pred, class_names):
    """Save the confusion matrix as an image and a CSV."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES_26)))

    # Image
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        title="Confusion Matrix (A-Z)",
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
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=6,
            )

    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close(fig)
    print(f"  saved confusion matrix: {CONFUSION_MATRIX_PATH}")

    # CSV
    np.savetxt(
        CONFUSION_CSV_PATH,
        cm,
        delimiter=",",
        fmt="%d",
        header=",".join(class_names),
        comments="",
    )
    print(f"  saved confusion matrix CSV: {CONFUSION_CSV_PATH}")


def save_classification_report(y_true, y_pred, class_names):
    """Save the per-class precision/recall/f1 report to ``reports/``."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES_26)),
        target_names=class_names,
        digits=4,
    )
    with open(CLASS_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  saved classification report: {CLASS_REPORT_PATH}")
    print("\nClassification Report (A-Z):\n")
    print(report)


def main():
    print("=" * 60)
    print("HandSpeak - CNN Training (26 classes A-Z)")
    print("=" * 60)

    # 1. Load preprocessed dataset.
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # 2. Build the CNN.
    print("\nBuilding CNN model ...")
    model = build_cnn_model(
        input_shape=INPUT_SHAPE,
        num_classes=NUM_CLASSES_26,
        learning_rate=1e-3,
    )
    model.summary()

    # 3. Callbacks.
    os.makedirs(MODEL_DIR, exist_ok=True)
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=MODEL_WEIGHTS_PATH,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
    ]

    # 4. Train.
    print(f"\nTraining for up to {EPOCHS} epochs (batch size {BATCH_SIZE}) ...")
    history = model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    # 5. Persist history + best model.
    save_history_json(history)
    # Save the full model (architecture + weights + optimizer state).
    model.save(MODEL_FULL_PATH)
    print(f"  saved best model: {MODEL_FULL_PATH}")

    # 6. Evaluate on test set.
    print("\nEvaluating best model on test set ...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_accuracy:.4f}")

    # 7. Predictions for reports.
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # 8. Generate reports.
    print("\nGenerating reports ...")
    plot_training_history(history)
    save_confusion_matrix(y_true, y_pred, LETTERS)
    save_classification_report(y_true, y_pred, LETTERS)

    print("\nDone. CNN training, evaluation, and reporting complete.")


if __name__ == "__main__":
    main()
