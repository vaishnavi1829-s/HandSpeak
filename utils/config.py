"""Central configuration and shared constants for the HandSpeak project.

Keeping all tunable values and paths in one place avoids magic numbers
scattered across the preprocessing, model, training, and inference code.
"""

import os

# ---------------------------------------------------------------------------
# Paths (relative to the project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

RAW_TRAIN_CSV = os.path.join(PROJECT_ROOT, "dataset", "raw", "sign_mnist_train.csv")
RAW_TEST_CSV = os.path.join(PROJECT_ROOT, "dataset", "raw", "sign_mnist_test.csv")

MODEL_DIR = os.path.join(PROJECT_ROOT, "trained_model")
MODEL_WEIGHTS_PATH = os.path.join(MODEL_DIR, "handspeak_cnn.weights.h5")
MODEL_FULL_PATH = os.path.join(MODEL_DIR, "handspeak_cnn.h5")
HISTORY_PATH = os.path.join(MODEL_DIR, "training_history.json")

# Improved model (retrained with fixed J/Z polarity + augmentation + regularization).
MODEL_IMPROVED_PATH = os.path.join(MODEL_DIR, "handspeak_cnn_improved.h5")
HISTORY_IMPROVED_PATH = os.path.join(MODEL_DIR, "training_history_improved.json")

# Webcam fine-tuned model (single 26-class A-Z model fine-tuned on dataset/webcam/).
# This is the model used for real-world webcam inference.
MODEL_WEBCAM_PATH = os.path.join(MODEL_DIR, "handspeak_cnn_webcam.h5")
HISTORY_WEBCAM_PATH = os.path.join(MODEL_DIR, "training_history_webcam.json")

# Indian Sign Language (ISL) model (single 26-class A-Z model trained on
# dataset/isl_webcam/). Kept separate from the ASL webcam model so the
# application can offer independent ASL and ISL recognition modes.
MODEL_ISL_PATH = os.path.join(MODEL_DIR, "handspeak_cnn_isl.h5")
HISTORY_ISL_PATH = os.path.join(MODEL_DIR, "training_history_isl.json")

CLASS_REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "classification_report.txt")
CLASS_REPORT_IMPROVED_PATH = os.path.join(PROJECT_ROOT, "reports", "classification_report_improved.txt")
CONFUSION_MATRIX_PATH = os.path.join(PROJECT_ROOT, "reports", "confusion_matrix.png")
CONFUSION_MATRIX_IMPROVED_PATH = os.path.join(PROJECT_ROOT, "reports", "confusion_matrix_improved.png")
CONFUSION_CSV_IMPROVED_PATH = os.path.join(PROJECT_ROOT, "reports", "confusion_matrix_improved.csv")
PLOT_IMPROVED_PATH = os.path.join(PROJECT_ROOT, "reports", "training_accuracy_loss_improved.png")

# ---------------------------------------------------------------------------
# Dataset / image constants
# ---------------------------------------------------------------------------
IMAGE_SIZE = 28          # Sign Language MNIST images are 28x28
CHANNELS = 1             # grayscale
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)

NUM_CLASSES = 26         # Full ASL alphabet A-Z (A=0 ... Z=25)

# ---------------------------------------------------------------------------
# Preprocessing / splitting constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
VALIDATION_SPLIT = 0.2   # fraction of the training CSV held out for validation

# ---------------------------------------------------------------------------
# Training constants
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
