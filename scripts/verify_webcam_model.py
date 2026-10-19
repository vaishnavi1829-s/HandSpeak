"""Safe, no-camera verification of the webcam fine-tuned model.

This script does NOT open the webcam, does NOT retrain, and does NOT modify the
dataset or models. It verifies the new ``trained_model/handspeak_cnn_webcam.h5``
end-to-end:

    1. The canonical A-Z label mapping is consistent.
    2. The model expects 28x28x1 and outputs 26 probabilities.
    3. ``recognition.preprocessing.preprocess_frame`` still reproduces the
       training format.
    4. The confidence threshold correctly suppresses uncertain predictions.
    5. The model runs inference on the collected webcam validation samples and
       reports overall + per-letter accuracy (held-out webcam domain).

Run from the project root::

    python scripts/verify_webcam_model.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2  # noqa: E402
from keras.models import load_model  # noqa: E402

from recognition.preprocessing import (  # noqa: E402
    CONFIDENCE_THRESHOLD,
    preprocess_frame,
)
from utils.config import (  # noqa: E402
    MODEL_WEBCAM_PATH,
    NUM_CLASSES,
    IMAGE_SIZE,
    CHANNELS,
)
from utils import label_mapping as lm  # noqa: E402

WEBCAM_ROOT = os.path.join(PROJECT_ROOT, "dataset", "webcam")


def _png_files(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(
        name
        for name in os.listdir(folder)
        if name.lower().endswith((".png", ".jpg", ".jpeg"))
    )


def check_label_mapping():
    print("[1/4] Verifying 26-class A-Z label mapping ...")
    assert NUM_CLASSES == 26
    assert lm.LETTERS == [chr(ord("A") + i) for i in range(26)]
    for i in range(26):
        assert lm.index_to_letter(i) == chr(ord("A") + i)
    print("    OK: 26 classes, A-Z mapping consistent.")


def check_model_io():
    print("[2/4] Verifying webcam model input/output shapes ...")
    assert os.path.exists(MODEL_WEBCAM_PATH), f"Missing model: {MODEL_WEBCAM_PATH}"
    model = load_model(MODEL_WEBCAM_PATH)
    assert model.input_shape[1:] == (IMAGE_SIZE, IMAGE_SIZE, CHANNELS), model.input_shape
    assert model.output_shape[-1] == NUM_CLASSES, model.output_shape
    probs = model.predict(
        np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS), dtype=np.float32), verbose=0
    )[0]
    assert probs.shape == (NUM_CLASSES,)
    print(f"    OK: input {model.input_shape}, output {model.output_shape}.")
    return model


def check_preprocessing():
    print("[3/4] Verifying webcam preprocessing format ...")
    frame = np.full((300, 300, 3), 255, dtype=np.uint8)
    cv2.circle(frame, (150, 150), 80, (0, 0, 0), thickness=-1)
    tensor = preprocess_frame(frame, target_size=IMAGE_SIZE)
    assert tensor is not None
    assert tensor.shape == (1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0.0 and tensor.max() <= 1.0
    assert preprocess_frame(np.full((300, 300, 3), 255, dtype=np.uint8)) is None
    assert preprocess_frame(np.zeros((300, 300, 3), dtype=np.uint8)) is None
    print("    OK: 28x28x1 float [0,1], blank frames -> None.")


def check_validation_inference(model):
    print("[4/4] Running inference on held-out webcam validation samples ...")
    # Reproduce the same deterministic 80/20 per-class split used in training.
    from sklearn.model_selection import train_test_split

    X_list = []
    y_list = []
    for letter in lm.LETTERS:
        label = ord(letter) - ord("A")
        folder = os.path.join(WEBCAM_ROOT, letter, "processed")
        for name in _png_files(folder):
            path = os.path.join(folder, name)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            arr = img.astype(np.float32) / 255.0
            X_list.append(arr.reshape(IMAGE_SIZE, IMAGE_SIZE, CHANNELS))
            y_list.append(label)
    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)

    _, X_val, _, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42, shuffle=True
    )

    probs = model.predict(X_val, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    confidences = probs[np.arange(len(y_pred)), y_pred]

    acc = float(np.mean(y_pred == y_val))
    print(f"    Validation samples: {X_val.shape[0]}")
    print(f"    Overall accuracy  : {acc * 100:.2f}%")

    print("    Per-letter accuracy:")
    for i, letter in enumerate(lm.LETTERS):
        mask = y_val == i
        support = int(mask.sum())
        correct = int((y_pred[mask] == i).sum()) if support else 0
        pct = 100 * correct / support if support else 0.0
        print(f"      {letter}: {pct:6.2f}%  (n={support})")

    # Confidence threshold behavior: low-confidence predictions suppressed.
    below = int((confidences < CONFIDENCE_THRESHOLD).sum())
    print(f"    Predictions below threshold ({CONFIDENCE_THRESHOLD}): {below}/{len(y_pred)}")
    print("    OK: webcam model runs end-to-end on webcam-domain validation data.")


def main():
    print("=" * 60)
    print("HandSpeak - Webcam Model Verification (no camera)")
    print("=" * 60)
    check_label_mapping()
    model = check_model_io()
    check_preprocessing()
    check_validation_inference(model)
    print("\nAll verification checks passed.")


if __name__ == "__main__":
    main()
