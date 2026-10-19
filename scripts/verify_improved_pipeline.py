"""No-camera verification of the improved HandSpeak pipeline.

This script does NOT open the webcam and does NOT retrain. It verifies:

1. The improved model exists and has the expected 28x28x1 -> 26 I/O shape.
2. The improved model loads and produces a valid probability distribution.
3. ``recognition.preprocessing.preprocess_frame`` still reproduces the
   hand-dark / background-light training format.
4. ``recognition.real_time_recognition`` loads the improved model path by
   default and its temporal smoothing returns a stable prediction.

Run from the project root::

    python scripts/verify_improved_pipeline.py
"""

import os
import sys
from collections import deque

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2  # noqa: E402
from keras.models import load_model  # noqa: E402

from recognition.preprocessing import preprocess_frame  # noqa: E402
from recognition.real_time_recognition import (  # noqa: E402
    smooth_prediction,
    load_trained_model,
)
from utils.config import (  # noqa: E402
    MODEL_IMPROVED_PATH,
    IMAGE_SIZE,
    CHANNELS,
    NUM_CLASSES,
)
from utils import label_mapping as lm  # noqa: E402


def main():
    print("=" * 60)
    print("HandSpeak - Improved Pipeline Verification (no camera)")
    print("=" * 60)

    # 1. Improved model file exists.
    assert os.path.exists(MODEL_IMPROVED_PATH), f"Missing {MODEL_IMPROVED_PATH}"

    # 2. Load and check I/O.
    model = load_model(MODEL_IMPROVED_PATH)
    assert model.input_shape[1:] == (IMAGE_SIZE, IMAGE_SIZE, CHANNELS), model.input_shape
    assert model.output_shape[-1] == NUM_CLASSES, model.output_shape
    print(f"[1/4] OK: model I/O {model.input_shape} -> {model.output_shape}")

    # 3. Dummy inference produces valid softmax.
    probs = model.predict(np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS),
                                   dtype=np.float32), verbose=0)[0]
    assert probs.shape == (NUM_CLASSES,)
    assert abs(float(probs.sum()) - 1.0) < 1e-3
    print("[2/4] OK: model produces valid 26-class probability distribution")

    # 4. Webcam preprocessing still matches training polarity.
    frame = np.full((300, 300, 3), 255, dtype=np.uint8)
    cv2.circle(frame, (150, 150), 80, (0, 0, 0), thickness=-1)
    tensor = preprocess_frame(frame, target_size=IMAGE_SIZE)
    assert tensor is not None
    assert tensor.shape == (1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
    img = tensor[0, :, :, 0]
    center = float(img[12:16, 12:16].mean())
    border = float(np.concatenate([img[0, :], img[-1, :], img[:, 0], img[:, -1]]).mean())
    assert center < border, "hand must be darker than background"
    print("[3/4] OK: webcam preprocessing yields hand-dark/background-light tensor")

    # 5. Temporal smoothing returns a stable prediction.
    history = deque(maxlen=7)
    for _ in range(4):
        history.append(("A", 0.95))
    history.append(("B", 0.80))
    history.append(("B", 0.80))
    letter, conf = smooth_prediction(history)
    assert letter == "A", f"expected 'A', got {letter}"
    assert 0.0 <= conf <= 1.0
    print("[4/4] OK: temporal smoothing selects stable majority ('A')")

    # 6. Default model path used by the recognizer is the improved model.
    import inspect
    src = inspect.getsource(load_trained_model)
    assert "MODEL_IMPROVED_PATH" in src, "recognizer must default to improved model"
    print("      OK: recognizer defaults to handspeak_cnn_improved.h5")

    print("\nAll improved-pipeline verification checks passed.")


if __name__ == "__main__":
    main()
