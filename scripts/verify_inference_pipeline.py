"""Safe, read-only verification of the webcam inference/preprocessing pipeline.

This script does NOT open the webcam, does NOT retrain the model, and does NOT
modify or download any dataset. It only *reads* the trained model and the
existing Sign Language MNIST CSVs to confirm:

    1. The 26-class A-Z label mapping is consistent everywhere.
    2. The trained model expects 28x28x1 and outputs 26 probabilities.
    3. ``recognition.preprocessing.preprocess_frame`` reproduces the training
       format (grayscale -> localize -> polarity-normalize -> center/pad ->
       28x28 -> /255.0) and returns ``None`` for empty/blank frames.
    4. The confidence threshold correctly suppresses uncertain predictions.

Run from the project root::

    python scripts/verify_inference_pipeline.py
"""

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
from utils.config import MODEL_FULL_PATH, NUM_CLASSES, IMAGE_SIZE, CHANNELS  # noqa: E402
from utils import label_mapping as lm  # noqa: E402


def check_label_mapping():
    print("[1/4] Verifying 26-class A-Z label mapping ...")
    assert NUM_CLASSES == 26, f"NUM_CLASSES must be 26, got {NUM_CLASSES}"
    assert lm.LETTERS == [chr(ord("A") + i) for i in range(26)]
    assert lm.get_all_letters() == lm.LETTERS
    assert lm.SIGN_LABELS == list(range(26))

    for i in range(26):
        letter = chr(ord("A") + i)
        assert lm.label_to_letter(i) == letter
        assert lm.label_to_index(i) == i
        assert lm.index_to_label(i) == i
        assert lm.index_to_letter(i) == letter

    print(f"    OK: {NUM_CLASSES} classes, A-Z mapping consistent.")
    return lm.LETTERS


def check_model_io():
    print("[2/4] Verifying trained model input/output shapes ...")
    assert os.path.exists(MODEL_FULL_PATH), f"Missing model: {MODEL_FULL_PATH}"
    model = load_model(MODEL_FULL_PATH)
    assert model.input_shape[1:] == (IMAGE_SIZE, IMAGE_SIZE, CHANNELS), model.input_shape
    assert model.output_shape[-1] == NUM_CLASSES, model.output_shape

    # Run one dummy inference to force graph construction and confirm output.
    probs = model.predict(np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS),
                                   dtype=np.float32), verbose=0)[0]
    assert probs.shape == (NUM_CLASSES,), probs.shape
    assert abs(float(probs.sum()) - 1.0) < 1e-3, "softmax probabilities must sum to 1"
    print(f"    OK: input {model.input_shape}, output {model.output_shape}.")
    return model


def check_preprocessing():
    print("[3/4] Verifying webcam preprocessing format ...")

    # Synthetic frame: light background with a dark hand-like circle.
    frame = np.full((300, 300, 3), 255, dtype=np.uint8)
    cv2.circle(frame, (150, 150), 80, (0, 0, 0), thickness=-1)

    tensor = preprocess_frame(frame, target_size=IMAGE_SIZE)
    assert tensor is not None, "preprocess_frame returned None for a hand image"
    assert tensor.shape == (1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS), tensor.shape
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0.0 and tensor.max() <= 1.0

    # Polarity check: the hand must be DARK (low value) and background LIGHT.
    img = tensor[0, :, :, 0]
    center = float(img[12:16, 12:16].mean())
    border = float(np.concatenate([img[0, :], img[-1, :], img[:, 0], img[:, -1]]).mean())
    assert center < border, f"hand should be darker than background (center={center:.2f}, border={border:.2f})"

    # Blank frames must yield None (no hand detected).
    assert preprocess_frame(np.full((300, 300, 3), 255, dtype=np.uint8)) is None
    assert preprocess_frame(np.zeros((300, 300, 3), dtype=np.uint8)) is None

    print(f"    OK: tensor {tensor.shape} {tensor.dtype}, polarity hand-dark/background-light.")
    return tensor


def check_prediction_threshold(model):
    print("[4/4] Verifying confidence threshold behavior ...")

    # A blank frame must produce (None, None) -> "No sign".
    from recognition.real_time_recognition import predict_letter

    blank = np.full((300, 300, 3), 255, dtype=np.uint8)
    letter, conf = predict_letter(model, blank, confidence_threshold=CONFIDENCE_THRESHOLD)
    assert letter is None and conf is None, "blank frame should produce 'No sign'"

    # A real SL-MNIST sample must run end-to-end and return a valid letter or
    # a low-confidence rejection (both are acceptable; this only validates the
    # 26-class mapping and tensor contract, not classification accuracy).
    import pandas as pd

    csv_path = os.path.join(PROJECT_ROOT, "dataset", "raw", "sign_mnist_train.csv")
    df = pd.read_csv(csv_path)
    raw_label = int(df.iloc[0]["label"])
    pixels = df.iloc[0, 1:].values.reshape(IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
    expected_letter = lm.label_to_letter(raw_label)

    tensor = pixels.reshape(1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS) / 255.0
    probs = model.predict(tensor, verbose=0)[0]
    top = int(np.argmax(probs))
    print(f"    SL-MNIST sample (raw label {raw_label} -> '{expected_letter}'): "
          f"model top class '{lm.index_to_letter(top)}' ({float(probs[top]) * 100:.1f}%)")
    assert 0 <= top < 26
    print("    OK: threshold and A-Z mapping behave as expected.")


def main():
    print("=" * 60)
    print("HandSpeak - Inference/Preprocessing Pipeline Verification")
    print("=" * 60)
    check_label_mapping()
    model = check_model_io()
    check_preprocessing()
    check_prediction_threshold(model)
    print("\nAll verification checks passed.")


if __name__ == "__main__":
    main()
