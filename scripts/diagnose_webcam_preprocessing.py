"""Read-only diagnostic for the webcam A->Y misclassification.

This script does NOT open the webcam, does NOT retrain, and does NOT modify or
download any dataset. It only *reads* the existing trained model, the SL-MNIST
CSVs, and the existing preprocessing modules to answer one question:

    Does the webcam preprocessing pipeline corrupt a clean training "A" (or
    other letters) so that the model misclassifies it?

Two inference paths are compared for the SAME known dataset samples:

    1. DIRECT path  : the 28x28x1 SL-MNIST tensor is fed straight to the CNN
                      (this is exactly what the model was trained on).
    2. WEBCAM path  : the 28x28 sample is upscaled into a 300x300 "frame",
                      then run through ``recognition.preprocessing.preprocess_frame``
                      (grayscale -> Otsu -> localize -> crop -> center -> resize
                       -> /255.0), then fed to the CNN.

If a sample is correct on the DIRECT path but wrong on the WEBCAM path, the
regression is inside the webcam preprocessing pipeline.

Run from the project root::

    python scripts/diagnose_webcam_preprocessing.py
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
    preprocess_frame,
    _foreground_mask,
    localize_hand,
)
from utils.config import MODEL_IMPROVED_PATH, IMAGE_SIZE, CHANNELS  # noqa: E402
from utils import label_mapping as lm  # noqa: E402


def top3(probs):
    idx = np.argsort(probs)[::-1][:3]
    return [(lm.index_to_letter(int(i)), float(probs[i])) for i in idx]


def load_slmnist_sample(label_index):
    """Return one 28x28 uint8 sample for the requested SL-MNIST raw label."""
    csv_path = os.path.join(PROJECT_ROOT, "dataset", "raw", "sign_mnist_train.csv")
    import pandas as pd

    df = pd.read_csv(csv_path)
    sub = df[df["label"] == label_index]
    row = sub.iloc[0]
    pixels = row[1:].values.reshape(IMAGE_SIZE, IMAGE_SIZE).astype(np.uint8)
    return pixels


def make_frame_from_sample(sample_28, canvas=300, hand_size=180):
    """Place a 28x28 SL-MNIST sample (dark hand, light bg) into a webcam-like frame.

    The sample is upscaled to ``hand_size`` and centered on a light ``canvas``
    square so there is real background margin around the hand, mimicking a
    centered ROI in the live feed.
    """
    up = cv2.resize(sample_28, (hand_size, hand_size), interpolation=cv2.INTER_CUBIC)
    frame = np.full((canvas, canvas, 3), 255, dtype=np.uint8)
    off = (canvas - hand_size) // 2
    frame[off:off + hand_size, off:off + hand_size] = up[..., None]
    return frame


def main():
    print("=" * 72)
    print("HandSpeak - Webcam Preprocessing Diagnostic (A->Y investigation)")
    print("=" * 72)

    model = load_model(MODEL_IMPROVED_PATH)
    print(f"Model: {os.path.basename(MODEL_IMPROVED_PATH)}")
    print(f"  input : {model.input_shape}")
    print(f"  output: {model.output_shape}")
    print(f"  label mapping: {lm.LETTERS}")

    # Letters to probe. A and Y first, then a spread of others.
    # NOTE: J (label 9) and Z (label 25) are NOT in the SL-MNIST CSV; they come
    # from dataset/jz/. Skip them here (the direct vs webcam comparison needs a
    # SL-MNIST sample to be meaningful).
    probes = ["A", "Y", "B", "C", "D", "E", "I", "S", "T", "X"]

    print("\n" + "-" * 72)
    print(f"{'Letter':<7}{'DIRECT top-3':<32}{'WEBCAM top-3':<32}")
    print("-" * 72)

    for letter in probes:
        label_index = lm.label_to_index(ord(letter) - ord("A"))
        sample = load_slmnist_sample(label_index)

        # DIRECT path: the raw SL-MNIST tensor (exactly the training format).
        direct_tensor = sample.astype(np.float32).reshape(1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS) / 255.0
        direct_probs = model.predict(direct_tensor, verbose=0)[0]
        direct_str = ", ".join(f"{l}:{p * 100:.1f}%" for l, p in top3(direct_probs))

        # WEBCAM path: upscale to a frame and run the real webcam preprocessing.
        frame = make_frame_from_sample(sample)
        web_tensor = preprocess_frame(frame, target_size=IMAGE_SIZE)
        if web_tensor is None:
            web_str = "None (no hand detected)"
        else:
            web_probs = model.predict(web_tensor, verbose=0)[0]
            web_str = ", ".join(f"{l}:{p * 100:.1f}%" for l, p in top3(web_probs))

        print(f"{letter:<7}{direct_str:<32}{web_str:<32}")

    # ------------------------------------------------------------------
    # Deep dive on the "A" sample: show exactly what the webcam path does.
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("Deep dive: what does the webcam path do to a clean SL-MNIST 'A'?")
    print("=" * 72)

    a_sample = load_slmnist_sample(0)  # label 0 -> 'A'
    frame = make_frame_from_sample(a_sample)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mask = _foreground_mask(gray)
    fg_fraction = float(np.count_nonzero(mask)) / float(mask.size)
    print(f"  Otsu foreground fraction (after small-region normalization): {fg_fraction:.3f}")

    localized = localize_hand(gray)
    if localized is None:
        print("  localize_hand returned None")
    else:
        print(f"  localized crop shape        : {localized.shape}")

        # Polarity decision internals (mirrors localize_hand logic).
        fg_mean = float(gray[mask > 0].mean()) if np.any(mask) else float("nan")
        bg_mean = float(gray[mask == 0].mean()) if np.any(mask == 0) else float("nan")
        print(f"  foreground mean (hand)      : {fg_mean:.1f}")
        print(f"  background mean             : {bg_mean:.1f}")
        print(f"  polarity inversion applied  : {fg_mean > bg_mean}")

        # What the final 28x28 tensor looks like.
        web_tensor = preprocess_frame(frame, target_size=IMAGE_SIZE)
        img = web_tensor[0, :, :, 0]
        center = float(img[10:18, 10:18].mean())
        border = float(np.concatenate([img[0, :], img[-1, :], img[:, 0], img[:, -1]]).mean())
        print(f"  final 28x28 center mean     : {center:.3f} (dark hand = low)")
        print(f"  final 28x28 border mean     : {border:.3f} (light bg = high)")
        print(f"  final polarity hand<bg      : {center < border}")

        # Compare round-trip vs the ORIGINAL 28x28 sample.
        orig = a_sample.astype(np.float32) / 255.0
        mae = float(np.mean(np.abs(img - orig)))
        print(f"  MAE vs original SL-MNIST A  : {mae:.3f} (0=identical)")

        # Prediction from the webcam tensor.
        probs = model.predict(web_tensor, verbose=0)[0]
        print(f"  WEBCAM-path 'A' prediction  : {top3(probs)}")


if __name__ == "__main__":
    main()
