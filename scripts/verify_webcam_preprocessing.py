"""Validate the updated webcam preprocessing against all 26 letters (A-Z).

This script does NOT open the webcam, does NOT retrain, and does NOT modify or
download any dataset. It reads:

  * the existing trained model  (``trained_model/handspeak_cnn_improved.h5``)
  * the SL-MNIST CSVs           (``dataset/raw/sign_mnist_*.csv``)  -> A-I, K-Y
  * the J/Z image folders       (``dataset/jz/J``, ``dataset/jz/Z``)

For every letter it runs BOTH inference paths and compares them:

  1. DIRECT  : the raw 28x28x1 training tensor straight into the CNN.
  2. WEBCAM  : the 28x28 sample is upscaled into a 300x300 "frame" and pushed
               through the UPDATED ``recognition.preprocessing.preprocess_frame``.

It reports per-letter accuracy, confidence, the difficult pairs (A/Y, C/O, D/P,
E/M/N/S/T, U/V), and flags any letter whose webcam-path accuracy degrades
relative to the direct path.

Run from the project root::

    python scripts/verify_webcam_preprocessing.py
"""

import os
import sys
from collections import Counter, defaultdict

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2  # noqa: E402
from keras.models import load_model  # noqa: E402
from PIL import Image  # noqa: E402

from recognition.preprocessing import preprocess_frame  # noqa: E402
from utils.config import MODEL_IMPROVED_PATH, IMAGE_SIZE, CHANNELS, NUM_CLASSES  # noqa: E402
from utils import label_mapping as lm  # noqa: E402

MAX_SAMPLES_PER_LETTER = 60


def load_slmnist_samples(label_index, limit=MAX_SAMPLES_PER_LETTER):
    """Yield 28x28 uint8 SL-MNIST samples for one raw label (0-8, 10-24)."""
    import pandas as pd

    csv_path = os.path.join(PROJECT_ROOT, "dataset", "raw", "sign_mnist_train.csv")
    df = pd.read_csv(csv_path)
    sub = df[df["label"] == label_index].iloc[:limit]
    for _, r in sub.iterrows():
        yield r[1:].values.reshape(IMAGE_SIZE, IMAGE_SIZE).astype(np.uint8)


def load_jz_samples(letter, limit=MAX_SAMPLES_PER_LETTER):
    """Yield 28x28 uint8 J/Z samples in the SAME polarity as SL-MNIST.

    The J/Z source images are light-hand-on-dark-background (opposite of
    SL-MNIST). ``combine_26_class`` inverts them before training, so we invert
    here too to obtain the training-equivalent polarity.
    """
    folder = os.path.join(PROJECT_ROOT, "dataset", "jz", letter)
    files = sorted(
        os.listdir(folder),
        key=lambda n: int("".join(ch for ch in n if ch.isdigit()) or 0),
    )[:limit]
    for name in files:
        path = os.path.join(folder, name)
        with Image.open(path) as img:
            gray = img.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
            arr = np.asarray(gray, dtype=np.float32) / 255.0
            inverted = 1.0 - arr  # hand -> dark, background -> light
        yield (inverted * 255.0).astype(np.uint8)


def make_frame(sample_28, canvas=300, hand_size=180):
    """Place a 28x28 sample into a webcam-like 300x300 light-background frame."""
    up = cv2.resize(sample_28, (hand_size, hand_size), interpolation=cv2.INTER_CUBIC)
    frame = np.full((canvas, canvas, 3), 255, dtype=np.uint8)
    off = (canvas - hand_size) // 2
    frame[off:off + hand_size, off:off + hand_size] = up[..., None]
    return frame


def iter_letter_samples(letter):
    """Yield 28x28 uint8 training samples for any letter A-Z."""
    label_index = lm.label_to_index(ord(letter) - ord("A"))
    if letter in ("J", "Z"):
        yield from load_jz_samples(letter)
    else:
        yield from load_slmnist_samples(label_index)


def main():
    print("=" * 78)
    print("HandSpeak - Webcam Preprocessing Verification (all 26 letters A-Z)")
    print("=" * 78)

    model = load_model(MODEL_IMPROVED_PATH)
    print(f"Model : {os.path.basename(MODEL_IMPROVED_PATH)}  "
          f"{model.input_shape} -> {model.output_shape}")
    print(f"Label : {lm.LETTERS}")
    print(f"Max samples per letter: {MAX_SAMPLES_PER_LETTER}\n")

    header = (
        f"{'Ltr':<4}{'n':>4} | {'DIRECT acc':>10} | "
        f"{'WEBCAM acc':>10} | {'avg conf':>9} | Webcam top confusion"
    )
    print(header)
    print("-" * 78)

    overall_direct_correct = 0
    overall_direct_total = 0
    overall_webcam_correct = 0
    overall_webcam_total = 0
    degraded = []

    for letter in lm.LETTERS:
        direct_correct = 0
        webcam_correct = 0
        webcam_conf = []
        webcam_top = Counter()
        n = 0

        for sample in iter_letter_samples(letter):
            n += 1
            # DIRECT path.
            direct_tensor = (
                sample.astype(np.float32).reshape(1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS) / 255.0
            )
            direct_probs = model.predict(direct_tensor, verbose=0)[0]
            if lm.index_to_letter(int(np.argmax(direct_probs))) == letter:
                direct_correct += 1

            # WEBCAM path (updated preprocessing).
            frame = make_frame(sample)
            web_tensor = preprocess_frame(frame, target_size=IMAGE_SIZE)
            if web_tensor is None:
                webcam_top["<None>"] += 1
                continue
            web_probs = model.predict(web_tensor, verbose=0)[0]
            top_idx = int(np.argmax(web_probs))
            top_letter = lm.index_to_letter(top_idx)
            webcam_top[top_letter] += 1
            webcam_conf.append(float(web_probs[top_idx]))
            if top_letter == letter:
                webcam_correct += 1

        d_acc = direct_correct / n if n else 0.0
        w_acc = webcam_correct / n if n else 0.0
        avg_conf = float(np.mean(webcam_conf)) if webcam_conf else 0.0

        # Top confusion (excluding the correct letter).
        confusion = [f"{k}:{v}" for k, v in webcam_top.most_common(4) if k != letter]
        confusion_str = ", ".join(confusion) if confusion else "-"

        print(
            f"{letter:<4}{n:>4} | {d_acc * 100:>9.1f}% | "
            f"{w_acc * 100:>9.1f}% | {avg_conf * 100:>8.1f}% | {confusion_str}"
        )

        overall_direct_correct += direct_correct
        overall_direct_total += n
        overall_webcam_correct += webcam_correct
        overall_webcam_total += n
        if w_acc < d_acc - 0.05:
            degraded.append((letter, d_acc, w_acc))

    print("-" * 78)
    print(f"DIRECT overall  : {overall_direct_correct}/{overall_direct_total} "
          f"({100 * overall_direct_correct / overall_direct_total:.2f}%)")
    print(f"WEBCAM overall  : {overall_webcam_correct}/{overall_webcam_total} "
          f"({100 * overall_webcam_correct / overall_webcam_total:.2f}%)")
    print(f"Degraded letters (webcam > 5pp below direct): "
          f"{[d[0] for d in degraded] if degraded else 'NONE'}")

    print("\nDifficult-pair inspection:")
    pairs = [("A", "Y"), ("C", "O"), ("D", "P"), ("U", "V"),
             ("E", "M"), ("M", "N"), ("N", "S"), ("S", "T")]
    for a, b in pairs:
        pa = dict((l, acc) for l, acc in _pair_probe(model, a))
        pb = dict((l, acc) for l, acc in _pair_probe(model, b))
        print(f"  {a} vs {b}:  '{a}' webcam-top-1 distribution {pa}")
        print(f"  {' ' * len(a)}     '{b}' webcam-top-1 distribution {pb}")


def _pair_probe(model, letter, limit=60):
    """Return [(top_letter, fraction)] for a letter through the webcam path."""
    counts = Counter()
    n = 0
    for sample in iter_letter_samples(letter):
        frame = make_frame(sample)
        tensor = preprocess_frame(frame, target_size=IMAGE_SIZE)
        if tensor is None:
            continue
        probs = model.predict(tensor, verbose=0)[0]
        counts[lm.index_to_letter(int(np.argmax(probs)))] += 1
        n += 1
    return [(k, round(100 * v / n, 1)) for k, v in counts.most_common(3)]


if __name__ == "__main__":
    main()
