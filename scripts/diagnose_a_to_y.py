"""Read-only investigation of the A -> Y webcam misclassification.

This script does NOT open the webcam, does NOT retrain, and does NOT modify or
download any dataset. It only reads the trained model, the SL-MNIST CSVs, and
the existing preprocessing code.

It answers three focused questions:

1. When *all* SL-MNIST "A" training samples are pushed through the webcam
   preprocessing path, what do they get classified as (is A->Y systematic)?
2. If the polarity is intentionally INVERTED (hand light / background dark),
   does an "A" become a "Y"? This probes the failure mode where the Otsu
   "smaller region" heuristic picks the wrong object on a real webcam frame.
3. What is the distribution of the final preprocessed "A" tensor (to compare
   its scale/contrast against the raw training tensor)?

Run from the project root::

    python scripts/diagnose_a_to_y.py
"""

import os
import sys
from collections import Counter

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2  # noqa: E402
from keras.models import load_model  # noqa: E402

from recognition.preprocessing import preprocess_frame  # noqa: E402
from utils.config import MODEL_IMPROVED_PATH, IMAGE_SIZE, CHANNELS  # noqa: E402
from utils import label_mapping as lm  # noqa: E402


def load_samples(label_index, limit=None):
    """Yield 28x28 uint8 samples for one SL-MNIST raw label."""
    import pandas as pd

    csv_path = os.path.join(PROJECT_ROOT, "dataset", "raw", "sign_mnist_train.csv")
    df = pd.read_csv(csv_path)
    sub = df[df["label"] == label_index]
    rows = sub if limit is None else sub.iloc[:limit]
    for _, r in rows.iterrows():
        yield r[1:].values.reshape(IMAGE_SIZE, IMAGE_SIZE).astype(np.uint8)


def make_frame(sample_28, canvas=300, hand_size=180):
    up = cv2.resize(sample_28, (hand_size, hand_size), interpolation=cv2.INTER_CUBIC)
    frame = np.full((canvas, canvas, 3), 255, dtype=np.uint8)
    off = (canvas - hand_size) // 2
    frame[off:off + hand_size, off:off + hand_size] = up[..., None]
    return frame


def letter_of(probs):
    return lm.index_to_letter(int(np.argmax(probs)))


def main():
    model = load_model(MODEL_IMPROVED_PATH)

    # ------------------------------------------------------------------
    # 1. All A samples through the WEBCAM path.
    # ------------------------------------------------------------------
    print("=" * 72)
    print("Q1: All SL-MNIST 'A' samples through the WEBCAM path")
    print("=" * 72)
    top_counts = Counter()
    confs = []
    n = 0
    for sample in load_samples(0):  # label 0 -> A
        frame = make_frame(sample)
        tensor = preprocess_frame(frame, target_size=IMAGE_SIZE)
        if tensor is None:
            top_counts["<None>"] += 1
            continue
        probs = model.predict(tensor, verbose=0)[0]
        top_counts[letter_of(probs)] += 1
        confs.append(float(probs.max()))
        n += 1
    print(f"  {n} A samples -> top-class distribution:")
    for letter, count in top_counts.most_common():
        print(f"    {letter:>6}: {count:4d}  ({100.0 * count / max(1, n):.1f}%)")
    if confs:
        print(f"  mean confidence of top class: {np.mean(confs):.3f}")

    # ------------------------------------------------------------------
    # 2. Inverted-polarity A -> does it become Y?
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("Q2: INVERTED-polarity 'A' (hand LIGHT / bg DARK) through webcam path")
    print("=" * 72)
    inv_counts = Counter()
    n2 = 0
    for sample in load_samples(0, limit=200):
        frame = make_frame(sample)
        inv_frame = cv2.bitwise_not(frame)  # hand becomes light, bg dark
        tensor = preprocess_frame(inv_frame, target_size=IMAGE_SIZE)
        if tensor is None:
            inv_counts["<None>"] += 1
            continue
        probs = model.predict(tensor, verbose=0)[0]
        inv_counts[letter_of(probs)] += 1
        n2 += 1
    print(f"  {n2} inverted-A samples -> top-class distribution:")
    for letter, count in inv_counts.most_common(12):
        print(f"    {letter:>6}: {count:4d}  ({100.0 * count / max(1, n2):.1f}%)")

    # ------------------------------------------------------------------
    # 3. Tensor statistics: raw training A vs webcam-preprocessed A.
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("Q3: Tensor statistics (raw training A vs webcam-preprocessed A)")
    print("=" * 72)
    raw_means, raw_mins, raw_maxs = [], [], []
    web_means, web_mins, web_maxs = [], [], []
    for sample in load_samples(0, limit=300):
        raw = sample.astype(np.float32) / 255.0
        raw_means.append(float(raw.mean()))
        raw_mins.append(float(raw.min()))
        raw_maxs.append(float(raw.max()))

        frame = make_frame(sample)
        tensor = preprocess_frame(frame, target_size=IMAGE_SIZE)
        if tensor is not None:
            img = tensor[0, :, :, 0]
            web_means.append(float(img.mean()))
            web_mins.append(float(img.min()))
            web_maxs.append(float(img.max()))

    def stat(name, arr):
        a = np.array(arr)
        print(f"  {name:>28}: mean={a.mean():.3f}  min={a.min():.3f}  max={a.max():.3f}")

    stat("raw training A  mean", raw_means)
    stat("webcam A        mean", web_means)
    stat("raw training A  min ", raw_mins)
    stat("webcam A        min ", web_mins)
    stat("raw training A  max ", raw_maxs)
    stat("webcam A        max ", web_maxs)

    print("\n(If the webcam-preprocessed A has a much higher mean pixel value")
    print(" than raw training A, the hand is being rendered too light ->")
    print(" the model sees a 'washed-out' sign, which explains the confusion.)")


if __name__ == "__main__":
    main()
