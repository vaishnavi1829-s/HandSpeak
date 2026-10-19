"""Quality audit for the ISL webcam v2 dataset (read-only).

Reports per letter:
    * total images
    * near-duplicate percentage (dHash Hamming <= 2)
    * MediaPipe detection success
    * 1-hand / 2-hand counts
    * blur (variance-of-Laplacian) issues
    * diversity score (mean pairwise dHash distance)

Writes ``results/isl_webcam_v2_quality_report.csv`` and a small markdown/JSON.

Run from the project root::

    python scripts/audit_isl_webcam_v2.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from recognition.landmark_extractor import (  # noqa: E402
    extract_two_hands_from_file,
    get_two_hand_landmarker,
)
from utils.label_mapping import LETTERS  # noqa: E402

V2_ROOT = os.path.join(PROJECT_ROOT, "dataset", "isl_webcam_v2")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

NEAR_DUP_HAMMING = 2
BLUR_THRESHOLD = 50.0  # variance of Laplacian below this = blurry


def _image_files(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(n for n in os.listdir(folder) if n.lower().endswith(IMG_EXTS))


def _dhash64(img):
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    bits = 0
    for r in range(8):
        for c in range(8):
            bits <<= 1
            if img[r, c] > img[r, c + 1]:
                bits |= 1
    return bits


def _hamming(a, b):
    return bin(a ^ b).count("1")


def _blur_score(img):
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def main():
    if not os.path.isdir(V2_ROOT):
        raise SystemExit(f"V2 root not found: {V2_ROOT}")

    landmarker = get_two_hand_landmarker()
    rows = []

    for letter in LETTERS:
        folder = os.path.join(V2_ROOT, letter, "roi")
        files = _image_files(folder)
        total = len(files)

        dhashes = []
        blur_scores = []
        detected = 0
        one_hand = 0
        two_hand = 0

        for name in files:
            path = os.path.join(folder, name)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            dhashes.append(_dhash64(img))
            blur_scores.append(_blur_score(img))

            res = extract_two_hands_from_file(path, landmarker=landmarker)
            if res is not None:
                n = int(res.left_present) + int(res.right_present)
                if n >= 1:
                    detected += 1
                if n == 1:
                    one_hand += 1
                elif n == 2:
                    two_hand += 1

        # Near-duplicate fraction.
        near = 0
        for i in range(len(dhashes)):
            for j in range(i + 1, len(dhashes)):
                if _hamming(dhashes[i], dhashes[j]) <= NEAR_DUP_HAMMING:
                    near += 1
                    break  # count each image once
        near_pct = 100.0 * near / total if total else 0.0

        # Diversity score (mean pairwise hamming distance / 64).
        diversity = 0.0
        if len(dhashes) > 1:
            pair_dists = []
            for i in range(len(dhashes)):
                for j in range(i + 1, len(dhashes)):
                    pair_dists.append(_hamming(dhashes[i], dhashes[j]))
            diversity = float(np.mean(pair_dists)) / 64.0

        blurry = sum(1 for b in blur_scores if b < BLUR_THRESHOLD)
        blur_pct = 100.0 * blurry / total if total else 0.0

        rows.append({
            "letter": letter,
            "total_images": total,
            "near_duplicate_pct": round(near_pct, 1),
            "mediapipe_success_pct": round(100.0 * detected / total, 1) if total else 0.0,
            "one_hand_count": one_hand,
            "two_hand_count": two_hand,
            "blurry_pct": round(blur_pct, 1),
            "diversity_score": round(diversity, 4),
        })

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "isl_webcam_v2_quality_report.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    json_path = os.path.join(RESULTS_DIR, "isl_webcam_v2_quality_report.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows}, fh, indent=2)

    print(f"{'L':<2}{'Total':>6}{'NearDup%':>10}{'MP%':>7}{'1H':>5}{'2H':>5}"
          f"{'Blur%':>7}{'Div':>7}")
    print("-" * 50)
    for r in rows:
        print(f"{r['letter']:<2}{r['total_images']:>6}{r['near_duplicate_pct']:>10.1f}"
              f"{r['mediapipe_success_pct']:>7.1f}{r['one_hand_count']:>5}"
              f"{r['two_hand_count']:>5}{r['blurry_pct']:>7.1f}{r['diversity_score']:>7.3f}")

    print(f"\nCSV  -> {csv_path}")
    print(f"JSON -> {json_path}")


if __name__ == "__main__":
    main()
