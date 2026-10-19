"""Audit the ISL webcam dataset for the two-hand landmark pipeline.

This script is READ-ONLY: it never modifies, moves, or deletes any image.  It
produces a per-letter data-quality report for A-Z covering:

    * total ROI images
    * MediaPipe hand count (0 / 1 / 2)
    * landmark detection success rate
    * valid two-hand samples
    * blurry / very dark / very bright flags (image-quality heuristics)
    * near-duplicate image counts (perceptual average-hash within a letter)
    * min / max / mean / median samples across classes

The report is saved to ``results/landmark/isl_2hand_dataset_audit.txt``.

Run from the project root::

    python scripts/audit_isl_twohand_dataset.py
"""

from __future__ import annotations

import os
import sys
import time
from collections import Counter

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

ISL_ROOT = os.path.join(PROJECT_ROOT, "dataset", "isl_webcam")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "landmark")
REPORT_PATH = os.path.join(RESULTS_DIR, "isl_2hand_dataset_audit.txt")

# Image-quality heuristics (BGR 0-255 ROI images, 300x300 typically).
BLUR_THRESHOLD = 100.0       # Laplacian variance below this is considered blurry.
DARK_THRESHOLD = 40.0        # mean gray below this is considered very dark.
BRIGHT_THRESHOLD = 215.0     # mean gray above this is considered very bright.

# Average-hash parameters for near-duplicate detection.
HASH_SIZE = 16               # 16x16 -> 256-bit hash.
HASH_HAMMING_DISTANCE = 4    # <= 4 bits different = near-duplicate.


def _image_files(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    return sorted(
        name
        for name in os.listdir(folder)
        if name.lower().endswith((".png", ".jpg", ".jpeg"))
    )


def _gray_mean(bgr: np.ndarray) -> float:
    return float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean())


def _laplacian_variance(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _average_hash(bgr: np.ndarray) -> int:
    """Return a 256-bit average hash of the downscaled grayscale image."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (HASH_SIZE, HASH_SIZE), interpolation=cv2.INTER_AREA)
    mean = small.mean()
    bits = (small > mean).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def count_near_duplicates(hashes: list[int]) -> int:
    """Return the number of images that have at least one near-duplicate.

    This is a simple O(n^2) pairwise check within a single letter folder, which
    is acceptable because each folder has on the order of ~75-84 images.
    """
    n = len(hashes)
    flagged = set()
    for i in range(n):
        for j in range(i + 1, n):
            if _hamming(hashes[i], hashes[j]) <= HASH_HAMMING_DISTANCE:
                flagged.add(i)
                flagged.add(j)
    return len(flagged)


def audit_letter(letter: str, landmarker) -> dict:
    folder = os.path.join(ISL_ROOT, letter, "roi")
    files = _image_files(folder)

    total = len(files)
    hand_0 = 0
    hand_1 = 0
    hand_2 = 0
    blurry = 0
    dark = 0
    bright = 0
    hashes: list[int] = []
    readable = 0

    for name in files:
        path = os.path.join(folder, name)
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            continue

        readable += 1

        # Image quality heuristics.
        mean_gray = _gray_mean(bgr)
        lap_var = _laplacian_variance(bgr)
        if lap_var < BLUR_THRESHOLD:
            blurry += 1
        if mean_gray < DARK_THRESHOLD:
            dark += 1
        if mean_gray > BRIGHT_THRESHOLD:
            bright += 1
        hashes.append(_average_hash(bgr))

        # Hand count using the same two-hand extractor used for training.
        result = extract_two_hands_from_file(path, landmarker=landmarker)
        if result is None:
            hand_0 += 1
        else:
            n = int(result.left_present) + int(result.right_present)
            if n == 2:
                hand_2 += 1
            else:
                hand_1 += 1

    near_dupes = count_near_duplicates(hashes)

    return {
        "letter": letter,
        "total": total,
        "readable": readable,
        "hand_0": hand_0,
        "hand_1": hand_1,
        "hand_2": hand_2,
        "blurry": blurry,
        "dark": dark,
        "bright": bright,
        "near_duplicates": near_dupes,
        "success_rate": (100.0 * hand_1 + hand_2) / total if total else 0.0,
        "valid_rate": (100.0 * hand_2) / total if total else 0.0,
    }


def main() -> None:
    if not os.path.isdir(ISL_ROOT):
        raise SystemExit(f"ISL dataset root not found: {ISL_ROOT}")

    landmarker = get_two_hand_landmarker()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Auditing ISL webcam dataset: {ISL_ROOT}")
    print(f"Using two-hand MediaPipe landmarker (num_hands=2)\n")

    rows = []
    start = time.time()
    for letter in LETTERS:
        row = audit_letter(letter, landmarker)
        rows.append(row)
        print(
            f"  {letter}: total={row['total']}  2hand={row['hand_2']}  "
            f"1hand={row['hand_1']}  0hand={row['hand_0']}  "
            f"valid%={row['valid_rate']:.1f}"
        )
    elapsed = time.time() - start

    # Statistics across classes.
    totals = [r["total"] for r in rows]
    two_hand_counts = [r["hand_2"] for r in rows]

    def _mean(vals):
        return float(np.mean(vals)) if vals else 0.0

    def _median(vals):
        return float(np.median(vals)) if vals else 0.0

    # 5 classes with the fewest valid two-hand samples.
    fewest = sorted(rows, key=lambda r: (r["hand_2"], r["letter"]))[:5]

    # Build the report text.
    lines = []
    lines.append("ISL Two-Hand Dataset Audit")
    lines.append("=" * 78)
    lines.append(f"Dataset root       : {ISL_ROOT}")
    lines.append(f"Classes            : A-Z (26)")
    lines.append(f"Elapsed seconds    : {elapsed:.1f}")
    lines.append("")
    lines.append("Hand-count legend:")
    lines.append("  0-hand  = MediaPipe detected zero hands")
    lines.append("  1-hand  = exactly one hand detected")
    lines.append("  2-hand  = exactly two hands detected (valid two-hand sample)")
    lines.append("")
    lines.append(
        f"{'Letter':<7}{'Total':>7}{'2-Hand':>8}{'1-Hand':>8}{'0-Hand':>8}"
        f"{'Valid%':>8}{'Blurry':>8}{'Dark':>6}{'Bright':>8}{'Dupes':>7}"
    )
    lines.append("-" * 78)
    for r in rows:
        lines.append(
            f"{r['letter']:<7}{r['total']:>7}{r['hand_2']:>8}{r['hand_1']:>8}"
            f"{r['hand_0']:>8}{r['valid_rate']:>7.1f}%{r['blurry']:>8}"
            f"{r['dark']:>6}{r['bright']:>8}{r['near_duplicates']:>7}"
        )
    lines.append("-" * 78)
    lines.append("")
    lines.append("Per-class statistics (total images):")
    lines.append(f"  minimum : {min(totals)}")
    lines.append(f"  maximum : {max(totals)}")
    lines.append(f"  mean    : {_mean(totals):.1f}")
    lines.append(f"  median  : {_median(totals):.1f}")
    lines.append("")
    lines.append("Per-class statistics (valid two-hand samples):")
    lines.append(f"  minimum : {min(two_hand_counts)}")
    lines.append(f"  maximum : {max(two_hand_counts)}")
    lines.append(f"  mean    : {_mean(two_hand_counts):.1f}")
    lines.append(f"  median  : {_median(two_hand_counts):.1f}")
    lines.append("")
    lines.append("Landmark detection success rate (1-hand + 2-hand as success):")
    total_imgs = sum(totals)
    total_detected = sum(r["hand_1"] + r["hand_2"] for r in rows)
    lines.append(f"  overall : {100.0 * total_detected / total_imgs:.2f}% "
                 f"({total_detected}/{total_imgs})")
    lines.append("")
    lines.append("5 classes with the fewest valid two-hand samples:")
    for r in fewest:
        lines.append(f"  {r['letter']}: {r['hand_2']} valid two-hand samples")
    lines.append("")
    lines.append("Recommendation (do NOT retrain yet):")
    lines.append("  Collect additional two-hand ISL webcam samples for the letters")
    lines.append("  with very few valid two-hand images. A reasonable balance target")
    lines.append("  is ~40-50 valid two-hand samples per class.")
    lines.append("")
    # Specific per-letter collection suggestions.
    target = 45
    deficit_rows = [r for r in rows if r["hand_2"] < target]
    lines.append(f"  Suggested additional captures to reach ~{target} two-hand "
                 f"samples/class:")
    for r in sorted(deficit_rows, key=lambda r: r["hand_2"]):
        needed = max(0, target - r["hand_2"])
        lines.append(f"    {r['letter']}: +{needed} two-hand samples "
                     f"(currently {r['hand_2']})")

    report = "\n".join(lines) + "\n"
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write(report)

    print(f"\nAudit complete in {elapsed:.1f}s")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
