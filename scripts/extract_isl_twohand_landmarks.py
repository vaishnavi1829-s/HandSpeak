"""Extract TWO-HAND ISL landmarks and build ``isl_landmarks_2hand.csv``.

This is a separate, experimental script.  It does NOT touch the existing
one-hand ISL CSV (``dataset/landmarks/isl_landmarks.csv``) and does NOT touch
the one-hand ISL model.

For every ISL webcam ROI image the script:

    1. Detects up to 2 hands with MediaPipe.
    2. Uses handedness to assign each hand to LEFT / RIGHT (not detection order).
    3. Applies the SAME wrist-relative + scale normalization used everywhere.
    4. Builds a deterministic 126-feature vector::

           lh_lm0_x ... lh_lm20_z  (LEFT hand, 63 features)
           rh_lm0_x ... rh_lm20_z  (RIGHT hand, 63 features)

Only images where BOTH hands are detected are written to the training CSV.
Images with 0 or 1 hand are logged (not silently treated as valid two-hand
samples) so the data-quality situation is fully transparent.

Outputs:

    dataset/landmarks/isl_landmarks_2hand.csv
    dataset/landmarks/isl_landmarks_2hand_summary.txt
    dataset/landmarks/failed/isl_twohand_failed.json

Run from the project root::

    python scripts/extract_isl_twohand_landmarks.py
    python scripts/extract_isl_twohand_landmarks.py --max-images 5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from recognition.landmark_extractor import (  # noqa: E402
    TWO_HAND_FEATURE_COLUMNS,
    extract_two_hands_from_file,
    get_two_hand_landmarker,
)
from utils.label_mapping import LETTERS  # noqa: E402

ISL_ROOT = os.path.join(PROJECT_ROOT, "dataset", "isl_webcam")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dataset", "landmarks")
FAILED_DIR = os.path.join(OUTPUT_DIR, "failed")

CSV_META_COLUMNS = [
    "label",
    "language",
    "source_path",
    "left_hand",
    "right_hand",
]
CSV_COLUMNS = CSV_META_COLUMNS + TWO_HAND_FEATURE_COLUMNS


def _image_files(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    return sorted(
        name
        for name in os.listdir(folder)
        if name.lower().endswith((".png", ".jpg", ".jpeg"))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Cap images per letter (useful for smoke testing).",
    )
    args = parser.parse_args()

    if not os.path.isdir(ISL_ROOT):
        raise SystemExit(f"ISL dataset root not found: {ISL_ROOT}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FAILED_DIR, exist_ok=True)

    csv_path = os.path.join(OUTPUT_DIR, "isl_landmarks_2hand.csv")
    summary_path = os.path.join(OUTPUT_DIR, "isl_landmarks_2hand_summary.txt")
    failed_path = os.path.join(FAILED_DIR, "isl_twohand_failed.json")

    landmarker = get_two_hand_landmarker()

    # Data-quality counters.
    total_images = 0
    two_hand = 0
    one_hand = 0
    zero_hand = 0
    two_hand_per_letter = {letter: 0 for letter in LETTERS}
    one_hand_per_letter = {letter: 0 for letter in LETTERS}
    zero_hand_per_letter = {letter: 0 for letter in LETTERS}
    failed_rows: list[dict] = []

    start = time.time()
    print(f"\n[ISL 2-HAND] Extracting from {ISL_ROOT}")
    print(f"  output -> {csv_path}")

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for letter in LETTERS:
            folder = os.path.join(ISL_ROOT, letter, "roi")
            files = _image_files(folder)
            if args.max_images is not None:
                files = files[: args.max_images]

            letter_two = 0
            letter_one = 0
            letter_zero = 0

            for name in files:
                source_path = os.path.join(folder, name)
                rel_path = os.path.relpath(source_path, PROJECT_ROOT)

                result = extract_two_hands_from_file(source_path, landmarker=landmarker)
                total_images += 1

                if result is None:
                    zero_hand += 1
                    letter_zero += 1
                    failed_rows.append(
                        {"label": letter, "source_path": rel_path, "hands": 0}
                    )
                    continue

                if result.left_present and result.right_present:
                    two_hand += 1
                    letter_two += 1

                    row = {
                        "label": letter,
                        "language": "ISL",
                        "source_path": rel_path,
                        "left_hand": "1",
                        "right_hand": "1",
                    }
                    for col, value in zip(TWO_HAND_FEATURE_COLUMNS, result.feature_vector):
                        row[col] = f"{value:.8f}"
                    writer.writerow(row)
                else:
                    one_hand += 1
                    letter_one += 1
                    failed_rows.append(
                        {
                            "label": letter,
                            "source_path": rel_path,
                            "hands": 1,
                            "left_hand": result.left_present,
                            "right_hand": result.right_present,
                        }
                    )

            two_hand_per_letter[letter] = letter_two
            one_hand_per_letter[letter] = letter_one
            zero_hand_per_letter[letter] = letter_zero
            print(
                f"  {letter}: 2-hand={letter_two}  1-hand={letter_one}  "
                f"0-hand={letter_zero}  (total={len(files)})"
            )

    elapsed = time.time() - start

    # Write summary.
    lines = []
    lines.append("ISL two-hand landmark extraction summary")
    lines.append("=" * 60)
    lines.append(f"total images        : {total_images}")
    lines.append(f"two-hand detected   : {two_hand}")
    lines.append(f"one-hand detected   : {one_hand}")
    lines.append(f"zero-hand detected  : {zero_hand}")
    if total_images:
        lines.append(f"two-hand rate       : {100.0 * two_hand / total_images:.2f}%")
    lines.append(f"elapsed seconds     : {elapsed:.2f}")
    lines.append("")
    lines.append("Per-letter two-hand sample counts:")
    for letter in LETTERS:
        lines.append(
            f"  {letter}: 2-hand={two_hand_per_letter[letter]}  "
            f"1-hand={one_hand_per_letter[letter]}  "
            f"0-hand={zero_hand_per_letter[letter]}"
        )

    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    with open(failed_path, "w", encoding="utf-8") as fh:
        json.dump(failed_rows, fh, indent=2)

    print(f"\n[ISL 2-HAND] Done in {elapsed:.2f}s")
    print(f"  two-hand   : {two_hand}")
    print(f"  one-hand   : {one_hand}")
    print(f"  zero-hand  : {zero_hand}")
    print(f"  summary    -> {summary_path}")
    print(f"  failed log -> {failed_path}")


if __name__ == "__main__":
    main()
