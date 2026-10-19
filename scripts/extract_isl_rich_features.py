"""Extract RICH ISL features and build ``isl_rich_features.csv``.

This is a NEW, experimental dataset.  It does NOT overwrite:

    dataset/landmarks/isl_landmarks.csv
    dataset/landmarks/isl_landmarks_2hand.csv

For every ISL webcam ROI image the script:

    1. Detects up to 2 hands with MediaPipe.
    2. Assigns each hand to LEFT / RIGHT using MediaPipe handedness (NOT
       detection order), reusing ``extract_two_hands_from_file``.
    3. Builds the rich 214-feature vector via
       ``recognition.isl_feature_extractor`` (normalized coordinates, joint
       angles, finger segment lengths, wrist-to-tip distances, fingertip-to-tip
       distances, cross-hand distances, and hand-presence flags).
    4. Writes one row per VALID sample (1 or 2 hands detected; zero-hand images
       are logged, not silently dropped from the report).

The feature schema/order is saved to
``dataset/landmarks/isl_feature_schema.json`` and MUST be reused at training,
validation, offline test, and real-time prediction time.

Outputs:

    dataset/landmarks/isl_rich_features.csv
    dataset/landmarks/isl_rich_features_summary.txt
    dataset/landmarks/isl_feature_schema.json
    dataset/landmarks/failed/isl_rich_failed.json

Run from the project root::

    python scripts/extract_isl_rich_features.py
    python scripts/extract_isl_rich_features.py --max-images 5
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
    extract_two_hands_from_file,
    get_two_hand_landmarker,
)
from recognition.isl_feature_extractor import (  # noqa: E402
    FEATURE_COUNT,
    RICH_FEATURE_COLUMNS,
    get_schema,
    rich_features_from_two_hand_result,
)
from utils.label_mapping import LETTERS  # noqa: E402

ISL_ROOT = os.path.join(PROJECT_ROOT, "dataset", "isl_webcam")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dataset", "landmarks")
FAILED_DIR = os.path.join(OUTPUT_DIR, "failed")

CSV_META_COLUMNS = [
    "label",
    "language",
    "source_path",
    "hands_detected",
    "left_hand",
    "right_hand",
]
CSV_COLUMNS = CSV_META_COLUMNS + RICH_FEATURE_COLUMNS

SCHEMA_PATH = os.path.join(OUTPUT_DIR, "isl_feature_schema.json")


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

    csv_path = os.path.join(OUTPUT_DIR, "isl_rich_features.csv")
    summary_path = os.path.join(OUTPUT_DIR, "isl_rich_features_summary.txt")
    failed_path = os.path.join(FAILED_DIR, "isl_rich_failed.json")

    # Save the schema FIRST so it exists even if extraction is interrupted.
    with open(SCHEMA_PATH, "w", encoding="utf-8") as fh:
        json.dump(get_schema(), fh, indent=2)
    print(f"[ISL RICH] Feature schema ({FEATURE_COUNT} features) -> {SCHEMA_PATH}")

    landmarker = get_two_hand_landmarker()

    valid_per_letter = {letter: 0 for letter in LETTERS}
    total = 0
    zero_hand = 0
    failed_rows: list[dict] = []

    start = time.time()
    print(f"\n[ISL RICH] Extracting from {ISL_ROOT}")
    print(f"  output -> {csv_path}")

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for letter in LETTERS:
            folder = os.path.join(ISL_ROOT, letter, "roi")
            files = _image_files(folder)
            if args.max_images is not None:
                files = files[: args.max_images]

            letter_valid = 0
            for name in files:
                source_path = os.path.join(folder, name)
                rel_path = os.path.relpath(source_path, PROJECT_ROOT)

                result = extract_two_hands_from_file(source_path, landmarker=landmarker)
                total += 1

                if result is None:
                    zero_hand += 1
                    failed_rows.append(
                        {"label": letter, "source_path": rel_path, "hands": 0}
                    )
                    continue

                n_hands = int(result.left_present) + int(result.right_present)
                features, feature_names = rich_features_from_two_hand_result(result)

                row = {
                    "label": letter,
                    "language": "ISL",
                    "source_path": rel_path,
                    "hands_detected": str(n_hands),
                    "left_hand": "1" if result.left_present else "0",
                    "right_hand": "1" if result.right_present else "0",
                }
                for col, value in zip(feature_names, features):
                    row[col] = f"{value:.8f}"

                writer.writerow(row)
                valid_per_letter[letter] += 1
                letter_valid += 1

            print(f"  {letter}: {letter_valid}/{len(files)} valid")

    elapsed = time.time() - start

    # Summary.
    lines = []
    lines.append("ISL rich-feature extraction summary")
    lines.append("=" * 60)
    lines.append(f"feature count      : {FEATURE_COUNT}")
    lines.append(f"total images       : {total}")
    lines.append(f"valid samples      : {sum(valid_per_letter.values())}")
    lines.append(f"zero-hand skipped  : {zero_hand}")
    lines.append(f"elapsed seconds    : {elapsed:.2f}")
    lines.append("")
    lines.append("Valid samples per letter:")
    for letter in LETTERS:
        lines.append(f"  {letter}: {valid_per_letter[letter]}")
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    with open(failed_path, "w", encoding="utf-8") as fh:
        json.dump(failed_rows, fh, indent=2)

    print(f"\n[ISL RICH] Done in {elapsed:.2f}s")
    print(f"  valid samples : {sum(valid_per_letter.values())}")
    print(f"  zero-hand     : {zero_hand}")
    print(f"  summary       -> {summary_path}")
    print(f"  failed log    -> {failed_path}")


if __name__ == "__main__":
    main()
