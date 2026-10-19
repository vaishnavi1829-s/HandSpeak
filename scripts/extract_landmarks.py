"""Generate MediaPipe hand-landmark datasets for ASL and ISL (separately).

This script reads the EXISTING webcam ROI images and produces two independent
CSV datasets (never mixing ASL and ISL):

    dataset/landmarks/asl_landmarks.csv
    dataset/landmarks/isl_landmarks.csv

Each CSV row contains the letter label, language/dataset name, source image
path, optional handedness/score metadata, and 63 landmark features
(``lm0_x`` ... ``lm20_z``) produced by
``recognition.landmark_extractor``.

A small extraction summary is also written next to each CSV:

    dataset/landmarks/asl_landmarks_summary.txt
    dataset/landmarks/isl_landmarks_summary.txt

This script only READS images and WRITES new CSV/summary files.  It does not
modify or delete any existing image or model.

Run from the project root::

    python scripts/extract_landmarks.py
    python scripts/extract_landmarks.py --dataset asl
    python scripts/extract_landmarks.py --dataset isl
    python scripts/extract_landmarks.py --max-images 3   # quick smoke test
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
    FEATURE_COLUMNS,
    extract_landmarks_from_file,
    get_hand_landmarker,
)
from utils.label_mapping import LETTERS  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASETS = {
    "asl": {
        "root": os.path.join(PROJECT_ROOT, "dataset", "webcam"),
        "language": "ASL",
    },
    "isl": {
        "root": os.path.join(PROJECT_ROOT, "dataset", "isl_webcam"),
        "language": "ISL",
    },
}

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dataset", "landmarks")
FAILED_DIR = os.path.join(OUTPUT_DIR, "failed")

CSV_META_COLUMNS = ["label", "language", "source_path", "handedness", "hand_score"]
CSV_COLUMNS = CSV_META_COLUMNS + FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _image_files(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    return sorted(
        name
        for name in os.listdir(folder)
        if name.lower().endswith((".png", ".jpg", ".jpeg"))
    )


def write_summary(
    path: str,
    dataset_name: str,
    counts: dict[str, int],
    successful: int,
    failed: int,
    failed_rows: list[dict],
    elapsed: float,
) -> None:
    total = successful + failed
    lines = []
    lines.append("Landmark extraction summary")
    lines.append("=" * 60)
    lines.append(f"dataset            : {dataset_name}")
    lines.append(f"total images       : {total}")
    lines.append(f"successful detections : {successful}")
    lines.append(f"failed detections  : {failed}")
    if total:
        lines.append(f"success rate       : {100.0 * successful / total:.2f}%")
    lines.append(f"elapsed seconds    : {elapsed:.2f}")
    lines.append("")
    lines.append("Samples per letter:")
    for letter in LETTERS:
        lines.append(f"  {letter}: {counts.get(letter, 0)}")

    if failed_rows:
        lines.append("")
        lines.append("Failed images (no hand detected or unreadable):")
        for row in failed_rows:
            lines.append(f"  [{row['label']}] {row['source_path']}")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def extract_dataset(dataset_name: str, max_images: int | None = None) -> None:
    cfg = DATASETS[dataset_name]
    root = cfg["root"]
    language = cfg["language"]

    if not os.path.isdir(root):
        raise SystemExit(f"Dataset root not found: {root}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FAILED_DIR, exist_ok=True)

    csv_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_landmarks.csv")
    summary_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_landmarks_summary.txt")
    failed_path = os.path.join(FAILED_DIR, f"{dataset_name}_failed.json")

    landmarker = get_hand_landmarker()

    counts: dict[str, int] = {letter: 0 for letter in LETTERS}
    successful = 0
    failed = 0
    failed_rows: list[dict] = []
    start = time.time()

    print(f"\n[{dataset_name.upper()}] Extracting landmarks from {root}")
    print(f"  output -> {csv_path}")

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for letter in LETTERS:
            folder = os.path.join(root, letter, "roi")
            files = _image_files(folder)

            if max_images is not None:
                files = files[:max_images]

            letter_total = len(files)
            letter_ok = 0

            for name in files:
                source_path = os.path.join(folder, name)
                rel_path = os.path.relpath(source_path, PROJECT_ROOT)

                result = extract_landmarks_from_file(source_path, landmarker=landmarker)

                if result is None:
                    failed += 1
                    failed_rows.append({"label": letter, "source_path": rel_path})
                    continue

                row = {
                    "label": letter,
                    "language": language,
                    "source_path": rel_path,
                    "handedness": result.handedness or "",
                    "hand_score": (
                        f"{result.hand_score:.6f}" if result.hand_score is not None else ""
                    ),
                }
                for col, value in zip(FEATURE_COLUMNS, result.feature_vector):
                    row[col] = f"{value:.8f}"

                writer.writerow(row)
                successful += 1
                letter_ok += 1

            counts[letter] = letter_ok
            print(f"  {letter}: {letter_ok}/{letter_total} detected")

    elapsed = time.time() - start

    write_summary(
        summary_path,
        dataset_name,
        counts,
        successful,
        failed,
        failed_rows,
        elapsed,
    )

    with open(failed_path, "w", encoding="utf-8") as fh:
        json.dump(failed_rows, fh, indent=2)

    print(f"\n[{dataset_name.upper()}] Done: {successful} successful, "
          f"{failed} failed in {elapsed:.2f}s")
    print(f"  summary -> {summary_path}")
    print(f"  failed  -> {failed_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["asl", "isl", "all"],
        default="all",
        help="Which dataset to process (default: all).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Cap images per letter (useful for smoke testing).",
    )
    args = parser.parse_args()

    names = ["asl", "isl"] if args.dataset == "all" else [args.dataset]
    for name in names:
        extract_dataset(name, max_images=args.max_images)


if __name__ == "__main__":
    main()
