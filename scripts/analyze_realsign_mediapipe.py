"""MediaPipe hand-count + landmark analysis for the RealSign ISL dataset.

Read-only analysis.  For each letter and each split it samples up to
``--max-per-class`` images (default 50) and reports, via MediaPipe two-hand
detection:

    * detected hand count (0 / 1 / 2)
    * handedness distribution
    * successful vs failed landmark extraction

Outputs ``results/realsign/isl_public_mediapipe.json`` plus a human-readable
``results/realsign/isl_public_mediapipe.txt``.

Run from the project root::

    python scripts/analyze_realsign_mediapipe.py
    python scripts/analyze_realsign_mediapipe.py --max-per-class 25
"""

from __future__ import annotations

import argparse
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

PUBLIC_ROOT = os.path.join(PROJECT_ROOT, "dataset", "isl_public")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "realsign")
SPLITS = ("Training", "Testing", "Validation")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def _image_files(folder: str, limit: int | None) -> list[str]:
    if not os.path.isdir(folder):
        return []
    files = sorted(n for n in os.listdir(folder) if n.lower().endswith(IMG_EXTS))
    if limit is not None:
        files = files[:limit]
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=50,
        help="Max images sampled per letter per split (default 50).",
    )
    args = parser.parse_args()

    if not os.path.isdir(PUBLIC_ROOT):
        raise SystemExit(f"Dataset not found: {PUBLIC_ROOT}")

    landmarker = get_two_hand_landmarker()

    report: dict = {"root": PUBLIC_ROOT, "max_per_class": args.max_per_class, "splits": {}}
    summary: dict = {}

    start = time.time()

    for split in SPLITS:
        split_dir = os.path.join(PUBLIC_ROOT, split)
        split_info: dict = {"classes": {}, "totals": {"images": 0, "zero": 0, "one": 0, "two": 0, "failed": 0}}

        for letter in LETTERS:
            folder = os.path.join(split_dir, letter)
            files = _image_files(folder, args.max_per_class)
            c = {"sampled": len(files), "zero": 0, "one": 0, "two": 0,
                 "failed": 0, "left": 0, "right": 0}
            for name in files:
                path = os.path.join(folder, name)
                split_info["totals"]["images"] += 1
                result = extract_two_hands_from_file(path, landmarker=landmarker)
                if result is None:
                    c["zero"] += 1
                    c["failed"] += 1
                    split_info["totals"]["zero"] += 1
                    split_info["totals"]["failed"] += 1
                    continue
                n_hands = int(result.left_present) + int(result.right_present)
                if n_hands == 1:
                    c["one"] += 1
                    split_info["totals"]["one"] += 1
                elif n_hands == 2:
                    c["two"] += 1
                    split_info["totals"]["two"] += 1
                else:
                    c["zero"] += 1
                    split_info["totals"]["zero"] += 1
                if result.left_present:
                    c["left"] += 1
                if result.right_present:
                    c["right"] += 1
            split_info["classes"][letter] = c

        report["splits"][split] = split_info
        summary[split] = split_info["totals"]

    elapsed = time.time() - start
    report["summary"] = summary
    report["elapsed_seconds"] = round(elapsed, 1)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, "isl_public_mediapipe.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    # Human-readable text.
    lines = []
    lines.append("RealSign ISL MediaPipe hand-count analysis")
    lines.append("=" * 80)
    lines.append(f"max_per_class : {args.max_per_class}")
    lines.append("")
    header = (
        f"{'Letter':<7}{'Sampled':>8}{'0-hand':>8}{'1-hand':>8}{'2-hand':>8}"
        f"{'Fail':>6}{'Left':>6}{'Right':>7}"
    )
    for split in SPLITS:
        si = report["splits"][split]
        lines.append(f"\n[{split}]")
        lines.append(header)
        lines.append("-" * 80)
        for letter in LETTERS:
            c = si["classes"][letter]
            lines.append(
                f"{letter:<7}{c['sampled']:>8}{c['zero']:>8}{c['one']:>8}"
                f"{c['two']:>8}{c['failed']:>6}{c['left']:>6}{c['right']:>7}"
            )
        t = si["totals"]
        lines.append("-" * 80)
        lines.append(
            f"{'TOTAL':<7}{t['images']:>8}{t['zero']:>8}{t['one']:>8}"
            f"{t['two']:>8}{t['failed']:>6}"
        )

    txt_path = os.path.join(RESULTS_DIR, "isl_public_mediapipe.txt")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nJSON -> {json_path}")
    print(f"TXT  -> {txt_path}")
    print(f"Elapsed {elapsed:.1f}s")


if __name__ == "__main__":
    main()
