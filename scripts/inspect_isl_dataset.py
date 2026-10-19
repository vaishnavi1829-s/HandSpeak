"""Inspect the collected INDIAN SIGN LANGUAGE (ISL) dataset under
``dataset/isl_webcam/``.

This is the STEP 1 (dataset verification) stage of the ISL pipeline. It only
READS the dataset; it never modifies, moves, or deletes any image.

Checks performed:

    1. All 26 letter folders (A-Z) exist.
    2. Counts processed 28x28 samples per letter (and ROI files where present).
    3. Detects missing / severely underrepresented classes.
    4. Detects corrupted, empty, unreadable, or invalid images (both the
       processed 28x28 inputs and the raw ROI captures).
    5. Reports image dimensions and channel layout.
    6. Confirms compatibility with ``recognition.preprocessing`` (28x28, single
       channel, values stored as uint8 0-255 and normalized later by /255.0).

Run from the project root::

    python scripts/inspect_isl_dataset.py
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.label_mapping import LETTERS  # noqa: E402

ISL_ROOT = os.path.join(PROJECT_ROOT, "dataset", "isl_webcam")
TARGET_SIZE = 28
MIN_SAMPLES_PER_CLASS = 10


def _png_files(folder: str) -> list[str]:
    """Return sorted image file names in ``folder`` (or an empty list)."""
    if not os.path.isdir(folder):
        return []
    return sorted(
        name
        for name in os.listdir(folder)
        if name.lower().endswith((".png", ".jpg", ".jpeg"))
    )


def inspect_letter(letter: str) -> dict:
    """Inspect one letter folder and return a summary dict."""
    letter_dir = os.path.join(ISL_ROOT, letter)
    processed_dir = os.path.join(letter_dir, "processed")
    roi_dir = os.path.join(letter_dir, "roi")

    processed_files = _png_files(processed_dir)
    roi_files = _png_files(roi_dir)

    result = {
        "letter": letter,
        "processed_count": len(processed_files),
        "roi_count": len(roi_files),
        "corrupted_processed": [],
        "corrupted_roi": [],
        "dimensions": set(),
        "channels": set(),
        "value_min": None,
        "value_max": None,
        "empty_or_blank": [],
    }

    # Inspect processed (28x28 CNN input) images.
    for name in processed_files:
        path = os.path.join(processed_dir, name)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            result["corrupted_processed"].append(name)
            continue
        if img.ndim == 3:
            result["channels"].add(img.shape[2])
        else:
            result["channels"].add(1)
        result["dimensions"].add((img.shape[0], img.shape[1]))
        mn, mx = float(img.min()), float(img.max())
        result["value_min"] = mn if result["value_min"] is None else min(result["value_min"], mn)
        result["value_max"] = mx if result["value_max"] is None else max(result["value_max"], mx)
        if mx - mn <= 0:
            result["empty_or_blank"].append(name)

    # Inspect ROI (raw capture) images.
    for name in roi_files:
        path = os.path.join(roi_dir, name)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            result["corrupted_roi"].append(name)

    return result


def main() -> None:
    print("=" * 70)
    print("ISL dataset inspection (dataset/isl_webcam/)")
    print("=" * 70)

    results = [inspect_letter(letter) for letter in LETTERS]

    # 1) Folder presence.
    missing_folders = [
        letter for letter in LETTERS
        if not os.path.isdir(os.path.join(ISL_ROOT, letter))
    ]
    print(f"\n[1] Letter folders present: {26 - len(missing_folders)}/26")
    if missing_folders:
        print(f"     MISSING folders: {''.join(missing_folders)}")
    else:
        print("     All 26 folders (A-Z) exist.")

    # 2) Per-letter counts.
    print("\n[2] Sample counts (processed 28x28 images):")
    counts = {}
    line_parts = []
    for r in results:
        counts[r["letter"]] = r["processed_count"]
        line_parts.append(f"{r['letter']}:{r['processed_count']}")
    print("     " + "  ".join(line_parts))
    total = sum(counts.values())
    print(f"     TOTAL processed samples: {total}")

    roi_total = sum(r["roi_count"] for r in results)
    print(f"     TOTAL roi captures     : {roi_total}")

    # 3) Missing / severely underrepresented classes.
    missing = [l for l, c in counts.items() if c == 0]
    underrepresented = [l for l, c in counts.items() if 0 < c < MIN_SAMPLES_PER_CLASS]
    print("\n[3] Missing / underrepresented classes:")
    if missing:
        print(f"     MISSING (0 samples)      : {''.join(missing)}")
    if underrepresented:
        print(f"     UNDERREPRESENTED (<{MIN_SAMPLES_PER_CLASS}) : {''.join(underrepresented)}")
    if not missing and not underrepresented:
        print(f"     None - every class has >= {MIN_SAMPLES_PER_CLASS} samples.")

    # 4) Corrupted / unreadable / invalid images.
    print("\n[4] Corrupted / unreadable / invalid images:")
    total_corrupt = 0
    for r in results:
        c = len(r["corrupted_processed"]) + len(r["corrupted_roi"])
        total_corrupt += c
        if c:
            print(f"     {r['letter']}: {c} corrupted "
                  f"(processed={len(r['corrupted_processed'])}, roi={len(r['corrupted_roi'])})")
    if total_corrupt == 0:
        print("     None - every processed and roi image is readable.")

    # 5) Dimensions and channels.
    print("\n[5] Dimensions & channels (processed images):")
    all_dims = set()
    all_chans = set()
    for r in results:
        all_dims |= r["dimensions"]
        all_chans |= r["channels"]
    print(f"     Unique dimensions: {sorted(all_dims) if all_dims else 'N/A'}")
    print(f"     Unique channels  : {sorted(all_chans) if all_chans else 'N/A'}")
    expected_dims = {(TARGET_SIZE, TARGET_SIZE)}
    dims_ok = all_dims == expected_dims
    chans_ok = all_chans == {1}
    print(f"     Expected 28x28 grayscale? dims_ok={dims_ok}, channels_ok={chans_ok}")

    # 6) Value range compatibility.
    print("\n[6] Value range (processed images):")
    overall_min = min((r["value_min"] for r in results if r["value_min"] is not None), default=None)
    overall_max = max((r["value_max"] for r in results if r["value_max"] is not None), default=None)
    print(f"     Global min={overall_min}, max={overall_max} (uint8 0-255 expected).")

    blank_total = sum(len(r["empty_or_blank"]) for r in results)
    if blank_total:
        print(f"     WARNING: {blank_total} blank/constant images detected "
              f"(they will be excluded during training).")
        for r in results:
            if r["empty_or_blank"]:
                print(f"       {r['letter']}: {len(r['empty_or_blank'])} blank")
    else:
        print("     No blank/constant images detected.")

    # Summary verdict.
    print("\n" + "-" * 70)
    print("VERDICT:")
    problems = []
    if missing_folders:
        problems.append(f"missing folders {''.join(missing_folders)}")
    if missing:
        problems.append(f"missing classes {''.join(missing)}")
    if total_corrupt:
        problems.append(f"{total_corrupt} corrupted images")
    if not dims_ok:
        problems.append("dimension mismatch")
    if not chans_ok:
        problems.append("channel mismatch")
    if problems:
        print(f"  ISSUES FOUND: {', '.join(problems)}")
    else:
        print("  OK - ISL dataset is compatible with the existing preprocessing/CNN pipeline.")
    print("=" * 70)


if __name__ == "__main__":
    main()
