"""Verify the extracted RealSign ISL dataset (read-only).

Checks, for ``dataset/isl_public``:

    * All 26 alphabet folders exist in Training / Testing / Validation.
    * Image counts per class.
    * Image dimensions (width x height) and format extension.
    * Corrupted images (cannot be decoded by OpenCV).
    * Duplicate images (exact byte/hash duplicates).
    * Blank / near-uniform images (low pixel variance).

Outputs a JSON report to ``results/realsign/isl_public_verification.json`` and
prints a concise summary.  This script only READS images; it never modifies
them.

Run from the project root::

    python scripts/verify_realsign_dataset.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from utils.label_mapping import LETTERS  # noqa: E402

PUBLIC_ROOT = os.path.join(PROJECT_ROOT, "dataset", "isl_public")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "realsign")
SPLITS = ("Training", "Testing", "Validation")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def _image_files(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    return sorted(
        n for n in os.listdir(folder) if n.lower().endswith(IMG_EXTS)
    )


def _is_blank(img: np.ndarray) -> bool:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(gray.std()) < 5.0


def main() -> None:
    if not os.path.isdir(PUBLIC_ROOT):
        raise SystemExit(f"Dataset not found: {PUBLIC_ROOT}. Extract Dataset.zip first.")

    report: dict = {
        "root": PUBLIC_ROOT,
        "splits": {},
        "summary": {},
        "corrupted": [],
        "blank": [],
        "duplicates": {},
    }

    start = time.time()
    global_hash_index: dict[str, list[str]] = {}

    for split in SPLITS:
        split_dir = os.path.join(PUBLIC_ROOT, split)
        split_info = {
            "classes": {},
            "corrupted": 0,
            "blank": 0,
            "total": 0,
            "dimensions": {},
        }
        for letter in LETTERS:
            folder = os.path.join(split_dir, letter)
            files = _image_files(folder)
            n = len(files)
            split_info["classes"][letter] = {
                "images": n,
                "formats": {},
                "dimensions": [],
                "corrupted": 0,
                "blank": 0,
            }
            split_info["total"] += n

            for name in files:
                path = os.path.join(folder, name)
                ext = os.path.splitext(name)[1].lower()
                split_info["classes"][letter]["formats"][ext] = (
                    split_info["classes"][letter]["formats"].get(ext, 0) + 1
                )

                # Hash (exact duplicate detection).
                try:
                    with open(path, "rb") as fh:
                        digest = hashlib.md5(fh.read()).hexdigest()
                    global_hash_index.setdefault(digest, []).append(
                        os.path.relpath(path, PUBLIC_ROOT)
                    )
                except OSError:
                    pass

                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is None:
                    split_info["classes"][letter]["corrupted"] += 1
                    split_info["corrupted"] += 1
                    report["corrupted"].append(os.path.relpath(path, PUBLIC_ROOT))
                    continue

                h, w = img.shape[:2]
                dim = f"{w}x{h}"
                split_info["classes"][letter]["dimensions"].append(dim)
                split_info["dimensions"][dim] = split_info["dimensions"].get(dim, 0) + 1

                if _is_blank(img):
                    split_info["classes"][letter]["blank"] += 1
                    split_info["blank"] += 1
                    report["blank"].append(os.path.relpath(path, PUBLIC_ROOT))

        report["splits"][split] = split_info

    # Aggregate duplicates across the whole dataset.
    duplicate_groups = {
        digest: paths for digest, paths in global_hash_index.items() if len(paths) > 1
    }
    report["duplicates"] = {
        "groups": len(duplicate_groups),
        "total_duplicate_images": sum(len(p) for p in duplicate_groups.values()),
        "examples": {
            digest: paths for digest, paths in list(duplicate_groups.items())[:20]
        },
    }

    # Summary.
    for split in SPLITS:
        s = report["splits"][split]
        report["summary"][split] = {
            "total_images": s["total"],
            "classes": len(s["classes"]),
            "corrupted": s["corrupted"],
            "blank": s["blank"],
            "missing_letters": [
                l for l, c in s["classes"].items() if c["images"] == 0
            ],
            "distinct_dimensions": len(s["dimensions"]),
        }

    elapsed = time.time() - start
    report["summary"]["elapsed_seconds"] = round(elapsed, 1)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "isl_public_verification.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    # Console summary.
    print("\nRealSign ISL dataset verification")
    print("=" * 70)
    for split in SPLITS:
        s = report["summary"][split]
        print(f"\n{split}: {s['total_images']} images, {s['classes']} classes")
        print(f"  corrupted={s['corrupted']}  blank={s['blank']}  "
              f"missing_letters={''.join(s['missing_letters']) or 'none'}")
        print(f"  distinct_dimensions={s['distinct_dimensions']}")
    print(f"\nDuplicates: {report['duplicates']['groups']} groups, "
          f"{report['duplicates']['total_duplicate_images']} files involved")
    print(f"\nReport -> {out_path}")
    print(f"Elapsed {elapsed:.1f}s")


if __name__ == "__main__":
    main()
