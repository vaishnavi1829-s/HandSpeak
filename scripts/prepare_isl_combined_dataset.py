"""Prepare the EXPERIMENTAL combined ISL dataset (read-only w.r.t. originals).

Creates ``dataset/isl_combined/`` with:

    train/         A-Z   (RealSign Training + webcam train split)
    validation/    A-Z   (RealSign Validation + webcam val split)
    test_webcam/   A-Z   (unseen webcam images ONLY — never used in training)

Leakage/dedup safeguards
-------------------------
* Exact (MD5) and near-duplicate (64-bit dHash) detection within the webcam
  dataset; only one representative per duplicate group is kept so the same
  image cannot be split across train/val/test.
* Cross-dataset exact-duplicate check between webcam and RealSign training.
* A deterministic stratified webcam split (random_state=42) is applied AFTER
  dedup, so the test_webcam images are truly unseen.

This script only READS the original datasets and WRITES new files under
``dataset/isl_combined/``.  It never modifies ``dataset/isl_webcam/`` or
``dataset/isl_public/``.

Run from the project root::

    python scripts/prepare_isl_combined_dataset.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from utils.label_mapping import LETTERS  # noqa: E402

WEBCAM_ROOT = os.path.join(PROJECT_ROOT, "dataset", "isl_webcam")
PUBLIC_ROOT = os.path.join(PROJECT_ROOT, "dataset", "isl_public")
OUT_ROOT = os.path.join(PROJECT_ROOT, "dataset", "isl_combined")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

TEST_FRACTION = 0.20
VAL_FRACTION = 0.15  # of the remaining webcam images after test holdout
MIN_TEST_PER_CLASS = 12
RANDOM_STATE = 42

# Near-duplicate detection uses a very tight Hamming threshold so we only
# collapse essentially-identical images (which could otherwise leak across the
# train/test split).  Sequential webcam frames legitimately differ by more than
# this and are kept as distinct training samples.
DHASH_NEAR_DUP_THRESHOLD = 2


def _image_files(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    return sorted(
        n for n in os.listdir(folder) if n.lower().endswith(IMG_EXTS)
    )


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _dhash64(path: str) -> int:
    """64-bit difference hash (perceptual) of an image."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return -1
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    bits = 0
    for r in range(8):
        for c in range(8):
            bits <<= 1
            if img[r, c] > img[r, c + 1]:
                bits |= 1
    return bits


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _dedup_by_md5(files: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    """Return (unique files, {md5: [paths]}) for duplicate groups."""
    seen: dict[str, list[str]] = {}
    for f in files:
        seen.setdefault(_md5(f), []).append(f)
    dup_groups = {k: v for k, v in seen.items() if len(v) > 1}
    unique = [v[0] for v in seen.values()]
    return unique, dup_groups


def _dedup_by_dhash(files: list[str]) -> tuple[list[str], list[list[str]]]:
    """Remove near-duplicates (dHash) greedily; return (unique, [groups])."""
    hashes = [(f, _dhash64(f)) for f in files]
    groups: list[list[str]] = []
    keep: list[str] = []
    used: set[int] = set()

    for i, (f, h) in enumerate(hashes):
        if i in used:
            continue
        if h == -1:
            keep.append(f)
            continue
        group = [f]
        for j in range(i + 1, len(hashes)):
            if j in used:
                continue
            h2 = hashes[j][1]
            if h2 == -1:
                continue
            if _hamming(h, h2) <= DHASH_NEAR_DUP_THRESHOLD:
                group.append(hashes[j][0])
                used.add(j)
        keep.append(f)
        if len(group) > 1:
            groups.append(group)
    return keep, groups


def _stratified_split(files: list[str], labels: list[str]) -> tuple[list, list, list]:
    """Deterministic stratified train/val/test split by label."""
    rng = np.random.RandomState(RANDOM_STATE)
    idx = np.arange(len(files))
    # group by label
    by_label: dict[str, list[int]] = {}
    for i, lab in zip(idx, labels):
        by_label.setdefault(lab, []).append(i)

    test_idx: list[int] = []
    val_idx: list[int] = []
    train_idx: list[int] = []

    for lab, ids in by_label.items():
        ids = list(ids)
        rng.shuffle(ids)
        n = len(ids)
        n_test = max(MIN_TEST_PER_CLASS, int(round(n * TEST_FRACTION)))
        n_test = min(n_test, n - 2 if n > 2 else n)
        n_test = max(0, n_test)
        test = ids[:n_test]
        rest = ids[n_test:]
        if rest:
            n_val = int(round(len(rest) * VAL_FRACTION))
            val = rest[:n_val]
            train = rest[n_val:]
        else:
            val, train = [], []
        test_idx.extend(test)
        val_idx.extend(val)
        train_idx.extend(train)

    return (
        [files[i] for i in train_idx],
        [files[i] for i in val_idx],
        [files[i] for i in test_idx],
    )


def _copy_files(files: list[str], dest_dir: str, letter: str) -> None:
    target = os.path.join(dest_dir, letter)
    os.makedirs(target, exist_ok=True)
    for src in files:
        shutil.copy2(src, os.path.join(target, os.path.basename(src)))


def main() -> None:
    if not os.path.isdir(WEBCAM_ROOT):
        raise SystemExit(f"Webcam ISL root not found: {WEBCAM_ROOT}")
    if not os.path.isdir(PUBLIC_ROOT):
        raise SystemExit(f"Public root not found: {PUBLIC_ROOT}")

    if os.path.isdir(OUT_ROOT):
        raise SystemExit(
            f"Output already exists: {OUT_ROOT}. Remove it manually if you want "
            f"to regenerate (no originals are touched)."
        )

    os.makedirs(OUT_ROOT, exist_ok=True)
    start = time.time()

    stats: dict[str, dict] = {l: {} for l in LETTERS}
    manifest_rows: list[dict] = []
    duplicate_report: dict = {
        "webcam_exact_duplicate_groups": {},
        "webcam_near_duplicate_groups": 0,
        "webcam_near_duplicate_files": 0,
        "cross_dataset_exact_duplicates": [],
    }

    # RealSign training MD5 set (for cross-dataset exact duplicate check).
    print("[PREP] Indexing RealSign training hashes...")
    realsign_train_md5: dict[str, str] = {}
    for letter in LETTERS:
        folder = os.path.join(PUBLIC_ROOT, "Training", letter)
        for name in _image_files(folder):
            p = os.path.join(folder, name)
            realsign_train_md5[_md5(p)] = p

    webcam_failed_holdout: list[str] = []

    for letter in LETTERS:
        webcam_folder = os.path.join(WEBCAM_ROOT, letter, "roi")
        webcam_files = [os.path.join(webcam_folder, n) for n in _image_files(webcam_folder)]

        # 1. Exact dedup within webcam.
        unique_md5, md5_groups = _dedup_by_md5(webcam_files)
        if md5_groups:
            duplicate_report["webcam_exact_duplicate_groups"][letter] = [
                [os.path.basename(p) for p in group] for group in md5_groups.values()
            ]

        # 2. Near-duplicate dedup within webcam.
        unique_dhash, near_groups = _dedup_by_dhash(unique_md5)
        duplicate_report["webcam_near_duplicate_groups"] += len(near_groups)
        duplicate_report["webcam_near_duplicate_files"] += sum(
            len(g) - 1 for g in near_groups
        )

        # 3. Cross-dataset exact-duplicate check (webcam vs RealSign train).
        for f in unique_dhash:
            if _md5(f) in realsign_train_md5:
                duplicate_report["cross_dataset_exact_duplicates"].append(
                    os.path.relpath(f, PROJECT_ROOT)
                )

        # 4. Stratified split of the dedup'd webcam pool.
        labels = [letter] * len(unique_dhash)
        train_w, val_w, test_w = _stratified_split(unique_dhash, labels)

        if len(test_w) < MIN_TEST_PER_CLASS:
            webcam_failed_holdout.append(letter)

        stats[letter]["webcam_total"] = len(webcam_files)
        stats[letter]["webcam_after_dedup"] = len(unique_dhash)
        stats[letter]["webcam_train"] = len(train_w)
        stats[letter]["webcam_val"] = len(val_w)
        stats[letter]["webcam_test"] = len(test_w)

        # 5. RealSign sources.
        realsign_train = [
            os.path.join(PUBLIC_ROOT, "Training", letter, n)
            for n in _image_files(os.path.join(PUBLIC_ROOT, "Training", letter))
        ]
        realsign_val = [
            os.path.join(PUBLIC_ROOT, "Validation", letter, n)
            for n in _image_files(os.path.join(PUBLIC_ROOT, "Validation", letter))
        ]
        stats[letter]["realsign_train"] = len(realsign_train)
        stats[letter]["realsign_val"] = len(realsign_val)

        # 6. Copy into combined structure.
        _copy_files(realsign_train, os.path.join(OUT_ROOT, "train"), letter)
        _copy_files(train_w, os.path.join(OUT_ROOT, "train"), letter)

        _copy_files(realsign_val, os.path.join(OUT_ROOT, "validation"), letter)
        _copy_files(val_w, os.path.join(OUT_ROOT, "validation"), letter)

        _copy_files(test_w, os.path.join(OUT_ROOT, "test_webcam"), letter)

        # 7. Manifest rows.
        for src in realsign_train:
            manifest_rows.append(
                {"letter": letter, "split": "train", "source": "realsign",
                 "source_path": os.path.relpath(src, PROJECT_ROOT)}
            )
        for src in train_w:
            manifest_rows.append(
                {"letter": letter, "split": "train", "source": "webcam",
                 "source_path": os.path.relpath(src, PROJECT_ROOT)}
            )
        for src in realsign_val:
            manifest_rows.append(
                {"letter": letter, "split": "validation", "source": "realsign",
                 "source_path": os.path.relpath(src, PROJECT_ROOT)}
            )
        for src in val_w:
            manifest_rows.append(
                {"letter": letter, "split": "validation", "source": "webcam",
                 "source_path": os.path.relpath(src, PROJECT_ROOT)}
            )
        for src in test_w:
            manifest_rows.append(
                {"letter": letter, "split": "test_webcam", "source": "webcam",
                 "source_path": os.path.relpath(src, PROJECT_ROOT)}
            )

        print(
            f"  {letter}: realsign(train={len(realsign_train)},val={len(realsign_val)}) "
            f"webcam(train={len(train_w)},val={len(val_w)},test={len(test_w)})"
        )

    # Manifest.
    manifest_path = os.path.join(OUT_ROOT, "manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["letter", "split", "source", "source_path"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    # Stats JSON + CSV.
    summary = {
        "webcam_classes_with_insufficient_holdout": webcam_failed_holdout,
        "per_letter": stats,
    }
    with open(os.path.join(OUT_ROOT, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    with open(os.path.join(OUT_ROOT, "stats.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "letter", "realsign_train", "realsign_val", "webcam_train",
            "webcam_val", "webcam_test", "combined_train",
        ])
        for letter in LETTERS:
            s = stats[letter]
            w.writerow([
                letter, s["realsign_train"], s["realsign_val"], s["webcam_train"],
                s["webcam_val"], s["webcam_test"],
                s["realsign_train"] + s["webcam_train"],
            ])

    # Duplicate report.
    with open(os.path.join(OUT_ROOT, "duplicates.json"), "w", encoding="utf-8") as fh:
        json.dump(duplicate_report, fh, indent=2)

    print(f"\n[PREP] Done in {time.time() - start:.1f}s")
    print(f"  combined root      : {OUT_ROOT}")
    print(f"  manifest           : {manifest_path}")
    if webcam_failed_holdout:
        print(f"  WARNING: insufficient webcam holdout for: {''.join(webcam_failed_holdout)}")
    else:
        print("  All letters have >= %d unseen webcam test images." % MIN_TEST_PER_CLASS)


if __name__ == "__main__":
    main()
