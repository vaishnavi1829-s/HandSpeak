"""Prepare the BALANCED combined ISL dataset (read-only w.r.t. originals).

Creates ``dataset/isl_balanced_combined/`` with train/validation/test_webcam
(A-Z).  The key difference from the previous experiment is BALANCE:

    * For every letter, the SAME number of RealSign and webcam training images
      is selected (N + N per class).
    * RealSign never dominates webcam.

Webcam selection
----------------
* Reuses the previous experiment's leakage-free ``test_webcam`` set (312 images,
  never used in training/validation/fitting).
* Remaining webcam images are near-duplicate aware (dHash Hamming <= 2).
* Diversity selection: greedily pick webcam images by maximizing a combined
  landmark-geometry + pixel-hash spread (so sequential near-identical frames
  are not all chosen).

Balance
-------
The largest common balanced number is computed as:

    N = min over letters of (dedup webcam pool excluding test, RealSign pool)

subject to reserving a small validation set.  The final N is reported.

Outputs are entirely new files under ``dataset/isl_balanced_combined/``.  No
original dataset is modified.

Run from the project root::

    python scripts/prepare_isl_balanced_dataset.py
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
PREV_COMBINED_ROOT = os.path.join(PROJECT_ROOT, "dataset", "isl_combined")
OUT_ROOT = os.path.join(PROJECT_ROOT, "dataset", "isl_balanced_combined")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DHASH_THRESHOLD = 2
VAL_PER_CLASS = 6
RANDOM_STATE = 42


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


def _load_prev_test_set() -> dict[str, set[str]]:
    """Return {letter: {absolute path}} of the previous test_webcam set."""
    manifest = os.path.join(PREV_COMBINED_ROOT, "manifest.csv")
    out: dict[str, set[str]] = {l: set() for l in LETTERS}
    if not os.path.exists(manifest):
        return out
    with open(manifest, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["split"] == "test_webcam" and row["source"] == "webcam":
                letter = row["letter"]
                out[letter].add(os.path.join(PROJECT_ROOT, row["source_path"]))
    return out


def _near_dup_dedup(files: list[str]) -> tuple[list[str], int]:
    """Greedy dHash dedup; returns (kept files, removed count)."""
    hashes = [(f, _dhash64(f)) for f in files]
    keep: list[str] = []
    removed = 0
    used: set[int] = set()
    for i, (f, h) in enumerate(hashes):
        if i in used:
            continue
        if h == -1:
            keep.append(f)
            continue
        keep.append(f)
        for j in range(i + 1, len(hashes)):
            if j in used:
                continue
            hj = hashes[j][1]
            if hj != -1 and _hamming(h, hj) <= DHASH_THRESHOLD:
                used.add(j)
                removed += 1
    return keep, removed


def _diversity_pick(files: list[str], k: int) -> list[str]:
    """Pick k files greedily to maximize landmark-geometry spread.

    Uses the rich landmark extractor for a 5-finger extension descriptor; falls
    back to a pixel hash when landmarks are unavailable.
    """
    if k <= 0:
        return []
    if len(files) <= k:
        return list(files)

    from recognition.landmark_extractor import (  # noqa: E402
        WRIST_IDX,
        extract_two_hands_from_file,
        get_two_hand_landmarker,
    )
    landmarker = get_two_hand_landmarker()

    descriptors: dict[str, np.ndarray] = {}
    FINGERS = {
        "thumb": [1, 2, 3, 4],
        "index": [5, 6, 7, 8],
        "middle": [9, 10, 11, 12],
        "ring": [13, 14, 15, 16],
        "pinky": [17, 18, 19, 20],
    }
    for f in files:
        result = extract_two_hands_from_file(f, landmarker=landmarker)
        vecs = []
        for raw in (result.left_landmarks if result else None,
                    result.right_landmarks if result else None):
            if raw is None:
                continue
            norm = raw - raw[WRIST_IDX]
            ref = np.linalg.norm(norm[9])
            if ref < 1e-6:
                ref = 1e-6
            norm = norm / ref
            ext = []
            for idx in FINGERS.values():
                pts = norm[idx]
                path = sum(np.linalg.norm(pts[i + 1] - pts[i]) for i in range(3))
                ext.append(np.linalg.norm(pts[-1] - pts[0]) / (path + 1e-6))
            vecs.append(np.asarray(ext, dtype=np.float32))
        if vecs:
            descriptors[f] = np.mean(vecs, axis=0)
        else:
            # fallback: pixel-based scalar signature
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            descriptors[f] = np.asarray([float(img.mean()) if img is not None else 0.0])

    chosen = [files[0]]
    remaining = set(files[1:])
    while len(chosen) < k and remaining:
        best = None
        best_score = -1.0
        for cand in remaining:
            d = descriptors.get(cand)
            if d is None:
                score = 0.0
            else:
                dists = [np.linalg.norm(d - descriptors.get(c, d)) for c in chosen]
                score = float(np.mean(dists))
            if score > best_score:
                best_score = score
                best = cand
        if best is None:
            break
        chosen.append(best)
        remaining.discard(best)
    return chosen


def main() -> None:
    if os.path.isdir(OUT_ROOT):
        raise SystemExit(
            f"Output already exists: {OUT_ROOT}. Remove it manually to regenerate."
        )
    os.makedirs(OUT_ROOT, exist_ok=True)

    prev_test = _load_prev_test_set()
    print(f"[BALANCED] Reusing previous test_webcam set: "
          f"{sum(len(v) for v in prev_test.values())} images")

    start = time.time()
    stats: dict[str, dict] = {}

    # Pass 1: compute dedup'd webcam pools excluding the test set.
    pools: dict[str, list[str]] = {}
    for letter in LETTERS:
        folder = os.path.join(WEBCAM_ROOT, letter, "roi")
        all_files = [os.path.join(folder, n) for n in _image_files(folder)]
        test = prev_test[letter]
        train_val = [f for f in all_files if f not in test]
        deduped, removed = _near_dup_dedup(train_val)
        pools[letter] = deduped
        stats[letter] = {
            "webcam_total": len(all_files),
            "webcam_test": len(test),
            "webcam_dedup_trainval": len(deduped),
            "webcam_dup_removed": removed,
        }

    # Largest common balanced number.  Reserve VAL_PER_CLASS for validation.
    min_webcam = min(len(pools[l]) for l in LETTERS)
    N_webcam_train = max(1, min_webcam - VAL_PER_CLASS)
    N = min(40, N_webcam_train)  # cap at 40 for a manageable experiment

    print(f"[BALANCED] min dedup'd webcam trainval pool = {min_webcam}")
    print(f"[BALANCED] balanced N (webcam train per class) = {N}")
    print(f"[BALANCED] validation per class = {VAL_PER_CLASS}")

    manifest_rows: list[dict] = []

    for letter in LETTERS:
        webcam_pool = pools[letter]
        # Diversity-select train + val.
        selected = _diversity_pick(webcam_pool, N + VAL_PER_CLASS)
        webcam_train = selected[:N]
        webcam_val = selected[N:N + VAL_PER_CLASS]

        # RealSign: pick N training + VAL_PER_CLASS validation.
        rs_train_folder = os.path.join(PUBLIC_ROOT, "Training", letter)
        rs_val_folder = os.path.join(PUBLIC_ROOT, "Validation", letter)
        rs_train_all = [os.path.join(rs_train_folder, n) for n in _image_files(rs_train_folder)]
        rs_val_all = [os.path.join(rs_val_folder, n) for n in _image_files(rs_val_folder)]
        rng = np.random.RandomState(RANDOM_STATE)
        rs_train = list(rng.choice(rs_train_all, size=N, replace=False))
        rs_val = list(rng.choice(rs_val_all, size=min(VAL_PER_CLASS, len(rs_val_all)), replace=False))

        stats[letter].update(
            {
                "realsign_train": len(rs_train),
                "realsign_val": len(rs_val),
                "webcam_train": len(webcam_train),
                "webcam_val": len(webcam_val),
            }
        )

        # Copy.
        for src, split in (
            (rs_train, "train"), (webcam_train, "train"),
            (rs_val, "validation"), (webcam_val, "validation"),
        ):
            dest = os.path.join(OUT_ROOT, split, letter)
            os.makedirs(dest, exist_ok=True)
            for s in src:
                shutil.copy2(s, os.path.join(dest, os.path.basename(s)))
                manifest_rows.append(
                    {"letter": letter, "split": split,
                     "source": "realsign" if "isl_public" in s else "webcam",
                     "source_path": os.path.relpath(s, PROJECT_ROOT)}
                )

        # Copy test_webcam.
        test_dest = os.path.join(OUT_ROOT, "test_webcam", letter)
        os.makedirs(test_dest, exist_ok=True)
        for s in sorted(prev_test[letter]):
            shutil.copy2(s, os.path.join(test_dest, os.path.basename(s)))
            manifest_rows.append(
                {"letter": letter, "split": "test_webcam", "source": "webcam",
                 "source_path": os.path.relpath(s, PROJECT_ROOT)}
            )

        print(f"  {letter}: webcam(train={len(webcam_train)},val={len(webcam_val)},"
              f"test={len(prev_test[letter])})  realsign(train={len(rs_train)},"
              f"val={len(rs_val)})")

    # Manifest + stats.
    with open(os.path.join(OUT_ROOT, "manifest.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["letter", "split", "source", "source_path"])
        w.writeheader()
        w.writerows(manifest_rows)

    with open(os.path.join(OUT_ROOT, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump({"balanced_N": N, "val_per_class": VAL_PER_CLASS, "per_letter": stats}, fh, indent=2)

    print(f"\n[BALANCED] Done in {time.time() - start:.1f}s -> {OUT_ROOT}")
    print(f"[BALANCED] balanced N = {N} RealSign + {N} webcam per class (train)")


if __name__ == "__main__":
    main()
