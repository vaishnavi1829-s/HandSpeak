"""Build a visual comparison gallery for the 10 letters flagged for manual review.

For each letter (C, D, G, J, K, N, Q, S, X, Y) this script creates ONE gallery
image under ``results/realsign/comparison/letter_<L>.png`` containing:

    * Row 1: 5 representative images from ``dataset/isl_webcam/<L>/roi``
    * Row 2: 5 representative images from ``dataset/isl_public/Training/<L>``

Images are preferred when MediaPipe detects at least one hand (so the sign is
visible).  The number of detected hands is annotated under each thumbnail.

This script is READ-ONLY: it never modifies either dataset.

Run from the project root::

    python scripts/build_realsign_comparison_gallery.py
"""

from __future__ import annotations

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

WEBCAM_ROOT = os.path.join(PROJECT_ROOT, "dataset", "isl_webcam")
PUBLIC_TRAIN_ROOT = os.path.join(PROJECT_ROOT, "dataset", "isl_public", "Training")
OUT_DIR = os.path.join(PROJECT_ROOT, "results", "realsign", "comparison")

TARGET_LETTERS = ["C", "D", "G", "J", "K", "N", "Q", "S", "X", "Y"]
EXAMPLES_PER_SIDE = 5
MAX_CANDIDATES = 30

THUMB_W, THUMB_H = 220, 220
LABEL_H = 34
TITLE_H = 56
PAD = 10


def _image_files(folder: str, limit: int) -> list[str]:
    if not os.path.isdir(folder):
        return []
    files = sorted(
        n for n in os.listdir(folder)
        if n.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    )
    return files[:limit]


def _detect_hands(landmarker, path: str) -> int:
    """Return detected hand count (-1 if image unreadable)."""
    result = extract_two_hands_from_file(path, landmarker=landmarker)
    if result is None:
        return 0
    return int(result.left_present) + int(result.right_present)


def _pick_with_detection(landmarker, folder: str) -> list[tuple[str, int]]:
    """Return up to EXAMPLES_PER_SIDE images that have >=1 hand, then fill."""
    files = _image_files(folder, MAX_CANDIDATES)
    good: list[tuple[str, int]] = []
    fallback: list[tuple[str, int]] = []

    for name in files:
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        hands = _detect_hands(landmarker, path)
        if hands >= 1:
            good.append((path, hands))
        else:
            fallback.append((path, hands))

    picked = good[:EXAMPLES_PER_SIDE]
    if len(picked) < EXAMPLES_PER_SIDE:
        picked += fallback[: EXAMPLES_PER_SIDE - len(picked)]
    return picked


def _thumb(path: str, hands: int, title: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        img = np.full((THUMB_H, THUMB_W, 3), 30, dtype=np.uint8)
    else:
        img = cv2.resize(img, (THUMB_W, THUMB_H))
    cell = np.full((THUMB_H + LABEL_H, THUMB_W, 3), 20, dtype=np.uint8)
    cell[:THUMB_H, :] = img
    cv2.putText(cell, title, (5, THUMB_H + 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return cell


def _build_gallery(letter: str, webcam_rows, public_rows) -> np.ndarray:
    n = EXAMPLES_PER_SIDE
    canvas = np.full(
        (TITLE_H + 2 * (THUMB_H + LABEL_H) + PAD, n * THUMB_W + (n + 1) * PAD, 3),
        40,
        dtype=np.uint8,
    )
    cv2.putText(canvas, f"LETTER {letter}", (PAD, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2, cv2.LINE_AA)

    y = TITLE_H
    cv2.putText(canvas, "My webcam ISL (dataset/isl_webcam)", (PAD, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
    for col, (path, hands) in enumerate(webcam_rows):
        x = PAD + col * (THUMB_W + PAD)
        cell = _thumb(path, hands, f"hands={hands}")
        canvas[y:y + THUMB_H + LABEL_H, x:x + THUMB_W] = cell

    y = TITLE_H + (THUMB_H + LABEL_H) + PAD
    cv2.putText(canvas, "RealSign public (dataset/isl_public/Training)", (PAD, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
    for col, (path, hands) in enumerate(public_rows):
        x = PAD + col * (THUMB_W + PAD)
        cell = _thumb(path, hands, f"hands={hands}")
        canvas[y:y + THUMB_H + LABEL_H, x:x + THUMB_W] = cell

    return canvas


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    landmarker = get_two_hand_landmarker()

    start = time.time()
    created: list[str] = []

    for letter in TARGET_LETTERS:
        webcam_folder = os.path.join(WEBCAM_ROOT, letter, "roi")
        public_folder = os.path.join(PUBLIC_TRAIN_ROOT, letter)

        webcam_rows = _pick_with_detection(landmarker, webcam_folder)
        public_rows = _pick_with_detection(landmarker, public_folder)

        if not webcam_rows or not public_rows:
            print(f"[SKIP] {letter}: missing source images "
                  f"(webcam={len(webcam_rows)}, public={len(public_rows)})")
            continue

        gallery = _build_gallery(letter, webcam_rows, public_rows)
        out_path = os.path.join(OUT_DIR, f"letter_{letter}.png")
        cv2.imwrite(out_path, gallery)
        created.append(out_path)
        print(f"[OK] {letter}: webcam={len(webcam_rows)} public={len(public_rows)} "
              f"-> {out_path}")

    print(f"\nCreated {len(created)} gallery image(s) in {OUT_DIR}")
    print(f"Elapsed {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
