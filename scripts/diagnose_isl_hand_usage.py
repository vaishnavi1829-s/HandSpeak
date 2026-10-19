"""Diagnose ISL hand usage (1-hand vs 2-hand) WITHOUT retraining anything.

This is a READ-ONLY diagnostic.  It does NOT modify datasets, models, or the
pipeline.  It produces:

    1. Annotated diagnostic images (landmarks + connections + handedness)
       under ``results/landmark/isl_dataset_diagnostics/<LETTER>/``.
    2. An independent "skin-blob" estimate of visible hands (YCrCb skin
       segmentation + connected-component analysis) to compare against
       MediaPipe's detected hand count.  This helps distinguish:
           * a genuinely one-hand sign
           * a two-hand sign where MediaPipe failed to detect the 2nd hand.
    3. ``results/landmark/isl_hand_usage_analysis.txt`` (per-letter table +
       ONE_HAND / TWO_HAND / NEEDS_REVIEW classification).
    4. ``results/landmark/isl_dataset_quality_summary.txt``.

Run from the project root::

    python scripts/diagnose_isl_hand_usage.py
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

from mediapipe.tasks.python import vision  # noqa: E402
from recognition.landmark_extractor import get_two_hand_landmarker  # noqa: E402
from utils.label_mapping import LETTERS  # noqa: E402

ISL_ROOT = os.path.join(PROJECT_ROOT, "dataset", "isl_webcam")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "landmark")
DIAG_DIR = os.path.join(RESULTS_DIR, "isl_dataset_diagnostics")
ANALYSIS_PATH = os.path.join(RESULTS_DIR, "isl_hand_usage_analysis.txt")
QUALITY_PATH = os.path.join(RESULTS_DIR, "isl_dataset_quality_summary.txt")

# Handedness colors (BGR): Left = magenta, Right = yellow.
COLOR_LEFT = (255, 0, 255)
COLOR_RIGHT = (0, 255, 255)
COLOR_TEXT = (255, 255, 255)

# Skin-segmentation thresholds in YCrCb.
CR_MIN, CR_MAX = 133, 173
CB_MIN, CB_MAX = 77, 127
MIN_SKIN_AREA = 2000          # minimum connected-component area (px) for a "hand".
DIAG_SAMPLES_PER_LETTER = 4   # number of annotated images saved per letter.


def _image_files(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    return sorted(
        name
        for name in os.listdir(folder)
        if name.lower().endswith((".png", ".jpg", ".jpeg"))
    )


def _skin_blob_count(bgr: np.ndarray) -> int:
    """Estimate the number of visible skin regions using YCrCb segmentation.

    This is an independent (non-MediaPipe) heuristic.  It is a rough proxy for
    "visible hands" and is used ONLY to flag possible MediaPipe detection
    failures; it is not a replacement for human review.
    """
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(
        ycrcb,
        np.array([0, CR_MIN, CB_MIN], dtype=np.uint8),
        np.array([255, CR_MAX, CB_MAX], dtype=np.uint8),
    )
    # Morphological cleanup.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = [c for c in contours if cv2.contourArea(c) >= MIN_SKIN_AREA]
    return len(big)


def _detect_hands(bgr: np.ndarray, landmarker):
    """Return (n_hands, handedness_list, hand_landmarks_list)."""
    from recognition.landmark_extractor import _mp_image_from_bgr, _landmarks_to_matrix

    mp_image = _mp_image_from_bgr(bgr)
    result = landmarker.detect(mp_image)

    handedness = []
    hand_landmarks = []
    if not result.hand_landmarks:
        return 0, handedness, hand_landmarks

    for idx in range(min(len(result.hand_landmarks), 2)):
        label = "Unknown"
        if result.handedness and idx < len(result.handedness):
            group = result.handedness[idx]
            if group:
                label = group[0].display_name
        handedness.append(label)
        hand_landmarks.append(_landmarks_to_matrix(result.hand_landmarks[idx]))

    return len(handedness), handedness, hand_landmarks


def _draw_hand(frame, landmarks, color):
    height, width = frame.shape[:2]
    pts = []
    for lm in landmarks:
        px = int(lm[0] * width)
        py = int(lm[1] * height)
        pts.append((px, py))
        cv2.circle(frame, (px, py), 3, color, -1)
    for conn in vision.HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(frame, pts[conn.start], pts[conn.end], color, 2)


def main() -> None:
    if not os.path.isdir(ISL_ROOT):
        raise SystemExit(f"ISL dataset root not found: {ISL_ROOT}")

    os.makedirs(DIAG_DIR, exist_ok=True)
    landmarker = get_two_hand_landmarker()

    print(f"Diagnosing ISL hand usage: {ISL_ROOT}\n")

    per_letter = []
    start = time.time()

    for letter in LETTERS:
        folder = os.path.join(ISL_ROOT, letter, "roi")
        files = _image_files(folder)
        letter_diag_dir = os.path.join(DIAG_DIR, letter)
        os.makedirs(letter_diag_dir, exist_ok=True)

        total = len(files)
        mp_0 = mp_1 = mp_2 = 0
        skin_0 = skin_1 = skin_2 = 0
        saved = 0
        detection_notes = []

        for name in files:
            path = os.path.join(folder, name)
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if bgr is None:
                continue

            n_hands, handedness, hand_landmarks = _detect_hands(bgr, landmarker)
            skin = _skin_blob_count(bgr)

            # MediaPipe counts.
            if n_hands == 0:
                mp_0 += 1
            elif n_hands == 1:
                mp_1 += 1
            else:
                mp_2 += 1

            # Skin-blob counts.
            if skin == 0:
                skin_0 += 1
            elif skin == 1:
                skin_1 += 1
            else:
                skin_2 += 1

            # Save a few annotated images per letter.
            if saved < DIAG_SAMPLES_PER_LETTER:
                annotated = bgr.copy()
                for lm, hand in zip(hand_landmarks, handedness):
                    color = COLOR_LEFT if hand == "Left" else COLOR_RIGHT
                    _draw_hand(annotated, lm, color)
                cv2.putText(
                    annotated, f"MP hands={n_hands}  skin_blobs={skin}",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA,
                )
                cv2.putText(
                    annotated, f"handedness={handedness}",
                    (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA,
                )
                out_name = f"{letter}_{name.split('_')[-1]}"
                cv2.imwrite(os.path.join(letter_diag_dir, out_name), annotated)
                saved += 1

        # Classify based on skin-blob proxy (independent of MediaPipe).
        skin_two_ratio = skin_2 / total if total else 0.0
        mp_two_ratio = mp_2 / total if total else 0.0

        if skin_two_ratio >= 0.4 and mp_two_ratio >= 0.3:
            usage = "TWO_HAND"
        elif skin_two_ratio < 0.15 and mp_two_ratio < 0.15:
            usage = "ONE_HAND"
        else:
            usage = "NEEDS_REVIEW"

        per_letter.append(
            {
                "letter": letter,
                "total": total,
                "mp_0": mp_0,
                "mp_1": mp_1,
                "mp_2": mp_2,
                "skin_0": skin_0,
                "skin_1": skin_1,
                "skin_2": skin_2,
                "skin_two_ratio": skin_two_ratio,
                "mp_two_ratio": mp_two_ratio,
                "usage": usage,
            }
        )

        print(
            f"  {letter}: total={total}  MP(0/1/2)={mp_0}/{mp_1}/{mp_2}  "
            f"skin(0/1/2)={skin_0}/{skin_1}/{skin_2}  -> {usage}"
        )

    elapsed = time.time() - start

    # ------------------------------------------------------------------
    # Hand-usage analysis table.
    # ------------------------------------------------------------------
    lines = []
    lines.append("ISL Hand-Usage Analysis")
    lines.append("=" * 90)
    lines.append(f"Dataset root      : {ISL_ROOT}")
    lines.append(f"Elapsed seconds   : {elapsed:.1f}")
    lines.append("")
    lines.append(
        f"{'Letter':<7}{'Total':>7}{'MP 0':>6}{'MP 1':>6}{'MP 2':>6}"
        f"{'Skin 0':>8}{'Skin 1':>8}{'Skin 2':>8}{'Class':>13}"
    )
    lines.append("-" * 90)
    for r in per_letter:
        lines.append(
            f"{r['letter']:<7}{r['total']:>7}{r['mp_0']:>6}{r['mp_1']:>6}"
            f"{r['mp_2']:>6}{r['skin_0']:>8}{r['skin_1']:>8}"
            f"{r['skin_2']:>8}{r['usage']:>13}"
        )
    lines.append("")
    lines.append("Legend:")
    lines.append("  MP  = MediaPipe detected hand count")
    lines.append("  Skin = independent YCrCb skin-segmentation blob count")
    lines.append("  Class = ONE_HAND / TWO_HAND / NEEDS_REVIEW")
    lines.append("")
    lines.append("Detection notes:")
    for r in per_letter:
        if r["usage"] == "NEEDS_REVIEW":
            lines.append(
                f"  {r['letter']}: MP 2-hand={r['mp_2']} vs Skin 2-hand={r['skin_2']} "
                f"(possible MediaPipe detection gap; review diagnostics/{r['letter']}/)"
            )

    analysis = "\n".join(lines) + "\n"
    with open(ANALYSIS_PATH, "w", encoding="utf-8") as fh:
        fh.write(analysis)

    # ------------------------------------------------------------------
    # Quality summary.
    # ------------------------------------------------------------------
    total_all = sum(r["total"] for r in per_letter)
    total_mp2 = sum(r["mp_2"] for r in per_letter)
    total_mp1 = sum(r["mp_1"] for r in per_letter)
    total_mp0 = sum(r["mp_0"] for r in per_letter)
    total_skin2 = sum(r["skin_2"] for r in per_letter)
    total_skin1 = sum(r["skin_1"] for r in per_letter)
    total_skin0 = sum(r["skin_0"] for r in per_letter)

    qlines = []
    qlines.append("ISL Dataset Quality Summary (hand usage)")
    qlines.append("=" * 90)
    qlines.append(f"Total ISL images                     : {total_all}")
    qlines.append("")
    qlines.append("MediaPipe detection:")
    qlines.append(f"  2 hands detected                   : {total_mp2} "
                 f"({100.0 * total_mp2 / total_all:.2f}%)")
    qlines.append(f"  1 hand detected                    : {total_mp1} "
                 f"({100.0 * total_mp1 / total_all:.2f}%)")
    qlines.append(f"  0 hands detected                   : {total_mp0} "
                 f"({100.0 * total_mp0 / total_all:.2f}%)")
    qlines.append("")
    qlines.append("Independent skin-segmentation estimate (proxy for visible hands):")
    qlines.append(f"  2 skin regions                     : {total_skin2} "
                 f"({100.0 * total_skin2 / total_all:.2f}%)")
    qlines.append(f"  1 skin region                      : {total_skin1} "
                 f"({100.0 * total_skin1 / total_all:.2f}%)")
    qlines.append(f"  0 skin regions                     : {total_skin0} "
                 f"({100.0 * total_skin0 / total_all:.2f}%)")
    qlines.append("")
    qlines.append("Per-letter classification:")
    one_hand = [r["letter"] for r in per_letter if r["usage"] == "ONE_HAND"]
    two_hand = [r["letter"] for r in per_letter if r["usage"] == "TWO_HAND"]
    review = [r["letter"] for r in per_letter if r["usage"] == "NEEDS_REVIEW"]
    qlines.append(f"  ONE_HAND      : {''.join(one_hand)}")
    qlines.append(f"  TWO_HAND      : {''.join(two_hand)}")
    qlines.append(f"  NEEDS_REVIEW  : {''.join(review)}")
    qlines.append("")
    qlines.append("MediaPipe detection appears problematic for (NEEDS_REVIEW):")
    qlines.append(f"  {''.join(review)}")
    qlines.append("")

    quality = "\n".join(qlines) + "\n"
    with open(QUALITY_PATH, "w", encoding="utf-8") as fh:
        fh.write(quality)

    print(f"\nDiagnostic complete in {elapsed:.1f}s")
    print(f"Analysis : {ANALYSIS_PATH}")
    print(f"Quality  : {QUALITY_PATH}")
    print(f"Diagnostics images: {DIAG_DIR}")


if __name__ == "__main__":
    main()
