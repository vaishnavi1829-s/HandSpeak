"""Diversity-aware ISL webcam dataset collection (dataset/isl_webcam_v2/).

This is a NEW, separate collection pipeline.  It does NOT write to
``dataset/isl_webcam/`` and does NOT touch any existing dataset or model.

Unlike the original collector (which saved every manual capture), this script
only saves a frame when it is **sufficiently different** from recently saved
samples.  When MediaPipe detects a hand, diversity is measured on normalized
landmarks (wrist/fingertip geometry); otherwise it falls back to perceptual
(dHash) similarity.

Saved layout::

    dataset/isl_webcam_v2/
        <LETTER>/roi/<LETTER>_<session>_<index>.png
        manifest.csv

Manifest columns::

    filename,letter,session,timestamp,hand_count,diversity_score

Run from the project root::

    python scripts/collect_isl_webcam_v2.py --letter A --target 150
    python scripts/collect_isl_webcam_v2.py --list
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from recognition.landmark_extractor import (  # noqa: E402
    WRIST_IDX,
    extract_two_hands_from_bgr,
    get_two_hand_landmarker,
)
from utils.label_mapping import LETTERS  # noqa: E402

V2_ROOT = os.path.join(PROJECT_ROOT, "dataset", "isl_webcam_v2")
MANIFEST_PATH = os.path.join(V2_ROOT, "manifest.csv")
MANIFEST_HEADER = [
    "filename", "letter", "session", "timestamp", "hand_count", "diversity_score",
]

DEFAULT_TARGET = 150
ROI_SIZE = 300
CAPTURE_DELAY = 0.5
WINDOW_NAME = "HandSpeak - ISL v2 Diversity Collection (Q=quit)"
ZERO_PAD = 4

# Diversity thresholds.
LANDMARK_MIN_DIST = 0.25   # minimum mean landmark distance to accept a frame
DHASH_MIN_HAMMING = 6      # minimum dHash Hamming distance (fallback)
RECENT_WINDOW = 8          # compare against the last N saved samples

VARIATION_HINTS = [
    "Move hand left/right",
    "Change distance from camera",
    "Change lighting angle",
    "Slight hand rotation",
    "Adjust finger spread",
    "Change background",
    "Vary wrist tilt",
    "Change hand size in frame",
]


def _roi_dir(letter: str) -> str:
    return os.path.join(V2_ROOT, letter, "roi")


def _ensure_dirs(letter: str) -> None:
    os.makedirs(_roi_dir(letter), exist_ok=True)


def _load_manifest() -> list[dict]:
    if not os.path.exists(MANIFEST_PATH):
        return []
    with open(MANIFEST_PATH, "r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_manifest(rows: list[dict]) -> None:
    os.makedirs(V2_ROOT, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r.get("letter", ""), r.get("filename", "")))
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_HEADER)
        w.writeheader()
        w.writerows(rows)


def _count_saved(letter: str) -> int:
    d = _roi_dir(letter)
    if not os.path.isdir(d):
        return 0
    return len([n for n in os.listdir(d) if n.lower().endswith((".png", ".jpg"))])


def _dhash64(img_gray: np.ndarray) -> int:
    img = cv2.resize(img_gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = 0
    for r in range(8):
        for c in range(8):
            bits <<= 1
            if img[r, c] > img[r, c + 1]:
                bits |= 1
    return bits


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _landmark_descriptor(result) -> np.ndarray | None:
    """Return a scale-normalized geometric descriptor of the detected hand(s)."""
    if result is None:
        return None
    vecs = []
    for raw in (result.left_landmarks, result.right_landmarks):
        if raw is None:
            continue
        norm = raw - raw[WRIST_IDX]
        ref = np.linalg.norm(norm[9])
        if ref < 1e-6:
            ref = 1e-6
        norm = norm / ref
        tips = norm[[4, 8, 12, 16, 20]]
        vecs.append(tips.flatten())
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def _diversity_score(
    descriptor: np.ndarray | None,
    dhash: int,
    recent_descriptors: list[np.ndarray],
    recent_dhashes: list[int],
) -> tuple[float, str]:
    """Return (score, method). Score >= 0; higher = more diverse."""
    if descriptor is not None and recent_descriptors:
        dists = [float(np.linalg.norm(descriptor - d)) for d in recent_descriptors]
        return float(np.mean(dists)), "landmark"
    if recent_dhashes:
        hams = [_hamming(dhash, d) for d in recent_dhashes]
        return float(np.mean(hams)) / 64.0, "dhash"
    return 1.0, "none"


def _is_diverse(
    descriptor: np.ndarray | None,
    dhash: int,
    recent_descriptors: list[np.ndarray],
    recent_dhashes: list[int],
) -> tuple[bool, float, str]:
    score, method = _diversity_score(descriptor, dhash, recent_descriptors, recent_dhashes)
    if method == "landmark":
        return score >= LANDMARK_MIN_DIST, score, method
    if method == "dhash":
        # score is mean hamming/64, so DHASH_MIN_HAMMING maps to a score.
        return score * 64.0 >= DHASH_MIN_HAMMING, score, method
    return True, score, method


def _run_session(
    letter: str, target: int, camera_index: int, session: str, manifest_rows: list[dict]
) -> None:
    _ensure_dirs(letter)
    landmarker = get_two_hand_landmarker()

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError("Could not open webcam.")

    saved = _count_saved(letter)
    recent_descriptors: list[np.ndarray] = []
    recent_dhashes: list[int] = []
    last_capture = 0.0
    hint_idx = 0
    flash = ""
    flash_until = 0.0
    skipped = 0

    print(f"\nCollecting ISL '{letter}' (target {target}, saved {saved}).")
    print("  SPACE=try capture  Q=quit  N/S=next letter  A-Z=jump")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                continue

            height, width = frame.shape[:2]
            roi_size = min(ROI_SIZE, width, height)
            x1 = (width - roi_size) // 2
            y1 = (height - roi_size) // 2
            roi = frame[y1:y1 + roi_size, x1:x1 + roi_size].copy()

            if time.time() >= flash_until:
                flash = ""

            # Overlay.
            cv2.rectangle(frame, (x1, y1), (x1 + roi_size, y1 + roi_size), (0, 255, 0), 2)
            lines = [
                f"Letter: {letter}",
                f"Saved: {saved}",
                f"Target: {target}",
                f"Skipped as duplicate: {skipped}",
                f"Hint: {VARIATION_HINTS[hint_idx % len(VARIATION_HINTS)]}",
                "SPACE=capture  N/S=next  A-Z=jump  Q=quit",
            ]
            for i, text in enumerate(lines):
                cv2.putText(frame, text, (10, 28 + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2, cv2.LINE_AA)
            if flash:
                cv2.putText(frame, flash, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key in (ord("n"), ord("s")):
                break
            if ord("a") <= key <= ord("z"):
                letter = chr(key).upper()
                _ensure_dirs(letter)
                saved = _count_saved(letter)
                recent_descriptors.clear()
                recent_dhashes.clear()
                skipped = 0
                continue

            if key not in (ord(" "), 13):
                continue

            now = time.time()
            if now - last_capture < CAPTURE_DELAY:
                continue
            last_capture = now

            # Detect hand(s) on the ROI.
            result = extract_two_hands_from_bgr(roi, landmarker=landmarker)
            hand_count = 0 if result is None else int(result.left_present) + int(result.right_present)

            descriptor = _landmark_descriptor(result)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            dhash = _dhash64(gray)

            diverse, score, method = _is_diverse(
                descriptor, dhash, recent_descriptors, recent_dhashes
            )

            if not diverse:
                skipped += 1
                flash = f"Duplicate (score {score:.2f}, {method}) - skipped"
                flash_until = time.time() + 0.8
                continue

            # Save.
            idx = saved + 1
            name = f"{letter}_{session}_{str(idx).zfill(ZERO_PAD)}.png"
            rel = os.path.join(letter, "roi", name)
            abs_path = os.path.join(V2_ROOT, rel)
            cv2.imwrite(abs_path, roi)

            manifest_rows.append({
                "filename": rel,
                "letter": letter,
                "session": session,
                "timestamp": f"{time.time():.3f}",
                "hand_count": str(hand_count),
                "diversity_score": f"{score:.4f}",
            })
            _write_manifest(manifest_rows)

            saved += 1
            if descriptor is not None:
                recent_descriptors.append(descriptor)
            recent_dhashes.append(dhash)
            if len(recent_descriptors) > RECENT_WINDOW:
                recent_descriptors.pop(0)
            if len(recent_dhashes) > RECENT_WINDOW:
                recent_dhashes.pop(0)
            hint_idx += 1

            flash = f"Saved (hands={hand_count}) {saved}/{target}"
            flash_until = time.time() + 0.8
            print(f"[{letter}] saved {saved}/{target} (hands={hand_count}, "
                  f"score={score:.3f}, method={method})")

            if saved >= target:
                print(f"[{letter}] target reached ({saved}).")
                break
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        capture.release()
        cv2.destroyAllWindows()

    print(f"[{letter}] session ended: saved={saved}/{target}, skipped={skipped}")


def _list_progress() -> None:
    counts = {l: _count_saved(l) for l in LETTERS}
    print("\nISL v2 progress:")
    print("  ".join(f"{l}:{counts[l]}" for l in LETTERS))
    print(f"Total: {sum(counts.values())}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--letter", default=None, help="Letter to collect first (A-Z).")
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET,
                        help=f"Target diverse samples per letter (default {DEFAULT_TARGET}).")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--session", default=None,
                        help="Session label (default auto = S1, S2, ...).")
    parser.add_argument("--list", action="store_true", help="Show progress and exit.")
    args = parser.parse_args()

    if args.list:
        _list_progress()
        return

    if args.letter:
        letter = args.letter.upper()
        if letter not in LETTERS:
            raise SystemExit(f"Invalid letter: {args.letter}")
    else:
        counts = {l: _count_saved(l) for l in LETTERS}
        letter = next((l for l in LETTERS if counts[l] < args.target), "A")
        print(f"Starting with first incomplete letter: {letter}")

    session = args.session
    if session is None:
        existing = _load_manifest()
        max_session = 0
        for row in existing:
            if row.get("letter") == letter and row.get("session", "").startswith("S"):
                try:
                    max_session = max(max_session, int(row["session"][1:]))
                except ValueError:
                    pass
        session = f"S{max_session + 1}"

    manifest_rows = _load_manifest()
    _run_session(letter, args.target, args.camera, session, manifest_rows)


if __name__ == "__main__":
    main()
