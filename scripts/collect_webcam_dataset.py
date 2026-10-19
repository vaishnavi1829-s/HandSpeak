"""Collect webcam-domain images for all 26 ASL letters (A-Z).

This script drives the laptop's built-in webcam and lets the user capture
real-world hand/sign samples one letter at a time. It is the **data collection
only** stage: it does NOT retrain, does NOT modify the existing model, and does
NOT touch the original datasets. Everything is written under ``dataset/webcam/``.

Design
------
* The existing ``recognition.preprocessing`` pipeline is reused verbatim so the
  saved 28x28 samples have exactly the format the CNN expects (grayscale,
  hand-dark / background-light, centered, padded, ``/255.0`` normalization).
* For every capture we save two files per letter::

      dataset/webcam/<LETTER>/processed/<LETTER>_NNNNNN.png   (28x28 CNN input)
      dataset/webcam/<LETTER>/roi/<LETTER>_NNNNNN.png         (original ROI)

  The original ROI lets you visually inspect the samples later.
* A single ``dataset/webcam/manifest.csv`` records every sample with its letter
  and the canonical A=0 ... Z=25 label index (from ``utils.label_mapping``).

Run from the project root::

    python scripts/collect_webcam_dataset.py            # interactive letter pick
    python scripts/collect_webcam_dataset.py --letter A # start at a letter
    python scripts/collect_webcam_dataset.py --list     # show progress only
    python scripts/collect_webcam_dataset.py --check-camera
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

# Allow this script to be run directly from the project root so that ``utils``,
# ``recognition``, and other packages resolve correctly.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from recognition.preprocessing import preprocess_frame  # noqa: E402
from utils.config import IMAGE_SIZE, NUM_CLASSES  # noqa: E402
from utils import label_mapping as lm  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WEBCAM_ROOT = os.path.join(PROJECT_ROOT, "dataset", "webcam")
MANIFEST_PATH = os.path.join(WEBCAM_ROOT, "manifest.csv")
MANIFEST_HEADER = ["filename", "letter", "label_index", "roi_filename"]

DEFAULT_TARGET = 75
ROI_SIZE = 300
CAPTURE_DELAY_SECONDS = 0.4
WINDOW_NAME = "HandSpeak - Webcam Dataset Collection (Q to quit)"
ZERO_PAD = 6

# Rotating prompts shown on screen to encourage natural variation between
# captures for the same letter.
VARIATION_HINTS = [
    "Move hand left/right in the box",
    "Change distance from camera",
    "Try different lighting angle",
    "Slight hand rotation",
    "Adjust finger spread",
    "Change background/position",
    "Vary hand orientation where natural",
    "Tilt wrist slightly",
]


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------
def _letter_dir(letter: str) -> str:
    return os.path.join(WEBCAM_ROOT, letter)


def _processed_dir(letter: str) -> str:
    return os.path.join(_letter_dir(letter), "processed")


def _roi_dir(letter: str) -> str:
    return os.path.join(_letter_dir(letter), "roi")


def ensure_dirs(letter: str) -> None:
    """Create (if needed) the processed/ and roi/ folders for one letter."""
    os.makedirs(_processed_dir(letter), exist_ok=True)
    os.makedirs(_roi_dir(letter), exist_ok=True)


def count_existing(letter: str) -> int:
    """Return the number of already-saved samples for ``letter``.

    Counts the processed 28x28 images, which are the authoritative saved
    samples. This allows resuming a partially completed collection session.
    """
    processed_dir = _processed_dir(letter)
    if not os.path.isdir(processed_dir):
        return 0
    return len(
        [
            name
            for name in os.listdir(processed_dir)
            if name.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )


def _format_index(index: int) -> str:
    return str(index).zfill(ZERO_PAD)


def load_manifest() -> list[dict]:
    """Read the existing manifest (if any) into a list of row dicts."""
    if not os.path.exists(MANIFEST_PATH):
        return []
    with open(MANIFEST_PATH, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return [row for row in reader]


def write_manifest(rows: list[dict]) -> None:
    """Write the full manifest, keeping rows sorted by filename."""
    os.makedirs(WEBCAM_ROOT, exist_ok=True)
    rows = sorted(rows, key=lambda r: r.get("filename", ""))
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def add_manifest_row(
    rows: list[dict], letter: str, index: int, rel_processed: str, rel_roi: str
) -> None:
    """Append one sample record to the in-memory manifest list."""
    label_index = ord(letter) - ord("A")
    rows.append(
        {
            "filename": rel_processed,
            "letter": letter,
            "label_index": str(label_index),
            "roi_filename": rel_roi,
        }
    )


def remove_manifest_row(rows: list[dict], rel_processed: str) -> None:
    """Remove the manifest record whose processed filename matches."""
    # Rebuild in place, keeping everything except the matching row.
    filtered = [row for row in rows if row.get("filename") != rel_processed]
    rows[:] = filtered


def letter_progress() -> dict[str, int]:
    """Return ``{letter: count}`` for all 26 letters (only existing folders)."""
    return {
        letter: count_existing(letter)
        for letter in lm.LETTERS
        if os.path.isdir(_processed_dir(letter))
    }


# ---------------------------------------------------------------------------
# Webcam helpers
# ---------------------------------------------------------------------------
def open_webcam(camera_index: int = 0) -> cv2.VideoCapture:
    """Open the webcam and return a ready capture handle.

    Raises
    ------
    RuntimeError
        If the camera cannot be opened.
    """
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(
            "Could not open the webcam. Check that it is connected, not in use "
            "by another application, and that OpenCV has camera access."
        )
    return capture


def draw_overlay(
    frame: np.ndarray,
    letter: str,
    count: int,
    target: int,
    hint: str,
    roi_box: tuple[int, int, int, int],
    flash: str = "",
) -> None:
    """Draw collection status and the ROI guide onto the display frame."""
    x1, y1, x2, y2 = roi_box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    header = f"TARGET: {letter}   ({count}/{target})"
    cv2.putText(
        frame, header, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2,
        cv2.LINE_AA,
    )

    bar_width = int(min(300, frame.shape[1] - 40))
    frac = min(1.0, count / max(1, target))
    cv2.rectangle(frame, (10, 44), (10 + bar_width, 56), (60, 60, 60), -1)
    cv2.rectangle(
        frame, (10, 44), (10 + int(bar_width * frac), 56), (0, 255, 0), -1,
    )

    cv2.putText(
        frame, f"Hint: {hint}", (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (255, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.putText(
        frame, "SPACE=capture  R=retake  N/S=next letter  Q=quit",
        (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (200, 200, 200), 1, cv2.LINE_AA,
    )

    if flash:
        cv2.putText(
            frame, flash, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (0, 0, 255), 2, cv2.LINE_AA,
        )


def _rel_processed(letter: str, index: int) -> str:
    name = f"{letter}_{_format_index(index)}.png"
    return os.path.join(letter, "processed", name)


def _rel_roi(letter: str, index: int) -> str:
    name = f"{letter}_{_format_index(index)}.png"
    return os.path.join(letter, "roi", name)


def capture_sample(
    letter: str, index: int, roi: np.ndarray, manifest_rows: list[dict]
) -> bool:
    """Preprocess and save one sample (28x28 + original ROI) and its manifest row.

    Returns ``True`` on success, ``False`` when no hand is detected in the ROI.
    """
    # 1. Run the existing preprocessing pipeline on the extracted ROI.
    tensor = preprocess_frame(roi, target_size=IMAGE_SIZE)
    if tensor is None:
        return False

    processed = (tensor[0, :, :, 0] * 255.0).astype(np.uint8)

    processed_path = os.path.join(WEBCAM_ROOT, _rel_processed(letter, index))
    roi_path = os.path.join(WEBCAM_ROOT, _rel_roi(letter, index))

    cv2.imwrite(processed_path, processed)
    cv2.imwrite(roi_path, roi)

    add_manifest_row(
        manifest_rows, letter, index, _rel_processed(letter, index), _rel_roi(letter, index)
    )
    write_manifest(manifest_rows)
    return True


def delete_sample(
    letter: str, index: int, manifest_rows: list[dict]
) -> None:
    """Delete the sample at ``index`` (both files) and drop its manifest row."""
    processed_path = os.path.join(WEBCAM_ROOT, _rel_processed(letter, index))
    roi_path = os.path.join(WEBCAM_ROOT, _rel_roi(letter, index))

    for path in (processed_path, roi_path):
        if os.path.exists(path):
            os.remove(path)

    remove_manifest_row(manifest_rows, _rel_processed(letter, index))
    write_manifest(manifest_rows)


# ---------------------------------------------------------------------------
# Collection loop
# ---------------------------------------------------------------------------
def run_collection(letter: str, target: int, camera_index: int, manifest_rows: list[dict]) -> None:
    """Run the interactive capture loop for a single target letter."""
    ensure_dirs(letter)

    capture = open_webcam(camera_index)
    print(f"\nCollecting letter '{letter}' (target {target}).")
    print("  SPACE/ENTER = capture    R = retake last    N/S = next letter    Q = quit")
    print("  Tip: vary hand position, distance, lighting, background, and rotation.")

    count = count_existing(letter)
    hint_idx = 0
    last_capture_time = 0.0
    flash = ""
    flash_until = 0.0
    last_index: int | None = None

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Warning: invalid/empty frame; skipping ...")
                continue

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            roi_size = min(ROI_SIZE, width, height)
            x1 = (width - roi_size) // 2
            y1 = (height - roi_size) // 2
            x2 = x1 + roi_size
            y2 = y1 + roi_size
            roi_box = (x1, y1, x2, y2)

            # Extract ROI before drawing any overlay so the guide box never
            # contaminates the saved hand pixels.
            roi = frame[y1:y2, x1:x2].copy()

            if time.time() >= flash_until:
                flash = ""

            draw_overlay(
                frame, letter, count, target, VARIATION_HINTS[hint_idx % len(VARIATION_HINTS)],
                roi_box, flash,
            )
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n'Q' pressed. Saving manifest and exiting ...")
                break

            if key in (ord(" "), 13):  # SPACE or ENTER
                now = time.time()
                if now - last_capture_time < CAPTURE_DELAY_SECONDS:
                    continue
                last_capture_time = now

                next_index = count + 1
                if capture_sample(letter, next_index, roi, manifest_rows):
                    count += 1
                    last_index = next_index
                    hint_idx += 1
                    flash = f"Saved {letter}_{_format_index(next_index)}  ({count}/{target})"
                    flash_until = time.time() + 0.8
                    print(f"[{letter}] saved sample {count}/{target} "
                          f"(index {_format_index(next_index)})")
                else:
                    flash = "No hand detected - adjust and retry"
                    flash_until = time.time() + 0.8
                    print(f"[{letter}] no hand detected; sample not saved.")

                if count >= target:
                    print(f"[{letter}] target reached ({count}/{target}).")
                    flash = f"Target reached! Press N for next letter"
                    flash_until = time.time() + 2.0

            elif key == ord("r"):
                if last_index is not None and last_index <= count and count > 0:
                    delete_sample(letter, last_index, manifest_rows)
                    print(f"[{letter}] retook last sample (deleted index "
                          f"{_format_index(last_index)}).")
                    count -= 1
                    last_index = None
                    flash = f"Retook last sample  ({count}/{target})"
                    flash_until = time.time() + 0.8
                else:
                    flash = "Nothing to retake"
                    flash_until = time.time() + 0.8

            elif key in (ord("n"), ord("s")):
                print(f"[{letter}] moving to next letter ({count}/{target} collected).")
                break

            elif ord("a") <= key <= ord("z"):
                new_letter = chr(key).upper()
                print(f"[{letter}] switching target to '{new_letter}' "
                      f"({count}/{target} collected).")
                letter = new_letter
                ensure_dirs(letter)
                count = count_existing(letter)
                last_index = None

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        capture.release()
        cv2.destroyAllWindows()

    print(f"Letter '{letter}' finished with {count}/{target} samples.")


# ---------------------------------------------------------------------------
# Progress / camera check
# ---------------------------------------------------------------------------
def print_progress() -> None:
    """Print a compact per-letter progress table."""
    progress = {letter: count_existing(letter) for letter in lm.LETTERS}
    total = sum(progress.values())
    print("\n" + "=" * 60)
    print("Webcam dataset progress (dataset/webcam/)")
    print("=" * 60)
    line = "  ".join(f"{letter}:{progress[letter]}" for letter in lm.LETTERS)
    print(line)
    print("-" * 60)
    print(f"Total samples collected: {total}  (target {DEFAULT_TARGET}/letter)")
    missing = [letter for letter, c in progress.items() if c == 0]
    if missing:
        print(f"Letters with 0 samples: {''.join(missing)}")
    print()


def check_camera(camera_index: int) -> None:
    """Attempt to open and immediately release the webcam."""
    print(f"Attempting to open webcam index {camera_index} ...")
    try:
        capture = open_webcam(camera_index)
    except RuntimeError as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)

    ok, frame = capture.read()
    capture.release()
    cv2.destroyAllWindows()
    if ok and frame is not None:
        print(f"OK: webcam opened and returned a frame of shape {frame.shape}.")
    else:
        print("WARNING: webcam opened but did not return a frame.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect webcam-domain A-Z sign samples for CNN fine-tuning."
    )
    parser.add_argument(
        "--letter",
        type=str,
        default=None,
        help="Letter to collect first (e.g. A). If omitted, an interactive prompt is shown.",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=DEFAULT_TARGET,
        help=f"Number of samples to collect per letter (default {DEFAULT_TARGET}).",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera index (default 0 = built-in webcam).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print per-letter progress and exit without opening the webcam.",
    )
    parser.add_argument(
        "--check-camera",
        action="store_true",
        help="Attempt to open the webcam once and exit (no collection).",
    )
    return parser.parse_args()


def _choose_letter() -> str:
    """Prompt the user for a starting letter, defaulting to the first incomplete."""
    progress = {letter: count_existing(letter) for letter in lm.LETTERS}
    incomplete = [letter for letter in lm.LETTERS if progress[letter] < DEFAULT_TARGET]
    default = incomplete[0] if incomplete else "A"

    print_progress()
    print("Enter the letter to collect (A-Z), or press Enter for "
          f"'{default}': ", end="")
    choice = input().strip().upper()
    if choice in lm.LETTERS:
        return choice
    if choice == "":
        return default
    print(f"'{choice}' is not a valid letter; defaulting to '{default}'.")
    return default


def main() -> None:
    args = parse_args()

    if args.list:
        print_progress()
        return

    if args.check_camera:
        check_camera(args.camera)
        return

    # Validate target.
    if args.target <= 0:
        print("ERROR: --target must be a positive integer.")
        sys.exit(1)

    manifest_rows = load_manifest()

    if args.letter:
        letter = args.letter.strip().upper()
        if letter not in lm.LETTERS:
            print(f"ERROR: '{args.letter}' is not a valid letter (A-Z).")
            sys.exit(1)
    else:
        letter = _choose_letter()

    print("\n" + "=" * 60)
    print("HandSpeak - Webcam Dataset Collection")
    print("=" * 60)
    print(f"Starting letter : {letter}")
    print(f"Target per letter: {args.target}")
    print(f"Save location   : {WEBCAM_ROOT}")
    print(f"Label mapping   : A=0 ... Z=25 (26 classes)")
    print("NOTE: This script only writes under dataset/webcam/. It does NOT")
    print("      retrain, and does NOT modify dataset/raw|train|test|validation|jz")
    print("      or any existing trained model.")
    print("=" * 60)

    run_collection(letter, args.target, args.camera, manifest_rows)

    print("\nCollection session ended. Summary:")
    print_progress()


if __name__ == "__main__":
    main()