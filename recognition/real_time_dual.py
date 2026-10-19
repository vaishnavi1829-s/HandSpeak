"""Live webcam sign recognition with ASL and ISL modes.

This is the STEP 7 (webcam testing) and STEP 8 (separate ASL/ISL modes) stage.

It drives the built-in webcam and runs live inference using the language-aware
``SignPredictor``. The user can switch between ASL and ISL at runtime:

    * Press ``1`` to switch to ASL (American Sign Language).
    * Press ``2`` to switch to ISL (Indian Sign Language).
    * Press ``Q`` to quit.

Each mode loads its own 26-class model; no model weights are shared across
modes. The live preview is horizontally mirrored for a natural feel. ASL
inference uses the mirrored frame (matching the ASL collection script), while
ISL inference uses the raw, unmirrored camera frame (matching the ISL
collection script, which does NOT mirror).

Run from the project root::

    python recognition/real_time_dual.py --language ASL
    python recognition/real_time_dual.py --language ISL
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from recognition.predictor import ASL, ISL, SignPredictor, SUPPORTED_LANGUAGES  # noqa: E402

ROI_SIZE = 300
WINDOW_NAME = "HandSpeak - Live Recognition (1=ASL, 2=ISL, Q=quit)"


def open_webcam(camera_index: int = 0) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(
            "Could not open the webcam. Check that it is connected and not in "
            "use by another application."
        )
    return capture


def draw_overlay(frame, language, label):
    """Draw the current mode and prediction onto the display frame."""
    cv2.putText(
        frame, f"Mode: {language}", (10, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, f"Prediction: {label}", (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, "1=ASL  2=ISL  Q=quit", (10, frame.shape[0] - 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA,
    )


def run(language: str, camera_index: int) -> None:
    predictor = SignPredictor(language=language)
    capture = open_webcam(camera_index)

    print("\nInstructions:")
    print("  - Place your hand/sign inside the green box.")
    print("  - Press '1' for ASL, '2' for ISL, 'Q' to quit.")
    print()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                continue

            # ASL inference uses a mirrored frame (matches ASL collection);
            # ISL inference uses the raw camera frame (matches ISL collection).
            if predictor.language == ASL:
                inference_frame = cv2.flip(frame, 1)
            else:
                inference_frame = frame

            # The user-facing preview is always mirrored for a natural feel.
            display_frame = cv2.flip(frame, 1)
            height, width = display_frame.shape[:2]

            roi_size = min(ROI_SIZE, width, height)
            x1 = (width - roi_size) // 2
            y1 = (height - roi_size) // 2
            x2 = x1 + roi_size
            y2 = y1 + roi_size

            roi = inference_frame[y1:y2, x1:x2].copy()

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = "No sign"
            if roi.size > 0:
                letter, confidence = predictor.predict_smoothed(roi)
                if letter is not None:
                    label = f"{letter}  ({confidence * 100:.1f}%)"

            draw_overlay(display_frame, predictor.language, label)
            cv2.imshow(WINDOW_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("1") and predictor.language != ASL:
                print("Switching to ASL ...")
                predictor = SignPredictor(language=ASL)
            elif key == ord("2") and predictor.language != ISL:
                print("Switching to ISL ...")
                predictor = SignPredictor(language=ISL)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        capture.release()
        cv2.destroyAllWindows()

    print("Webcam released. Recognition session ended.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live ASL/ISL webcam recognition with runtime mode switch."
    )
    parser.add_argument(
        "--language",
        type=str,
        default=ASL,
        choices=SUPPORTED_LANGUAGES,
        help="Initial recognition mode (default ASL).",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera index (default 0).",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("HandSpeak - Real-Time Webcam Recognition (ASL + ISL)")
    print("=" * 60)
    run(args.language.upper(), args.camera)


if __name__ == "__main__":
    main()
