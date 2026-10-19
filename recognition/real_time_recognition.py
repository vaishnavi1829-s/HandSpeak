"""Real-time webcam sign recognition for the HandSpeak project.

This module loads the already-trained CNN from ``trained_model/handspeak_cnn.h5``
and runs live inference on frames captured from the built-in laptop webcam.

It does NOT retrain the model, does NOT modify the dataset or preprocessing
files, and does NOT start Flask, TTS, speech recognition, translation, or any
other application layer.

Preprocessing is delegated to :mod:`recognition.preprocessing`, which
reproduces the exact SL-MNIST training format (grayscale, thresholding, hand
localization, polarity inversion, centering/padding, 28x28 resize, /255.0
normalization).

The live loop:

    1. Opens the default webcam with OpenCV (``cv2.VideoCapture(0)``).
    2. Captures frames continuously (mirrored so the feed feels natural).
    3. Draws a fixed square region-of-interest (ROI) in the center of the
       frame where the user should place their hand/sign.
    4. Extracts the ROI and preprocesses it to the CNN input spec.
    5. Predicts one of the 26 classes A-Z using the trained model.
    6. Applies a confidence threshold; low-confidence inputs display "No sign".
    7. Draws the predicted letter and confidence score on the frame.
    8. Exits when the user presses the ``Q`` key.

Run from the project root::

    python recognition/real_time_recognition.py
"""

from __future__ import annotations

import os
import sys
from collections import Counter, deque

import cv2
import numpy as np

# Allow this script to be run directly from the project root so that the
# ``utils``, ``models``, and ``recognition`` packages resolve correctly.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from keras.models import load_model  # noqa: E402

from recognition.preprocessing import (  # noqa: E402
    CONFIDENCE_THRESHOLD,
    preprocess_frame,
)
from utils.config import (  # noqa: E402
    MODEL_WEBCAM_PATH,
    IMAGE_SIZE,
    CHANNELS,
)
from utils.label_mapping import LETTERS  # noqa: E402

# Size of the square hand/sign region captured from the webcam frame.
ROI_SIZE = 300

# Webcam window title.
WINDOW_NAME = "HandSpeak - Real-Time Sign Recognition (Q to quit)"

# Number of consecutive frames kept for temporal smoothing / majority voting.
SMOOTHING_WINDOW = 7

# Minimum number of votes a letter must receive to be shown (stability gate).
MIN_VOTES = 4


def load_trained_model(model_path=MODEL_WEBCAM_PATH):
    """Load the webcam fine-tuned CNN model.

    Parameters
    ----------
    model_path : str
        Path to the saved full model (``.h5`` containing architecture + weights).

    Returns
    -------
    keras.Model
        The compiled trained model, ready for inference.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist at ``model_path``.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at: {model_path}. "
            "Run the improved CNN training script first, or verify the path."
        )

    print(f"Loading trained model from: {model_path}")
    model = load_model(model_path)
    print("Model loaded successfully.")
    return model


def predict_letter(model, roi, confidence_threshold=CONFIDENCE_THRESHOLD):
    """Predict the most likely sign letter for a given ROI.

    Parameters
    ----------
    model : keras.Model
        Trained CNN model.
    roi : numpy.ndarray
        BGR hand/sign image region.
    confidence_threshold : float
        Minimum top-class probability required to report a letter.

    Returns
    -------
    tuple[str, float] | tuple[None, None]
        The predicted letter (A-Z) and its confidence, or ``(None, None)`` when
        no hand is detected or confidence is below the threshold.
    """
    input_tensor = preprocess_frame(roi, target_size=IMAGE_SIZE)
    if input_tensor is None:
        return None, None

    probabilities = model.predict(input_tensor, verbose=0)[0]
    class_index = int(np.argmax(probabilities))
    confidence = float(probabilities[class_index])

    if confidence < confidence_threshold:
        return None, None

    return LETTERS[class_index], confidence


def smooth_prediction(history, window=SMOOTHING_WINDOW, min_votes=MIN_VOTES):
    """Return the most stable prediction over recent frames.

    Parameters
    ----------
    history : collections.deque
        Deque holding recent (letter, confidence) pairs (maxlen=window).
    window : int
        Number of recent frames to consider.
    min_votes : int
        Minimum votes the top letter must receive to be considered stable.

    Returns
    -------
    tuple[str, float] | tuple[None, None]
        The smoothed letter and its average confidence, or ``(None, None)`` if
        there is no stable majority.
    """
    letters = [entry[0] for entry in history if entry[0] is not None]
    if not letters:
        return None, None

    counts = Counter(letters)
    top_letter, votes = counts.most_common(1)[0]
    if votes < min_votes:
        return None, None

    confs = [entry[1] for entry in history if entry[0] == top_letter]
    avg_conf = float(sum(confs) / len(confs)) if confs else 0.0
    return top_letter, avg_conf


def _update_history(history, letter, confidence):
    """Append a raw prediction to the smoothing deque (keeps fixed length)."""
    history.append((letter, confidence))


def open_webcam(camera_index=0):
    """Open the webcam and return a ready ``cv2.VideoCapture`` handle.

    Raises
    ------
    RuntimeError
        If the camera cannot be opened.
    """
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(
            "Could not open the webcam. Check that it is connected, not in "
            "use by another application, and that OpenCV has camera access."
        )
    return capture


def main():
    print("=" * 60)
    print("HandSpeak - Real-Time Webcam Sign Recognition")
    print("=" * 60)

    # 1. Load the existing trained model (no training happens here).
    model = load_trained_model()

    # 2. Open the built-in webcam.
    capture = open_webcam()

    print("\nInstructions:")
    print("  - Place your hand/sign inside the green box on screen.")
    print("  - Press 'Q' to quit.")
    print()

    # Rolling window of recent (letter, confidence) raw predictions.
    history = deque(maxlen=SMOOTHING_WINDOW)

    try:
        while True:
            # 3. Capture the next frame.
            ok, frame = capture.read()
            if not ok:
                print("Warning: received an invalid/empty frame; skipping ...")
                continue

            # Mirror the frame horizontally so the feed behaves like a mirror.
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            # 4. Define a centered square region-of-interest for the sign.
            roi_size = min(ROI_SIZE, width, height)
            x1 = (width - roi_size) // 2
            y1 = (height - roi_size) // 2
            x2 = x1 + roi_size
            y2 = y1 + roi_size

            # 5. Extract the ROI BEFORE any overlay is drawn, so the green box
            #    (and any text) never contaminates the hand pixels sent to the
            #    CNN. The overlay is drawn only on the display frame afterwards.
            roi = frame[y1:y2, x1:x2].copy()

            # Draw the ROI on the display frame.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                "Place hand here",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            label = "No sign"
            if roi.size > 0:
                letter, confidence = predict_letter(model, roi)
                _update_history(history, letter, confidence)
                smooth_letter, smooth_conf = smooth_prediction(history)
                if smooth_letter is not None:
                    label = f"{smooth_letter}  ({smooth_conf * 100:.1f}%)"

            # 6. Display the prediction on the frame.
            cv2.putText(
                frame,
                f"Prediction: {label}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # 7. Show the live feed.
            cv2.imshow(WINDOW_NAME, frame)

            # 8. Exit cleanly when the user presses 'Q'.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n'Q' pressed. Exiting ...")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting ...")
    finally:
        capture.release()
        cv2.destroyAllWindows()

    print("Webcam released. Recognition session ended.")


if __name__ == "__main__":
    main()
