"""Real-time webcam recognition using MediaPipe hand landmarks + Random Forest.

ASL mode         -> ONE hand  -> 63 features  -> asl_landmark_rf.joblib
ISL mode         -> TWO hands -> 126 features -> isl_landmark_rf_2hand.joblib
ISL Rich mode    -> ONE or TWO hands -> 214 geometric features ->
                    isl_landmark_rf_rich.joblib

The existing CNN models are never loaded, modified, or overwritten.  This is a
separate landmark-based recognizer.

Run from the project root::

    python recognition/real_time_landmark.py
"""

from __future__ import annotations

import os
import sys
import time
from collections import Counter, deque

import cv2
import joblib
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mediapipe.tasks.python import vision  # noqa: E402

from recognition.landmark_extractor import (  # noqa: E402
    get_hand_landmarker,
    get_two_hand_landmarker,
    extract_landmarks_from_bgr,
    extract_two_hands_from_bgr,
)
from recognition.isl_feature_extractor import (  # noqa: E402
    rich_features_from_two_hand_result,
)

# ---------------------------------------------------------------------------
# Configuration (easy to change)
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.65
SMOOTHING_WINDOW = 5
MIN_VOTES = 3
MISS_RESET_THRESHOLD = 3

WINDOW_NAME = "HandSpeak - Landmark Recognition"

# Model paths (ASL one-hand, ISL two-hand, ISL rich-feature; kept separate).
MODEL_PATHS = {
    "ASL": os.path.join(PROJECT_ROOT, "trained_model", "asl_landmark_rf.joblib"),
    "ISL": os.path.join(PROJECT_ROOT, "trained_model", "isl_landmark_rf_rich.joblib"),
    "ISL Rich": os.path.join(
        PROJECT_ROOT, "trained_model", "isl_landmark_rf_rich.joblib"
    ),
}

MEDIAPIPE_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "mediapipe", "hand_landmarker.task"
)

# Colors (BGR).
COLOR_LANDMARK = (0, 255, 0)
COLOR_CONNECTION = (255, 255, 0)
COLOR_LEFT = (255, 0, 255)       # magenta for left hand
COLOR_RIGHT = (0, 255, 255)      # yellow for right hand
COLOR_TEXT = (255, 255, 255)
COLOR_WARN = (0, 0, 255)
COLOR_ACCENT = (0, 255, 255)


def load_model(mode: str):
    """Load the Random Forest model for the given mode."""
    model_path = MODEL_PATHS[mode]
    if not os.path.exists(model_path):
        hint = {
            "ASL": "asl (python scripts/train_landmark_models.py --dataset asl)",
            "ISL": "isl two-hand (python scripts/train_isl_twohand_model.py)",
            "ISL Rich": "isl rich (python scripts/train_isl_rich_model.py)",
        }[mode]
        raise FileNotFoundError(
            f"Landmark model not found: {model_path}\n"
            f"Train it first: {hint}"
        )
    print(f"[{mode}] Loading model: {model_path}")
    return joblib.load(model_path)


def load_mediapipe_one_hand():
    if not os.path.exists(MEDIAPIPE_MODEL_PATH):
        raise FileNotFoundError(
            f"MediaPipe model asset not found: {MEDIAPIPE_MODEL_PATH}"
        )
    return get_hand_landmarker(MEDIAPIPE_MODEL_PATH)


def load_mediapipe_two_hand():
    if not os.path.exists(MEDIAPIPE_MODEL_PATH):
        raise FileNotFoundError(
            f"MediaPipe model asset not found: {MEDIAPIPE_MODEL_PATH}"
        )
    return get_two_hand_landmarker(MEDIAPIPE_MODEL_PATH)


def select_mode() -> str | None:
    """Show the startup menu and return the selected mode (or None to quit)."""
    print("=================================")
    print(" HandSpeak Landmark Recognition")
    print("=================================")
    print("1 - ASL (one hand)")
    print("2 - ISL (two hands)")
    print("3 - ISL Rich Features (one/two hands, geometric features)")
    print("Q - Quit")
    print("=================================")

    while True:
        choice = input(
            "Select mode (1=ASL, 2=ISL, 3=ISL Rich, Q=Quit): "
        ).strip().lower()
        if choice in ("q", "quit"):
            return None
        if choice == "1":
            return "ASL"
        if choice == "2":
            return "ISL"
        if choice == "3":
            return "ISL Rich"
        print("Invalid choice. Please enter 1, 2, 3, or Q.")


def draw_landmarks(frame, landmarks, color: tuple, mirror_x: bool = False) -> None:
    """Draw 21 landmarks and connections for one hand.

    ``mirror_x`` maps MediaPipe coordinates that were computed on an unflipped
    frame onto a horizontally mirrored display frame (``x_display = 1 - x``).
    """
    height, width = frame.shape[:2]
    points = []
    for lm in landmarks:
        px = int((1.0 - lm[0]) * width if mirror_x else lm[0] * width)
        py = int(lm[1] * height)
        points.append((px, py))
        cv2.circle(frame, (px, py), 4, color, -1)

    connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
    for conn in connections:
        cv2.line(frame, points[conn.start], points[conn.end], color, 2)


def smooth_prediction(history: deque, model, feature_vector: np.ndarray):
    """Majority-vote smoothing over the recent prediction history."""
    X = feature_vector.reshape(1, -1)
    proba = model.predict_proba(X)[0]
    letter = model.classes_[int(np.argmax(proba))]
    confidence = float(np.max(proba))

    history.append(letter)
    while len(history) > SMOOTHING_WINDOW:
        history.popleft()

    if len(history) < SMOOTHING_WINDOW:
        return None, None

    votes = Counter(history)
    top_letter, top_count = votes.most_common(1)[0]
    if top_count < MIN_VOTES:
        return None, None

    return top_letter, confidence


def run(mode: str) -> None:
    """Run the webcam loop for the selected mode."""
    model = load_model(mode)
    one_hand_landmarker = load_mediapipe_one_hand()
    two_hand_landmarker = load_mediapipe_two_hand()

    capture = cv2.VideoCapture(CAMERA_INDEX)
    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open webcam at index {CAMERA_INDEX}. "
            "Check that it is connected and not in use by another application."
        )

    print(f"\n[{mode}] Webcam opened. Press Q to quit, R to reset history, "
          f"1/2/3 to switch mode.")
    print(f"[{mode}] Confidence threshold = {CONFIDENCE_THRESHOLD}, "
          f"smoothing window = {SMOOTHING_WINDOW}")

    history: deque[str] = deque(maxlen=SMOOTHING_WINDOW)
    misses = 0
    last_frame_time = time.time()
    fps = 0.0
    active_mode = mode

    while True:
        ok, camera_frame = capture.read()
        if not ok:
            continue

        # ASL inference uses a mirrored frame (matches ASL collection);
        # ISL/ISL Rich inference uses the raw camera frame (matches ISL
        # collection, which is saved unflipped).
        if active_mode == "ASL":
            inference_frame = cv2.flip(camera_frame, 1)
        else:
            inference_frame = camera_frame

        # The user-facing preview is always mirrored for a natural webcam feel.
        display_frame = cv2.flip(camera_frame, 1)
        height, width = display_frame.shape[:2]

        # If inference ran on an unflipped frame but the preview is mirrored,
        # landmark x-coordinates must be mirrored before drawing.
        mirror_landmarks = active_mode != "ASL"

        if active_mode == "ASL":
            # ---- ASL: one hand, 63 features ----
            result = extract_landmarks_from_bgr(inference_frame, landmarker=one_hand_landmarker)
            if result is None:
                history.clear()
                hands_text = "Hands detected: 0"
                status = "No hand detected"
                cv2.putText(display_frame, status, (15, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, COLOR_WARN, 2, cv2.LINE_AA)
                confidence = None
            else:
                draw_landmarks(display_frame, result.landmarks, COLOR_LANDMARK,
                               mirror_x=mirror_landmarks)
                hands_text = "Hands detected: 1"
                letter, confidence = smooth_prediction(history, model, result.feature_vector)
                if letter is None:
                    status = "No confident sign"
                    cv2.putText(display_frame, status, (15, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, COLOR_WARN, 2, cv2.LINE_AA)
                else:
                    status = letter
                    cv2.putText(display_frame, status, (15, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                1.8, COLOR_ACCENT, 3, cv2.LINE_AA)

                if confidence is not None:
                    cv2.putText(display_frame, f"Confidence: {confidence * 100:.1f}%",
                                (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                COLOR_TEXT, 2, cv2.LINE_AA)
                    if confidence < CONFIDENCE_THRESHOLD:
                        cv2.putText(display_frame, "Low confidence", (15, 105),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WARN, 2,
                                    cv2.LINE_AA)

        else:
            # ---- ISL (two hands, 126 features) OR ISL Rich (geometric features) ----
            result = extract_two_hands_from_bgr(inference_frame, landmarker=two_hand_landmarker)

            if result is None:
                misses += 1
                if misses >= MISS_RESET_THRESHOLD:
                    history.clear()
                    misses = 0
                hands_text = "Hands detected: 0"
                status = "No hands detected"
                cv2.putText(display_frame, status, (15, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, COLOR_WARN, 2, cv2.LINE_AA)
                confidence = None
            else:
                # Draw whichever hands are present (left = magenta, right = yellow).
                if result.left_present and result.left_landmarks is not None:
                    draw_landmarks(display_frame, result.left_landmarks, COLOR_LEFT,
                                   mirror_x=mirror_landmarks)
                if result.right_present and result.right_landmarks is not None:
                    draw_landmarks(display_frame, result.right_landmarks, COLOR_RIGHT,
                                   mirror_x=mirror_landmarks)

                n_hands = int(result.left_present) + int(result.right_present)
                hands_text = f"Hands detected: {n_hands}"

                if active_mode == "ISL Rich":
                    # Rich mode supports one OR two hands (no fabrication).
                    misses = 0
                    features, _ = rich_features_from_two_hand_result(result)
                    letter, confidence = smooth_prediction(history, model, features)
                    if letter is None:
                        status = "No confident sign"
                        cv2.putText(display_frame, status, (15, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.9, COLOR_WARN, 2, cv2.LINE_AA)
                    else:
                        status = letter
                        cv2.putText(display_frame, status, (15, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                    1.8, COLOR_ACCENT, 3, cv2.LINE_AA)

                    if confidence is not None:
                        cv2.putText(display_frame, f"Confidence: {confidence * 100:.1f}%",
                                    (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    COLOR_TEXT, 2, cv2.LINE_AA)
                        if confidence < CONFIDENCE_THRESHOLD:
                            cv2.putText(display_frame, "Low confidence", (15, 105),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WARN, 2,
                                        cv2.LINE_AA)
                else:
                    # ISL two-hand mode: require both hands before predicting.
                    if n_hands < 2:
                        misses += 1
                        if misses >= MISS_RESET_THRESHOLD:
                            history.clear()
                            misses = 0
                        status = "Waiting for second hand..."
                        cv2.putText(display_frame, status, (15, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.9, COLOR_WARN, 2, cv2.LINE_AA)
                        confidence = None
                    else:
                        misses = 0
                        features, _ = rich_features_from_two_hand_result(result)
                        letter, confidence = smooth_prediction(
                            history, model, features
                        )
                        if letter is None:
                            status = "No confident sign"
                            cv2.putText(display_frame, status, (15, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.9, COLOR_WARN, 2, cv2.LINE_AA)
                        else:
                            status = letter
                            cv2.putText(display_frame, status, (15, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                        1.8, COLOR_ACCENT, 3, cv2.LINE_AA)

                        if confidence is not None:
                            cv2.putText(display_frame, f"Confidence: {confidence * 100:.1f}%",
                                        (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                        COLOR_TEXT, 2, cv2.LINE_AA)
                            if confidence < CONFIDENCE_THRESHOLD:
                                cv2.putText(display_frame, "Low confidence", (15, 105),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WARN, 2,
                                            cv2.LINE_AA)

        # FPS.
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_frame_time, 1e-6))
        last_frame_time = now

        # UI overlay (top-right).
        cv2.putText(display_frame, f"Mode: {active_mode}", (width - 190, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_ACCENT, 2, cv2.LINE_AA)
        cv2.putText(display_frame, hands_text, (width - 190, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2, cv2.LINE_AA)
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (width - 190, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2, cv2.LINE_AA)
        cv2.putText(display_frame, "Q=Quit  R=Reset  1=ASL  2=ISL  3=ISL Rich",
                    (15, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    COLOR_TEXT, 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        elif key in (ord("r"), ord("R")):
            history.clear()
            misses = 0
            print(f"[{active_mode}] Prediction history reset.")
        elif key == ord("1"):
            if active_mode != "ASL":
                active_mode = "ASL"
                model = load_model(active_mode)
                history.clear()
                misses = 0
                print("[ASL] Switched to ASL model.")
        elif key == ord("2"):
            if active_mode != "ISL":
                active_mode = "ISL"
                model = load_model(active_mode)
                history.clear()
                misses = 0
                print("[ISL] Switched to ISL two-hand model.")
        elif key == ord("3"):
            if active_mode != "ISL Rich":
                active_mode = "ISL Rich"
                model = load_model(active_mode)
                history.clear()
                misses = 0
                print("[ISL Rich] Switched to ISL rich-feature model.")

    capture.release()
    cv2.destroyAllWindows()
    print(f"[{active_mode}] Exited.")


def main() -> None:
    mode = select_mode()
    if mode is None:
        print("Exiting.")
        return
    try:
        run(mode)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"\nError: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
