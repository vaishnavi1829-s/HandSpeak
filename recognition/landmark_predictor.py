"""Landmark-based sign predictor for ASL + ISL, matching the working webcam demo.

This module is the shared inference layer that reproduces the exact recognition
pipeline used by :mod:`recognition.real_time_landmark` (the working VS Code
webcam script):

    * ASL -> ONE hand   -> 63 wrist-relative/scale-normalized features
             -> ``trained_model/asl_landmark_rf.joblib``
    * ISL -> one or two hands -> 214 rich geometric features
             -> ``trained_model/isl_landmark_rf_rich.joblib``

The CNN models (``handspeak_cnn*.h5``) are never loaded here.  ASL and ISL use
completely separate Random Forest models; the ASL model is never used for ISL.

Mirroring matches the working demo exactly:

    * ASL inference uses a horizontally flipped frame (ASL collection is
      mirrored).
    * ISL inference uses the raw, unmirrored frame (ISL collection is saved
      unflipped).

The temporal smoothing below reproduces ``smooth_prediction`` from
``recognition/real_time_landmark.py``: a 5-frame rolling majority vote with a
minimum of 3 matching votes.  The confidence reported is the top-class
probability of the most recent frame (identical to the working script, which
returns ``confidence`` from the current frame rather than averaging).
"""

from __future__ import annotations

import os
import sys
from collections import Counter, deque

import cv2
import joblib
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from recognition.landmark_extractor import (  # noqa: E402
    extract_landmarks_from_bgr,
    extract_two_hands_from_bgr,
    get_hand_landmarker,
    get_two_hand_landmarker,
)
from recognition.isl_feature_extractor import (  # noqa: E402
    rich_features_from_two_hand_result,
)

ASL = "ASL"
ISL = "ISL"
SUPPORTED_LANGUAGES = (ASL, ISL)

MODEL_DIR = os.path.join(PROJECT_ROOT, "trained_model")

# Separate model per language (the ASL model is never used for ISL).
MODEL_PATHS = {
    ASL: os.path.join(MODEL_DIR, "asl_landmark_rf.joblib"),
    ISL: os.path.join(MODEL_DIR, "isl_landmark_rf_rich.joblib"),
}

MEDIAPIPE_MODEL_PATH = os.path.join(
    PROJECT_ROOT, "mediapipe", "hand_landmarker.task"
)

# Matches ``recognition/real_time_landmark.py``. This is a soft display threshold,
# not a hard suppression gate (the working demo still reports the predicted letter
# and marks it "Low confidence" when below this value).
CONFIDENCE_THRESHOLD = 0.65

# Temporal smoothing constants (identical to ``real_time_landmark.py``).
SMOOTHING_WINDOW = 5
MIN_VOTES = 3

TOP_K = 5


class LandmarkSignPredictor:
    """Loads one landmark Random Forest model and predicts letters from BGR images.

    Parameters
    ----------
    language:
        ``"ASL"`` or ``"ISL"``.
    """

    #: Expose the smoothing window for callers (e.g. the Flask endpoint).
    SMOOTHING_WINDOW = SMOOTHING_WINDOW
    MIN_VOTES = MIN_VOTES

    def __init__(self, language: str = ASL):
        language = (language or ASL).upper()
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{language}'. "
                f"Choose one of {SUPPORTED_LANGUAGES}."
            )

        self.language = language
        self.model_path = MODEL_PATHS[language]
        self._model = self._load_model()
        self._one_hand_landmarker = get_hand_landmarker(MEDIAPIPE_MODEL_PATH)
        self._two_hand_landmarker = get_two_hand_landmarker(MEDIAPIPE_MODEL_PATH)

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Landmark model not found for '{self.language}': "
                f"{self.model_path}."
            )
        print(f"[{self.language}] loading landmark model from {self.model_path}")
        return joblib.load(self.model_path)

    @property
    def classes(self) -> list[str]:
        """Return the model's class labels in output order (A-Z)."""
        return [str(cls) for cls in self._model.classes_]

    def _predict_raw(self, bgr: np.ndarray) -> dict:
        """Run the raw per-frame pipeline and return a JSON-ready result dict.

        ``prediction`` is ``None`` only when no hand is detected.  Otherwise it
        is always the top class (the working script does not hard-suppress low
        confidence, it only annotates it).
        """
        if bgr is None or bgr.size == 0:
            return self._no_result("Could not read the image.")

        if self.language == ASL:
            # ASL uses a mirrored frame (matches ASL collection / working demo).
            inference_frame = cv2.flip(bgr, 1)
            result = extract_landmarks_from_bgr(
                inference_frame, landmarker=self._one_hand_landmarker
            )
            if result is None:
                return self._no_result("No hand/sign detected in the image.")
            feature_vector = result.feature_vector
            hands_detected = 1
        else:
            # ISL uses the raw (unmirrored) frame.
            result = extract_two_hands_from_bgr(
                bgr, landmarker=self._two_hand_landmarker
            )
            if result is None:
                return self._no_result("No hand/sign detected in the image.")
            feature_vector, _ = rich_features_from_two_hand_result(result)
            hands_detected = int(result.left_present) + int(result.right_present)

        X = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)
        probabilities = self._model.predict_proba(X)[0]
        top_indices = np.argsort(probabilities)[::-1][:TOP_K]

        classes = self.classes
        top_k = [
            {
                "letter": classes[int(idx)],
                "probability": float(probabilities[int(idx)]),
            }
            for idx in top_indices
        ]

        top = top_k[0]
        return {
            "language": self.language,
            "prediction": top["letter"],
            "confidence": top["probability"],
            "top_k": top_k,
            "hands_detected": hands_detected,
        }

    def predict(self, bgr: np.ndarray) -> dict:
        """Predict one frame without temporal smoothing (static image path)."""
        return self._predict_raw(bgr)

    def predict_frames(self, frames: list[np.ndarray]) -> dict:
        """Apply the working script's 5-frame majority-vote smoothing.

        Each frame is processed with the raw pipeline and appended to a rolling
        ``deque(maxlen=SMOOTHING_WINDOW)``.  A frame with no detected hand clears
        the history (exactly as ``real_time_landmark.run`` does).  Once the
        history is full, the majority letter must have at least ``MIN_VOTES``
        votes or the result is ``None`` ("no confident sign").

        The returned ``confidence`` is the top-class probability of the LAST
        processed frame, matching the working script's ``smooth_prediction``.
        """
        if not frames:
            return self._no_result("No frames were provided.")

        # Fewer frames than the smoothing window -> fall back to a raw result so
        # single-image uploads still work.
        if len(frames) < SMOOTHING_WINDOW:
            result = self._predict_raw(frames[-1])
            result["smoothing"] = False
            result["note"] = "insufficient_frames_for_smoothing"
            return result

        history: deque[str] = deque(maxlen=SMOOTHING_WINDOW)
        last_confidence = 0.0
        last_top_k: list[dict] = []
        last_hands = 0

        for frame in frames:
            raw = self._predict_raw(frame)
            if raw.get("prediction") is None:
                # Working script clears history on a no-hand frame.
                history.clear()
                continue

            history.append(raw["prediction"])
            last_confidence = raw["confidence"]
            last_top_k = raw["top_k"]
            last_hands = raw["hands_detected"]

        if len(history) < SMOOTHING_WINDOW:
            return self._no_result("No confident sign (insufficient valid frames).")

        votes = Counter(history)
        top_letter, top_count = votes.most_common(1)[0]

        if top_count < MIN_VOTES:
            result = self._no_result(
                f"No confident sign (majority vote {top_count}/{SMOOTHING_WINDOW})."
            )
            result["top_k"] = last_top_k
            result["smoothing"] = True
            result["smoothing_window"] = SMOOTHING_WINDOW
            result["smoothing_min_votes"] = MIN_VOTES
            result["smoothing_votes"] = top_count
            return result

        return {
            "language": self.language,
            "prediction": top_letter,
            "confidence": float(last_confidence),
            "top_k": last_top_k,
            "hands_detected": last_hands,
            "smoothing": True,
            "smoothing_window": SMOOTHING_WINDOW,
            "smoothing_min_votes": MIN_VOTES,
            "smoothing_votes": top_count,
        }

    def _no_result(self, message: str) -> dict:
        return {
            "language": self.language,
            "prediction": None,
            "confidence": 0.0,
            "top_k": [],
            "hands_detected": 0,
            "error": message,
        }


def load_predictor(language: str) -> LandmarkSignPredictor:
    """Convenience factory returning a ready ``LandmarkSignPredictor``."""
    return LandmarkSignPredictor(language=language)
