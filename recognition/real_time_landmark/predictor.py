"""Language-aware sign predictor shared by the live demo and the Flask app.

This module owns model loading and per-frame prediction for BOTH recognition
modes:

    * ASL (American Sign Language) -> ``trained_model/handspeak_cnn_webcam.h5``
    * ISL (Indian Sign Language)  -> ``trained_model/handspeak_cnn_isl.h5``

Both models share the same architecture (28x28x1 input, 26 output classes) and
the same ``recognition.preprocessing`` pipeline, so the only difference between
modes is which weights file is loaded.

This separation satisfies the "ASL + ISL separate modes" requirement: a caller
instantiates a ``SignPredictor`` per language and the application routes frames
to the correct model explicitly.
"""

from __future__ import annotations

from collections import Counter, deque

import numpy as np

from keras.models import load_model

from recognition.preprocessing import CONFIDENCE_THRESHOLD, preprocess_frame
from utils.config import (
    IMAGE_SIZE,
    MODEL_WEBCAM_PATH,
    MODEL_ISL_PATH,
)
from utils.label_mapping import LETTERS

# Language identifiers accepted throughout the application.
ASL = "ASL"
ISL = "ISL"
SUPPORTED_LANGUAGES = (ASL, ISL)

# Model path per language.
LANGUAGE_MODEL_PATHS = {
    ASL: MODEL_WEBCAM_PATH,
    ISL: MODEL_ISL_PATH,
}

# Temporal smoothing constants (same behaviour as the ASL-only demo).
SMOOTHING_WINDOW = 7
MIN_VOTES = 4


class SignPredictor:
    """Loads one sign-language model and predicts letters from webcam ROIs.

    Parameters
    ----------
    language : str
        ``"ASL"`` or ``"ISL"``.
    confidence_threshold : float
        Minimum top-class probability required to report a letter.
    """

    def __init__(self, language: str = ASL,
                 confidence_threshold: float = CONFIDENCE_THRESHOLD):
        language = (language or ASL).upper()
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{language}'. "
                f"Choose one of {SUPPORTED_LANGUAGES}."
            )

        self.language = language
        self.confidence_threshold = confidence_threshold
        self.model_path = LANGUAGE_MODEL_PATHS[language]
        self.letters = list(LETTERS)
        self._history: deque = deque(maxlen=SMOOTHING_WINDOW)
        self._model = self._load_model()

    def _load_model(self):
        import os

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found for '{self.language}': {self.model_path}. "
                "Train the model first (scripts/train_webcam_cnn.py for ASL, "
                "scripts/train_isl_cnn.py for ISL)."
            )
        print(f"[{self.language}] loading model from {self.model_path}")
        return load_model(self.model_path)

    def predict(self, roi: np.ndarray):
        """Return ``(letter, confidence)`` or ``(None, None)`` for a BGR ROI.

        The ROI is preprocessed with the shared ``recognition.preprocessing``
        pipeline, then passed through the loaded model. Low-confidence
        predictions are suppressed to ``None``.
        """
        input_tensor = preprocess_frame(roi, target_size=IMAGE_SIZE)
        if input_tensor is None:
            return None, None

        probabilities = self._model.predict(input_tensor, verbose=0)[0]
        class_index = int(np.argmax(probabilities))
        confidence = float(probabilities[class_index])

        if confidence < self.confidence_threshold:
            return None, None

        return self.letters[class_index], confidence

    def predict_smoothed(self, roi: np.ndarray):
        """Predict and apply temporal majority-vote smoothing.

        Returns
        -------
        tuple[str, float] | tuple[None, None]
            The smoothed letter and its average confidence, or ``(None, None)``
            when there is no stable majority.
        """
        letter, confidence = self.predict(roi)
        self._history.append((letter, confidence))

        letters = [entry[0] for entry in self._history if entry[0] is not None]
        if not letters:
            return None, None

        counts = Counter(letters)
        top_letter, votes = counts.most_common(1)[0]
        if votes < MIN_VOTES:
            return None, None

        confs = [entry[1] for entry in self._history if entry[0] == top_letter]
        avg_conf = float(sum(confs) / len(confs)) if confs else 0.0
        return top_letter, avg_conf

    def reset_history(self) -> None:
        """Clear the temporal smoothing window (e.g. on mode/language switch)."""
        self._history.clear()


def load_predictor(language: str) -> SignPredictor:
    """Convenience factory returning a ready ``SignPredictor``."""
    return SignPredictor(language=language)
