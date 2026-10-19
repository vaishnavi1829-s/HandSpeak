
"""HandSpeak Flask application: ASL + ISL sign language recognition.

This is the STEP 9 (application integration) stage. It exposes a web UI that
supports two SEPARATE recognition modes:

    * ASL (American Sign Language)
        -> MediaPipe one-hand landmarks (63 features)
        -> ``trained_model/asl_landmark_rf.joblib``
    * ISL (Indian Sign Language)
        -> MediaPipe one/two-hand rich landmarks (214 features)
        -> ``trained_model/isl_landmark_rf_rich.joblib``

Both modes use the same landmark-based Random Forest pipeline as the working
VS Code webcam demo (:mod:`recognition.real_time_landmark`). The older CNN
models (``handspeak_cnn*.h5``) are not used by this app.

Run from the project root::

    python app.py
"""

from __future__ import annotations

import os

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from recognition.landmark_predictor import (
    ASL,
    ISL,
    MODEL_PATHS,
    SUPPORTED_LANGUAGES,
    LandmarkSignPredictor,
)

# Match the JS capture loop (5 frames per "Predict Sign" click).
WEBCAM_CAPTURE_COUNT = 5

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload limit

# Lazy-loaded predictors (one per language), cached after first use.
_PREDICTORS: dict[str, LandmarkSignPredictor] = {}


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _get_predictor(language: str) -> LandmarkSignPredictor:
    """Return (and cache) the landmark predictor for ``language``."""
    language = (language or ASL).upper()
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language '{language}'.")
    if language not in _PREDICTORS:
        _PREDICTORS[language] = LandmarkSignPredictor(language=language)
    return _PREDICTORS[language]


def _decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode uploaded image bytes into a BGR ``np.ndarray``."""
    buf = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode the uploaded image.")
    return bgr


def _predict_image_bytes(data: bytes, language: str) -> dict:
    """Decode an uploaded image and return a landmark-based prediction dict."""
    predictor = _get_predictor(language)
    return predictor.predict(_decode_image_bytes(data))


def _predict_webcam_bytes(frames: list[bytes], language: str) -> dict:
    """Decode captured webcam frames and apply temporal smoothing.

    ``frames`` is the ordered list of JPEG frame bytes captured by the browser
    immediately around the "Predict Sign" click.  The landmark predictor applies
    the working script's 5-frame majority-vote smoothing to these frames.
    """
    predictor = _get_predictor(language)
    bgr_frames = [_decode_image_bytes(frame) for frame in frames]
    return predictor.predict_frames(bgr_frames)


@app.route("/")
def index():
    return render_template("index.html", languages=list(SUPPORTED_LANGUAGES))


@app.route("/health")
def health():
    """Liveness probe returning the available models and modes."""
    available = {
        lang: os.path.exists(MODEL_PATHS[lang]) for lang in SUPPORTED_LANGUAGES
    }
    return jsonify(
        {
            "status": "ok",
            "languages": list(SUPPORTED_LANGUAGES),
            "models_available": available,
            "pipeline": "mediapipe-landmarks + random-forest",
        }
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Predict the sign in an uploaded image or a burst of captured webcam frames.

    Multipart form fields:
        image    : the uploaded image file (static image upload).
        frames   : one or more JPEG files (ordered webcam capture burst).
        language : "ASL" or "ISL" (default ASL).

    When ``frames`` is supplied, the working script's 5-frame majority-vote
    smoothing is applied across the burst.  When only a single ``image`` is
    supplied, a single-frame prediction is returned.
    """
    language = (request.form.get("language") or ASL).upper()
    if language not in SUPPORTED_LANGUAGES:
        return jsonify({"error": f"Unsupported language '{language}'."}), 400

    frames = request.files.getlist("frames")

    try:
        if frames:
            if len(frames) != WEBCAM_CAPTURE_COUNT:
                return jsonify(
                    {
                        "error": (
                            f"Expected {WEBCAM_CAPTURE_COUNT} frames for webcam "
                            f"smoothing, got {len(frames)}."
                        )
                    }
                ), 400
            result = _predict_webcam_bytes([f.read() for f in frames], language)
        else:
            if "image" not in request.files:
                return jsonify({"error": "No image file provided."}), 400
            file = request.files["image"]
            if file.filename == "" or not _allowed_file(file.filename):
                return jsonify({"error": "Invalid or unsupported image file."}), 400
            result = _predict_image_bytes(file.read(), language)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    return jsonify(result)


if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(host="127.0.0.1", port=5000, debug=False)
