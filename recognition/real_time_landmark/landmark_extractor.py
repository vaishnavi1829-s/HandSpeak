"""Reusable MediaPipe hand-landmark extraction and normalization utilities.

This module is the single source of truth for the landmark-based hand-sign
pipeline.  The exact same preprocessing/normalization logic is used for:

    1. Dataset generation      (``scripts/extract_landmarks.py``)
    2. Offline testing         (``scripts/test_landmark_models.py``)
    3. Real-time webcam usage  (future, not wired into the live app yet)

Design notes
------------
* MediaPipe Hands returns 21 landmarks.  Each landmark has normalized ``x``,
  ``y`` (0..1 relative to the image) and ``z`` (approximately relative to the
  wrist, in the same scale as ``x``).
* We make the **wrist (landmark 0) the origin** by subtracting the wrist
  coordinates from every landmark.
* We then divide by a robust hand-size reference so that the distance between
  the camera and the hand does not strongly affect the prediction.  The
  reference used here is the Euclidean distance between the wrist (0) and the
  middle-finger MCP (9), clamped to a small epsilon to avoid division by zero.
* The result is a 63-dimensional feature vector::

      lm0_x, lm0_y, lm0_z, lm1_x, ..., lm20_x, lm20_y, lm20_z

  Note ``lm0`` is always (0, 0, 0) after wrist-relative normalization.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

try:
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    _LEGACY_API = False
except Exception:  # pragma: no cover - only for much older mediapipe builds
    mp_python = None
    vision = None
    _LEGACY_API = True

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_LANDMARKS = 21
FEATURE_DIM = NUM_LANDMARKS * 3  # 63

# Landmark 0 is the wrist (origin), landmark 9 is the middle-finger MCP.
WRIST_IDX = 0
REFERENCE_IDX = 9

# Hand-size reference epsilon avoids divide-by-zero on degenerate detections.
SCALE_EPSILON = 1e-6

# Default model asset path (downloaded to the project-local mediapipe folder so
# the pipeline does not rely on a machine-global path).
DEFAULT_MODEL_PATH = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    "mediapipe",
    "hand_landmarker.task",
)

FEATURE_COLUMNS = [
    f"lm{i}_{axis}" for i in range(NUM_LANDMARKS) for axis in ("x", "y", "z")
]

# Two-hand (ISL) feature columns: LEFT hand (lh_*) then RIGHT hand (rh_*).
LEFT_HAND_FEATURE_COLUMNS = [
    f"lh_lm{i}_{axis}" for i in range(NUM_LANDMARKS) for axis in ("x", "y", "z")
]
RIGHT_HAND_FEATURE_COLUMNS = [
    f"rh_lm{i}_{axis}" for i in range(NUM_LANDMARKS) for axis in ("x", "y", "z")
]
TWO_HAND_FEATURE_COLUMNS = LEFT_HAND_FEATURE_COLUMNS + RIGHT_HAND_FEATURE_COLUMNS
TWO_HAND_FEATURE_DIM = len(TWO_HAND_FEATURE_COLUMNS)  # 126


@dataclass
class LandmarkResult:
    """Normalized landmark detection result for a single image.

    Attributes
    ----------
    feature_vector:
        1-D ``np.ndarray`` of shape ``(63,)`` with wrist-relative, scale-
        normalized landmark coordinates.
    landmarks:
        Raw ``(21, 3)`` ``np.ndarray`` of normalized ``x/y/z`` values before
        wrist-relative normalization (useful for debugging/visualization).
    handedness:
        ``"Left"``, ``"Right"``, or ``None`` when unavailable.
    hand_score:
        Detection confidence of the selected hand, or ``None``.
    """

    feature_vector: np.ndarray
    landmarks: np.ndarray
    handedness: Optional[str] = None
    hand_score: Optional[float] = None


# ---------------------------------------------------------------------------
# MediaPipe model singleton
# ---------------------------------------------------------------------------
_HANDS_INSTANCE: Optional[object] = None
_HANDS_MODEL_PATH: Optional[str] = None


def _create_hands(model_path: str, num_hands: int = 1):
    """Create a MediaPipe ``HandLandmarker`` using the Tasks Python API.

    Parameters
    ----------
    model_path:
        Path to a ``hand_landmarker.task`` model.
    num_hands:
        Maximum number of hands to detect (1 for ASL, 2 for ISL two-hand).
    """
    if _LEGACY_API or mp_python is None or vision is None:
        raise RuntimeError(
            "This pipeline requires the MediaPipe Tasks API "
            "('mediapipe.tasks.python.vision.HandLandmarker')."
        )
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=num_hands,
        min_hand_detection_confidence=0.2,
        min_hand_presence_confidence=0.2,
        min_tracking_confidence=0.2,
    )
    return vision.HandLandmarker.create_from_options(options)


def get_hand_landmarker(model_path: Optional[str] = None):
    """Return a cached one-hand ``HandLandmarker`` instance.

    Parameters
    ----------
    model_path:
        Path to a ``hand_landmarker.task`` model.  Defaults to the
        project-local ``mediapipe/hand_landmarker.task``.

    Returns
    -------
    ``mediapipe.tasks.python.vision.HandLandmarker``
    """
    global _HANDS_INSTANCE, _HANDS_MODEL_PATH

    model_path = model_path or DEFAULT_MODEL_PATH
    if _HANDS_INSTANCE is None or _HANDS_MODEL_PATH != model_path:
        _HANDS_INSTANCE = _create_hands(model_path, num_hands=1)
        _HANDS_MODEL_PATH = model_path
    return _HANDS_INSTANCE


# Two-hand landmarker singleton (separate from the one-hand singleton).
_HANDS_INSTANCE_2: Optional[object] = None
_HANDS_MODEL_PATH_2: Optional[str] = None


def get_two_hand_landmarker(model_path: Optional[str] = None):
    """Return a cached two-hand ``HandLandmarker`` instance.

    This is used by the ISL two-hand pipeline (max 2 detected hands).
    """
    global _HANDS_INSTANCE_2, _HANDS_MODEL_PATH_2

    model_path = model_path or DEFAULT_MODEL_PATH
    if _HANDS_INSTANCE_2 is None or _HANDS_MODEL_PATH_2 != model_path:
        _HANDS_INSTANCE_2 = _create_hands(model_path, num_hands=2)
        _HANDS_MODEL_PATH_2 = model_path
    return _HANDS_INSTANCE_2


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
def _mp_image_from_bgr(bgr: np.ndarray):
    """Convert a BGR OpenCV image to a MediaPipe Tasks SRGB ``Image``."""
    if bgr is None:
        raise ValueError("Image is None.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def _landmarks_to_matrix(hand_landmarks) -> np.ndarray:
    """Convert a MediaPipe hand-landmarks list to a ``(21, 3)`` matrix."""
    coords = np.empty((NUM_LANDMARKS, 3), dtype=np.float32)
    for i, lm in enumerate(hand_landmarks):
        coords[i, 0] = lm.x
        coords[i, 1] = lm.y
        coords[i, 2] = lm.z
    return coords


def _handedness_label(result) -> Optional[str]:
    """Return ``"Left"``/``"Right"`` for the first hand, else ``None``."""
    try:
        if not result.handedness:
            return None
        first = result.handedness[0]
        if not first:
            return None
        return first[0].display_name
    except Exception:
        return None


def _hand_score(result) -> Optional[float]:
    """Return the detection confidence for the first hand, else ``None``."""
    try:
        if not result.handedness:
            return None
        first = result.handedness[0]
        if not first:
            return None
        return float(first[0].score)
    except Exception:
        return None


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Apply wrist-relative + scale normalization to raw landmarks.

    Parameters
    ----------
    landmarks:
        ``(21, 3)`` array of normalized ``x/y/z`` coordinates.

    Returns
    -------
    ``np.ndarray``
        Flattened ``(63,)`` feature vector with ``lm0`` at the origin.
    """
    if landmarks.shape != (NUM_LANDMARKS, 3):
        raise ValueError(
            f"Expected landmarks shape (21, 3), got {landmarks.shape}."
        )

    wrist = landmarks[WRIST_IDX].copy()

    # 1) Wrist-relative translation.
    centered = landmarks - wrist

    # 2) Scale normalization using a robust hand-size reference.
    ref_vec = centered[REFERENCE_IDX]
    scale = float(np.linalg.norm(ref_vec))
    if scale < SCALE_EPSILON:
        scale = SCALE_EPSILON
    normalized = centered / scale

    return normalized.astype(np.float32).flatten()


def extract_landmarks_from_bgr(
    bgr: np.ndarray,
    landmarker=None,
    model_path: Optional[str] = None,
) -> Optional[LandmarkResult]:
    """Detect one hand in a BGR image and return normalized landmarks.

    Parameters
    ----------
    bgr:
        OpenCV BGR image (any size; MediaPipe resizes internally).
    landmarker:
        Optional pre-created ``HandLandmarker``.  If omitted, a cached
        singleton is used.
    model_path:
        Optional model path used when creating the cached landmarker.

    Returns
    -------
    ``LandmarkResult`` or ``None`` if no hand was detected.
    """
    if bgr is None:
        return None

    landmarker = landmarker or get_hand_landmarker(model_path)
    mp_image = _mp_image_from_bgr(bgr)
    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None

    hand = result.hand_landmarks[0]
    raw = _landmarks_to_matrix(hand)
    features = normalize_landmarks(raw)

    return LandmarkResult(
        feature_vector=features,
        landmarks=raw,
        handedness=_handedness_label(result),
        hand_score=_hand_score(result),
    )


def extract_landmarks_from_file(
    image_path: str,
    landmarker=None,
    model_path: Optional[str] = None,
) -> Optional[LandmarkResult]:
    """Read an image from disk and extract normalized landmarks.

    Returns ``None`` when the file cannot be read or no hand is detected.
    """
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return extract_landmarks_from_bgr(bgr, landmarker=landmarker, model_path=model_path)


# ---------------------------------------------------------------------------
# Two-hand (ISL) extraction
# ---------------------------------------------------------------------------
@dataclass
class TwoHandResult:
    """Normalized two-hand detection result.

    Attributes
    ----------
    feature_vector:
        1-D ``np.ndarray`` of shape ``(126,)`` with the LEFT hand's 63
        normalized features followed by the RIGHT hand's 63 normalized
        features.  A missing hand is represented by an all-zero block.
    left_landmarks:
        Raw ``(21, 3)`` normalized left-hand coordinates, or ``None``.
    right_landmarks:
        Raw ``(21, 3)`` normalized right-hand coordinates, or ``None``.
    left_present:
        Whether a left hand was detected.
    right_present:
        Whether a right hand was detected.
    """

    feature_vector: np.ndarray
    left_landmarks: Optional[np.ndarray] = None
    right_landmarks: Optional[np.ndarray] = None
    left_present: bool = False
    right_present: bool = False


def _handedness_by_index(result, index: int) -> Optional[str]:
    """Return the handedness display name for hand ``index``, else ``None``."""
    try:
        if not result.handedness:
            return None
        group = result.handedness[index]
        if not group:
            return None
        return group[0].display_name
    except Exception:
        return None


def _hand_score_by_index(result, index: int) -> Optional[float]:
    """Return the detection confidence for hand ``index``, else ``None``."""
    try:
        if not result.handedness:
            return None
        group = result.handedness[index]
        if not group:
            return None
        return float(group[0].score)
    except Exception:
        return None


def extract_two_hands_from_bgr(
    bgr: np.ndarray,
    landmarker=None,
    model_path: Optional[str] = None,
) -> Optional[TwoHandResult]:
    """Detect up to two hands and build a deterministic 126-feature vector.

    The feature order is ALWAYS::

        lh_lm0_x ... lh_lm20_z  (LEFT hand, 63 features)
        rh_lm0_x ... rh_lm20_z  (RIGHT hand, 63 features)

    Hands are assigned to LEFT/RIGHT using MediaPipe handedness, NOT detection
    order.  If a hand is missing, its 63-feature block is all zeros.  Returns
    ``None`` only when MediaPipe detects zero hands.
    """
    if bgr is None:
        return None

    landmarker = landmarker or get_two_hand_landmarker(model_path)
    mp_image = _mp_image_from_bgr(bgr)
    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None

    left_features = np.zeros(FEATURE_DIM, dtype=np.float32)
    right_features = np.zeros(FEATURE_DIM, dtype=np.float32)
    left_raw: Optional[np.ndarray] = None
    right_raw: Optional[np.ndarray] = None
    left_present = False
    right_present = False

    for index in range(min(len(result.hand_landmarks), 2)):
        handedness = _handedness_by_index(result, index)
        raw = _landmarks_to_matrix(result.hand_landmarks[index])
        features = normalize_landmarks(raw)

        if handedness == "Left":
            left_features = features
            left_raw = raw
            left_present = True
        elif handedness == "Right":
            right_features = features
            right_raw = raw
            right_present = True
        else:
            # Fallback: if handedness is unavailable, assign by detection order
            # (index 0 -> left slot) so we still produce a usable vector rather
            # than dropping data.
            if not left_present:
                left_features = features
                left_raw = raw
                left_present = True
            elif not right_present:
                right_features = features
                right_raw = raw
                right_present = True

    feature_vector = np.concatenate([left_features, right_features]).astype(np.float32)

    return TwoHandResult(
        feature_vector=feature_vector,
        left_landmarks=left_raw,
        right_landmarks=right_raw,
        left_present=left_present,
        right_present=right_present,
    )


def extract_two_hands_from_file(
    image_path: str,
    landmarker=None,
    model_path: Optional[str] = None,
) -> Optional[TwoHandResult]:
    """Read an image from disk and extract a deterministic two-hand feature vector.

    Returns ``None`` when the file cannot be read or zero hands are detected.
    """
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return extract_two_hands_from_bgr(bgr, landmarker=landmarker, model_path=model_path)
