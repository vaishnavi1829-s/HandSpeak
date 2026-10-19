"""Rich ISL feature engineering on top of MediaPipe hand landmarks.

This module is a NEW, experimental feature extractor.  It does NOT modify the
existing ``recognition/landmark_extractor.py`` pipeline; instead it REUSES the
existing wrist-relative + scale normalization (``normalize_landmarks``) and the
existing ``lh_*``/``rh_*`` coordinate column names so the rich representation
stays consistent with the rest of the landmark pipeline.

The rich vector adds interpretable geometric features on top of the raw
normalized coordinates.  It supports BOTH one-hand and two-hand ISL signs.

Feature groups (per detected hand, in fixed order)
--------------------------------------------------
A. 63 normalized landmark coordinates   (``lh_lm0_x ... lh_lm20_z`` /
    ``rh_lm0_x ... rh_lm20_z``)
B. 14 finger/joint angles               (``*_thumb_angle_1/2``,
    ``*_{index,middle,ring,pinky}_angle_1/2/3``)
C. 15 finger segment lengths            (``*_{finger}_seg_1/2/3``)
D.  5 wrist-to-fingertip distances      (``*_wrist_{finger}_tip``)
E.  7 fingertip-to-fingertip distances  (``*_tip_{f1}_{f2}``)

Per hand that is 63 + 41 = 104 features.  A full two-hand vector is therefore
104 * 2 = 208 features, plus:

F.  4 cross-hand distances (``cross_wrist_wrist``, ``cross_index_index``,
    ``cross_middle_middle``, ``cross_thumb_thumb``)
G.  2 hand-presence indicators (``left_hand_present``, ``right_hand_present``)

=> 214 features in total (the exact count is printed at runtime and stored in
   ``dataset/landmarks/isl_feature_schema.json``).

Missing-hand representation
---------------------------
If only one hand is detected, the other hand's 104-feature block is all zeros
and its presence flag is 0.  A hand is NEVER fabricated or duplicated.  This is
safe for a RandomForestClassifier because the vector length is constant.

Cross-hand distances are normalized by the average hand-size reference of the
two detected hands so camera distance does not dominate them.  When fewer than
two hands exist, all cross-hand distances are 0.
"""

from __future__ import annotations

import os
import sys

from typing import Optional

import numpy as np

# Allow ``python recognition/isl_feature_extractor.py`` to run from any cwd.
if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from recognition.landmark_extractor import (
    NUM_LANDMARKS,
    REFERENCE_IDX,
    SCALE_EPSILON,
    WRIST_IDX,
    LEFT_HAND_FEATURE_COLUMNS,
    RIGHT_HAND_FEATURE_COLUMNS,
    normalize_landmarks,
)

# ---------------------------------------------------------------------------
# MediaPipe landmark indices (21-landmark hand model).
# ---------------------------------------------------------------------------
WRIST = 0
THUMB = [1, 2, 3, 4]      # CMC, MCP, IP, tip
INDEX = [5, 6, 7, 8]      # MCP, PIP, DIP, tip
MIDDLE = [9, 10, 11, 12]
RING = [13, 14, 15, 16]
PINKY = [17, 18, 19, 20]

# Order used for per-finger angle / length / wrist-distance feature names.
FINGERS = {
    "thumb": THUMB,
    "index": INDEX,
    "middle": MIDDLE,
    "ring": RING,
    "pinky": PINKY,
}

# Fingers that use a 3-joint (MCP/PIP/DIP) angle layout.
_THREE_JOINT_FINGERS = ("index", "middle", "ring", "pinky")

# Fingertip-to-fingertip pairs included in the feature vector.
_TIP_PAIRS = (
    ("thumb", "index"),
    ("thumb", "middle"),
    ("thumb", "ring"),
    ("thumb", "pinky"),
    ("index", "middle"),
    ("middle", "ring"),
    ("ring", "pinky"),
)

CROSS_HAND_FEATURE_NAMES = (
    "cross_wrist_wrist",
    "cross_index_index",
    "cross_middle_middle",
    "cross_thumb_thumb",
)

PRESENCE_FEATURE_NAMES = ("left_hand_present", "right_hand_present")


# ---------------------------------------------------------------------------
# Low-level geometry helpers
# ---------------------------------------------------------------------------
def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return the angle at vertex ``b`` formed by ``a-b-c``, in degrees."""
    ba = a - b
    bc = c - b
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-12:
        return 0.0
    cos_value = float(np.dot(ba, bc) / denom)
    cos_value = max(-1.0, min(1.0, cos_value))
    return float(np.degrees(np.arccos(cos_value)))


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two points."""
    return float(np.linalg.norm(a - b))


def _hand_scale(landmarks: np.ndarray) -> float:
    """Robust hand-size reference (same reference as the base pipeline)."""
    ref = landmarks[REFERENCE_IDX] - landmarks[WRIST_IDX]
    scale = float(np.linalg.norm(ref))
    return scale if scale >= SCALE_EPSILON else SCALE_EPSILON


def _normalized_matrix(landmarks: np.ndarray) -> np.ndarray:
    """Reuse the base wrist-relative + scale normalization, as a (21, 3) matrix."""
    flat = normalize_landmarks(landmarks)
    return flat.reshape(NUM_LANDMARKS, 3)


# ---------------------------------------------------------------------------
# Single-hand feature construction
# ---------------------------------------------------------------------------
def _single_hand_feature_names(prefix: str) -> list[str]:
    """Deterministic feature names for one hand (prefix ``"lh"`` or ``"rh"``)."""
    names: list[str] = []

    # A. Normalized coordinates.
    names.extend(
        LEFT_HAND_FEATURE_COLUMNS if prefix == "lh" else RIGHT_HAND_FEATURE_COLUMNS
    )

    # B. Joint angles.
    names.append(f"{prefix}_thumb_angle_1")
    names.append(f"{prefix}_thumb_angle_2")
    for finger in _THREE_JOINT_FINGERS:
        for j in (1, 2, 3):
            names.append(f"{prefix}_{finger}_angle_{j}")

    # C. Finger segment lengths.
    for finger in ("thumb",) + _THREE_JOINT_FINGERS:
        for j in (1, 2, 3):
            names.append(f"{prefix}_{finger}_seg_{j}")

    # D. Wrist-to-fingertip distances.
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        names.append(f"{prefix}_wrist_{finger}_tip")

    # E. Fingertip-to-fingertip distances.
    for f1, f2 in _TIP_PAIRS:
        names.append(f"{prefix}_tip_{f1}_{f2}")

    return names


def _single_hand_features(landmarks: np.ndarray, prefix: str) -> np.ndarray:
    """Return the 104-feature vector for one hand in deterministic order."""
    norm = _normalized_matrix(landmarks)
    values: list[float] = []

    # A. Normalized coordinates (63).
    values.extend(norm.flatten().tolist())

    # B. Joint angles.
    # Thumb: angle at CMC (wrist-CMC-MCP) and at MCP (CMC-MCP-IP).
    values.append(_angle_deg(norm[WRIST], norm[THUMB[0]], norm[THUMB[1]]))
    values.append(_angle_deg(norm[THUMB[0]], norm[THUMB[1]], norm[THUMB[2]]))
    for finger in _THREE_JOINT_FINGERS:
        idx = FINGERS[finger]
        values.append(_angle_deg(norm[WRIST], norm[idx[0]], norm[idx[1]]))  # MCP
        values.append(_angle_deg(norm[idx[0]], norm[idx[1]], norm[idx[2]]))  # PIP
        values.append(_angle_deg(norm[idx[1]], norm[idx[2]], norm[idx[3]]))  # DIP

    # C. Finger segment lengths.
    for finger in ("thumb",) + _THREE_JOINT_FINGERS:
        idx = FINGERS[finger]
        values.append(_dist(norm[idx[0]], norm[idx[1]]))
        values.append(_dist(norm[idx[1]], norm[idx[2]]))
        values.append(_dist(norm[idx[2]], norm[idx[3]]))

    # D. Wrist-to-fingertip distances.
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        tip = FINGERS[finger][3]
        values.append(_dist(norm[WRIST], norm[tip]))

    # E. Fingertip-to-fingertip distances.
    tips = {finger: FINGERS[finger][3] for finger in FINGERS}
    for f1, f2 in _TIP_PAIRS:
        values.append(_dist(norm[tips[f1]], norm[tips[f2]]))

    return np.asarray(values, dtype=np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_rich_features(
    left_landmarks: Optional[np.ndarray],
    right_landmarks: Optional[np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """Build the rich one/two-hand feature vector.

    Parameters
    ----------
    left_landmarks:
        Raw ``(21, 3)`` MediaPipe normalized ``x/y/z`` coordinates for the LEFT
        hand, or ``None`` when no left hand was detected.
    right_landmarks:
        Raw ``(21, 3)`` coordinates for the RIGHT hand, or ``None``.

    Returns
    -------
    ``(feature_vector, feature_names)`` where ``feature_vector`` is a float32
    array of fixed length (214) and ``feature_names`` is the ordered schema.
    """
    names: list[str] = []
    values: list[float] = []

    left_present = left_landmarks is not None
    right_present = right_landmarks is not None

    _zero = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)

    # Left-hand block (104 features; all zero when missing).
    names.extend(_single_hand_feature_names("lh"))
    lh = left_landmarks if left_present else _zero
    values.extend(_single_hand_features(lh, "lh").tolist())

    # Right-hand block (104 features; all zero when missing).
    names.extend(_single_hand_feature_names("rh"))
    rh = right_landmarks if right_present else _zero
    values.extend(_single_hand_features(rh, "rh").tolist())

    # Cross-hand distances (normalized by the average hand-size reference).
    names.extend(CROSS_HAND_FEATURE_NAMES)
    if left_present and right_present:
        scale = 0.5 * (_hand_scale(left_landmarks) + _hand_scale(right_landmarks))
        if scale < SCALE_EPSILON:
            scale = SCALE_EPSILON
        values.append(_dist(left_landmarks[WRIST], right_landmarks[WRIST]) / scale)
        values.append(_dist(left_landmarks[INDEX[3]], right_landmarks[INDEX[3]]) / scale)
        values.append(_dist(left_landmarks[MIDDLE[3]], right_landmarks[MIDDLE[3]]) / scale)
        values.append(_dist(left_landmarks[THUMB[3]], right_landmarks[THUMB[3]]) / scale)
    else:
        values.extend([0.0, 0.0, 0.0, 0.0])

    # Hand-presence indicators.
    names.extend(PRESENCE_FEATURE_NAMES)
    values.append(1.0 if left_present else 0.0)
    values.append(1.0 if right_present else 0.0)

    return np.asarray(values, dtype=np.float32), names


def rich_features_from_two_hand_result(result) -> tuple[np.ndarray, list[str]]:
    """Convenience wrapper that consumes a ``TwoHandResult`` from the base module."""
    return build_rich_features(result.left_landmarks, result.right_landmarks)


# Build the canonical schema/order once at import time (fixed, deterministic).
_ZERO = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
_, RICH_FEATURE_COLUMNS = build_rich_features(_ZERO, _ZERO)
FEATURE_COUNT = len(RICH_FEATURE_COLUMNS)


def get_schema() -> dict:
    """Return a JSON-serializable description of the rich feature schema."""
    groups = {
        "normalized_coordinates": [
            n
            for n in RICH_FEATURE_COLUMNS
            if n.startswith("lh_lm") or n.startswith("rh_lm")
        ],
        "joint_angles": [n for n in RICH_FEATURE_COLUMNS if "_angle_" in n],
        "finger_segment_lengths": [n for n in RICH_FEATURE_COLUMNS if "_seg_" in n],
        "wrist_to_fingertip_distances": [
            n for n in RICH_FEATURE_COLUMNS if "_wrist_" in n and "_tip" in n
        ],
        "fingertip_to_fingertip_distances": [
            n for n in RICH_FEATURE_COLUMNS if n.startswith(("lh_tip", "rh_tip"))
        ],
        "cross_hand_distances": [
            n for n in RICH_FEATURE_COLUMNS if n.startswith("cross_")
        ],
        "hand_presence": [
            n for n in RICH_FEATURE_COLUMNS if n.endswith("_hand_present")
        ],
    }
    return {
        "feature_count": FEATURE_COUNT,
        "hand_configurations_supported": ["one_hand", "two_hand"],
        "missing_hand_representation": (
            "all-zero 104-feature block for the missing hand; presence flag = 0"
        ),
        "normalization": "wrist-relative + scale by wrist-to-middle-MCP distance",
        "cross_hand_normalization": "average hand-size reference of the two hands",
        "feature_names": RICH_FEATURE_COLUMNS,
        "groups": groups,
    }


if __name__ == "__main__":  # pragma: no cover - quick self-check
    print(f"Rich ISL feature count: {FEATURE_COUNT}")
    print(f"First 5 features: {RICH_FEATURE_COLUMNS[:5]}")
    print(f"Last 5 features:  {RICH_FEATURE_COLUMNS[-5:]}")
