"""Webcam-frame preprocessing that reproduces the SL-MNIST training format.

The combined 26-class dataset (see :mod:`preprocessing.combine_26_class`) stores
every sample as a 28x28 single-channel image whose pixel values were produced by
dividing raw 0-255 grayscale values by 255.0. In that dataset the **hand is
dark** (low pixel value) and the **background is light** (high pixel value), and
the hand is a crisp, centered, high-contrast silhouette.

Live webcam frames do not share that layout out of the box, so this module
reproduces the training format deterministically:

    1. BGR -> grayscale.
    2. Build a robust foreground mask of the hand using Otsu thresholding plus
       a *border-based polarity decision* (NOT the old "smaller region"
       heuristic, which broke when the hand covered more than half the ROI).
    3. Clean the mask with conservative morphology so finger gaps survive.
    4. Crop tightly around the hand bounding box.
    5. Add controlled padding while keeping the hand centered and preserving
       its aspect ratio (no stretching).
    6. Render a centered, padded, "hand-dark / background-light" crop that
       *preserves the continuous grayscale* of the hand (edges, shading, and
       finger texture) to stay in the SL-MNIST contrast distribution.
    7. Resize to 28x28 (INTER_AREA) and normalize by /255.0.

Hand localization prefers MediaPipe Hands when it is actually installed, and
falls back to a self-contained OpenCV path otherwise (MediaPipe is optional).
"""

from __future__ import annotations

import cv2
import numpy as np

# Final CNN input size (matches ``utils.config.IMAGE_SIZE`` / training).
TARGET_SIZE = 28

# Fraction of the hand bounding-box size added as padding on each side before
# the hand is centered on the square canvas. Mirrors the natural margin in
# SL-MNIST images without swallowing the hand in background.
PADDING_RATIO = 0.20

# Minimum/maximum fraction of the ROI that must be classified as foreground for
# us to trust that a hand is actually present. Values outside this range produce
# ``None`` so the caller can display "No sign".
MIN_FOREGROUND_FRACTION = 0.005
MAX_FOREGROUND_FRACTION = 0.95

# Predictions with a top-class probability below this are reported as
# "No sign" instead of a letter.
CONFIDENCE_THRESHOLD = 0.6

# The SL-MNIST training images are continuous grayscale (their histogram spans
# roughly 25-229, with anti-aliased edges and internal shading). Binarizing the
# hand to a fixed dark/light silhouette was found to destroy that texture and
# cause severe domain-shift misclassification, so the renderer below preserves
# the actual grayscale crop and only normalizes its polarity.


def _to_gray(frame_bgr):
    """Convert a BGR webcam frame to single-channel grayscale."""
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


def _border_mean(gray, thickness=3):
    """Return the mean grayscale value of the frame's outer ring.

    In a centered ROI the border is almost always background, so this is a
    reliable background reference regardless of how large the hand is.
    """
    h, w = gray.shape
    if h < 2 * thickness or w < 2 * thickness:
        return float(gray.mean())
    ring = np.concatenate(
        [
            gray[0:thickness, :].ravel(),
            gray[-thickness:, :].ravel(),
            gray[:, 0:thickness].ravel(),
            gray[:, -thickness:].ravel(),
        ]
    )
    return float(ring.mean()) if ring.size else float(gray.mean())


def _foreground_mask(gray):
    """Return a cleaned binary mask of the hand (foreground == 255) or ``None``.

    The polarity of the hand is determined by comparing the two Otsu-separated
    intensity classes against the border (background) mean: the class whose mean
    is *farthest* from the background is the hand. This works for both a dark
    hand on a light background and a light hand on a dark background, and does
    not assume the hand is smaller than half the frame.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    light_pixels = binary > 0
    dark_pixels = binary == 0
    if not np.any(light_pixels) or not np.any(dark_pixels):
        return None

    border = _border_mean(gray)
    light_mean = float(gray[light_pixels].mean())
    dark_mean = float(gray[dark_pixels].mean())

    # The class closest to the border is background; the other is the hand.
    if abs(light_mean - border) <= abs(dark_mean - border):
        mask = cv2.bitwise_not(binary)  # hand is the dark class -> set to 255
    else:
        mask = binary  # hand is the light class

    # Conservative morphology: open removes speckle, close fills small holes.
    # A small kernel preserves the gaps between fingers (important for V/K/R).
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def _largest_contour_bbox(mask):
    """Return the bounding box ``(x, y, w, h)`` of the largest object.

    Returns ``None`` if the mask contains no usable foreground.
    """
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    if w < 3 or h < 3:
        return None
    return x, y, w, h


def _mediapipe_hand_bbox(gray):
    """Optional MediaPipe-based hand bounding box.

    Returns ``(x, y, w, h)`` or ``None``. The import is attempted lazily and
    tolerates either the classic or modern MediaPipe import layout so the
    OpenCV fallback is used automatically when MediaPipe is unavailable.
    """
    try:  # pragma: no cover - depends on optional dependency
        import mediapipe as mp  # noqa: PLC0415

        hands_cls = getattr(mp.solutions, "hands", None)
        if hands_cls is None:  # older package layout
            from mediapipe.python.solutions import hands as hands_cls  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None

    try:  # pragma: no cover
        with hands_cls.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
        ) as hands:
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            result = hands.process(rgb)
            if not result.multi_hand_landmarks:
                return None

            landmarks = result.multi_hand_landmarks[0].landmark
            h, w = gray.shape[:2]
            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]
            x_min = int(min(xs) * w)
            x_max = int(max(xs) * w)
            y_min = int(min(ys) * h)
            y_max = int(max(ys) * h)
            return x_min, y_min, max(1, x_max - x_min), max(1, y_max - y_min)
    except Exception:  # noqa: BLE001
        return None


def _render_silhouette(gray, mask, bbox):
    """Render a centered, padded, aspect-preserving grayscale hand crop.

    Produces a square grayscale canvas where the hand is dark and the background
    is light, matching the SL-MNIST contrast distribution. Unlike the previous
    binary-silhouette approach, the *actual grayscale values* of the cropped
    hand are preserved (anti-aliased edges, shading, and finger texture all
    survive) and only the polarity is normalized, which keeps the webcam path in
    the same continuous-intensity domain the CNN was trained on.

    The hand is not stretched: the tight crop is padded and centered on the
    square canvas, then the caller resizes the whole square uniformly.
    """
    x, y, w, h = bbox

    # Controlled padding around the hand (clamped to the image bounds).
    pad = int(round(max(w, h) * PADDING_RATIO))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(gray.shape[1], x + w + pad)
    y2 = min(gray.shape[0], y + h + pad)

    crop_gray = gray[y1:y2, x1:x2].astype(np.float32)
    crop_mask = mask[y1:y2, x1:x2] > 0

    if not np.any(crop_mask):
        return None

    # Background reference: the mean of the non-hand pixels inside the padded
    # crop (falls back to 255 if the hand fills the whole crop).
    bg_mean = (
        float(crop_gray[~crop_mask].mean()) if np.any(~crop_mask) else 255.0
    )
    fg_mean = float(crop_gray[crop_mask].mean())

    # Normalize polarity: hand (foreground) must be dark, background light.
    flip = fg_mean > bg_mean
    if flip:
        crop_gray = 255.0 - crop_gray
        bg_fill = 255.0 - bg_mean
    else:
        bg_fill = bg_mean

    # Square canvas, larger side drives the square so the hand is never
    # stretched by the final uniform resize. The canvas background is filled
    # with the measured background intensity (not a hard-coded value).
    side = max(crop_gray.shape[0], crop_gray.shape[1])
    canvas = np.full((side, side), bg_fill, dtype=np.float32)
    y_off = (side - crop_gray.shape[0]) // 2
    x_off = (side - crop_gray.shape[1]) // 2
    canvas[y_off:y_off + crop_gray.shape[0], x_off:x_off + crop_gray.shape[1]] = (
        crop_gray
    )
    return canvas.astype(np.uint8)


def localize_hand(gray):
    """Locate the hand and return a square, padded, dark-hand/light-background crop.

    Returns
    -------
    numpy.ndarray or None
        A square grayscale crop (uint8) with the hand dark and the background
        light, or ``None`` when no hand is detected.
    """
    mask = _foreground_mask(gray)
    if mask is None:
        return None

    fg_fraction = float(np.count_nonzero(mask)) / float(mask.size)
    if fg_fraction < MIN_FOREGROUND_FRACTION:
        return None
    if fg_fraction > MAX_FOREGROUND_FRACTION:
        return None

    bbox = _largest_contour_bbox(mask)
    if bbox is None:
        # MediaPipe is optional and only used when contour detection fails.
        bbox = _mediapipe_hand_bbox(gray)
    if bbox is None:
        return None

    rendered = _render_silhouette(gray, mask, bbox)
    return rendered


def preprocess_frame(frame_bgr, target_size=TARGET_SIZE):
    """Convert a BGR webcam frame into a CNN input tensor.

    Parameters
    ----------
    frame_bgr : numpy.ndarray
        BGR image from ``cv2.VideoCapture``.
    target_size : int
        Height/width to resize to (default 28, matching training).

    Returns
    -------
    numpy.ndarray or None
        A float32 tensor of shape ``(1, target_size, target_size, 1)`` with
        values in [0, 1], or ``None`` when no hand is present.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None

    gray = _to_gray(frame_bgr)
    localized = localize_hand(gray)
    if localized is None:
        return None

    resized = cv2.resize(
        localized, (target_size, target_size), interpolation=cv2.INTER_AREA
    )
    normalized = resized.astype(np.float32) / 255.0
    return normalized.reshape(1, target_size, target_size, 1)
