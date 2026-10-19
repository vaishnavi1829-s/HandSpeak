# ISL Orientation Fix

## Summary

The real-time recognition pipeline was applying a horizontal mirror to every
webcam frame. This was correct for ASL (whose collection script mirrors frames),
but incorrect for ISL (whose collection scripts save raw, unflipped frames).
This caused a training/inference domain shift for ISL: the model was trained on
unflipped images but received flipped images at inference time.

The fix makes mirroring language-aware:

- **ASL** → inference frame is mirrored (unchanged behavior).
- **ISL** → inference frame is unmirrored (matches training data).
- The **preview** stays mirrored for a natural webcam feel in both modes.

## Previous data flow

```
Camera
  ↓
frame capture
  ↓
cv2.flip(frame, 1)          # applied UNCONDITIONALLY
  ↓
ROI crop (flipped)
  ↓
MediaPipe / CNN             # received FLIPPED input for BOTH ASL and ISL
  ↓
ISL model                   # trained on UNFLIPPED images
```

Result: ISL training data is unflipped, but inference received flipped frames.

## Corrected data flow

### ASL (unchanged inference orientation)

```
Camera
  ↓
frame capture
  ↓
inference_frame = cv2.flip(frame, 1)
  ↓
ROI crop (flipped)
  ↓
ASL model / MediaPipe       # flipped input (matches ASL collection)
```

### ISL (now unflipped inference)

```
Camera
  ↓
frame capture
  ↓
inference_frame = frame      # NO horizontal flip
  ↓
ROI crop (unflipped)
  ↓
ISL model / MediaPipe        # unflipped input (matches ISL collection)
```

### Display (both modes)

```
Camera
  ↓
frame capture
  ↓
display_frame = cv2.flip(frame, 1)   # always mirrored for natural preview
  ↓
overlay + cv2.imshow
```

## Files changed

### 1. [`recognition/real_time_dual.py`](recognition/real_time_dual.py)

- Split the single flipped `frame` into two variables:
  - `inference_frame`: flipped only when `predictor.language == ASL`; raw for ISL.
  - `display_frame`: always mirrored.
- The ROI sent to `predictor.predict_smoothed()` is cropped from
  `inference_frame`, so ISL CNN inference now uses unflipped pixels.
- The green ROI rectangle, overlay text, and `cv2.imshow()` use `display_frame`,
  so the preview remains mirrored.

### 2. [`recognition/real_time_landmark.py`](recognition/real_time_landmark.py)

- Renamed the captured frame to `camera_frame`.
- `inference_frame`: flipped only for `active_mode == "ASL"`; raw for ISL and
  ISL Rich.
- `display_frame`: always mirrored.
- `extract_landmarks_from_bgr()` (ASL) and `extract_two_hands_from_bgr()`
  (ISL/ISL Rich) now receive `inference_frame`, so ISL landmark extraction uses
  the unflipped frame.
- Added `mirror_x` support to `draw_landmarks()`. Because ISL landmarks are now
  computed on the unflipped frame but drawn on the mirrored preview, their
  x-coordinates are mirrored (`x_display = 1 - x`) before drawing so the overlay
  still aligns visually.

### 3. Left unchanged intentionally

- [`recognition/real_time_recognition.py`](recognition/real_time_recognition.py)
  (ASL-only CNN demo) keeps its unconditional flip. It does not serve ISL.
- [`scripts/collect_isl_webcam_v2.py`](scripts/collect_isl_webcam_v2.py) (V2
  collector) is not modified and continues to save unflipped ROI images.
- [`scripts/collect_webcam_dataset.py`](scripts/collect_webcam_dataset.py)
  (ASL collector) keeps its flip.
- [`recognition/landmark_extractor.py`](recognition/landmark_extractor.py) is
  not modified; it already receives whatever frame the caller passes.

## Why the mismatch mattered

Many ISL signs are asymmetric. A horizontal flip reverses the sign's
left/right orientation, so a model trained on unflipped images sees a different
visual configuration at inference time. This is a domain shift that degrades
accuracy for asymmetric letters even when training and inference preprocessing
are otherwise identical.

## How ASL and ISL now differ

| | ASL | ISL |
|---|---|---|
| Inference frame | Mirrored (`cv2.flip(..., 1)`) | Raw (no flip) |
| Preview frame | Mirrored | Mirrored |
| MediaPipe input | Mirrored | Unflipped |
| Model input | Mirrored | Unflipped |
| Matches collection script | Yes (ASL collector flips) | Yes (ISL collectors do not flip) |

## MediaPipe handedness after the change

No handedness reordering was added. MediaPipe's handedness assumes mirrored
(selfie) input:

- ASL path still feeds a mirrored frame, so handedness labels remain consistent
  with the user's actual left/right.
- ISL path now feeds an unflipped frame, so MediaPipe handedness labels are
  inverted relative to the user's actual left/right. This is the same condition
  as the V2 collector (which also feeds unflipped frames), so ISL inference and
  ISL training data now agree with each other.

No dataset or model was modified.

## Verification results

1. **Syntax check**: `python -m py_compile recognition/real_time_dual.py
   recognition/real_time_landmark.py` → exit code 0.
2. **Import check**: `import recognition.real_time_dual` and
   `import recognition.real_time_landmark` → "import OK", exit code 0.
3. **ASL path**: [`recognition/real_time_dual.py`](recognition/real_time_dual.py:85)
   applies `cv2.flip(frame, 1)` when `predictor.language == ASL`.
4. **ISL CNN path**: [`recognition/real_time_dual.py`](recognition/real_time_dual.py:100)
   crops the ROI from `inference_frame`, which is the raw frame for ISL.
5. **ISL landmark path**: [`recognition/real_time_landmark.py`](recognition/real_time_landmark.py:255)
   passes `inference_frame` (raw for ISL) to `extract_two_hands_from_bgr()`.
6. **Preview**: [`recognition/real_time_dual.py`](recognition/real_time_dual.py:91)
   and [`recognition/real_time_landmark.py`](recognition/real_time_landmark.py:213)
   both build `display_frame` with a flip, keeping the preview mirrored.
