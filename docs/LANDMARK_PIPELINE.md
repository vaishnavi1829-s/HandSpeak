# Landmark-Based Hand Sign Recognition Pipeline

This document describes the experimental MediaPipe hand-landmark pipeline that
runs **alongside** (not instead of) the existing CNN pipeline.  The CNN models
and datasets are left completely untouched.

## Why MediaPipe landmarks?

The existing CNN works on 28x28 grayscale images.  Landmarks are a complementary,
lightweight representation: MediaPipe estimates the 3D position of 21 key points
on the hand.  This representation is:

- **Compact** — only 63 numbers per frame (vs. 784 pixels for the CNN input).
- **Robust to background/lighting** — it tracks hand geometry rather than pixels.
- **Fast to train** — a Random Forest trains in seconds on CPU.
- **Interpretable** — each feature corresponds to a real anatomical point.

## The 21 hand landmarks

MediaPipe Hands returns 21 landmarks per detected hand:

| Index | Landmark |
|-------|----------|
| 0 | Wrist |
| 1–4 | Thumb (CMC, MCP, IP, tip) |
| 5–8 | Index finger (MCP, PIP, DIP, tip) |
| 9–12 | Middle finger (MCP, PIP, DIP, tip) |
| 13–16 | Ring finger (MCP, PIP, DIP, tip) |
| 17–20 | Pinky (MCP, PIP, DIP, tip) |

Each landmark has normalized `x`, `y` (relative to the image) and `z`
(depth, roughly in the same scale as `x`).

## 63 base x/y/z features

Every image is converted to a feature vector:

```
lm0_x, lm0_y, lm0_z, lm1_x, lm1_y, lm1_z, ..., lm20_x, lm20_y, lm20_z
```

That is **21 points × 3 coordinates = 63 features**.

## Wrist-relative normalization

To make the features invariant to where the hand appears in the frame and how
far it is from the camera:

1. **Translation** — subtract the wrist (`landmark 0`) coordinates from every
   landmark.  This makes the wrist the origin (`lm0 = 0, 0, 0`).
2. **Scale normalization** — divide by the Euclidean distance between the wrist
   (0) and the middle-finger MCP (9).  This is a robust hand-size reference, so
   distance from the camera does not strongly affect prediction.

The exact same normalization is applied during dataset generation, offline
testing, and (future) real-time prediction via the shared module
[`recognition/landmark_extractor.py`](../recognition/landmark_extractor.py).

## ASL One-Hand vs ISL Two-Hand Architecture

ASL and ISL use **different hand configurations** in this project:

| Language | Hands | Landmarks | Features | Model |
|----------|-------|-----------|----------|-------|
| ASL | 1 hand | 21 | 63 | `trained_model/asl_landmark_rf.joblib` |
| ISL (new) | 2 hands | 42 total | 126 | `trained_model/isl_landmark_rf_2hand.joblib` |
| ISL (legacy one-hand) | 1 hand | 21 | 63 | `trained_model/isl_landmark_rf.joblib` (preserved) |

**Why different hand configurations?**  In this project's dataset, ASL letters
are captured as a single dominant-hand sign, while ISL is being modeled as a
two-hand system (left + right hand) because several ISL alphabet signs are
two-handed.  The two languages are therefore kept fully separate with distinct
feature layouts:

- ASL: `lm0_x ... lm20_z` (63 features)
- ISL two-hand: `lh_lm0_x ... lh_lm20_z` then `rh_lm0_x ... rh_lm20_z` (126 features)

The legacy one-hand ISL model and CSV remain untouched as a backup.

## ASL and ISL are trained separately

ASL and ISL can have **different mappings** between a hand shape and a letter.
Therefore the pipeline produces completely independent datasets, models, and
evaluation artifacts:

| Language | Dataset | Model |
|----------|---------|-------|
| ASL | `dataset/landmarks/asl_landmarks.csv` | `trained_model/asl_landmark_rf.joblib` |
| ISL (one-hand, legacy) | `dataset/landmarks/isl_landmarks.csv` | `trained_model/isl_landmark_rf.joblib` |
| ISL (two-hand, new) | `dataset/landmarks/isl_landmarks_2hand.csv` | `trained_model/isl_landmark_rf_2hand.joblib` |

## Random Forest classifier

The first baseline uses `RandomForestClassifier`:

- `n_estimators=400`
- `class_weight="balanced"` (the classes are not perfectly balanced)
- `random_state=42` for reproducibility

## Dataset generation process

[`scripts/extract_landmarks.py`](../scripts/extract_landmarks.py) iterates over
the existing webcam ROI images:

- ASL: `dataset/webcam/<LETTER>/roi/*.png`
- ISL: `dataset/isl_webcam/<LETTER>/roi/*.png`

For each image it detects one hand, extracts 21 landmarks, normalizes them, and
writes one CSV row.  Images where MediaPipe cannot detect a hand are **logged**
to `dataset/landmarks/failed/<lang>_failed.json` and a summary is written next to
each CSV.

## Train/test methodology

[`scripts/train_landmark_models.py`](../scripts/train_landmark_models.py) uses a
**stratified** train/validation/test split (`80/15/5` of the detected samples,
with `test_size=0.2` and `val_size=0.15`) so every letter is represented in every
split.  The split is deterministic thanks to `random_state=42` and identical CSV
ordering.  Training uses only the corresponding language dataset, so there is no
data leakage between ASL and ISL.

## Evaluation metrics

[`scripts/evaluate_landmark_models.py`](../scripts/evaluate_landmark_models.py)
recomputes the same stratified test split and reports, per language:

- Test accuracy
- Macro and weighted precision / recall / F1-score
- Full classification report
- Confusion matrix (PNG + CSV)
- Per-letter accuracy
- `*_metrics.json` summary

Outputs live under `results/landmark/`.

## How to run

### 1. Extract landmarks

```bash
# Both ASL and ISL
python scripts/extract_landmarks.py

# Or one at a time
python scripts/extract_landmarks.py --dataset asl
python scripts/extract_landmarks.py --dataset isl
```

### 2. Train the models

```bash
# Both ASL and ISL
python scripts/train_landmark_models.py

# Or one at a time
python scripts/train_landmark_models.py --dataset asl
python scripts/train_landmark_models.py --dataset isl
```

### 3. Evaluate the models

```bash
python scripts/evaluate_landmark_models.py
python scripts/evaluate_landmark_models.py --dataset asl
python scripts/evaluate_landmark_models.py --dataset isl
```

### 4. Offline prediction test

```bash
# ASL sample across several letters
python scripts/test_landmark_models.py --dataset asl

# ISL sample
python scripts/test_landmark_models.py --dataset isl

# A single image
python scripts/test_landmark_models.py --dataset asl --image dataset/webcam/A/roi/A_000002.png

# A whole folder
python scripts/test_landmark_models.py --dataset isl --folder dataset/isl_webcam/B/roi
```

## Real-Time Webcam Recognition

The live recognizer is a **separate** script from the existing CNN webcam app:

```bash
python recognition/real_time_landmark.py
```

### How the webcam pipeline works

```
Webcam
   ↓
OpenCV frame
   ↓
MediaPipe Hand Landmarker
   ↓
ASL: 1 hand  (21 landmarks -> 63 features)
ISL: 2 hands (left + right -> 126 features)
   ↓
Same wrist-relative + scale normalization used during training
   ↓
ASL or ISL Random Forest
   ↓
A-Z prediction
   ↓
Display letter + confidence on screen
```

The frame is mirrored for a natural feel.  Landmarks and hand connections are
drawn on top of the feed so detection can be visually verified.

### ASL / ISL mode selection

On startup the program prints a menu:

```
1 - ASL
2 - ISL
Q - Quit
```

Selecting `1` loads [`trained_model/asl_landmark_rf.joblib`](../trained_model/asl_landmark_rf.joblib)
(one-hand, 63 features).  Selecting `2` loads
[`trained_model/isl_landmark_rf_2hand.joblib`](../trained_model/isl_landmark_rf_2hand.joblib)
(two-hand, 126 features).  The two models are never mixed.  During runtime you
can switch with the `1` and `2` keys (the prediction history is reset on
switch).

### Temporal smoothing

Raw per-frame predictions flicker, so the script keeps the latest
`SMOOTHING_WINDOW` (default 7) predictions in a deque and applies majority
voting.  A letter is displayed only when it appears at least `MIN_VOTES`
(default 4) times in the window, which stabilizes the output.

### Confidence threshold

A per-frame confidence below `CONFIDENCE_THRESHOLD` (default 0.65) suppresses
the prediction and displays "Low confidence" / "No confident sign".  This avoids
committing to a guess on ambiguous frames.

### Keyboard controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `R` | Reset prediction history |
| `1` | Switch to ASL model |
| `2` | Switch to ISL model |

### Configuration

The top of [`recognition/real_time_landmark.py`](../recognition/real_time_landmark.py)
exposes these tunables:

- `CAMERA_INDEX = 0`
- `CONFIDENCE_THRESHOLD = 0.65`
- `SMOOTHING_WINDOW = 7`

## ISL Rich Feature Engineering

The one-hand and two-hand ISL models above use only raw wrist-relative landmark
coordinates.  Some ISL signs are distinguished primarily by **finger
configuration / hand geometry** (e.g. which fingers are folded, how far the
thumb is from the index, whether the hands touch).  The rich-feature pipeline is
an experiment to see whether explicit geometric features help.

### Feature extractor

The reusable module [`recognition/isl_feature_extractor.py`](../recognition/isl_feature_extractor.py)
builds a deterministic 214-feature vector from MediaPipe landmarks.  It supports
**one hand or two hands** and reuses the existing wrist-relative + scale
normalization from [`recognition/landmark_extractor.py`](../recognition/landmark_extractor.py).

Per hand (104 features each):

| Group | Count | Description |
|-------|-------|-------------|
| Normalized coordinates | 63 | `lh_lm0_x ... lh_lm20_z` / `rh_lm0_x ... rh_lm20_z` |
| Joint angles | 14 | thumb (2 angles), index/middle/ring/pinky (3 angles each, at MCP/PIP/DIP) |
| Finger segment lengths | 15 | 3 segments × 5 fingers (thumb, index, middle, ring, pinky) |
| Wrist-to-fingertip distances | 5 | wrist → each fingertip |
| Fingertip-to-fingertip distances | 7 | thumb↔index/middle/ring/pinky, index↔middle, middle↔ring, ring↔pinky |

Then, across both hands:

| Group | Count | Description |
|-------|-------|-------------|
| Cross-hand distances | 4 | wrist↔wrist, index↔index, middle↔middle, thumb↔thumb (normalized by average hand size) |
| Hand-presence flags | 2 | `left_hand_present`, `right_hand_present` (from MediaPipe handedness, not detection order) |

Total = 104 + 104 + 4 + 2 = **214 features**.

### Missing-hand representation

If only one hand is detected, the other hand's 104-feature block is all zeros
and its presence flag is 0.  A hand is **never fabricated or duplicated**, which
is safe for the fixed-length Random Forest input.

### Feature schema

The canonical feature order is stored in
[`dataset/landmarks/isl_feature_schema.json`](../dataset/landmarks/isl_feature_schema.json)
and is used identically for training, validation, offline testing, and real-time
prediction.

### Dataset

[`scripts/extract_isl_rich_features.py`](../scripts/extract_isl_rich_features.py)
writes [`dataset/landmarks/isl_rich_features.csv`](../dataset/landmarks/isl_rich_features.csv)
with columns `label`, `source_path`, `hands_detected`, `left_hand`, `right_hand`,
and all 214 features.  All A-Z classes are kept; zero-hand images are logged to
`dataset/landmarks/failed/isl_rich_failed.json` (not silently dropped from the
report).

### Random Forest classification

[`scripts/train_isl_rich_model.py`](../scripts/train_isl_rich_model.py) trains a
`RandomForestClassifier` (400 trees, `class_weight="balanced"`,
`random_state=42`) on the 214 features.  The model is saved to
`trained_model/isl_landmark_rf_rich.joblib` and a deterministic stratified
train/val/test split (including source paths) is saved to
`dataset/landmarks/isl_rich_split.json` so the same test set can be reproduced.

### Comparison with the previous landmark model

[`scripts/compare_isl_rich_vs_old.py`](../scripts/compare_isl_rich_vs_old.py)
evaluates both the old one-hand model (`isl_landmark_rf.joblib`) and the new
rich model on a **common** held-out test set (the intersection of their test
splits) so the comparison has no data leakage.  It writes:

- `results/landmark/isl_rich_classification_report.txt`
- `results/landmark/isl_rich_confusion_matrix.png` / `.csv`
- `results/landmark/isl_rich_metrics.json`
- `results/landmark/isl_rich_vs_old_comparison.txt`
- `results/landmark/isl_rich_vs_old_per_letter.csv`

### Rich-feature commands

```bash
# 1. Extract rich ISL features (writes isl_rich_features.csv + schema)
python scripts/extract_isl_rich_features.py

# 2. Train the new rich-feature ISL model
python scripts/train_isl_rich_model.py

# 3. Compare old vs new model on the same test set
python scripts/compare_isl_rich_vs_old.py

# 4. Test a single ISL image with the rich model
python scripts/test_landmark_models.py --dataset isl_rich --image dataset/isl_webcam/C/roi/C_000001.png
```

### Real-time rich-feature mode

[`recognition/real_time_landmark.py`](../recognition/real_time_landmark.py) now
has a third selectable mode:

```
1 - ASL (one hand)
2 - ISL (two hands)
3 - ISL Rich Features (one/two hands, geometric features)
```

ISL Rich mode detects one or two hands, builds the same 214-feature vector, and
loads `trained_model/isl_landmark_rf_rich.joblib`.  Landmarks/connections are
drawn for every detected hand, and temporal smoothing is preserved.

## Two-hand ISL commands

```bash
# 1. Extract ISL two-hand landmarks (writes isl_landmarks_2hand.csv)
python scripts/extract_isl_twohand_landmarks.py

# 2. Train the new two-hand ISL model
python scripts/train_isl_twohand_model.py

# 3. Evaluate the two-hand ISL model
python scripts/evaluate_isl_twohand_model.py

# 4. Test a single ISL two-hand image
python scripts/test_landmark_models.py --dataset isl_2hand --image dataset/isl_webcam/A/roi/A_000001.png

# 5. Run real-time recognition (ASL one-hand / ISL two-hand)
python recognition/real_time_landmark.py
```

## Notes / future work

- Text-to-Speech is a future enhancement and is not implemented here.
- The `mediapipe/hand_landmarker.task` model asset is required for landmark
  detection and is kept in the project-local `mediapipe/` folder.
- The existing CNN webcam recognizer and its models are left untouched; this
  real-time landmark recognizer is an independent, parallel path.
- The two-hand ISL model is **experimental**: the existing ISL webcam dataset is
  predominantly single-hand, so only letters with at least 2 valid two-hand
  samples were included during training.  Letters excluded are documented in
  [`results/landmark/isl_2hand_metrics.json`](../results/landmark/isl_2hand_metrics.json).
