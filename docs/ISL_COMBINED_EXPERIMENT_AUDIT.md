# ISL Combined Experiment Audit

This audit explains why the combined RealSign + webcam model dropped to 89.01% unseen-webcam accuracy versus 98.88% for the existing model.

## 1. Original webcam counts (A-Z)

The webcam ISL dataset (`dataset/isl_webcam/`) contains **75–84 images per class** (see table below).

## 2. Webcam images actually used in combined training

Only **7–37 webcam images per class** reached the combined training set after (a) near-duplicate removal and (b) the train/validation/test split. The rest were excluded from training.

## 3. Per-letter audit table

| Letter | Original webcam | Webcam used (train) | Webcam excluded | RealSign used | Ratio RealSign:Webcam |
|--------|----------------:|--------------------:|----------------:|--------------:|----------------------:|
| A | 76 | 11 | 65 | 700 | 63.6 |
| B | 77 | 7 | 70 | 700 | 100.0 |
| C | 76 | 25 | 51 | 700 | 28.0 |
| D | 75 | 15 | 60 | 700 | 46.7 |
| E | 75 | 8 | 67 | 700 | 87.5 |
| F | 84 | 19 | 65 | 700 | 36.8 |
| G | 75 | 30 | 45 | 700 | 23.3 |
| H | 75 | 13 | 62 | 699 | 53.8 |
| I | 79 | 30 | 49 | 699 | 23.3 |
| J | 80 | 19 | 61 | 700 | 36.8 |
| K | 75 | 20 | 55 | 700 | 35.0 |
| L | 84 | 21 | 63 | 700 | 33.3 |
| M | 75 | 31 | 44 | 700 | 22.6 |
| N | 75 | 31 | 44 | 700 | 22.6 |
| O | 78 | 13 | 65 | 700 | 53.8 |
| P | 75 | 31 | 44 | 700 | 22.6 |
| Q | 75 | 32 | 43 | 700 | 21.9 |
| R | 75 | 24 | 51 | 700 | 29.2 |
| S | 76 | 26 | 50 | 700 | 26.9 |
| T | 76 | 20 | 56 | 700 | 35.0 |
| U | 75 | 32 | 43 | 700 | 21.9 |
| V | 76 | 14 | 62 | 700 | 50.0 |
| W | 75 | 24 | 51 | 700 | 29.2 |
| X | 75 | 31 | 44 | 700 | 22.6 |
| Y | 75 | 37 | 38 | 700 | 18.9 |
| Z | 77 | 26 | 51 | 700 | 26.9 |

## 4. Overall RealSign : Webcam ratio

- RealSign training images used: **18198**
- Webcam training images used: **590**
- Overall ratio RealSign:Webcam = **30.8:1**

## 5. Why webcam training count became 7–37 per class

- Original webcam images: **1989** (75–84/class).
- **Near-duplicate removal (dHash, Hamming ≤2)** removed **984** images (984 files across 388 groups). The webcam ROI frames are sequential captures of the same signer, so many are near-identical.
- After dedup, each class was stratified-split into train/val/test (random_state=42): ~12 test, ~3 val, remainder train.
- This left only **590** webcam training images total (7–37/class).

## 6. Exclusion breakdown

Webcam images were excluded from training due to:

1. **Near-duplicate removal** (not landmark failure or hand-count filtering).
2. **Train/validation/test stratified splitting** (a fixed ~12 images/class were held out to `test_webcam/`, plus a small val set).

The following were **NOT** the cause:
- MediaPipe landmark extraction failures: only 0-hand frames were skipped during feature extraction (train 17626/18788 ≈ 6.2%, test 273/312 ≈ 12.5%).
- Hand-count filtering: the rich extractor keeps BOTH 1-hand and 2-hand detections (train hand distribution: 8848 one-hand, 8778 two-hand).
- Cross-dataset exact duplicates: **0** found.

## 7. Feature representation vs existing 98.88% model

| Property | Existing 98.88% model | Combined experimental |
|----------|----------------------|-----------------------|
| Feature count | 63 (single-hand xyz) | 214 (rich geometric) |
| One/two-hand | one-hand only | one AND two hand |
| Normalization | wrist-relative + scale | same + angles/distances |
| Landmark ordering | lm0_x..lm20_z | lh/rh rich order |
| Classifier | RandomForest | RandomForest |
| Hyperparameters | 400 trees, class_weight=balanced | 400 trees, class_weight=balanced |
| Training data | **webcam-only** | **RealSign-dominated** |

The existing 98.88% model uses **63 single-hand features on webcam-only data**; the combined model uses **214 rich features on RealSign-dominated data**. The existing rich model (214 features, webcam-only) scored 96.70% on the same test, so the feature change alone accounts for only ~2 points.

## 8. Data leakage check

- The `test_webcam/` source paths had **0 overlap** with train/validation feature rows.
- `test_webcam/` was created from a stratified split AFTER dedup and was never used in training, validation, augmentation, or model fitting.
- No cross-dataset exact duplicates between webcam and RealSign.

## 9. Root-cause analysis

The drop from 98.88% → 89.01% is caused by a **domain/convention mismatch**, not data leakage and not primarily the feature representation:

1. **RealSign dominance (30:1).** The combined training set contains 18,198 RealSign images vs only 590 webcam images. The model learns RealSign's visual patterns, which differ from webcam patterns — especially for the REVIEW letters (C, D, F, G, I, J, K, L, N, Q, S, X, Y).
2. **Severe reduction of webcam training data.** Near-duplicate removal collapsed ~half the webcam images (sequential same-signer frames), leaving only 7–37 webcam training images per class — too few to preserve the webcam domain.
3. **Feature representation difference (secondary).** The 63-feature one-hand model → 214-feature rich model transition costs ~2 points (98.88% → 96.70%), but the remaining ~7.7 points come from the RealSign data mixing.

**Conclusion:** the combined model is worse because the public RealSign data (a) overwhelms the small webcam sample and (b) uses different sign/hand conventions for several letters. It is not a leakage issue.
