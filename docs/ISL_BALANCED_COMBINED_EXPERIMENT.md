# ISL Balanced Combined-Data Experiment

This document records the **balanced** combination of RealSign public ISL data
with the webcam ISL dataset, and the resulting comparison against the existing
production model.

**No original dataset, existing model, or script was modified, deleted, or
replaced.** This is a completely separate experimental dataset/model.

---

## 1. Objective

Test whether **balancing** RealSign and webcam training samples (instead of the
previous 30.8:1 imbalance) improves or maintains real-webcam recognition.

---

## 2. Dataset construction

New dataset: `dataset/isl_balanced_combined/`

```
dataset/isl_balanced_combined/
├── train/          (A-Z)
├── validation/     (A-Z)
└── test_webcam/    (A-Z)   <- UNSEEN webcam images only
```

### Balance

The largest common balanced number across all 26 classes was determined as:

- Minimum deduplicated webcam train/val pool (after reserving the fixed unseen
  test set) = **16** (bottleneck: letter C).
- Reserving **6** webcam images/class for validation leaves
  **N = 10 webcam training images per class**.

Therefore the balanced training configuration is:

- **10 RealSign + 10 webcam training images per class** (1:1 ratio).
- **6 RealSign + 6 webcam validation images per class**.

The cap of 40 was not reached because only 16 diverse webcam images/class were
available after near-duplicate removal and test-set reservation.

### Webcam selection (diversity-aware)

- The previous experiment's leakage-free `test_webcam` set (312 images, 12/class)
  was **reused** — these images were never used in training/validation/fitting.
- Remaining webcam images were **near-duplicate aware** (dHash Hamming ≤2).
- Training/validation webcam images were selected to **maximize landmark-
  geometry diversity** (5-finger extension descriptor), so sequential
  near-identical frames are not all chosen.

### RealSign selection

- 10 training images per class sampled deterministically (random_state=42) from
  RealSign Training.
- 6 validation images per class sampled from RealSign Validation.

---

## 3. Exact samples per class

| Split | Source | Images/class | Total |
|-------|--------|-------------:|------:|
| train | RealSign | 10 | 260 |
| train | Webcam | 10 | 260 |
| validation | RealSign | 6 | 156 |
| validation | Webcam | 6 | 156 |
| test_webcam | Webcam | 12 | 312 |

Totals: **train = 520**, **validation = 312**, **test_webcam = 312**.

RealSign:webcam ratio in training = **1:1** (10:10 per class).

---

## 4. Feature representation & model

- Features: **214 rich geometric features** (normalized coordinates, joint
  angles, finger segment lengths, wrist-to-tip distances, fingertip-to-tip
  distances, cross-hand distances, hand-presence). Supports one-hand and
  two-hand signs naturally.
- Classifier: `RandomForestClassifier`
  - `n_estimators=400`
  - `class_weight="balanced"`
  - `random_state=42`
- Model saved as: `trained_model/isl_balanced_combined_experimental.joblib`.

Feature extraction success:
- train: 451/520 (86.7%)
- validation: 288/312 (92.3%)
- test_webcam: 273/312 (87.5%)

---

## 5. Results (unseen webcam test set)

| Model | Accuracy | Macro P | Macro R | Macro F1 |
|-------|---------:|--------:|--------:|--------:|
| Existing production (`isl_landmark_rf.joblib`) | **0.9888** | 0.9897 | 0.9737 | 0.9781 |
| Balanced combined experimental | **0.7875** | 0.8018 | 0.7887 | 0.7694 |

Validation accuracy of the balanced model (webcam + RealSign val): 0.7743.

### Per-letter comparison (existing vs balanced)

| Letter | Existing | Balanced | Change |
|--------|---------:|---------:|-------:|
| A | 1.000 | 0.800 | −0.200 |
| B | 1.000 | 0.900 | −0.100 |
| C | 0.500 | 1.000 | **+0.500** |
| D | 1.000 | 0.778 | −0.222 |
| E | 1.000 | 0.917 | −0.083 |
| F | 1.000 | 0.818 | −0.182 |
| G | 1.000 | 1.000 | +0.000 |
| H | 1.000 | 0.833 | −0.167 |
| I | 1.000 | 0.364 | −0.636 |
| J | 1.000 | 0.900 | −0.100 |
| K | 0.917 | 0.917 | +0.000 |
| L | 1.000 | 1.000 | +0.000 |
| M | 1.000 | 0.417 | −0.583 |
| N | 1.000 | 0.250 | −0.750 |
| O | 1.000 | 1.000 | +0.000 |
| P | 0.900 | 0.300 | −0.600 |
| Q | 1.000 | 0.900 | −0.100 |
| R | 1.000 | 0.917 | −0.083 |
| S | 1.000 | 1.000 | +0.000 |
| T | 1.000 | 1.000 | +0.000 |
| U | 1.000 | 1.000 | +0.000 |
| V | 1.000 | 0.900 | −0.100 |
| W | 1.000 | 0.750 | −0.250 |
| X | 1.000 | 0.300 | −0.700 |
| Y | 1.000 | 0.714 | −0.286 |
| Z | 1.000 | 0.833 | −0.167 |

Improved: **C** (+0.500).
Unchanged: G, K, L, O, S, T, U.
Worse: A, B, D, E, F, H, I, J, M, N, P, Q, R, V, W, X, Y, Z.

Confusion matrix:
`results/realsign/balanced_combined_confusion_matrix.png`.

---

## 6. Analysis

- The balanced model still performs **worse** (0.7875 vs 0.9888) than the
  existing webcam-only production model.
- The result shows that **imbalance was not the only problem** — the remaining
  cause is a **domain/convention mismatch**: RealSign images use different
  sign conventions and visual styles for many letters (especially the REVIEW
  letters), which actively confuses the classifier even when balanced 1:1.
- The balanced experiment uses very few training samples (10 RealSign + 10
  webcam/class), which also limits the model's ability to learn robust
  per-class boundaries.

---

## 7. Evidence-based conclusion

**C. Performance became worse.**

- The balanced combined model (0.7875) underperforms the existing production
  model (0.9888) by **−0.2013** on the unseen webcam test set.
- Only letter **C** improved; letters **I, M, N, P, X** regressed severely
  (−0.58 to −0.75).
- The remaining issue is **class-specific domain/convention mismatch**, not
  feature representation (the existing rich model already reached 96.70% with
  the same features on webcam-only data) and not imbalance (this experiment
  was balanced 1:1).
- **Recommendation:** Keep the existing production model. Do not combine the
  RealSign dataset with the webcam dataset until the sign-convention
  differences for the REVIEW letters are resolved and/or substantially more
  diverse webcam data is collected.

---

## 8. Files created (originals untouched)

- `dataset/isl_balanced_combined/` (train/validation/test_webcam + manifest + stats)
- `dataset/landmarks/isl_balanced_*_features.csv`
- `trained_model/isl_balanced_combined_experimental.joblib`
- `results/realsign/balanced_combined_comparison.csv`
- `results/realsign/balanced_combined_metrics.json`
- `results/realsign/balanced_combined_confusion_matrix.png`
- `scripts/prepare_isl_balanced_dataset.py`
- `scripts/train_isl_balanced_model.py`
