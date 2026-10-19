# ISL Combined Dataset Experiment

This document records the **experimental** combination of the webcam ISL
dataset (`dataset/isl_webcam/`) with the RealSign public ISL dataset
(`dataset/isl_public/`) and the resulting model comparison.

**No original dataset was modified, deleted, or moved.** A new experimental
model was trained under a new filename; the production models were not
replaced.

---

## 1. Objective

Determine whether adding RealSign public data to the webcam ISL data improves
**real webcam recognition** for A-Z, while allowing both one-hand and two-hand
signs naturally.

---

## 2. Dataset preparation

A new dataset was created at `dataset/isl_combined/`:

```
dataset/isl_combined/
├── train/          (A-Z)
├── validation/     (A-Z)
└── test_webcam/    (A-Z)   <- UNSEEN webcam images only
```

Rules applied:

- **RealSign Training** → `train/` (main public source).
- **RealSign Validation** → `validation/`.
- **Webcam images** were stratified-split (random_state=42) after dedup:
  - ~12 unseen webcam images per class held out to `test_webcam/`.
  - remaining webcam images split into `train/` and `validation/`.
- Exact (MD5) and near-duplicate (dHash, Hamming ≤2) detection within webcam.
- Cross-dataset exact-duplicate check (webcam vs RealSign training).
- `test_webcam/` is **never** used in training/validation/augmentation.

### Leakage / duplicate findings

| Check | Result |
|-------|--------|
| Webcam exact duplicate groups | 0 |
| Webcam near-duplicate groups | 388 (984 files involved) |
| Cross-dataset exact duplicates | 0 |
| Classes with insufficient webcam holdout | none (all ≥12) |

> Near-duplicates are sequential webcam frames that are perceptually very
> similar; only essentially-identical pairs were collapsed so that no near-
> identical image could appear in both train and test.

---

## 3. Dataset counts

| Letter | RealSign train | RealSign val | Webcam train | Webcam val | Webcam test | Combined train |
|--------|---------------:|-------------:|-------------:|-----------:|------------:|---------------:|
| A | 700 | 100 | 11 | 2 | 12 | 711 |
| B | 700 | 100 | 7 | 1 | 12 | 707 |
| C | 700 | 100 | 25 | 4 | 12 | 725 |
| D | 700 | 100 | 15 | 3 | 12 | 715 |
| E | 700 | 100 | 8 | 2 | 12 | 708 |
| F | 700 | 100 | 19 | 3 | 12 | 719 |
| G | 700 | 100 | 30 | 5 | 12 | 730 |
| H | 699 | 100 | 13 | 2 | 12 | 712 |
| I | 699 | 100 | 30 | 5 | 12 | 729 |
| J | 700 | 101 | 19 | 3 | 12 | 719 |
| K | 700 | 75 | 20 | 3 | 12 | 720 |
| L | 700 | 100 | 21 | 4 | 12 | 721 |
| M | 700 | 100 | 31 | 6 | 12 | 731 |
| N | 700 | 100 | 31 | 5 | 12 | 731 |
| O | 700 | 100 | 13 | 2 | 12 | 713 |
| P | 700 | 100 | 31 | 6 | 12 | 731 |
| Q | 700 | 101 | 32 | 6 | 12 | 732 |
| R | 700 | 100 | 24 | 4 | 12 | 724 |
| S | 700 | 100 | 26 | 4 | 12 | 726 |
| T | 700 | 100 | 20 | 4 | 12 | 720 |
| U | 700 | 101 | 32 | 6 | 12 | 732 |
| V | 700 | 100 | 14 | 2 | 12 | 714 |
| W | 700 | 100 | 24 | 4 | 12 | 724 |
| X | 700 | 100 | 31 | 6 | 12 | 731 |
| Y | 700 | 101 | 37 | 6 | 12 | 737 |
| Z | 700 | 100 | 26 | 5 | 12 | 726 |

Totals: **train = 18,788**, **validation = 2,682**, **test_webcam = 312**.

Class imbalance: RealSign contributes ~700 per class, webcam contributes a
small number (7–37 train images per class). The combined training set is
therefore heavily dominated by RealSign.

---

## 4. Preprocessing

The **rich 214-feature landmark representation** was used (the same reusable
extractor from [`recognition/isl_feature_extractor.py`](../recognition/isl_feature_extractor.py),
which supports one and two hands without forcing a two-hand-only layout).

Feature extraction was cached to:

- `dataset/landmarks/isl_combined_train_features.csv`
- `dataset/landmarks/isl_combined_val_features.csv`
- `dataset/landmarks/isl_combined_test_webcam_features.csv`

Original images were never modified; feature CSVs are new files.

---

## 5. Training (experimental model)

- Model: `RandomForestClassifier` (400 trees, `class_weight="balanced"`,
  `random_state=42`).
- Features: 214 rich geometric features.
- Saved as: `trained_model/isl_combined_landmark_experimental.joblib`.
- Validation accuracy (webcam + RealSign validation mix): **0.8957**.

This validation number includes RealSign validation images, so it does **not**
reflect real webcam performance.

---

## 6. Unseen webcam test results (the decisive comparison)

All three models were evaluated on the same `test_webcam/` images.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|-------|---------:|----------------:|-------------:|---------:|
| Existing one-hand (`isl_landmark_rf.joblib`) | **0.9888** | 0.9897 | 0.9737 | 0.9781 |
| Existing rich (`isl_landmark_rf_rich.joblib`) | 0.9670 | 0.9722 | 0.9647 | 0.9664 |
| Combined experimental | 0.8901 | 0.8642 | 0.8601 | 0.8545 |

The combined model is **worse** than both existing ISL models on real webcam
images.

### Per-letter comparison (existing one-hand vs combined)

| Letter | Existing | Combined | Change |
|--------|---------:|---------:|-------:|
| A | 1.000 | 0.700 | −0.300 |
| B | 1.000 | 0.900 | −0.100 |
| C | 0.500 | 0.000 | −0.500 |
| D | 1.000 | 0.778 | −0.222 |
| E | 1.000 | 0.583 | −0.417 |
| F | 1.000 | 0.818 | −0.182 |
| G | 1.000 | 1.000 | +0.000 |
| H | 1.000 | 0.833 | −0.167 |
| I | 1.000 | 0.909 | −0.091 |
| J | 1.000 | 1.000 | +0.000 |
| K | 0.917 | 1.000 | **+0.083** |
| L | 1.000 | 1.000 | +0.000 |
| M | 1.000 | 0.833 | −0.167 |
| N | 1.000 | 0.917 | −0.083 |
| O | 1.000 | 0.833 | −0.167 |
| P | 0.900 | 0.800 | −0.100 |
| Q | 1.000 | 1.000 | +0.000 |
| R | 1.000 | 1.000 | +0.000 |
| S | 1.000 | 1.000 | +0.000 |
| T | 1.000 | 1.000 | +0.000 |
| U | 1.000 | 1.000 | +0.000 |
| V | 1.000 | 0.900 | −0.100 |
| W | 1.000 | 1.000 | +0.000 |
| X | 1.000 | 0.700 | −0.300 |
| Y | 1.000 | 0.857 | −0.143 |
| Z | 1.000 | 1.000 | +0.000 |

Letters that improved: **K** (+0.083).
Letters that became worse: A, B, C, D, E, F, H, I, M, N, O, P, V, X, Y.

Confusion matrix:
`results/realsign/combined_model_confusion_matrix.png`.

---

## 7. Final recommendation

- **Did combining the datasets improve real webcam recognition?** **No.** The
  combined experimental model accuracy on unseen webcam images dropped from
  0.9888 (existing one-hand) to 0.8901.
- **Which letters improved?** Only **K** showed a small improvement; most
  letters stayed the same or got worse.
- **Which letters became worse?** C, E, X, A, D, F, H, M, O, V, Y (and several
  smaller drops).
- **Is overfitting visible?** The validation accuracy (0.8957) is close to the
  webcam test accuracy, but the model is dominated by RealSign visual patterns
  which differ from webcam patterns. This is a **domain-mismatch / sign-
  convention mismatch** rather than classic overfitting to the train set.
- **Is additional webcam data needed?** **Yes.** The webcam training pool per
  class is very small (7–37 images), and the public data does not transfer well
  to webcam-style captures.
- **Should the combined model be used in production?** **No.** Keep the existing
  production model. The experimental model remains separate
  (`isl_combined_landmark_experimental.joblib`) for review only.

The earlier compatibility findings are consistent with this result: the
`REVIEW` letters (C, D, F, G, I, J, K, N, Q, S, X, Y, plus L) carry the
largest sign-convention and hand-count differences, and naive combination
hurts real webcam accuracy.

---

## 8. Files created (originals untouched)

- `dataset/isl_combined/` (train / validation / test_webcam + manifest + stats + duplicates)
- `dataset/landmarks/isl_combined_*_features.csv`
- `trained_model/isl_combined_landmark_experimental.joblib`
- `results/realsign/combined_model_comparison.md`
- `results/realsign/combined_model_metrics.json`
- `results/realsign/combined_model_confusion_matrix.png`
- `scripts/prepare_isl_combined_dataset.py`
- `scripts/train_isl_combined_model.py`
