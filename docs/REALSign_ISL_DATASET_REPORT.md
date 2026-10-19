# RealSign ISL Dataset — Compatibility Report

This report documents the **download → extract → verify → MediaPipe analysis →
compatibility** workflow for the public
[RealSign Indian Sign Language Dataset](https://github.com/RealSign62/RealSign-Indian-Sign-Language-Dataset).

**No existing dataset, model, script, or application functionality was
modified, deleted, or overwritten.** The existing webcam ISL dataset
(`dataset/isl_webcam/`) is untouched, and **no datasets were merged and no model
was retrained**.

---

## 1. Source

- Repository: https://github.com/RealSign62/RealSign-Indian-Sign-Language-Dataset
- License: **CC0-1.0** (public domain dedication)
- Archive: `Dataset.zip` (Git LFS pointer in the repo)
- Actual archive size: **656,689,688 bytes (~626.3 MB)**
- Downloaded via the GitHub LFS media endpoint
  (`media.githubusercontent.com`).

The download is resumable via [`scripts/download_realsign.py`](../scripts/download_realsign.py).

---

## 2. Dataset structure

Extracted to `dataset/isl_public/`:

```
dataset/isl_public/
├── Training/        (A–Z)
├── Testing/         (A–Z)
├── Validation/      (A–Z)
└── Letters/         (26 reference JPGs, one per letter)
```

All 26 alphabet folders exist in Training, Testing, and Validation.

---

## 3. Class counts

| Split | Images | Classes | Notes |
|-------|-------:|--------:|-------|
| Training | 18,198 | 26 | ~700 per class (H and I have 699) |
| Testing | 5,200 | 26 | 200 per class |
| Validation | 2,579 | 26 | ~100 per class (J/Q/U/Y = 101, K = 75) |
| **Total** | **25,977** | 26 | plus 26 reference images in `Letters/` |

Note: the repository README describes ~700 train / ~200 test / ~100 validation
per class. The extracted counts match closely, with minor per-class deviations.

---

## 4. Image statistics

- **Format:** 100% JPEG (`.jpg`).
- **Dimensions:** highly variable (thousands of distinct dimensions). The most
  common are ~193×287, ~281×313, ~166×254, ~336×297, ~218×352. Images are
  NOT uniformly sized — any consumer must resize before model input.
- **Corrupted images:** 2 files could not be decoded by OpenCV:
  - `Testing/E/301.jpg`
  - `Testing/K/301.jpg`
- **Blank / near-uniform images:** 0 detected (low-variance threshold).
- **Exact duplicate images:** 787 duplicate groups involving 1,678 files.
  These are byte-identical files (likely reused across splits). They should be
  considered during any future train/test split to avoid leakage.

---

## 5. MediaPipe detection statistics

MediaPipe two-hand detection was run over a 50-image-per-class sample for each
split (3,900 images total). Results:

| Split | 0-hand | 1-hand | 2-hand | Failed |
|-------|-------:|-------:|-------:|-------:|
| Training | 28 | 547 | 725 | 28 |
| Testing | 9 | 513 | 778 | 9 |
| Validation | 183 | 633 | 484 | 183 |
| **Total (sampled)** | **220** | **1,693** | **1,987** | **220** |

- **Successful landmark extraction:** 3,680 / 3,900 (94.4%).
- **Failed extraction (0 hands):** 220 / 3,900 (5.6%). Failures are
  concentrated in Validation (183), and in specific letters (notably B, D, S, V,
  Y, I).
- The dataset contains **both one-hand and two-hand signs**, so it cannot be
  forced into a single-hand or two-hand-only representation.

Full per-letter breakdown: [`results/realsign/isl_public_mediapipe.txt`](../results/realsign/isl_public_mediapipe.txt).

---

## 6. One-hand / two-hand analysis (per letter)

The public dataset has clearly **one-hand-dominant** letters (C, I, K, L, M, O,
R, T, U, V, W) and **two-hand-dominant** letters (A, B, D, E, F, G, H, P, S, X,
Y, Z), plus several mixed letters (J, N, Q).

This matches the existing webcam ISL dataset in broad terms but with important
per-letter differences (see §8).

---

## 7. Sign-pattern compatibility (A–Z)

Compatibility is a **heuristic** based on the dominant hand count (0/1/2) in
each dataset. It flags letters where the two datasets disagree, not definitive
sign-convention equality.

| Letter | Pub 1H% | Pub 2H% | Web 1H% | Web 2H% | Compatibility |
|--------|--------:|--------:|--------:|--------:|---------------|
| A | 14.7 | 84.0 | 18.4 | 73.7 | MATCH |
| B | 2.7 | 84.0 | 14.3 | 80.5 | MATCH |
| C | 92.7 | 1.3 | 25.0 | 1.3 | **REVIEW** |
| D | 6.7 | 76.7 | 70.7 | 8.0 | **REVIEW** |
| E | 17.3 | 79.3 | 33.3 | 66.7 | MATCH |
| F | 7.3 | 92.0 | 29.8 | 47.6 | MATCH |
| G | 14.0 | 83.3 | 86.7 | 9.3 | **REVIEW** |
| H | 0.0 | 100.0 | 44.0 | 56.0 | MATCH |
| I | 74.0 | 0.0 | 64.6 | 10.1 | MATCH |
| J | 24.0 | 72.7 | 93.8 | 0.0 | **REVIEW** |
| K | 64.7 | 35.3 | 18.7 | 81.3 | **REVIEW** |
| L | 98.7 | 0.0 | 100.0 | 0.0 | MATCH |
| M | 92.7 | 7.3 | 56.0 | 44.0 | MATCH |
| N | 30.0 | 66.7 | 77.3 | 22.7 | **REVIEW** |
| O | 90.7 | 0.7 | 100.0 | 0.0 | MATCH |
| P | 14.0 | 83.3 | 41.3 | 46.7 | MATCH |
| Q | 14.7 | 78.7 | 68.0 | 26.7 | **REVIEW** |
| R | 59.3 | 40.7 | 98.7 | 1.3 | MATCH |
| S | 1.3 | 83.3 | 84.2 | 15.8 | **REVIEW** |
| T | 83.3 | 16.7 | 52.6 | 46.1 | MATCH |
| U | 92.0 | 0.0 | 98.7 | 0.0 | MATCH |
| V | 85.3 | 0.0 | 93.4 | 0.0 | MATCH |
| W | 88.0 | 10.7 | 72.0 | 0.0 | MATCH |
| X | 26.0 | 74.0 | 66.7 | 16.0 | **REVIEW** |
| Y | 12.7 | 78.0 | 50.7 | 2.7 | **REVIEW** |
| Z | 22.0 | 76.0 | 23.4 | 76.6 | MATCH |

Full table: [`results/realsign/isl_compatibility.csv`](../results/realsign/isl_compatibility.csv).

---

## 8. Letters requiring manual verification

The following letters show a clear **one-hand vs two-hand disagreement** between
the public RealSign dataset and the existing webcam ISL dataset, and must be
manually reviewed before any merge or joint training:

- **C** — public is 1-hand, webcam is mixed (25% 1-hand).
- **D** — public is 2-hand, webcam is mostly 1-hand.
- **G** — public is 2-hand, webcam is mostly 1-hand.
- **J** — public is 2-hand, webcam is mostly 1-hand.
- **K** — public is mixed 1-hand, webcam is mostly 2-hand.
- **N** — public is 2-hand, webcam is mostly 1-hand.
- **Q** — public is 2-hand, webcam is mostly 1-hand.
- **S** — public is 2-hand, webcam is mostly 1-hand.
- **X** — public is 2-hand, webcam is mostly 1-hand.
- **Y** — public is 2-hand, webcam is mixed 1-hand.

> These differences may be due to (a) genuinely different sign conventions,
> (b) MediaPipe hand-detection gaps, or (c) dataset capture differences. They
> MUST be visually inspected before the datasets are combined.

---

## 9. Recommendation for training

1. **Do not merge the datasets yet.** The two datasets appear to follow
   different hand conventions for at least 10 letters.
2. Manually review the `REVIEW` letters (C, D, G, J, K, N, Q, S, X, Y) against
   the existing ISL convention used by the webcam dataset.
3. If merged later, **resize all images** (they are not uniform) and **handle
   the 2 corrupted files** (`Testing/E/301.jpg`, `Testing/K/301.jpg`).
4. Apply **duplicate-aware splitting** — 787 exact-duplicate groups exist, so a
   naive random split could leak duplicates across train/test.
5. Keep the public dataset as a **supplementary/held-out** source first (e.g.,
   as an external validation set) rather than training directly on it, until the
   sign-convention question is resolved.

---

## 10. Files created (no existing files modified)

| File | Purpose |
|------|---------|
| [`scripts/download_realsign.py`](../scripts/download_realsign.py) | Resumable downloader for `Dataset.zip` |
| [`scripts/verify_realsign_dataset.py`](../scripts/verify_realsign_dataset.py) | Structure/count/dimension/corruption/duplicate verification |
| [`scripts/analyze_realsign_mediapipe.py`](../scripts/analyze_realsign_mediapipe.py) | MediaPipe 0/1/2-hand analysis |
| [`scripts/build_realsign_compatibility.py`](../scripts/build_realsign_compatibility.py) | Public vs webcam compatibility table |
| [`dataset/Dataset.zip`](../dataset/Dataset.zip) | Downloaded archive (626.3 MB) |
| `dataset/isl_public/` | Extracted RealSign dataset (Training/Testing/Validation/Letters) |
| `dataset/_realsign_readme.md` | Upstream README copy |
| [`results/realsign/isl_public_verification.json`](../results/realsign/isl_public_verification.json) | Verification results |
| [`results/realsign/isl_public_mediapipe.json`](../results/realsign/isl_public_mediapipe.json) | MediaPipe per-letter stats |
| [`results/realsign/isl_public_mediapipe.txt`](../results/realsign/isl_public_mediapipe.txt) | Human-readable MediaPipe stats |
| [`results/realsign/isl_compatibility.csv`](../results/realsign/isl_compatibility.csv) | A–Z compatibility table |
| [`results/realsign/isl_compatibility.json`](../results/realsign/isl_compatibility.json) | Compatibility JSON |
| [`docs/REALSign_ISL_DATASET_REPORT.md`](../docs/REALSign_ISL_DATASET_REPORT.md) | This report |

The existing `dataset/webcam/`, `dataset/isl_webcam/`, all CNN models, all
landmark models, and all prior scripts/results remain intact.
