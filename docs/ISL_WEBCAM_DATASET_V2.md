# ISL Webcam Dataset v2 — Diversity-Aware Collection

This document describes the new, **separate** ISL webcam collection pipeline
that prioritizes **diversity** over raw frame count.

The existing dataset (`dataset/isl_webcam/`) and the existing production model
are **not modified, deleted, or replaced** by this pipeline.

---

## 1. Why v2?

The original collector saved every manual capture. Because the webcam produces
sequential near-identical frames, many saved images are near-duplicates. After
duplicate filtering, some classes had few genuinely diverse samples.

v2 fixes this by only saving a frame when it is **sufficiently different** from
recently saved samples.

---

## 2. Output location

```
dataset/isl_webcam_v2/
    A/roi/A_S1_0001.png
    B/roi/B_S1_0001.png
    ...
    manifest.csv
```

This is completely separate from `dataset/isl_webcam/`.

---

## 3. Diversity method

When MediaPipe detects at least one hand, diversity is measured using a
**scale-normalized landmark descriptor** (wrist-normalized fingertip positions
of all detected hands). A frame is saved only if its mean distance to the last
8 saved frames is >= 0.25.

When no hand is detected, the script falls back to a **perceptual dHash**
similarity and requires a Hamming distance >= 6.

Both methods prevent near-duplicate frames from being saved.

---

## 4. Live collection display

The collection window shows:

- Letter
- Saved
- Target
- Skipped as duplicate
- Current variation hint

Controls: `SPACE` = try capture, `N/S` = next letter, `A-Z` = jump, `Q` = quit.

---

## 5. Manifest

`dataset/isl_webcam_v2/manifest.csv` records:

| Column | Description |
|--------|-------------|
| filename | relative path under v2 root |
| letter | A-Z |
| session | S1, S2, ... (auto-increments per letter) |
| timestamp | capture time |
| hand_count | 0/1/2 |
| diversity_score | computed diversity metric |

---

## 6. Collection protocol (recommended)

Target: **~150 diverse images per letter**.

For each letter:

1. **Multiple sessions** — collect in 3–5 sessions (S1, S2, ...) so lighting,
   background, and hand position change naturally between sessions.
2. **Different hand positions** — move the hand left/right/up/down in the frame.
3. **Different camera distances** — near, medium, far.
4. **Natural wrist orientation** — tilt/rotate the wrist slightly while keeping
   the sign valid.
5. **Different hand sizes in frame** — vary distance to the camera.
6. **Lighting/background variation** — change room lighting or camera direction.
7. **Different people** (if available) — different hand shapes/skin tones.

**Important:** never artificially change the actual ISL finger configuration.
The sign must remain valid.

---

## 7. Quality audit

Run:

```bash
python scripts/audit_isl_webcam_v2.py
```

It reports per letter:

- total images
- near-duplicate percentage (dHash Hamming ≤2)
- MediaPipe detection success %
- 1-hand count
- 2-hand count
- blur percentage (variance-of-Laplacian threshold)
- diversity score (mean pairwise dHash distance)

Output: `results/isl_webcam_v2_quality_report.csv` (+ `.json`).

---

## 8. Commands

```bash
# Interactive collection (starts at first incomplete letter)
python scripts/collect_isl_webcam_v2.py --target 150

# Collect a specific letter
python scripts/collect_isl_webcam_v2.py --letter A --target 150

# New session explicitly
python scripts/collect_isl_webcam_v2.py --letter A --session S2 --target 150

# Show progress
python scripts/collect_isl_webcam_v2.py --list

# Audit quality
python scripts/audit_isl_webcam_v2.py
```

---

## 9. Notes

- Do **not** train a model yet.
- Do **not** merge with RealSign yet.
- The existing `dataset/isl_webcam/` remains unchanged.
