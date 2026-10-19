# RealSign ISL — Final Compatibility Report

This report determines whether the **webcam ISL dataset** (`dataset/isl_webcam/`) and the **RealSign public ISL dataset** (`dataset/isl_public/`) represent the same alphabet sign convention and could safely share class labels in a future training stage.

**No datasets were merged, modified, deleted, or retrained.** The existing webcam dataset and RealSign dataset are both preserved.

## Method

For each letter, up to 40 images per dataset were processed with MediaPipe (two-hand detection). For every detected hand we computed a scale-invariant **finger-extension descriptor** — the ratio of the direct MCP→tip distance to the polyline MCP→PIP→DIP→tip distance for each of the 5 fingers (thumb uses CMC→tip / CMC→MCP→IP→tip). Values near 1.0 indicate an extended/straight finger; values near 0 indicate a curled finger.

This focuses on **actual hand/finger geometry**, not raw pixel appearance, so camera angle, skin tone, lighting, background, hand size, and image quality are intentionally de-emphasized.

### Classification rules

- **COMPATIBLE** — dominant hand count matches AND ≤1 finger differs in extension pattern.
- **REVIEW** — hand count differs OR ≥2 fingers differ.
- **INCOMPATIBLE** — ≥4 fingers differ AND similarity < 0.5.

> A different **hand count** is treated as `REVIEW` (not automatically `INCOMPATIBLE`) because it can be caused by either a genuine one-hand vs two-hand sign difference OR a MediaPipe detection gap.

## Detailed per-letter table

| Letter | Webcam hand | RealSign hand | Webcam finger config | RealSign finger config | Similarity | Compatibility | Reason |
|--------|------------|---------------|----------------------|-------------------------|------------|---------------|--------|
| A | 2 | 2 | thumb=extended, index=extended, middle=extended, ring=curled, pinky=extended | thumb=extended, index=extended, middle=extended, ring=curled, pinky=curled | 0.98 | COMPATIBLE | hand count matches (2 hand); 1/5 fingers differ in extension pattern; landmark similarity 0.98 |
| B | 2 | 2 | thumb=extended, index=extended, middle=extended, ring=extended, pinky=extended | thumb=extended, index=extended, middle=extended, ring=extended, pinky=extended | 0.95 | COMPATIBLE | hand count matches (2 hand); finger extension patterns match; landmark similarity 0.95 |
| C | 0 | 1 | thumb=extended, index=extended, middle=extended, ring=curled, pinky=curled | thumb=extended, index=extended, middle=extended, ring=extended, pinky=extended | 0.87 | REVIEW | hand count differs (webcam 0 vs RealSign 1); 2/5 fingers differ in extension pattern; landmark similarity 0.87 |
| D | 1 | 2 | thumb=extended, index=extended, middle=extended, ring=extended, pinky=extended | thumb=extended, index=extended, middle=extended, ring=curled, pinky=curled | 0.97 | REVIEW | hand count differs (webcam 1 vs RealSign 2); 2/5 fingers differ in extension pattern; landmark similarity 0.97 |
| E | 2 | 2 | thumb=extended, index=extended, middle=curled, ring=curled, pinky=curled | thumb=extended, index=extended, middle=curled, ring=curled, pinky=curled | 0.99 | COMPATIBLE | hand count matches (2 hand); finger extension patterns match; landmark similarity 0.99 |
| F | 0 | 2 | thumb=extended, index=extended, middle=extended, ring=curled, pinky=curled | thumb=extended, index=extended, middle=extended, ring=curled, pinky=curled | 0.98 | REVIEW | hand count differs (webcam 0 vs RealSign 2); finger extension patterns match; landmark similarity 0.98 |
| G | 1 | 2 | thumb=extended, index=curled, middle=curled, ring=curled, pinky=curled | thumb=extended, index=curled, middle=curled, ring=curled, pinky=curled | 0.98 | REVIEW | hand count differs (webcam 1 vs RealSign 2); finger extension patterns match; landmark similarity 0.98 |
| H | 2 | 2 | thumb=extended, index=extended, middle=extended, ring=extended, pinky=extended | thumb=extended, index=extended, middle=extended, ring=extended, pinky=extended | 0.99 | COMPATIBLE | hand count matches (2 hand); finger extension patterns match; landmark similarity 0.99 |
| I | 1 | 0 | thumb=extended, index=extended, middle=curled, ring=curled, pinky=extended | thumb=extended, index=extended, middle=extended, ring=curled, pinky=curled | 0.96 | REVIEW | hand count differs (webcam 1 vs RealSign 0); 2/5 fingers differ in extension pattern; landmark similarity 0.96 |
| J | 1 | 2 | thumb=extended, index=extended, middle=extended, ring=extended, pinky=extended | thumb=extended, index=extended, middle=curled, ring=curled, pinky=curled | 0.85 | REVIEW | hand count differs (webcam 1 vs RealSign 2); 3/5 fingers differ in extension pattern; landmark similarity 0.85 |
| K | 2 | 1 | thumb=extended, index=extended, middle=extended, ring=curled, pinky=curled | thumb=extended, index=extended, middle=extended, ring=curled, pinky=extended | 0.96 | REVIEW | hand count differs (webcam 2 vs RealSign 1); 1/5 fingers differ in extension pattern; landmark similarity 0.96 |
| L | 1 | 1 | thumb=extended, index=extended, middle=curled, ring=curled, pinky=curled | thumb=extended, index=extended, middle=extended, ring=extended, pinky=extended | 0.93 | REVIEW | hand count matches (1 hand); 3/5 fingers differ in extension pattern; landmark similarity 0.93 |
| M | 1 | 1 | thumb=extended, index=extended, middle=extended, ring=extended, pinky=extended | thumb=extended, index=extended, middle=extended, ring=extended, pinky=extended | 0.99 | COMPATIBLE | hand count matches (1 hand); finger extension patterns match; landmark similarity 0.99 |
| N | 1 | 2 | thumb=extended, index=extended, middle=extended, ring=extended, pinky=extended | thumb=extended, index=extended, middle=extended, ring=extended, pinky=extended | 0.95 | REVIEW | hand count differs (webcam 1 vs RealSign 2); finger extension patterns match; landmark similarity 0.95 |
| O | 1 | 1 | thumb=extended, index=extended, middle=extended, ring=extended, pinky=extended | thumb=extended, index=extended, middle=extended, ring=extended, pinky=extended | 0.91 | COMPATIBLE | hand count matches (1 hand); finger extension patterns match; landmark similarity 0.91 |
| P | 1 | 2 | thumb=extended, index=extended, middle=extended, ring=extended, pinky=extended | thumb=extended, index=extended, middle=extended, ring=extended, pinky=extended | 0.98 | REVIEW | hand count differs (webcam 1 vs RealSign 2); finger extension patterns match; landmark similarity 0.98 |
| Q | 1 | 2 | thumb=extended, index=extended, middle=extended, ring=extended, pinky=extended | thumb=extended, index=extended, middle=extended, ring=extended, pinky=extended | 0.98 | REVIEW | hand count differs (webcam 1 vs RealSign 2); finger extension patterns match; landmark similarity 0.98 |
| R | 1 | 1 | thumb=extended, index=extended, middle=extended, ring=extended, pinky=extended | thumb=extended, index=extended, middle=extended, ring=extended, pinky=extended | 0.96 | COMPATIBLE | hand count matches (1 hand); finger extension patterns match; landmark similarity 0.96 |
| S | 1 | 2 | thumb=extended, index=extended, middle=extended, ring=extended, pinky=extended | thumb=extended, index=curled, middle=curled, ring=curled, pinky=extended | 0.70 | REVIEW | hand count differs (webcam 1 vs RealSign 2); 3/5 fingers differ in extension pattern; landmark similarity 0.70 |
| T | 1 | 1 | thumb=extended, index=extended, middle=extended, ring=curled, pinky=curled | thumb=extended, index=extended, middle=extended, ring=curled, pinky=curled | 0.99 | COMPATIBLE | hand count matches (1 hand); finger extension patterns match; landmark similarity 0.99 |
| U | 1 | 1 | thumb=extended, index=extended, middle=extended, ring=extended, pinky=extended | thumb=extended, index=extended, middle=extended, ring=extended, pinky=extended | 0.99 | COMPATIBLE | hand count matches (1 hand); finger extension patterns match; landmark similarity 0.99 |
| V | 1 | 1 | thumb=extended, index=extended, middle=extended, ring=extended, pinky=extended | thumb=extended, index=extended, middle=extended, ring=extended, pinky=extended | 0.96 | COMPATIBLE | hand count matches (1 hand); finger extension patterns match; landmark similarity 0.96 |
| W | 1 | 1 | thumb=extended, index=extended, middle=extended, ring=extended, pinky=extended | thumb=extended, index=extended, middle=extended, ring=extended, pinky=extended | 0.94 | COMPATIBLE | hand count matches (1 hand); finger extension patterns match; landmark similarity 0.94 |
| X | 1 | 2 | thumb=extended, index=extended, middle=extended, ring=extended, pinky=extended | thumb=extended, index=extended, middle=extended, ring=extended, pinky=extended | 0.96 | REVIEW | hand count differs (webcam 1 vs RealSign 2); finger extension patterns match; landmark similarity 0.96 |
| Y | 1 | 2 | thumb=extended, index=extended, middle=extended, ring=curled, pinky=curled | thumb=extended, index=extended, middle=extended, ring=extended, pinky=extended | 0.90 | REVIEW | hand count differs (webcam 1 vs RealSign 2); 2/5 fingers differ in extension pattern; landmark similarity 0.90 |
| Z | 2 | 2 | thumb=extended, index=extended, middle=extended, ring=extended, pinky=extended | thumb=extended, index=extended, middle=extended, ring=extended, pinky=extended | 1.00 | COMPATIBLE | hand count matches (2 hand); finger extension patterns match; landmark similarity 1.00 |

## Summary

- **Compatible letters (12):** A, B, E, H, M, O, R, T, U, V, W, Z
- **Review letters (14):** C, D, F, G, I, J, K, L, N, P, Q, S, X, Y
- **Incompatible letters (0):** none

## Previously flagged letters

The 10 letters flagged earlier (C, D, G, J, K, N, Q, S, X, Y) all remain `REVIEW` — none were auto-marked `INCOMPATIBLE`. The differences are driven primarily by **hand count** and, for a few letters, by additional finger-extension differences.

## Recommendation

**Do not merge the datasets yet.** The following classes are the strongest candidates for safe combination in a future training stage (same dominant hand count and near-identical finger configuration):

- **Safe-to-combine candidates:** A, B, E, H, M, O, R, T, U, V, W, Z

The `REVIEW` letters need manual visual confirmation (using the galleries in `results/realsign/comparison/`) to decide whether the hand-count difference is a real sign-convention difference or only a MediaPipe detection gap.

## Outputs

- `results/realsign/final_compatibility.csv`
- `results/realsign/final_compatibility_raw.json`
- `docs/REALSign_ISL_FINAL_COMPATIBILITY_REPORT.md`
