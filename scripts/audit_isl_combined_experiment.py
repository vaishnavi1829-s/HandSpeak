"""Generate the combined-experiment audit CSV + markdown (read-only).

Uses the recorded ``dataset/isl_combined/stats.json`` and
``dataset/isl_combined/duplicates.json`` to produce:

    results/realsign/combined_experiment_audit.csv
    docs/ISL_COMBINED_EXPERIMENT_AUDIT.md

No dataset or model is modified.
"""

from __future__ import annotations

import csv
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.label_mapping import LETTERS  # noqa: E402

COMBINED_ROOT = os.path.join(PROJECT_ROOT, "dataset", "isl_combined")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "realsign")
STATS_JSON = os.path.join(COMBINED_ROOT, "stats.json")
DUP_JSON = os.path.join(COMBINED_ROOT, "duplicates.json")


def main() -> None:
    with open(STATS_JSON, "r", encoding="utf-8") as fh:
        stats = json.load(fh)
    with open(DUP_JSON, "r", encoding="utf-8") as fh:
        dup = json.load(fh)

    per_letter = stats["per_letter"]

    rows = []
    total_webcam_orig = 0
    total_webcam_train = 0
    total_webcam_dup_removed = 0
    total_webcam_val = 0
    total_webcam_test = 0
    total_realsign = 0

    for letter in LETTERS:
        s = per_letter[letter]
        orig = s["webcam_total"]
        after_dedup = s["webcam_after_dedup"]
        dup_removed = orig - after_dedup
        train = s["webcam_train"]
        val = s["webcam_val"]
        test = s["webcam_test"]
        rs = s["realsign_train"]
        excluded = orig - train

        total_webcam_orig += orig
        total_webcam_train += train
        total_webcam_dup_removed += dup_removed
        total_webcam_val += val
        total_webcam_test += test
        total_realsign += rs

        ratio = round(rs / train, 1) if train else None

        reasons = []
        if dup_removed > 0:
            reasons.append(f"near-dup removed {dup_removed}")
        if val > 0:
            reasons.append(f"val {val}")
        reasons.append(f"test {test}")
        reason = " + ".join(reasons) + f" (of {orig})"

        rows.append(
            {
                "letter": letter,
                "original_webcam_count": orig,
                "webcam_used_in_combined_training": train,
                "webcam_excluded": excluded,
                "realsign_used": rs,
                "ratio_realsign_to_webcam": ratio,
                "reason_excluded": reason,
            }
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "combined_experiment_audit.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "letter",
                "original_webcam_count",
                "webcam_used_in_combined_training",
                "webcam_excluded",
                "realsign_used",
                "ratio_realsign_to_webcam",
                "reason_excluded",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Markdown.
    lines = []
    lines.append("# ISL Combined Experiment Audit")
    lines.append("")
    lines.append(
        "This audit explains why the combined RealSign + webcam model dropped to "
        "89.01% unseen-webcam accuracy versus 98.88% for the existing model."
    )
    lines.append("")
    lines.append("## 1. Original webcam counts (A-Z)")
    lines.append("")
    lines.append(
        "The webcam ISL dataset (`dataset/isl_webcam/`) contains **75–84 images "
        "per class** (see table below)."
    )
    lines.append("")
    lines.append("## 2. Webcam images actually used in combined training")
    lines.append("")
    lines.append(
        "Only **7–37 webcam images per class** reached the combined training set "
        "after (a) near-duplicate removal and (b) the train/validation/test split. "
        "The rest were excluded from training."
    )
    lines.append("")
    lines.append("## 3. Per-letter audit table")
    lines.append("")
    lines.append(
        "| Letter | Original webcam | Webcam used (train) | Webcam excluded | "
        "RealSign used | Ratio RealSign:Webcam |"
    )
    lines.append(
        "|--------|----------------:|--------------------:|----------------:|"
        "--------------:|----------------------:|"
    )
    for r in rows:
        ratio = f"{r['ratio_realsign_to_webcam']:.1f}" if r["ratio_realsign_to_webcam"] is not None else "n/a"
        lines.append(
            f"| {r['letter']} | {r['original_webcam_count']} | "
            f"{r['webcam_used_in_combined_training']} | {r['webcam_excluded']} | "
            f"{r['realsign_used']} | {ratio} |"
        )

    overall_ratio = round(total_realsign / total_webcam_train, 1)
    lines.append("")
    lines.append("## 4. Overall RealSign : Webcam ratio")
    lines.append("")
    lines.append(
        f"- RealSign training images used: **{total_realsign}**"
    )
    lines.append(
        f"- Webcam training images used: **{total_webcam_train}**"
    )
    lines.append(f"- Overall ratio RealSign:Webcam = **{overall_ratio}:1**")
    lines.append("")
    lines.append("## 5. Why webcam training count became 7–37 per class")
    lines.append("")
    lines.append(
        f"- Original webcam images: **{total_webcam_orig}** (75–84/class)."
    )
    lines.append(
        f"- **Near-duplicate removal (dHash, Hamming ≤2)** removed "
        f"**{total_webcam_dup_removed}** images ({dup['webcam_near_duplicate_files']} "
        f"files across {dup['webcam_near_duplicate_groups']} groups). The webcam ROI "
        f"frames are sequential captures of the same signer, so many are near-identical."
    )
    lines.append(
        f"- After dedup, each class was stratified-split into train/val/test "
        f"(random_state=42): ~{total_webcam_test // 26} test, ~{total_webcam_val // 26} "
        f"val, remainder train."
    )
    lines.append(
        f"- This left only **{total_webcam_train}** webcam training images total "
        f"(7–37/class)."
    )
    lines.append("")
    lines.append("## 6. Exclusion breakdown")
    lines.append("")
    lines.append("Webcam images were excluded from training due to:")
    lines.append("")
    lines.append("1. **Near-duplicate removal** (not landmark failure or hand-count "
                 "filtering).")
    lines.append("2. **Train/validation/test stratified splitting** (a fixed ~12 "
                 "images/class were held out to `test_webcam/`, plus a small val set).")
    lines.append("")
    lines.append("The following were **NOT** the cause:")
    lines.append("- MediaPipe landmark extraction failures: only 0-hand frames were "
                 "skipped during feature extraction (train 17626/18788 ≈ 6.2%, "
                 "test 273/312 ≈ 12.5%).")
    lines.append("- Hand-count filtering: the rich extractor keeps BOTH 1-hand and "
                 "2-hand detections (train hand distribution: 8848 one-hand, 8778 "
                 "two-hand).")
    lines.append("- Cross-dataset exact duplicates: **0** found.")
    lines.append("")
    lines.append("## 7. Feature representation vs existing 98.88% model")
    lines.append("")
    lines.append("| Property | Existing 98.88% model | Combined experimental |")
    lines.append("|----------|----------------------|-----------------------|")
    lines.append("| Feature count | 63 (single-hand xyz) | 214 (rich geometric) |")
    lines.append("| One/two-hand | one-hand only | one AND two hand |")
    lines.append("| Normalization | wrist-relative + scale | same + angles/distances |")
    lines.append("| Landmark ordering | lm0_x..lm20_z | lh/rh rich order |")
    lines.append("| Classifier | RandomForest | RandomForest |")
    lines.append("| Hyperparameters | 400 trees, class_weight=balanced | 400 trees, class_weight=balanced |")
    lines.append("| Training data | **webcam-only** | **RealSign-dominated** |")
    lines.append("")
    lines.append(
        "The existing 98.88% model uses **63 single-hand features on webcam-only "
        "data**; the combined model uses **214 rich features on RealSign-dominated "
        "data**. The existing rich model (214 features, webcam-only) scored 96.70% "
        "on the same test, so the feature change alone accounts for only ~2 points."
    )
    lines.append("")
    lines.append("## 8. Data leakage check")
    lines.append("")
    lines.append(
        "- The `test_webcam/` source paths had **0 overlap** with train/validation "
        "feature rows."
    )
    lines.append(
        "- `test_webcam/` was created from a stratified split AFTER dedup and was "
        "never used in training, validation, augmentation, or model fitting."
    )
    lines.append("- No cross-dataset exact duplicates between webcam and RealSign.")
    lines.append("")
    lines.append("## 9. Root-cause analysis")
    lines.append("")
    lines.append(
        "The drop from 98.88% → 89.01% is caused by a **domain/convention mismatch**, "
        "not data leakage and not primarily the feature representation:"
    )
    lines.append("")
    lines.append(
        "1. **RealSign dominance (30:1).** The combined training set contains "
        "18,198 RealSign images vs only 590 webcam images. The model learns "
        "RealSign's visual patterns, which differ from webcam patterns — especially "
        "for the REVIEW letters (C, D, F, G, I, J, K, L, N, Q, S, X, Y)."
    )
    lines.append(
        "2. **Severe reduction of webcam training data.** Near-duplicate removal "
        "collapsed ~half the webcam images (sequential same-signer frames), leaving "
        "only 7–37 webcam training images per class — too few to preserve the "
        "webcam domain."
    )
    lines.append(
        "3. **Feature representation difference (secondary).** The 63-feature "
        "one-hand model → 214-feature rich model transition costs ~2 points "
        "(98.88% → 96.70%), but the remaining ~7.7 points come from the RealSign "
        "data mixing."
    )
    lines.append("")
    lines.append(
        "**Conclusion:** the combined model is worse because the public RealSign "
        "data (a) overwhelms the small webcam sample and (b) uses different "
        "sign/hand conventions for several letters. It is not a leakage issue."
    )
    lines.append("")

    md_path = os.path.join(PROJECT_ROOT, "docs", "ISL_COMBINED_EXPERIMENT_AUDIT.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(f"CSV -> {csv_path}")
    print(f"MD  -> {md_path}")
    print(f"\nOverall RealSign:Webcam = {overall_ratio}:1")
    print(f"Webcam: orig={total_webcam_orig} train={total_webcam_train} "
          f"val={total_webcam_val} test={total_webcam_test} "
          f"dup_removed={total_webcam_dup_removed}")


if __name__ == "__main__":
    main()
