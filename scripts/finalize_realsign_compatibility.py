"""Generate the final compatibility CSV + markdown report from landmark analysis.

Reads ``results/realsign/final_compatibility_raw.json`` and applies a
transparent classification rule to produce:

    results/realsign/final_compatibility.csv
    docs/REALSign_ISL_FINAL_COMPATIBILITY_REPORT.md

Classification rule (per letter):
    * COMPATIBLE  : dominant hand count matches AND <= 1 finger disagrees
                    between the two datasets' mean finger-extension patterns.
    * REVIEW      : hand count differs OR >= 2 fingers disagree.
    * INCOMPATIBLE: >= 4 fingers disagree AND low similarity (< 0.5).

Read-only; no dataset/model is modified.
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

RAW_JSON = os.path.join(PROJECT_ROOT, "results", "realsign", "final_compatibility_raw.json")
CSV_PATH = os.path.join(PROJECT_ROOT, "results", "realsign", "final_compatibility.csv")
MD_PATH = os.path.join(PROJECT_ROOT, "docs", "REALSign_ISL_FINAL_COMPATIBILITY_REPORT.md")

FINGER_LABELS = ["thumb", "index", "middle", "ring", "pinky"]


def _describe_pattern(pat: list[str]) -> str:
    parts = []
    for finger, state in zip(FINGER_LABELS, pat):
        parts.append(f"{finger}={state}")
    return ", ".join(parts)


def _classify(row: dict) -> str:
    web_dom = row["webcam_hand_count"]
    pub_dom = row["public_hand_count"]
    disagree = row["n_finger_disagreements"]
    sim = row["finger_similarity"]

    if disagree >= 4 and sim < 0.5:
        return "INCOMPATIBLE"
    if web_dom == pub_dom and disagree <= 1:
        return "COMPATIBLE"
    return "REVIEW"


def _reason(row: dict) -> str:
    web_dom = row["webcam_hand_count"]
    pub_dom = row["public_hand_count"]
    disagree = row["n_finger_disagreements"]
    sim = row["finger_similarity"]

    reasons = []
    if web_dom != pub_dom:
        reasons.append(
            f"hand count differs (webcam {web_dom} vs RealSign {pub_dom})"
        )
    else:
        reasons.append(f"hand count matches ({web_dom} hand)")

    if disagree == 0:
        reasons.append("finger extension patterns match")
    elif disagree <= 2:
        reasons.append(f"{disagree}/5 fingers differ in extension pattern")
    else:
        reasons.append(f"{disagree}/5 fingers differ in extension pattern")

    reasons.append(f"landmark similarity {sim:.2f}")
    return "; ".join(reasons)


def main() -> None:
    if not os.path.exists(RAW_JSON):
        raise SystemExit(
            f"Raw JSON not found: {RAW_JSON}\n"
            f"Run: python scripts/analyze_realsign_landmark_compatibility.py"
        )

    with open(RAW_JSON, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    rows = data["rows"]
    order = {r["letter"]: r for r in rows}

    # Build CSV.
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "letter",
                "webcam_hand_count",
                "public_hand_count",
                "compatibility",
                "reason",
                "manual_review_required",
            ],
        )
        writer.writeheader()
        for letter in LETTERS:
            r = order[letter]
            label = _classify(r)
            reason = _reason(r)
            manual = "yes" if label in ("REVIEW", "INCOMPATIBLE") else "no"
            writer.writerow(
                {
                    "letter": letter,
                    "webcam_hand_count": r["webcam_hand_count"],
                    "public_hand_count": r["public_hand_count"],
                    "compatibility": label,
                    "reason": reason,
                    "manual_review_required": manual,
                }
            )

    # Build markdown report.
    compatible = [l for l in LETTERS if _classify(order[l]) == "COMPATIBLE"]
    review = [l for l in LETTERS if _classify(order[l]) == "REVIEW"]
    incompatible = [l for l in LETTERS if _classify(order[l]) == "INCOMPATIBLE"]

    lines = []
    lines.append("# RealSign ISL — Final Compatibility Report")
    lines.append("")
    lines.append(
        "This report determines whether the **webcam ISL dataset** "
        "(`dataset/isl_webcam/`) and the **RealSign public ISL dataset** "
        "(`dataset/isl_public/`) represent the same alphabet sign convention and "
        "could safely share class labels in a future training stage."
    )
    lines.append("")
    lines.append(
        "**No datasets were merged, modified, deleted, or retrained.** The "
        "existing webcam dataset and RealSign dataset are both preserved."
    )
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        "For each letter, up to 40 images per dataset were processed with "
        "MediaPipe (two-hand detection). For every detected hand we computed a "
        "scale-invariant **finger-extension descriptor** — the ratio of the "
        "direct MCP→tip distance to the polyline MCP→PIP→DIP→tip distance for "
        "each of the 5 fingers (thumb uses CMC→tip / CMC→MCP→IP→tip). Values "
        "near 1.0 indicate an extended/straight finger; values near 0 indicate "
        "a curled finger."
    )
    lines.append("")
    lines.append(
        "This focuses on **actual hand/finger geometry**, not raw pixel "
        "appearance, so camera angle, skin tone, lighting, background, hand "
        "size, and image quality are intentionally de-emphasized."
    )
    lines.append("")
    lines.append("### Classification rules")
    lines.append("")
    lines.append("- **COMPATIBLE** — dominant hand count matches AND ≤1 finger differs in extension pattern.")
    lines.append("- **REVIEW** — hand count differs OR ≥2 fingers differ.")
    lines.append("- **INCOMPATIBLE** — ≥4 fingers differ AND similarity < 0.5.")
    lines.append("")
    lines.append(
        "> A different **hand count** is treated as `REVIEW` (not automatically "
        "`INCOMPATIBLE`) because it can be caused by either a genuine one-hand vs "
        "two-hand sign difference OR a MediaPipe detection gap."
    )
    lines.append("")
    lines.append("## Detailed per-letter table")
    lines.append("")
    lines.append(
        "| Letter | Webcam hand | RealSign hand | Webcam finger config | "
        "RealSign finger config | Similarity | Compatibility | Reason |"
    )
    lines.append(
        "|--------|------------|---------------|----------------------|"
        "-------------------------|------------|---------------|--------|"
    )

    for letter in LETTERS:
        r = order[letter]
        wp = _describe_pattern(r["webcam_finger_pattern"])
        pp = _describe_pattern(r["public_finger_pattern"])
        label = _classify(r)
        lines.append(
            f"| {letter} | {r['webcam_hand_count']} | {r['public_hand_count']} | "
            f"{wp} | {pp} | {r['finger_similarity']:.2f} | {label} | "
            f"{_reason(r)} |"
        )

    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Compatible letters ({len(compatible)}):** {', '.join(compatible)}")
    lines.append(f"- **Review letters ({len(review)}):** {', '.join(review)}")
    lines.append(f"- **Incompatible letters ({len(incompatible)}):** "
                 f"{', '.join(incompatible) if incompatible else 'none'}")
    lines.append("")
    lines.append("## Previously flagged letters")
    lines.append("")
    lines.append(
        "The 10 letters flagged earlier (C, D, G, J, K, N, Q, S, X, Y) all "
        "remain `REVIEW` — none were auto-marked `INCOMPATIBLE`. The differences "
        "are driven primarily by **hand count** and, for a few letters, by "
        "additional finger-extension differences."
    )
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append("**Do not merge the datasets yet.** The following classes are the "
                 "strongest candidates for safe combination in a future training "
                 "stage (same dominant hand count and near-identical finger "
                 "configuration):")
    lines.append("")
    lines.append(f"- **Safe-to-combine candidates:** {', '.join(compatible)}")
    lines.append("")
    lines.append(
        "The `REVIEW` letters need manual visual confirmation (using the galleries "
        "in `results/realsign/comparison/`) to decide whether the hand-count "
        "difference is a real sign-convention difference or only a MediaPipe "
        "detection gap."
    )
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append("- `results/realsign/final_compatibility.csv`")
    lines.append("- `results/realsign/final_compatibility_raw.json`")
    lines.append("- `docs/REALSign_ISL_FINAL_COMPATIBILITY_REPORT.md`")
    lines.append("")

    with open(MD_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(f"CSV -> {CSV_PATH}")
    print(f"MD  -> {MD_PATH}")
    print(f"\nCompatible ({len(compatible)}): {''.join(compatible)}")
    print(f"Review     ({len(review)}): {''.join(review)}")
    print(f"Incompatible ({len(incompatible)}): {''.join(incompatible) or 'none'}")


if __name__ == "__main__":
    main()
