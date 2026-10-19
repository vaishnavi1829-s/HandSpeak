"""Build the RealSign vs webcam ISL compatibility report (read-only).

Combines:
    * MediaPipe hand-count analysis of the RealSign public dataset
      (``results/realsign/isl_public_mediapipe.json``).
    * MediaPipe hand-count analysis of the EXISTING webcam ISL dataset
      (``results/landmark/isl_hand_usage_analysis.txt`` already reports the
      MediaPipe 0/1/2 hand distribution per letter; this script recomputes a
      quick summary from that file rather than re-running detection).

Writes ``results/realsign/isl_compatibility.csv`` (per-letter table) and
``results/realsign/isl_compatibility.json``.  This does NOT merge datasets and
does NOT modify any image/model.

Run from the project root::

    python scripts/build_realsign_compatibility.py
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.label_mapping import LETTERS  # noqa: E402

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "realsign")
PUBLIC_MP_JSON = os.path.join(RESULTS_DIR, "isl_public_mediapipe.json")
WEBCAM_HAND_USAGE_TXT = os.path.join(
    PROJECT_ROOT, "results", "landmark", "isl_hand_usage_analysis.txt"
)

# SPLITS to aggregate the public dataset over (Training + Testing + Validation).
SPLITS = ("Training", "Testing", "Validation")


def _load_public_hand_counts() -> dict[str, dict]:
    with open(PUBLIC_MP_JSON, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    per_letter: dict[str, dict] = {l: {"sampled": 0, "zero": 0, "one": 0, "two": 0, "failed": 0} for l in LETTERS}
    for split in SPLITS:
        classes = data["splits"][split]["classes"]
        for letter in LETTERS:
            c = classes.get(letter, {})
            per_letter[letter]["sampled"] += c.get("sampled", 0)
            per_letter[letter]["zero"] += c.get("zero", 0)
            per_letter[letter]["one"] += c.get("one", 0)
            per_letter[letter]["two"] += c.get("two", 0)
            per_letter[letter]["failed"] += c.get("failed", 0)
    return per_letter


def _load_webcam_hand_counts() -> dict[str, dict]:
    """Parse the existing ISL hand-usage analysis text file."""
    per_letter: dict[str, dict] = {l: {"mp_0": 0, "mp_1": 0, "mp_2": 0, "total": 0} for l in LETTERS}
    if not os.path.exists(WEBCAM_HAND_USAGE_TXT):
        return per_letter

    with open(WEBCAM_HAND_USAGE_TXT, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip()
            # Rows look like:  A           76     6    14    56       0      73       3 NEEDS_REVIEW
            m = re.match(
                r"^\s*([A-Z])\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)",
                line,
            )
            if not m:
                continue
            letter = m.group(1)
            if letter not in per_letter:
                continue
            total = int(m.group(2))
            mp0 = int(m.group(3))
            mp1 = int(m.group(4))
            mp2 = int(m.group(5))
            per_letter[letter] = {
                "total": total,
                "mp_0": mp0,
                "mp_1": mp1,
                "mp_2": mp2,
            }
    return per_letter


def _compat_label(public: dict, webcam: dict) -> str:
    """Heuristic compatibility label based on dominant hand count."""
    def dominant(zero, one, two):
        counts = {"0": zero, "1": one, "2": two}
        top = max(counts, key=counts.get)
        return top

    pub_total = public["sampled"]
    if pub_total == 0:
        return "UNKNOWN"

    pub_dom = dominant(public["zero"], public["one"], public["two"])

    web_total = webcam.get("total", 0)
    if web_total == 0:
        return "UNKNOWN"
    web_dom = dominant(webcam["mp_0"], webcam["mp_1"], webcam["mp_2"])

    if pub_dom == web_dom:
        return "MATCH"
    # One side is clearly two-hand, the other clearly one-hand.
    if {pub_dom, web_dom} == {"1", "2"}:
        return "REVIEW"
    if pub_dom == "0" or web_dom == "0":
        return "REVIEW"
    return "REVIEW"


def main() -> None:
    if not os.path.exists(PUBLIC_MP_JSON):
        raise SystemExit(
            f"Public MediaPipe JSON not found: {PUBLIC_MP_JSON}\n"
            f"Run: python scripts/analyze_realsign_mediapipe.py --max-per-class 50"
        )

    public = _load_public_hand_counts()
    webcam = _load_webcam_hand_counts()

    rows = []
    for letter in LETTERS:
        p = public[letter]
        w = webcam[letter]

        p_sampled = p["sampled"]
        p_one_pct = 100.0 * p["one"] / p_sampled if p_sampled else 0.0
        p_two_pct = 100.0 * p["two"] / p_sampled if p_sampled else 0.0

        w_total = w["total"]
        w_one_pct = 100.0 * w["mp_1"] / w_total if w_total else 0.0
        w_two_pct = 100.0 * w["mp_2"] / w_total if w_total else 0.0

        rows.append(
            {
                "letter": letter,
                "public_images_sampled": p_sampled,
                "public_1hand_pct": round(p_one_pct, 1),
                "public_2hand_pct": round(p_two_pct, 1),
                "webcam_images": w_total,
                "webcam_1hand_pct": round(w_one_pct, 1),
                "webcam_2hand_pct": round(w_two_pct, 1),
                "compatibility": _compat_label(p, w),
            }
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)

    csv_path = os.path.join(RESULTS_DIR, "isl_compatibility.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(RESULTS_DIR, "isl_compatibility.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "splits_aggregated": SPLITS}, fh, indent=2)

    # Print.
    print(f"{'Letter':<7}{'PubImages':>10}{'Pub1H%':>8}{'Pub2H%':>8}"
          f"{'WebImages':>10}{'Web1H%':>8}{'Web2H%':>8}{'Compat':>10}")
    print("-" * 70)
    for r in rows:
        print(f"{r['letter']:<7}{r['public_images_sampled']:>10}"
              f"{r['public_1hand_pct']:>8.1f}{r['public_2hand_pct']:>8.1f}"
              f"{r['webcam_images']:>10}{r['webcam_1hand_pct']:>8.1f}"
              f"{r['webcam_2hand_pct']:>8.1f}{r['compatibility']:>10}")

    print(f"\nCSV  -> {csv_path}")
    print(f"JSON -> {json_path}")


if __name__ == "__main__":
    main()
