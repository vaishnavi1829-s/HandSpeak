"""Quantitative landmark compatibility analysis: webcam ISL vs RealSign ISL.

For every letter A-Z this script computes, for each dataset, a rotation- and
scale-invariant **finger-configuration descriptor** plus the detected hand-count
distribution.  It then compares the two datasets per letter and emits a
suggested compatibility label.  The final human report is generated separately;
this script provides the underlying measurements.

Descriptor (per detected hand):
    finger extension ratio = direct(MCP -> tip) / path(MCP->PIP->DIP->tip)
    (thumb: CMC -> tip / CMC->MCP->IP->tip)

A value near 1.0 means the finger is straight/extended; near 0 means curled.

Read-only: no dataset, image, or model is modified.

Run from the project root::

    python scripts/analyze_realsign_landmark_compatibility.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np  # noqa: E402

from recognition.landmark_extractor import (  # noqa: E402
    WRIST_IDX,
    extract_two_hands_from_file,
    get_two_hand_landmarker,
)
from utils.label_mapping import LETTERS  # noqa: E402

WEBCAM_ROOT = os.path.join(PROJECT_ROOT, "dataset", "isl_webcam")
PUBLIC_ROOT = os.path.join(PROJECT_ROOT, "dataset", "isl_public", "Training")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "realsign")

SAMPLE_PER_CLASS = 40
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# MediaPipe indices: [MCP, PIP, DIP, TIP] for fingers; thumb uses [CMC, MCP, IP, TIP].
FINGER_IDX = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}
FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]
EPS = 1e-6


def _image_files(folder: str, limit: int) -> list[str]:
    if not os.path.isdir(folder):
        return []
    files = sorted(
        n for n in os.listdir(folder) if n.lower().endswith(IMG_EXTS)
    )
    return files[:limit]


def _finger_extensions(norm: np.ndarray) -> np.ndarray:
    """Return 5 finger-extension ratios for a normalized (21,3) hand."""
    out = []
    for finger in FINGER_ORDER:
        idx = FINGER_IDX[finger]
        pts = norm[idx]
        path = sum(np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1))
        direct = np.linalg.norm(pts[-1] - pts[0])
        out.append(direct / (path + EPS))
    return np.asarray(out, dtype=np.float32)


def _analyze_dataset(root: str, letters: list[str]) -> dict:
    landmarker = get_two_hand_landmarker()
    out: dict = {}
    for letter in letters:
        folder = os.path.join(root, letter, "roi" if "isl_webcam" in root else "")
        # webcam uses <root>/<L>/roi; public uses <root>/<L>
        if "isl_webcam" in root:
            folder = os.path.join(root, letter, "roi")
        else:
            folder = os.path.join(root, letter)

        files = _image_files(folder, SAMPLE_PER_CLASS)
        hand_counts = {"0": 0, "1": 0, "2": 0}
        ext_vectors: list[np.ndarray] = []

        for name in files:
            path = os.path.join(folder, name)
            result = extract_two_hands_from_file(path, landmarker=landmarker)
            if result is None:
                hand_counts["0"] += 1
                continue

            n = int(result.left_present) + int(result.right_present)
            hand_counts[str(n)] += 1

            for raw in (result.left_landmarks, result.right_landmarks):
                if raw is None:
                    continue
                norm = raw - raw[WRIST_IDX]
                ref = np.linalg.norm(norm[9])  # same scale reference as pipeline
                if ref < EPS:
                    ref = EPS
                norm = norm / ref
                ext_vectors.append(_finger_extensions(norm))

        total = len(files)
        if total == 0:
            out[letter] = {
                "sampled": 0,
                "hand_counts": hand_counts,
                "hand_2hand_pct": 0.0,
                "finger_mean": [0.0] * 5,
                "finger_std": [0.0] * 5,
            }
            continue

        mean = (
            np.mean(ext_vectors, axis=0).tolist()
            if ext_vectors
            else [0.0] * 5
        )
        std = (
            np.std(ext_vectors, axis=0).tolist()
            if ext_vectors
            else [0.0] * 5
        )
        two_pct = 100.0 * hand_counts["2"] / total if total else 0.0

        out[letter] = {
            "sampled": total,
            "hand_counts": hand_counts,
            "hand_2hand_pct": round(two_pct, 1),
            "finger_mean": [round(x, 3) for x in mean],
            "finger_std": [round(x, 3) for x in std],
        }
    return out


def _dominant_hand_count(info: dict) -> int:
    hc = info["hand_counts"]
    return max(("0", "1", "2"), key=lambda k: hc.get(k, 0))


def _extended_pattern(mean: list[float], thr: float = 0.55) -> list[str]:
    return ["extended" if v >= thr else "curled" for v in mean]


def _suggest_compat(web: dict, pub: dict) -> tuple[str, float, int]:
    """Return (label, similarity, n_finger_disagreements)."""
    if web["sampled"] == 0 or pub["sampled"] == 0:
        return "REVIEW", 0.0, 5

    wm = np.asarray(web["finger_mean"])
    pm = np.asarray(pub["finger_mean"])
    denom = np.linalg.norm(wm) * np.linalg.norm(pm)
    cos = float(np.dot(wm, pm) / denom) if denom > EPS else 0.0
    mean_abs_diff = float(np.mean(np.abs(wm - pm)))

    w_pat = _extended_pattern(web["finger_mean"])
    p_pat = _extended_pattern(pub["finger_mean"])
    disagree = sum(1 for w, p in zip(w_pat, p_pat) if w != p)

    web_dom = _dominant_hand_count(web)
    pub_dom = _dominant_hand_count(pub)
    hand_same = web_dom == pub_dom

    # Similarity combines cosine and mean-abs-diff.
    similarity = float(0.5 * cos + 0.5 * (1.0 - min(1.0, mean_abs_diff)))

    if disagree <= 1 and hand_same and similarity >= 0.7:
        label = "COMPATIBLE"
    elif disagree >= 4 or similarity < 0.5:
        label = "INCOMPATIBLE"
    else:
        label = "REVIEW"

    return label, round(similarity, 3), disagree


def main() -> None:
    if not os.path.isdir(WEBCAM_ROOT):
        raise SystemExit(f"Webcam ISL root not found: {WEBCAM_ROOT}")
    if not os.path.isdir(PUBLIC_ROOT):
        raise SystemExit(f"Public root not found: {PUBLIC_ROOT}")

    start = time.time()
    print("[ANALYZE] Webcam ISL dataset...")
    web = _analyze_dataset(WEBCAM_ROOT, LETTERS)
    print("[ANALYZE] RealSign Training dataset...")
    pub = _analyze_dataset(PUBLIC_ROOT, LETTERS)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows = []
    for letter in LETTERS:
        w, p = web[letter], pub[letter]
        label, similarity, disagree = _suggest_compat(w, p)
        web_dom = _dominant_hand_count(w)
        pub_dom = _dominant_hand_count(p)
        rows.append(
            {
                "letter": letter,
                "webcam_hand_count": int(web_dom),
                "public_hand_count": int(pub_dom),
                "webcam_2hand_pct": w["hand_2hand_pct"],
                "public_2hand_pct": p["hand_2hand_pct"],
                "finger_similarity": similarity,
                "n_finger_disagreements": disagree,
                "webcam_finger_mean": w["finger_mean"],
                "public_finger_mean": p["finger_mean"],
                "webcam_finger_pattern": _extended_pattern(w["finger_mean"]),
                "public_finger_pattern": _extended_pattern(p["finger_mean"]),
                "compatibility": label,
            }
        )

    report = {"rows": rows, "sample_per_class": SAMPLE_PER_CLASS}
    json_path = os.path.join(RESULTS_DIR, "final_compatibility_raw.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    # Print table.
    hdr = (f"{'L':<2}{'W#':>4}{'P#':>4}{'W2h%':>6}{'P2h%':>6}"
           f"{'Sim':>6}{'Dis':>4}  {'Web pattern':<36}{'Pub pattern':<36}{'Label':>12}")
    print(hdr)
    print("-" * 120)
    for r in rows:
        wp = ",".join(x[0].upper() for x in r["webcam_finger_pattern"])
        pp = ",".join(x[0].upper() for x in r["public_finger_pattern"])
        print(f"{r['letter']:<2}{r['webcam_hand_count']:>4}{r['public_hand_count']:>4}"
              f"{r['webcam_2hand_pct']:>6.0f}{r['public_2hand_pct']:>6.0f}"
              f"{r['finger_similarity']:>6.3f}{r['n_finger_disagreements']:>4}"
              f"  {wp:<36}{pp:<36}{r['compatibility']:>12}")

    print(f"\nJSON -> {json_path}")
    print(f"Elapsed {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
