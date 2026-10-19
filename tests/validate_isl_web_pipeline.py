"""Validation: confirm the web ISL pipeline matches the working demo pipeline.

Usage:
    python tests/validate_isl_web_pipeline.py
"""

from __future__ import annotations

import glob
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from recognition.landmark_extractor import (  # noqa: E402
    extract_two_hands_from_bgr,
    get_two_hand_landmarker,
)
from recognition.isl_feature_extractor import (  # noqa: E402
    rich_features_from_two_hand_result,
)
from recognition.landmark_predictor import LandmarkSignPredictor  # noqa: E402

ISL_ROOT = os.path.join(PROJECT_ROOT, "dataset", "isl_webcam")


def main() -> None:
    predictor = LandmarkSignPredictor("ISL")
    landmarker = get_two_hand_landmarker()

    files = sorted(glob.glob(os.path.join(ISL_ROOT, "*", "roi", "*.png")))
    print(f"Total ISL roi images: {len(files)}")

    tested = 0
    agreed = 0
    mismatches = 0
    no_hand = 0

    for path in files[:300]:
        img = cv2.imread(path)
        if img is None:
            continue

        web = predictor.predict(img)

        # Working script's ISL pipeline: raw (unmirrored) frame, two-hand
        # extraction, then rich geometric features.
        manual = extract_two_hands_from_bgr(img, landmarker=landmarker)
        if manual is None:
            no_hand += 1
            continue

        features, _ = rich_features_from_two_hand_result(manual)
        proba = predictor._model.predict_proba(features.reshape(1, -1))[0]
        working_letter = predictor._model.classes_[int(np.argmax(proba))]

        web_letter = web.get("prediction")
        if web_letter is None:
            mismatches += 1
            if mismatches <= 10:
                print(f"MISMATCH (web=None, work={working_letter}): {path}")
            continue

        tested += 1
        if web_letter == working_letter:
            agreed += 1
        else:
            mismatches += 1
            if mismatches <= 10:
                print(f"MISMATCH (web={web_letter}, work={working_letter}): {path}")

    print(f"tested={tested} agreed={agreed} mismatches={mismatches} "
          f"no_hand={no_hand}")


if __name__ == "__main__":
    main()
