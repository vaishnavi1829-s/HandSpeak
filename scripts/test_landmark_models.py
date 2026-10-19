"""Offline prediction test for the landmark Random Forest models.

Select ASL or ISL, point the script at test images (or use the default test
folders under the webcam datasets), and it will:

    1. Load the corresponding ``*_landmark_rf.joblib`` model.
    2. Extract landmarks using the SAME preprocessing/normalization module
       used for dataset generation and training.
    3. Predict the letter and print actual vs. predicted label with the
       predicted-class probability.

Usage examples::

    python scripts/test_landmark_models.py --dataset asl
    python scripts/test_landmark_models.py --dataset isl
    python scripts/test_landmark_models.py --dataset asl --image dataset/webcam/A/roi/A_000002.png
    python scripts/test_landmark_models.py --dataset isl --folder dataset/isl_webcam/B/roi

This is an OFFLINE test only; it does not open the webcam.
"""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from recognition.landmark_extractor import (  # noqa: E402
    extract_landmarks_from_file,
    extract_two_hands_from_file,
)
from recognition.isl_feature_extractor import (  # noqa: E402
    rich_features_from_two_hand_result,
)
from utils.label_mapping import LETTERS  # noqa: E402

MODEL_DIR = os.path.join(PROJECT_ROOT, "trained_model")

DATASETS = {
    "asl": {
        "model": os.path.join(MODEL_DIR, "asl_landmark_rf.joblib"),
        "root": os.path.join(PROJECT_ROOT, "dataset", "webcam"),
        "language": "ASL",
        "two_hand": False,
        "rich": False,
    },
    "isl": {
        "model": os.path.join(MODEL_DIR, "isl_landmark_rf.joblib"),
        "root": os.path.join(PROJECT_ROOT, "dataset", "isl_webcam"),
        "language": "ISL",
        "two_hand": False,
        "rich": False,
    },
    "isl_2hand": {
        "model": os.path.join(MODEL_DIR, "isl_landmark_rf_2hand.joblib"),
        "root": os.path.join(PROJECT_ROOT, "dataset", "isl_webcam"),
        "language": "ISL-2HAND",
        "two_hand": True,
        "rich": False,
    },
    "isl_rich": {
        "model": os.path.join(MODEL_DIR, "isl_landmark_rf_rich.joblib"),
        "root": os.path.join(PROJECT_ROOT, "dataset", "isl_webcam"),
        "language": "ISL-RICH",
        "two_hand": True,
        "rich": True,
    },
}


def _image_files(folder: str, limit: int | None) -> list[str]:
    if not os.path.isdir(folder):
        return []
    files = sorted(
        name
        for name in os.listdir(folder)
        if name.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if limit is not None:
        files = files[:limit]
    return files


def infer_label_from_path(image_path: str) -> str | None:
    """Infer the actual A-Z label from the dataset folder structure.

    The webcam datasets are laid out as ``<root>/<LETTER>/roi/<file>``, so we
    walk up the path looking for a single-character folder name that is one of
    the 26 known letters.
    """
    path = os.path.abspath(image_path)
    while True:
        path, leaf = os.path.split(path)
        if not leaf:
            return None
        if len(leaf) == 1 and leaf in LETTERS:
            return leaf


def predict_image(
    model, image_path: str, two_hand: bool = False, rich: bool = False
) -> None:
    if rich:
        result = extract_two_hands_from_file(image_path)
        if result is None:
            print(f"  [SKIP] no hands detected: {image_path}")
            return
        n_hands = int(result.left_present) + int(result.right_present)
        features, _ = rich_features_from_two_hand_result(result)
        X = features.reshape(1, -1)
        feature_count = X.shape[1]
    elif two_hand:
        result = extract_two_hands_from_file(image_path)
        if result is None:
            print(f"  [SKIP] no hands detected: {image_path}")
            return
        n_hands = int(result.left_present) + int(result.right_present)
        if n_hands < 2:
            print(f"  [SKIP] only {n_hands} hand(s) detected (need 2): {image_path}")
            return
        X = result.feature_vector.reshape(1, -1)
        feature_count = X.shape[1]
    else:
        result = extract_landmarks_from_file(image_path)
        if result is None:
            print(f"  [SKIP] no hand detected: {image_path}")
            return
        X = result.feature_vector.reshape(1, -1)
        n_hands = 1
        feature_count = X.shape[1]

    proba = model.predict_proba(X)[0]
    pred_label = model.classes_[int(np.argmax(proba))]
    conf = float(np.max(proba))

    actual = infer_label_from_path(image_path)

    print(f"  Actual:         {actual or '?'}")
    print(f"  Predicted:      {pred_label}")
    print(f"  Confidence:     {conf:.4f}")
    print(f"  Hands detected: {n_hands}")
    print(f"  Feature count:  {feature_count}")
    print(f"  ({image_path})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["asl", "isl", "isl_2hand", "isl_rich"],
        required=True,
        help="Which language model to test (asl, isl, isl_2hand, isl_rich).",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a single image to predict.",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Folder of images to predict (actual labels inferred from parent folder).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max images to test per folder when using default test folders.",
    )
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]

    if not os.path.exists(cfg["model"]):
        raise SystemExit(f"Model not found: {cfg['model']}. Train it first.")

    print(f"[{cfg['language']}] Loading model {cfg['model']}")
    model = joblib.load(cfg["model"])
    print(f"[{cfg['language']}] Classes: {''.join(model.classes_.tolist())}\n")

    two_hand = cfg["two_hand"]
    rich = cfg.get("rich", False)

    if args.image:
        predict_image(model, args.image, two_hand=two_hand, rich=rich)
        return

    if args.folder:
        files = [os.path.join(args.folder, n) for n in _image_files(args.folder, None)]
        print(f"[{cfg['language']}] Testing {len(files)} image(s) from {args.folder}\n")
        for f in files:
            predict_image(model, f, two_hand=two_hand, rich=rich)
        return

    # Default: run a small sample across a few letters.
    print(f"[{cfg['language']}] Testing default sample across a few letters.\n")
    for letter in ["A", "B", "C", "D", "E"]:
        folder = os.path.join(cfg["root"], letter, "roi")
        files = _image_files(folder, args.limit)
        print(f"  Letter {letter}:")
        for name in files:
            predict_image(
                model, os.path.join(folder, name), two_hand=two_hand, rich=rich
            )


if __name__ == "__main__":
    main()
