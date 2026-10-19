"""Train and evaluate the EXPERIMENTAL combined ISL rich-feature model.

Pipeline
--------
1. Extract 214-feature rich vectors for:
       dataset/isl_combined/train/
       dataset/isl_combined/validation/
       dataset/isl_combined/test_webcam/
   Features are cached to ``dataset/landmarks/isl_combined_*_features.csv``.
2. Train a NEW RandomForestClassifier on the train split (validation used for
   early feedback only).
3. Evaluate on the UNSEEN webcam test set (``test_webcam``).
4. Evaluate the SAME test set with the existing ISL models for comparison:
       trained_model/isl_landmark_rf.joblib        (63-feature, one-hand)
       trained_model/isl_landmark_rf_rich.joblib   (214-feature, rich)

Outputs
-------
    trained_model/isl_combined_landmark_experimental.joblib
    results/realsign/combined_model_metrics.json
    results/realsign/combined_model_comparison.md
    results/realsign/combined_model_confusion_matrix.png

This script never modifies existing datasets or models.

Run from the project root::

    python scripts/train_isl_combined_model.py
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

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from recognition.landmark_extractor import (  # noqa: E402
    FEATURE_COLUMNS,
    extract_landmarks_from_file,
    extract_two_hands_from_file,
    get_hand_landmarker,
    get_two_hand_landmarker,
)
from recognition.isl_feature_extractor import (  # noqa: E402
    RICH_FEATURE_COLUMNS,
    rich_features_from_two_hand_result,
)
from utils.label_mapping import LETTERS  # noqa: E402

COMBINED_ROOT = os.path.join(PROJECT_ROOT, "dataset", "isl_combined")
LANDMARKS_DIR = os.path.join(PROJECT_ROOT, "dataset", "landmarks")
MODEL_DIR = os.path.join(PROJECT_ROOT, "trained_model")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "realsign")

EXPERIMENTAL_MODEL = os.path.join(MODEL_DIR, "isl_combined_landmark_experimental.joblib")
EXISTING_ONEHAND_MODEL = os.path.join(MODEL_DIR, "isl_landmark_rf.joblib")
EXISTING_RICH_MODEL = os.path.join(MODEL_DIR, "isl_landmark_rf_rich.joblib")

SPLITS = ("train", "validation", "test_webcam")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

RANDOM_STATE = 42
RF_PARAMS = {
    "n_estimators": 400,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "class_weight": "balanced",
}


def _image_files(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    return sorted(
        n for n in os.listdir(folder) if n.lower().endswith(IMG_EXTS)
    )


def _extract_split(split: str, landmarker, csv_path: str) -> pd.DataFrame:
    """Extract rich features for one split; cache to CSV."""
    if os.path.exists(csv_path):
        print(f"  [cache] {split} -> {csv_path}")
        return pd.read_csv(csv_path)

    rows = []
    root = os.path.join(COMBINED_ROOT, split)
    n_total = 0
    for letter in LETTERS:
        folder = os.path.join(root, letter)
        files = _image_files(folder)
        for name in files:
            path = os.path.join(folder, name)
            result = extract_two_hands_from_file(path, landmarker=landmarker)
            n_total += 1
            if result is None:
                continue
            features, _ = rich_features_from_two_hand_result(result)
            row = {
                "label": letter,
                "source_path": os.path.relpath(path, PROJECT_ROOT),
                "hands_detected": int(result.left_present) + int(result.right_present),
            }
            for col, val in zip(RICH_FEATURE_COLUMNS, features):
                row[col] = f"{val:.8f}"
            rows.append(row)

    df = pd.DataFrame(rows, columns=["label", "source_path", "hands_detected"] + RICH_FEATURE_COLUMNS)
    os.makedirs(LANDMARKS_DIR, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"  [extract] {split}: {len(df)}/{n_total} with >=1 hand -> {csv_path}")
    return df


def _confusion_matrix_png(cm, classes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm, interpolation="nearest", cmap="viridis")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)),
           xticklabels=classes, yticklabels=classes,
           ylabel="True", xlabel="Predicted")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="w" if cm[i, j] > cm.max() / 2 else "k", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _metrics(y_true, y_pred, labels):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "n": int(len(y_true)),
        "per_letter_accuracy": {
            l: (float(accuracy_score(y_true[y_true == l], y_pred[y_true == l]))
                if (y_true == l).sum() else None)
            for l in labels
        },
    }


def _eval_onehand_model(model, test_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the 63-feature one-hand model by re-extracting single-hand features."""
    landmarker = get_hand_landmarker()
    ys, preds = [], []
    for letter in LETTERS:
        for name in _image_files(os.path.join(test_dir, letter)):
            path = os.path.join(test_dir, letter, name)
            res = extract_landmarks_from_file(path, landmarker=landmarker)
            if res is None:
                continue
            ys.append(letter)
            preds.append(model.predict(res.feature_vector.reshape(1, -1))[0])
    return np.asarray(ys), np.asarray(preds)


def main() -> None:
    os.makedirs(LANDMARKS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("[COMBINED] Extracting rich features...")
    landmarker = get_two_hand_landmarker()
    start = time.time()

    train_csv = os.path.join(LANDMARKS_DIR, "isl_combined_train_features.csv")
    val_csv = os.path.join(LANDMARKS_DIR, "isl_combined_val_features.csv")
    test_csv = os.path.join(LANDMARKS_DIR, "isl_combined_test_webcam_features.csv")

    train_df = _extract_split("train", landmarker, train_csv)
    val_df = _extract_split("validation", landmarker, val_csv)
    test_df = _extract_split("test_webcam", landmarker, test_csv)
    print(f"[COMBINED] Feature extraction done in {time.time() - start:.1f}s")

    X_train = train_df[RICH_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_train = train_df["label"].to_numpy()
    X_val = val_df[RICH_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_val = val_df["label"].to_numpy()

    print(f"[COMBINED] train={len(X_train)}  val={len(X_val)}  test={len(test_df)}")

    clf = RandomForestClassifier(**RF_PARAMS)
    print("[COMBINED] Training experimental RandomForest...")
    clf.fit(X_train, y_train)
    joblib.dump(clf, EXPERIMENTAL_MODEL)
    val_acc = float(clf.score(X_val, y_val))
    print(f"[COMBINED] Experimental model saved -> {EXPERIMENTAL_MODEL}")
    print(f"[COMBINED] Validation accuracy (webcam+realsign val): {val_acc:.4f}")

    # Evaluate experimental on unseen webcam test.
    X_test = test_df[RICH_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_test = test_df["label"].to_numpy()
    y_pred_exp = clf.predict(X_test)
    exp_metrics = _metrics(y_test, y_pred_exp, LETTERS)

    # Evaluate existing one-hand model on the SAME test set.
    print("[COMBINED] Evaluating existing one-hand ISL model...")
    old_one = joblib.load(EXISTING_ONEHAND_MODEL)
    y_test_old, y_pred_old = _eval_onehand_model(old_one, os.path.join(COMBINED_ROOT, "test_webcam"))
    old_metrics = _metrics(y_test_old, y_pred_old, LETTERS)

    # Evaluate existing rich model on the SAME test set (rich features already cached).
    print("[COMBINED] Evaluating existing rich ISL model...")
    old_rich = joblib.load(EXISTING_RICH_MODEL)
    y_pred_old_rich = old_rich.predict(X_test)
    old_rich_metrics = _metrics(y_test, y_pred_old_rich, LETTERS)

    # Confusion matrix for experimental.
    cm = confusion_matrix(y_test, y_pred_exp, labels=LETTERS)
    _confusion_matrix_png(
        cm, LETTERS, os.path.join(RESULTS_DIR, "combined_model_confusion_matrix.png")
    )

    # Build comparison table.
    per_letter = []
    for letter in LETTERS:
        old_a = old_metrics["per_letter_accuracy"].get(letter)
        exp_a = exp_metrics["per_letter_accuracy"].get(letter)
        change = (exp_a - old_a) if (old_a is not None and exp_a is not None) else None
        per_letter.append(
            {
                "letter": letter,
                "existing_accuracy": old_a,
                "combined_accuracy": exp_a,
                "change": change,
            }
        )

    comparison_md = [
        "# Combined-Data Model Comparison (unseen webcam test set)",
        "",
        f"Unseen webcam test images: {len(y_test)} (all A-Z classes).",
        "",
        "| Letter | Existing Model Accuracy | Combined Model Accuracy | Change |",
        "|--------|------------------------|-------------------------|--------|",
    ]
    for r in per_letter:
        o = f"{r['existing_accuracy']:.3f}" if r["existing_accuracy"] is not None else "n/a"
        c = f"{r['combined_accuracy']:.3f}" if r["combined_accuracy"] is not None else "n/a"
        ch = f"{r['change']:+.3f}" if r["change"] is not None else "n/a"
        comparison_md.append(f"| {r['letter']} | {o} | {c} | {ch} |")

    comparison_md.append("")
    comparison_md.append("## Overall")
    comparison_md.append("")
    comparison_md.append(
        f"- Existing one-hand model accuracy: **{old_metrics['accuracy']:.4f}** "
        f"(n={old_metrics['n']})"
    )
    comparison_md.append(
        f"- Existing rich model accuracy: **{old_rich_metrics['accuracy']:.4f}** "
        f"(n={old_rich_metrics['n']})"
    )
    comparison_md.append(
        f"- Combined experimental model accuracy: **{exp_metrics['accuracy']:.4f}** "
        f"(n={exp_metrics['n']})"
    )
    comparison_md.append("")
    comparison_md.append("## Macro metrics (unseen webcam test)")
    comparison_md.append("")
    comparison_md.append(
        "| Model | Accuracy | Precision | Recall | F1 |"
    )
    comparison_md.append("|-------|----------|-----------|--------|----|")
    for name, m in (
        ("Existing one-hand", old_metrics),
        ("Existing rich", old_rich_metrics),
        ("Combined experimental", exp_metrics),
    ):
        comparison_md.append(
            f"| {name} | {m['accuracy']:.4f} | {m['precision_macro']:.4f} | "
            f"{m['recall_macro']:.4f} | {m['f1_macro']:.4f} |"
        )

    md_path = os.path.join(RESULTS_DIR, "combined_model_comparison.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(comparison_md) + "\n")

    # JSON metrics.
    metrics_out = {
        "validation_accuracy": val_acc,
        "existing_onehand": old_metrics,
        "existing_rich": old_rich_metrics,
        "combined_experimental": exp_metrics,
        "per_letter_comparison": per_letter,
    }
    with open(os.path.join(RESULTS_DIR, "combined_model_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics_out, fh, indent=2)

    print("\n[COMBINED] ============ UNSEEN WEBCAM TEST RESULTS ============")
    print(f"  Existing one-hand : acc={old_metrics['accuracy']:.4f}  "
          f"P={old_metrics['precision_macro']:.4f}  R={old_metrics['recall_macro']:.4f}  "
          f"F1={old_metrics['f1_macro']:.4f}")
    print(f"  Existing rich     : acc={old_rich_metrics['accuracy']:.4f}  "
          f"P={old_rich_metrics['precision_macro']:.4f}  R={old_rich_metrics['recall_macro']:.4f}  "
          f"F1={old_rich_metrics['f1_macro']:.4f}")
    print(f"  Combined exp.     : acc={exp_metrics['accuracy']:.4f}  "
          f"P={exp_metrics['precision_macro']:.4f}  R={exp_metrics['recall_macro']:.4f}  "
          f"F1={exp_metrics['f1_macro']:.4f}")
    print(f"\n  comparison -> {md_path}")
    print(f"  metrics    -> {os.path.join(RESULTS_DIR, 'combined_model_metrics.json')}")


if __name__ == "__main__":
    main()
