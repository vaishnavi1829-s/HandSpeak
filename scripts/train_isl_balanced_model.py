"""Train + evaluate the BALANCED combined ISL experimental model.

Extracts 214-feature rich vectors for ``dataset/isl_balanced_combined``,
trains a NEW RandomForest, and evaluates both the existing production model and
the new balanced model on the SAME unseen webcam test set.

Outputs:
    trained_model/isl_balanced_combined_experimental.joblib
    results/realsign/balanced_combined_comparison.csv
    results/realsign/balanced_combined_metrics.json
    results/realsign/balanced_combined_confusion_matrix.png

Read-only w.r.t. originals; no existing model is replaced.
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
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from recognition.landmark_extractor import (  # noqa: E402
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

BALANCED_ROOT = os.path.join(PROJECT_ROOT, "dataset", "isl_balanced_combined")
LANDMARKS_DIR = os.path.join(PROJECT_ROOT, "dataset", "landmarks")
MODEL_DIR = os.path.join(PROJECT_ROOT, "trained_model")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "realsign")

EXPERIMENTAL_MODEL = os.path.join(MODEL_DIR, "isl_balanced_combined_experimental.joblib")
EXISTING_ONEHAND = os.path.join(MODEL_DIR, "isl_landmark_rf.joblib")

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


def _image_files(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(n for n in os.listdir(folder) if n.lower().endswith(IMG_EXTS))


def _extract(split, landmarker, csv_path):
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    rows = []
    root = os.path.join(BALANCED_ROOT, split)
    total = 0
    for letter in LETTERS:
        for name in _image_files(os.path.join(root, letter)):
            path = os.path.join(root, letter, name)
            result = extract_two_hands_from_file(path, landmarker=landmarker)
            total += 1
            if result is None:
                continue
            feats, _ = rich_features_from_two_hand_result(result)
            row = {"label": letter, "source_path": os.path.relpath(path, PROJECT_ROOT)}
            for col, val in zip(RICH_FEATURE_COLUMNS, feats):
                row[col] = f"{val:.8f}"
            rows.append(row)
    df = pd.DataFrame(rows, columns=["label", "source_path"] + RICH_FEATURE_COLUMNS)
    os.makedirs(LANDMARKS_DIR, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"  [{split}] {len(df)}/{total} with >=1 hand")
    return df


def _confusion_png(cm, classes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm, interpolation="nearest", cmap="viridis")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)),
           xticklabels=classes, yticklabels=classes, ylabel="True", xlabel="Predicted")
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
        "per_letter": {
            l: (float(accuracy_score(y_true[y_true == l], y_pred[y_true == l]))
                if (y_true == l).sum() else None)
            for l in labels
        },
    }


def _eval_onehand(model, test_dir):
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


def main():
    os.makedirs(LANDMARKS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    landmarker = get_two_hand_landmarker()
    start = time.time()
    train_df = _extract("train", landmarker, os.path.join(LANDMARKS_DIR, "isl_balanced_train_features.csv"))
    val_df = _extract("validation", landmarker, os.path.join(LANDMARKS_DIR, "isl_balanced_val_features.csv"))
    test_df = _extract("test_webcam", landmarker, os.path.join(LANDMARKS_DIR, "isl_balanced_test_webcam_features.csv"))
    print(f"[BALANCED] feature extraction {time.time()-start:.1f}s")

    X_train = train_df[RICH_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_train = train_df["label"].to_numpy()
    X_val = val_df[RICH_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_val = val_df["label"].to_numpy()

    clf = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X_train, y_train)
    joblib.dump(clf, EXPERIMENTAL_MODEL)
    val_acc = float(clf.score(X_val, y_val))
    print(f"[BALANCED] trained model -> {EXPERIMENTAL_MODEL}")
    print(f"[BALANCED] validation accuracy: {val_acc:.4f}")

    X_test = test_df[RICH_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_test = test_df["label"].to_numpy()
    y_pred_bal = clf.predict(X_test)
    bal_metrics = _metrics(y_test, y_pred_bal, LETTERS)

    # Existing one-hand model on same test.
    old = joblib.load(EXISTING_ONEHAND)
    y_old_true, y_old_pred = _eval_onehand(old, os.path.join(BALANCED_ROOT, "test_webcam"))
    old_metrics = _metrics(y_old_true, y_old_pred, LETTERS)

    cm = confusion_matrix(y_test, y_pred_bal, labels=LETTERS)
    _confusion_png(cm, LETTERS, os.path.join(RESULTS_DIR, "balanced_combined_confusion_matrix.png"))

    # Per-letter CSV.
    rows = []
    for letter in LETTERS:
        o = old_metrics["per_letter"].get(letter)
        b = bal_metrics["per_letter"].get(letter)
        change = (b - o) if (o is not None and b is not None) else None
        rows.append({
            "letter": letter,
            "existing_accuracy": o,
            "balanced_combined_accuracy": b,
            "change": change,
        })
    csv_path = os.path.join(RESULTS_DIR, "balanced_combined_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    metrics_out = {
        "balanced_N": 10,
        "val_per_class": 6,
        "validation_accuracy": val_acc,
        "existing": old_metrics,
        "balanced_combined": bal_metrics,
        "per_letter_comparison": rows,
    }
    with open(os.path.join(RESULTS_DIR, "balanced_combined_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics_out, fh, indent=2)

    print("\n[BALANCED] ===== UNSEEN WEBCAM TEST =====")
    print(f"  Existing : acc={old_metrics['accuracy']:.4f} P={old_metrics['precision_macro']:.4f} "
          f"R={old_metrics['recall_macro']:.4f} F1={old_metrics['f1_macro']:.4f}")
    print(f"  Balanced : acc={bal_metrics['accuracy']:.4f} P={bal_metrics['precision_macro']:.4f} "
          f"R={bal_metrics['recall_macro']:.4f} F1={bal_metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
