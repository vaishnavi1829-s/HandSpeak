# ISL (Indian Sign Language) Pipeline

This document describes the full 26-class ISL pipeline for HandSpeak, mirroring
the existing ASL workflow but keeping the two languages fully separate.

## Dataset

- Source: `dataset/isl_webcam/`
- Structure: `dataset/isl_webcam/<LETTER>/processed/<LETTER>_NNNNNN.png`
  (28x28 grayscale CNN input) and `dataset/isl_webcam/<LETTER>/roi/...`
  (original 300x300 ROI for inspection).
- Manifest: `dataset/isl_webcam/manifest.csv` records every sample with its
  letter and canonical A=0 ... Z=25 label index.

## Pipeline Steps

1. **Dataset verification**
   ```bash
   python scripts/inspect_isl_dataset.py
   ```
   Verifies 26/26 letter folders, per-letter counts, image readability,
   dimensions (28x28), channels (1), and value ranges.

2. **Preprocessing check**
   The ISL collection script reuses `recognition.preprocessing.preprocess_frame`
   verbatim, so the saved samples already match the CNN input spec
   (grayscale, hand-dark/background-light, centered, padded, `/255.0`).

3. **Train / Validation split**
   Reproducible 80/20 per-class stratified split (`random_state=42`).

4. **Data augmentation**
   Mild offline augmentation applied to the training split only (small
   rotation/translation/zoom/brightness; no horizontal flip).

5. **Train ISL 26-class CNN**
   ```bash
   python scripts/train_isl_cnn.py
   ```
   Trains a fresh 26-class model (same architecture as `models/cnn_model.py`)
   with EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint.

6. **Accuracy + Confusion Matrix**
   Reports are written to `reports/`:
   - `classification_report_isl.txt`
   - `confusion_matrix_isl.csv` / `confusion_matrix_isl.png`
   - `per_letter_accuracy_isl.txt`
   - `training_accuracy_loss_isl.png`

7. **Webcam testing (live ISL recognition)**
   ```bash
   python recognition/real_time_dual.py --language ISL
   ```

8. **ASL + ISL separate modes**
   Both modes are exposed by `recognition/predictor.py`. Press `1` (ASL) or
   `2` (ISL) in the live demo to switch. Models:
   - ASL -> `trained_model/handspeak_cnn_webcam.h5`
   - ISL -> `trained_model/handspeak_cnn_isl.h5`

9. **Application integration**
   ```bash
   python app.py
   ```
   Web UI at `http://127.0.0.1:5000/` with a language selector and image upload
   prediction. JSON API at `POST /api/predict` (fields: `image`, `language`).

## Models

| Language | Model path | Status |
|----------|-----------|--------|
| ASL      | `trained_model/handspeak_cnn_webcam.h5` | existing |
| ISL      | `trained_model/handspeak_cnn_isl.h5`     | trained (this pipeline) |
