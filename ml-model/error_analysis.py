"""
Forest Fire Detection - Error Analysis Script
==============================================
Reproducible ML error analysis on the deterministic test split.
Loads the trained Keras model directly for efficient batch inference.
"""
import os
import json
import numpy as np
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

IMG_SIZE = (224, 224)
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
FIRE_DIR = os.path.join(DATASET_DIR, "fire")
NO_FIRE_DIR = os.path.join(DATASET_DIR, "no_fire")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "fire_model.h5")
ERROR_ANALYSIS_PATH = os.path.join(os.path.dirname(__file__), "error_analysis.json")

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15 / 0.85


def _stratify_or_none(labels):
    """Return labels for stratify=, or None when stratification is unsafe."""
    labels = np.asarray(labels)
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return None
    if np.min(np.bincount(labels.astype(int))) < 2:
        return None
    return labels


def load_dataset():
    """Load all images with filenames, return arrays and metadata."""
    images, labels, filenames = [], [], []

    def load_folder(folder, label, prefix):
        if not os.path.isdir(folder):
            return 0
        count = 0
        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            try:
                img = load_img(os.path.join(folder, fname), target_size=IMG_SIZE)
                arr = img_to_array(img).astype("float32")
                images.append(arr)
                labels.append(label)
                filenames.append(prefix + fname)
                count += 1
            except Exception as e:
                print(f"Skipping {fname}: {e}")
        return count

    n_fire = load_folder(FIRE_DIR, 1, "fire/")
    n_no_fire = load_folder(NO_FIRE_DIR, 0, "no_fire/")
    print(f"Loaded {n_fire} fire images, {n_no_fire} no_fire images.")
    return np.array(images), np.array(labels), np.array(filenames)


def compute_quality_metrics(img_path):
    """Compute image quality metrics matching backend/app.py logic."""
    try:
        from PIL import Image
        img = Image.open(img_path)
        width, height = img.size

        # Blur score (Laplacian variance)
        gray = img.convert("L")
        gray_arr = np.array(gray, dtype=np.float32)
        laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        padded = np.pad(gray_arr, 1, mode='reflect')
        h, w = gray_arr.shape
        laplacian = np.zeros((h, w), dtype=np.float32)
        for i in range(h):
            for j in range(w):
                window = padded[i:i+3, j:j+3]
                laplacian[i, j] = np.sum(window * laplacian_kernel)
        blur_score = float(np.var(laplacian))

        # Brightness and contrast
        if img.mode != "RGB":
            img = img.convert("RGB")
        rgb_arr = np.array(img, dtype=np.float32)
        brightness = float(np.mean(rgb_arr))
        luminance = 0.299 * rgb_arr[:,:,0] + 0.587 * rgb_arr[:,:,1] + 0.114 * rgb_arr[:,:,2]
        contrast = float(np.std(luminance))

        # Warnings
        warnings = []
        if brightness < 30:
            warnings.append("Image appears very dark (possible nighttime or underexposed)")
        elif brightness > 225:
            warnings.append("Image appears very bright (possible overexposed)")
        if contrast < 10:
            warnings.append("Image has very low contrast (possible fog, smoke, or blur)")
        elif contrast > 80:
            warnings.append("Image has very high contrast")
        if blur_score < 20 and blur_score > 0:
            warnings.append("Image appears significantly blurred")
        if blur_score == 0.0:
            warnings.append("Image appears to be a solid color (not a real photo)")

        return {
            "width": width,
            "height": height,
            "blur_score": round(blur_score, 2),
            "brightness": round(brightness, 1),
            "contrast": round(contrast, 1),
            "warnings": warnings
        }
    except Exception as e:
        return {
            "width": 0, "height": 0,
            "blur_score": 0.0, "brightness": 0.0, "contrast": 0.0,
            "warnings": [f"Quality computation failed: {e}"]
        }


def main():
    print("Loading dataset...")
    X, y, filenames = load_dataset()
    if len(X) == 0:
        print("No images found!")
        return

    y = np.asarray(y)
    n_fire = int((y == 1).sum())
    n_no_fire = int((y == 0).sum())

    # Replicate EXACT split from train_model.py
    X_temp, X_test, y_temp, y_test, filenames_temp, filenames_test = train_test_split(
        X, y, filenames, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=_stratify_or_none(y),
    )
    X_train, X_val, y_train, y_val, filenames_train, filenames_val = train_test_split(
        X_temp, y_temp, filenames_temp, test_size=VAL_SIZE, random_state=RANDOM_STATE,
        stratify=_stratify_or_none(y_temp),
    )
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Test fire: {int((y_test == 1).sum())}, Test no_fire: {int((y_test == 0).sum())}")

    # Preprocess test set
    X_test_p = preprocess_input(X_test.astype("float32"))

    # Load Keras model
    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f"Model loaded: {model.count_params():,} params")

    # Batch inference on test set
    print("Running batch inference on test set...")
    batch_size = 32
    probs = model.predict(X_test_p, batch_size=batch_size, verbose=1).flatten()

    # Apply 0.5 threshold
    y_pred = (probs > 0.5).astype(int)
    y_true = y_test.astype(int)

    # Compute aggregate metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    test_size = len(y_true)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n=== Test Set Metrics ===")
    print(f"Test size: {test_size}")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Confusion Matrix: [[{tn}, {fp}], [{fn}, {tp}]]")

    # Verify math
    assert tp + tn + fp + fn == test_size, "Confusion matrix doesn't sum to test_size"
    assert abs(accuracy - (tp + tn) / test_size) < 1e-6, "Accuracy mismatch"
    if tp + fp > 0:
        assert abs(precision - tp / (tp + fp)) < 1e-6, "Precision mismatch"
    if tp + fn > 0:
        assert abs(recall - tp / (tp + fn)) < 1e-6, "Recall mismatch"
    print("[OK] All metric calculations verified")

    # Per-image results
    per_image = []
    false_positives = []
    false_negatives = []

    for i in range(test_size):
        fname = filenames_test[i]
        true_label = int(y_true[i])
        prob_fire = float(probs[i])
        pred_label = int(y_pred[i])
        correct = bool(pred_label == true_label)

        # Determine full path for quality metrics
        is_fire = true_label == 1
        folder = FIRE_DIR if is_fire else NO_FIRE_DIR
        img_path = os.path.join(folder, os.path.basename(fname))

        quality = compute_quality_metrics(img_path)

        entry = {
            "filename": fname,
            "true_label": true_label,
            "true_class": "fire" if true_label == 1 else "no_fire",
            "prob_fire": round(prob_fire, 4),
            "predicted_label": pred_label,
            "predicted_class": "fire" if pred_label == 1 else "no_fire",
            "correct": correct,
            "quality": quality
        }
        per_image.append(entry)

        if true_label == 0 and pred_label == 1:
            false_positives.append(entry)
        elif true_label == 1 and pred_label == 0:
            false_negatives.append(entry)

    # Descriptive error statistics
    fp_count = len(false_positives)
    fn_count = len(false_negatives)

    # Confidence distributions for errors
    fp_probs = [e["prob_fire"] for e in false_positives]
    fn_probs = [1.0 - e["prob_fire"] for e in false_negatives]

    # Quality warning associations
    fp_warnings_count = sum(1 for e in false_positives if e["quality"]["warnings"])
    fn_warnings_count = sum(1 for e in false_negatives if e["quality"]["warnings"])

    warning_types_fp = {}
    warning_types_fn = {}
    for e in false_positives:
        for w in e["quality"]["warnings"]:
            warning_types_fp[w] = warning_types_fp.get(w, 0) + 1
    for e in false_negatives:
        for w in e["quality"]["warnings"]:
            warning_types_fn[w] = warning_types_fn.get(w, 0) + 1

    descriptive_stats = {
        "fp_count": fp_count,
        "fn_count": fn_count,
        "fp_prob_fire_stats": {
            "min": round(min(fp_probs), 4) if fp_probs else None,
            "max": round(max(fp_probs), 4) if fp_probs else None,
            "mean": round(np.mean(fp_probs), 4) if fp_probs else None,
            "median": round(np.median(fp_probs), 4) if fp_probs else None,
        },
        "fn_prob_no_fire_stats": {
            "min": round(min(fn_probs), 4) if fn_probs else None,
            "max": round(max(fn_probs), 4) if fn_probs else None,
            "mean": round(np.mean(fn_probs), 4) if fn_probs else None,
            "median": round(np.median(fn_probs), 4) if fn_probs else None,
        },
        "fp_with_quality_warnings": fp_warnings_count,
        "fn_with_quality_warnings": fn_warnings_count,
        "fp_warning_types": warning_types_fp,
        "fn_warning_types": warning_types_fn,
    }

    # Build complete analysis artifact
    analysis = {
        "evaluation_metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_path": MODEL_PATH,
            "model_type": "MobileNetV2 transfer learning (frozen base)",
            "input_shape": [224, 224, 3],
            "decision_threshold": 0.5,
        },
        "dataset": {
            "total_images": len(X),
            "fire_count": n_fire,
            "no_fire_count": n_no_fire,
        },
        "split": {
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "random_state": RANDOM_STATE,
            "stratified": True,
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": test_size,
        },
        "aggregate_metrics": {
            "test_size": test_size,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        },
        "per_image_results": per_image,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "descriptive_error_statistics": descriptive_stats,
    }

    # Save to JSON
    with open(ERROR_ANALYSIS_PATH, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)
    print(f"\n[OK] Error analysis saved to {ERROR_ANALYSIS_PATH}")

    # Summary
    print(f"\n=== Summary ===")
    print(f"Test set size: {test_size}")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print(f"False positives: {fp_count}")
    print(f"False negatives: {fn_count}")
    if fp_count > 0:
        print(f"  FP prob_fire range: {min(fp_probs):.4f} - {max(fp_probs):.4f}")
        print(f"  FP with quality warnings: {fp_warnings_count}")
    if fn_count > 0:
        print(f"  FN prob_no_fire range: {min(fn_probs):.4f} - {max(fn_probs):.4f}")
        print(f"  FN with quality warnings: {fn_warnings_count}")


if __name__ == "__main__":
    main()