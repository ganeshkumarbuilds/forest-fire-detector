"""
Feature 11: Offline Color Ablation Experiment (grayscale sensitivity).
============================================================
OFFLINE only. No retraining, no recalibration, no threshold change,
no production-path change, no backend/frontend change, no overwrite
of previous artifacts, no causal claims.

Run:
    venv\\Scripts\\python.exe ml-model/color_ablation.py
"""
import json
import os
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

IMG_SIZE = (224, 224)
HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(HERE, "dataset")
FIRE_DIR = os.path.join(DATASET_DIR, "fire")
NO_FIRE_DIR = os.path.join(DATASET_DIR, "no_fire")
MODEL_PATH = os.path.join(HERE, "fire_model.h5")
ERROR_ANALYSIS_PATH = os.path.join(HERE, "error_analysis.json")
OUTPUT_PATH = os.path.join(HERE, "color_ablation.json")

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15 / 0.85
THRESHOLD = 0.5
BATCH_SIZE = 32


def _stratify_or_none(labels):
    labels = np.asarray(labels)
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return None
    if np.min(np.bincount(labels.astype(int))) < 2:
        return None
    return labels


def load_filenames_and_labels():
    fire_files = sorted([f for f in os.listdir(FIRE_DIR)
                         if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    nofire_files = sorted([f for f in os.listdir(NO_FIRE_DIR)
                           if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    all_files = ["fire/" + f for f in fire_files] + \
                ["no_fire/" + f for f in nofire_files]
    all_labels = [1] * len(fire_files) + [0] * len(nofire_files)
    return np.array(all_files), np.array(all_labels)


def resolve_path(prefixed):
    if prefixed.startswith("fire/"):
        return os.path.join(FIRE_DIR, os.path.basename(prefixed))
    return os.path.join(NO_FIRE_DIR, os.path.basename(prefixed))


def preprocess_clean(prefixed):
    """Same as robustness evaluation: original RGB -> resize 224 BILINEAR."""
    pil = Image.open(resolve_path(prefixed)).convert("RGB")
    resized = pil.resize(IMG_SIZE, Image.BILINEAR)
    arr = np.array(resized).astype("float32")
    return preprocess_input(arr)


def preprocess_grayscale(prefixed):
    """Original 250x250 RGB -> L -> RGB(3ch) -> resize 224 BILINEAR -> preprocess."""
    pil = Image.open(resolve_path(prefixed)).convert("RGB")
    gray3 = pil.convert("L").convert("RGB")
    resized = gray3.resize(IMG_SIZE, Image.BILINEAR)
    arr = np.array(resized).astype("float32")
    return preprocess_input(arr)


def compute_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "test_size": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred,
                                                zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred,
                                           zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def main():
    all_files, all_labels = load_filenames_and_labels()
    assert len(all_files) == 1832, f"Expected 1832, got {len(all_files)}"

    idx = np.arange(len(all_files))
    idx_temp, idx_test, y_temp, y_test, f_temp, f_test = train_test_split(
        idx, all_labels, all_files, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=_stratify_or_none(all_labels))
    idx_train, idx_val, _, _, _, _ = train_test_split(
        idx_temp, y_temp, f_temp, test_size=VAL_SIZE,
        random_state=RANDOM_STATE, stratify=_stratify_or_none(y_temp))
    assert len(idx_test) == 275 and len(idx_val) == 275 and len(idx_train) == 1282
    y_test = y_test.astype(int)
    assert int((y_test == 1).sum()) == 139
    assert int((y_test == 0).sum()) == 136
    print(f"Test: 275 (fire 139, no_fire 136)")

    print(f"Loading model {MODEL_PATH} (compile=False, no retraining)...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    print("Preprocessing CLEAN test set (robustness-style)...")
    X_clean = np.array([preprocess_clean(f) for f in f_test])
    print("Preprocessing GRAYSCALE test set...")
    X_gray = np.array([preprocess_grayscale(f) for f in f_test])

    print("Running clean batch inference...")
    clean_probs = model.predict(X_clean, batch_size=BATCH_SIZE,
                                verbose=1).flatten().astype(float)
    print("Running grayscale batch inference...")
    gray_probs = model.predict(X_gray, batch_size=BATCH_SIZE,
                               verbose=1).flatten().astype(float)
    clean_pred = (clean_probs > THRESHOLD).astype(int)
    gray_pred = (gray_probs > THRESHOLD).astype(int)

    clean_metrics = compute_metrics(y_test, clean_pred)
    gray_metrics = compute_metrics(y_test, gray_pred)
    assert (clean_metrics["tp"] + clean_metrics["tn"] +
            clean_metrics["fp"] + clean_metrics["fn"]) == 275
    assert (gray_metrics["tp"] + gray_metrics["tn"] +
            gray_metrics["fp"] + gray_metrics["fn"]) == 275
    # Verify clean matches existing baseline [[130,6],[0,139]].
    assert clean_metrics["confusion_matrix"] == [[130, 6], [0, 139]], \
        f"Clean baseline mismatch: {clean_metrics['confusion_matrix']}"

    deltas = {
        "accuracy_delta": round(gray_metrics["accuracy"] -
                                clean_metrics["accuracy"], 4),
        "precision_delta": round(gray_metrics["precision"] -
                                 clean_metrics["precision"], 4),
        "recall_delta": round(gray_metrics["recall"] -
                              clean_metrics["recall"], 4),
        "f1_delta": round(gray_metrics["f1"] - clean_metrics["f1"], 4),
        "fp_delta": int(gray_metrics["fp"] - clean_metrics["fp"]),
        "fn_delta": int(gray_metrics["fn"] - clean_metrics["fn"]),
    }
    change_rate = round(float(np.mean(gray_pred != clean_pred)), 4)
    avg_abs = round(float(np.mean(np.abs(gray_probs - clean_probs))), 4)

    fire_mask = y_test == 1
    nofire_mask = y_test == 0
    per_class = {
        "fire_change_rate": round(float(np.mean(
            gray_pred[fire_mask] != clean_pred[fire_mask])), 4),
        "no_fire_change_rate": round(float(np.mean(
            gray_pred[nofire_mask] != clean_pred[nofire_mask])), 4),
        "fire_n": int(fire_mask.sum()),
        "no_fire_n": int(nofire_mask.sum()),
    }

    # Six clean FPs from error_analysis.json (no inference on n=6).
    with open(ERROR_ANALYSIS_PATH, "r", encoding="utf-8") as f:
        analysis = json.load(f)
    clean_fps = analysis.get("false_positives", [])
    assert len(clean_fps) == 6
    fp_lookup = {r["filename"]: r for r in clean_fps}
    f_list = [str(x) for x in f_test.tolist()]
    fp_details = []
    for fp in clean_fps:
        fn = fp["filename"]
        i = f_list.index(fn)
        cp = round(float(clean_probs[i]), 4)
        gp = round(float(gray_probs[i]), 4)
        cpr, gpr = int(clean_pred[i]), int(gray_pred[i])
        fp_details.append({
            "filename": fn,
            "clean_prob_fire": cp,
            "grayscale_prob_fire": gp,
            "delta_probability": round(gp - cp, 4),
            "clean_prediction": cpr,
            "grayscale_prediction": gpr,
            "changed": bool(cpr != gpr),
        })
        stored = float(fp_lookup[fn]["prob_fire"])
        if abs(cp - stored) > 0.05:
            print(f"WARNING: clean prob drift for {fn}: "
                  f"run={cp:.4f} stored={stored:.4f}")
    n_changed = sum(1 for d in fp_details if d["changed"])
    fp_stability = {
        "total_clean_fp": 6,
        "changed": int(n_changed),
        "unchanged": int(6 - n_changed),
        "summary": f"{n_changed} of 6 clean false positives changed class "
                   f"under grayscale.",
        "details": fp_details,
        "note": "n=6: descriptive only, no statistical inference.",
    }

    per_image = []
    for k in range(len(f_test)):
        per_image.append({
            "filename": str(f_test[k]),
            "true_label": int(y_test[k]),
            "clean_prob_fire": round(float(clean_probs[k]), 4),
            "grayscale_prob_fire": round(float(gray_probs[k]), 4),
            "clean_pred": int(clean_pred[k]),
            "grayscale_pred": int(gray_pred[k]),
            "changed": bool(clean_pred[k] != gray_pred[k]),
        })

    try:
        import sklearn as _skl
        skl_ver = _skl.__version__
    except Exception:
        skl_ver = "unknown"
    try:
        import PIL as _pil
        pil_ver = _pil.__version__
    except Exception:
        pil_ver = "unknown"

    artifact = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_size": 1832,
            "test_size": 275,
            "fire_test": 139,
            "no_fire_test": 136,
            "random_state": RANDOM_STATE,
            "threshold": THRESHOLD,
            "input_size": [224, 224, 3],
            "grayscale_transformation": 'original 250x250 RGB -> PIL '
                'convert("L") (ITU-R 601 luma) -> convert("RGB") to '
                "replicate across 3 channels",
            "resize_method": "BILINEAR to 224x224",
            "preprocessing": "MobileNetV2 preprocess_input (x/127.5 - 1) "
                             "applied AFTER grayscale+resize; grayscale is "
                             "never applied after normalization",
            "batch_size": BATCH_SIZE,
            "model_path": MODEL_PATH,
            "sklearn_version": skl_ver,
            "pillow_version": pil_ver,
            "tf_version": tf.__version__,
        },
        "clean_metrics": clean_metrics,
        "grayscale_metrics": gray_metrics,
        "deltas": deltas,
        "prediction_change_rate": change_rate,
        "per_class_change_rates": per_class,
        "avg_absolute_probability_change": avg_abs,
        "clean_fp_stability": fp_stability,
        "per_image": per_image,
        "limitations": [
            "Grayscale removes hue and saturation jointly.",
            "Grayscale also changes brightness/contrast distributions.",
            "The model was not trained on grayscale images.",
            "Therefore degradation can reflect distribution shift rather "
            "than shortcut reliance.",
            "Results establish sensitivity to color removal, not causality.",
            "Results apply only to this frozen model and this test "
            "distribution.",
            "This is not an OOD benchmark.",
        ],
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    print(f"\n[OK] Color ablation saved to {OUTPUT_PATH}")
    print(f"Clean: {clean_metrics}")
    print(f"Gray:  {gray_metrics}")
    print(f"Deltas: {deltas}")
    print(f"Change rate: {change_rate} "
          f"(fire {per_class['fire_change_rate']}, "
          f"no_fire {per_class['no_fire_change_rate']}), "
          f"avg|dp|={avg_abs}")
    print(fp_stability["summary"])


if __name__ == "__main__":
    main()
