"""
Feature 13: Controlled Saturation Ablation (offline sensitivity experiment).
============================================================
Frozen fire_model.h5 response to progressive chromatic attenuation via
PIL ImageEnhance.Color. No retraining, no model/backend/frontend changes,
no overwrite of prior artifacts, no causal claims.

Run:
    venv\\Scripts\\python.exe ml-model/saturation_ablation.py
"""
import json
import os
import platform
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image, ImageEnhance

IMG_SIZE = (224, 224)
HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(HERE, "dataset")
FIRE_DIR = os.path.join(DATASET_DIR, "fire")
NO_FIRE_DIR = os.path.join(DATASET_DIR, "no_fire")
MODEL_PATH = os.path.join(HERE, "fire_model.h5")
ERROR_ANALYSIS_PATH = os.path.join(HERE, "error_analysis.json")
OUTPUT_PATH = os.path.join(HERE, "saturation_ablation.json")

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15 / 0.85
THRESHOLD = 0.5
BATCH_SIZE = 32
LEVELS = [1.00, 0.75, 0.50, 0.25, 0.00]


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


def preprocess_sat(prefixed, factor):
    """Original RGB -> PIL Color(factor) -> resize once -> preprocess."""
    pil = Image.open(resolve_path(prefixed)).convert("RGB")
    mod = ImageEnhance.Color(pil).enhance(float(factor))
    resized = mod.resize(IMG_SIZE, Image.BILINEAR)
    return preprocess_input(np.array(resized).astype("float32"))


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
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(f"Dataset not found: {DATASET_DIR}")

    all_files, all_labels = load_filenames_and_labels()
    assert len(all_files) == 1832, f"Expected 1832, got {len(all_files)}"

    idx = np.arange(len(all_files))
    idx_temp, idx_test, y_temp, y_test, f_temp, f_test = train_test_split(
        idx, all_labels, all_files, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=_stratify_or_none(all_labels))
    idx_train, idx_val, _, _, _, _ = train_test_split(
        idx_temp, y_temp, f_temp, test_size=VAL_SIZE,
        random_state=RANDOM_STATE, stratify=_stratify_or_none(y_temp))
    assert len(idx_test) == 275 and len(idx_val) == 275
    assert len(idx_train) == 1282
    y_test = y_test.astype(int)
    assert int((y_test == 1).sum()) == 139
    assert int((y_test == 0).sum()) == 136

    print("Loading model (compile=False, no retraining)...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    probs, preds = {}, {}
    for lv in LEVELS:
        print(f"Preprocessing + inference for saturation {lv:.2f}...")
        X = np.array([preprocess_sat(f, lv) for f in f_test])
        p = model.predict(X, batch_size=BATCH_SIZE,
                          verbose=1).flatten().astype(float)
        key = f"{lv:.2f}"
        probs[key] = p
        preds[key] = (p > THRESHOLD).astype(int)

    metrics = {k: compute_metrics(y_test, preds[k]) for k in probs}
    for m in metrics.values():
        assert m["tp"] + m["tn"] + m["fp"] + m["fn"] == 275
    clean = metrics["1.00"]
    assert clean["confusion_matrix"] == [[130, 6], [0, 139]], \
        f"Clean baseline mismatch: {clean['confusion_matrix']}"
    assert (clean["accuracy"], clean["precision"], clean["recall"],
            clean["f1"]) == (0.9782, 0.9586, 1.0, 0.9789)

    fire_mask = y_test == 1
    nofire_mask = y_test == 0
    deltas, changes, per_class, mdp = {}, {}, {}, {}
    for k in probs:
        if k == "1.00":
            continue
        m = metrics[k]
        deltas[k] = {
            "accuracy_delta": round(m["accuracy"] - clean["accuracy"], 4),
            "precision_delta": round(m["precision"] - clean["precision"], 4),
            "recall_delta": round(m["recall"] - clean["recall"], 4),
            "f1_delta": round(m["f1"] - clean["f1"], 4),
            "fp_delta": int(m["fp"] - clean["fp"]),
            "fn_delta": int(m["fn"] - clean["fn"]),
        }
        changes[k] = round(float(np.mean(preds[k] != preds["1.00"])), 4)
        mdp[k] = round(float(np.mean(np.abs(probs[k] - probs["1.00"]))), 4)
        per_class[k] = {
            "fire_change_rate": round(float(np.mean(
                preds[k][fire_mask] != preds["1.00"][fire_mask])), 4),
            "no_fire_change_rate": round(float(np.mean(
                preds[k][nofire_mask] != preds["1.00"][nofire_mask])), 4),
            "fire_n": 139, "no_fire_n": 136,
            "fire_changed": int((preds[k][fire_mask] !=
                                 preds["1.00"][fire_mask]).sum()),
            "no_fire_changed": int((preds[k][nofire_mask] !=
                                     preds["1.00"][nofire_mask]).sum()),
        }

    with open(ERROR_ANALYSIS_PATH, "r", encoding="utf-8") as f:
        analysis = json.load(f)
    clean_fps = analysis.get("false_positives", [])
    assert len(clean_fps) == 6
    f_list = [str(x) for x in f_test.tolist()]
    fp_stability = {}
    for k in probs:
        if k == "1.00":
            continue
        details = []
        for fp in clean_fps:
            fn = fp["filename"]
            i = f_list.index(fn)
            cp = round(float(probs["1.00"][i]), 4)
            lp = round(float(probs[k][i]), 4)
            cpr, lpr = int(preds["1.00"][i]), int(preds[k][i])
            details.append({
                "filename": fn,
                "clean_prob_fire": cp,
                "level_prob_fire": lp,
                "delta_probability": round(lp - cp, 4),
                "clean_prediction": cpr,
                "level_prediction": lpr,
                "changed": bool(cpr != lpr),
            })
        n_ch = sum(1 for d in details if d["changed"])
        fp_stability[k] = {
            "total_clean_fp": 6,
            "changed": int(n_ch),
            "unchanged": int(6 - n_ch),
            "summary": f"{n_ch} of 6 clean false positives changed at "
                       f"saturation {k}.",
            "details": details,
            "note": "Descriptive only; n=6 is small.",
        }

    per_image = []
    for i in range(len(f_test)):
        row = {"filename": str(f_test[i]),
               "true_label": int(y_test[i]),
               "clean_prob_fire": round(float(probs["1.00"][i]), 4),
               "clean_prediction": int(preds["1.00"][i]),
               "saturation_1.00_prob_fire": round(float(probs["1.00"][i]), 4),
               "saturation_1.00_prediction": int(preds["1.00"][i])}
        for k in ("0.75", "0.50", "0.25", "0.00"):
            row[f"saturation_{k}_prob_fire"] = round(float(probs[k][i]), 4)
            row[f"saturation_{k}_prediction"] = int(preds[k][i])
            row[f"saturation_{k}_changed"] = bool(preds[k][i] != preds["1.00"][i])
            row[f"saturation_{k}_abs_prob_change"] = round(
                float(abs(probs[k][i] - probs["1.00"][i])), 4)
        per_image.append(row)

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
            "batch_size": BATCH_SIZE,
            "model_path": MODEL_PATH,
            "transformation": "original 250x250 RGB -> "
                              "ImageEnhance.Color(enhance=factor) -> resize "
                              "224x224 BILINEAR -> float32 -> "
                              "preprocess_input",
            "pil_color_factor": "1.00=original; 0.00=luma gray; linear RGB "
                                "interpolation, not HSV/HSL",
            "levels": [f"{lv:.2f}" for lv in LEVELS],
            "resize_method": "BILINEAR, exactly once",
            "preprocessing": "MobileNetV2 preprocess_input (x/127.5 - 1) "
                             "after resize; never perturb normalized tensors",
            "tensorflow_version": tf.__version__,
            "python_version": platform.python_version(),
        },
        "clean_metrics": clean,
        "levels": {k: metrics[k] for k in ("1.00", "0.75", "0.50",
                                           "0.25", "0.00")},
        "deltas_vs_clean": deltas,
        "change_rates": changes,
        "mean_abs_prob_change": mdp,
        "per_class_change_rates": per_class,
        "clean_fp_stability": fp_stability,
        "per_image": per_image,
        "limitations": [
            "The experiment measures sensitivity to controlled "
            "color/saturation reduction, not causal feature importance.",
            "ImageEnhance.Color is PIL's RGB-space interpolation toward "
            "grayscale/luma; it is not HSV/HSL perceptual saturation scaling.",
            "Luminance is approximately preserved but not mathematically "
            "guaranteed to be bit-identical because of image-operation "
            "rounding.",
            "The transformed images are still distribution-shifted relative "
            "to the natural RGB training data.",
            "Any performance degradation can therefore combine information "
            "removal with generic OOD/backbone effects.",
            "Fire genuinely contains chromatic information, so sensitivity "
            "to saturation does not imply a learned shortcut.",
            "Dataset-level color differences can contribute to the observed "
            "behavior.",
            "Results apply only to this frozen model and this test split.",
            "The six-clean-FP analysis is descriptive because n=6 is small.",
        ],
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    print(f"[OK] Saturation ablation saved to {OUTPUT_PATH}")
    for k in ("1.00", "0.75", "0.50", "0.25", "0.00"):
        print(f"  {k}: {metrics[k]}")
    for k in ("0.75", "0.50", "0.25", "0.00"):
        print(f"  {k}: change={changes[k]} |dp|={mdp[k]} "
              f"fire={per_class[k]['fire_change_rate']} "
              f"nofire={per_class[k]['no_fire_change_rate']} | "
              f"{fp_stability[k]['summary']}")


if __name__ == "__main__":
    main()
