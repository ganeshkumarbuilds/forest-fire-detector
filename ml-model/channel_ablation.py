"""
Feature 12: RGB Channel Ablation (offline sensitivity experiment).
============================================================
Measures sensitivity of frozen fire_model.h5 to removing individual
RGB channels. No retraining, no model/backend/frontend changes,
no overwrite of prior artifacts, no causal claims.

Run:
    venv\\Scripts\\python.exe ml-model/channel_ablation.py
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
from PIL import Image

IMG_SIZE = (224, 224)
HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(HERE, "dataset")
FIRE_DIR = os.path.join(DATASET_DIR, "fire")
NO_FIRE_DIR = os.path.join(DATASET_DIR, "no_fire")
MODEL_PATH = os.path.join(HERE, "fire_model.h5")
ERROR_ANALYSIS_PATH = os.path.join(HERE, "error_analysis.json")
OUTPUT_PATH = os.path.join(HERE, "channel_ablation.json")

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
    pil = Image.open(resolve_path(prefixed)).convert("RGB")
    resized = pil.resize(IMG_SIZE, Image.BILINEAR)
    return preprocess_input(np.array(resized).astype("float32"))


def preprocess_channel(prefixed, ch):
    """Extract channel ch (0=R,1=G,2=B), replicate [c,c,c], resize once."""
    pil = Image.open(resolve_path(prefixed)).convert("RGB")
    a = np.array(pil)
    rep = np.repeat(a[:, :, ch:ch + 1], 3, axis=2).astype("uint8")
    resized = Image.fromarray(rep).resize(IMG_SIZE, Image.BILINEAR)
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

    print("Preprocessing clean + R/G/B-only sets...")
    X_clean = np.array([preprocess_clean(f) for f in f_test])
    X_r = np.array([preprocess_channel(f, 0) for f in f_test])
    X_g = np.array([preprocess_channel(f, 1) for f in f_test])
    X_b = np.array([preprocess_channel(f, 2) for f in f_test])

    print("Running batch inference (clean)...")
    p_clean = model.predict(X_clean, batch_size=BATCH_SIZE,
                            verbose=1).flatten().astype(float)
    print("Running batch inference (R-only)...")
    p_r = model.predict(X_r, batch_size=BATCH_SIZE,
                        verbose=1).flatten().astype(float)
    print("Running batch inference (G-only)...")
    p_g = model.predict(X_g, batch_size=BATCH_SIZE,
                        verbose=1).flatten().astype(float)
    print("Running batch inference (B-only)...")
    p_b = model.predict(X_b, batch_size=BATCH_SIZE,
                        verbose=1).flatten().astype(float)

    c_pred = (p_clean > THRESHOLD).astype(int)
    r_pred = (p_r > THRESHOLD).astype(int)
    g_pred = (p_g > THRESHOLD).astype(int)
    b_pred = (p_b > THRESHOLD).astype(int)

    clean_m = compute_metrics(y_test, c_pred)
    r_m = compute_metrics(y_test, r_pred)
    g_m = compute_metrics(y_test, g_pred)
    b_m = compute_metrics(y_test, b_pred)
    for m in (clean_m, r_m, g_m, b_m):
        assert m["tp"] + m["tn"] + m["fp"] + m["fn"] == 275
    assert clean_m["confusion_matrix"] == [[130, 6], [0, 139]], \
        f"Clean baseline mismatch: {clean_m['confusion_matrix']}"
    assert (clean_m["accuracy"], clean_m["precision"], clean_m["recall"],
            clean_m["f1"]) == (0.9782, 0.9586, 1.0, 0.9789)

    def rel(abl_m, abl_p, abl_pred):
        return {
            "accuracy_delta": round(abl_m["accuracy"] - clean_m["accuracy"], 4),
            "precision_delta": round(abl_m["precision"] - clean_m["precision"], 4),
            "recall_delta": round(abl_m["recall"] - clean_m["recall"], 4),
            "f1_delta": round(abl_m["f1"] - clean_m["f1"], 4),
            "fp_delta": int(abl_m["fp"] - clean_m["fp"]),
            "fn_delta": int(abl_m["fn"] - clean_m["fn"]),
            "prediction_change_rate": round(float(np.mean(abl_pred != c_pred)), 4),
            "avg_absolute_probability_change": round(
                float(np.mean(np.abs(abl_p - p_clean))), 4),
        }

    fire_mask = y_test == 1
    nofire_mask = y_test == 0

    def per_class(abl_pred):
        return {
            "fire_change_rate": round(float(np.mean(
                abl_pred[fire_mask] != c_pred[fire_mask])), 4),
            "no_fire_change_rate": round(float(np.mean(
                abl_pred[nofire_mask] != c_pred[nofire_mask])), 4),
            "fire_n": 139, "no_fire_n": 136,
            "fire_changed": int((abl_pred[fire_mask] !=
                                 c_pred[fire_mask]).sum()),
            "no_fire_changed": int((abl_pred[nofire_mask] !=
                                    c_pred[nofire_mask]).sum()),
        }

    with open(ERROR_ANALYSIS_PATH, "r", encoding="utf-8") as f:
        analysis = json.load(f)
    clean_fps = analysis.get("false_positives", [])
    assert len(clean_fps) == 6
    f_list = [str(x) for x in f_test.tolist()]

    fp_stability = {}
    for tag, abl_p, abl_pred in (("r_only", p_r, r_pred),
                                 ("g_only", p_g, g_pred),
                                 ("b_only", p_b, b_pred)):
        details = []
        for fp in clean_fps:
            fn = fp["filename"]
            i = f_list.index(fn)
            cp = round(float(p_clean[i]), 4)
            ap = round(float(abl_p[i]), 4)
            cpr, apr = int(c_pred[i]), int(abl_pred[i])
            details.append({
                "filename": fn,
                "clean_prob_fire": cp,
                "ablated_prob_fire": ap,
                "delta_probability": round(ap - cp, 4),
                "clean_prediction": cpr,
                "ablated_prediction": apr,
                "changed": bool(cpr != apr),
            })
        n_ch = sum(1 for d in details if d["changed"])
        fp_stability[tag] = {
            "total_clean_fp": 6,
            "changed": int(n_ch),
            "unchanged": int(6 - n_ch),
            "summary": f"{n_ch} of 6 clean false positives changed under "
                       f"{tag}.",
            "details": details,
            "note": "Descriptive only; n=6 is small.",
        }

    per_image = []
    for k in range(len(f_test)):
        per_image.append({
            "filename": str(f_test[k]),
            "true_label": int(y_test[k]),
            "clean_prob_fire": round(float(p_clean[k]), 4),
            "clean_prediction": int(c_pred[k]),
            "r_only_prob_fire": round(float(p_r[k]), 4),
            "r_only_prediction": int(r_pred[k]),
            "r_only_changed": bool(r_pred[k] != c_pred[k]),
            "r_only_abs_prob_change": round(float(abs(p_r[k] - p_clean[k])), 4),
            "g_only_prob_fire": round(float(p_g[k]), 4),
            "g_only_prediction": int(g_pred[k]),
            "g_only_changed": bool(g_pred[k] != c_pred[k]),
            "g_only_abs_prob_change": round(float(abs(p_g[k] - p_clean[k])), 4),
            "b_only_prob_fire": round(float(p_b[k]), 4),
            "b_only_prediction": int(b_pred[k]),
            "b_only_changed": bool(b_pred[k] != c_pred[k]),
            "b_only_abs_prob_change": round(float(abs(p_b[k] - p_clean[k])), 4),
        })

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
            "preprocessing": "MobileNetV2 preprocess_input (x/127.5 - 1) "
                             "applied after resize; never ablate after "
                             "normalization",
            "resize_method": "BILINEAR to 224x224, exactly once",
            "transform_r_only": "original RGB -> extract R -> replicate "
                                "[R,R,R] uint8 RGB -> resize -> preprocess",
            "transform_g_only": "original RGB -> extract G -> replicate "
                                "[G,G,G] uint8 RGB -> resize -> preprocess",
            "transform_b_only": "original RGB -> extract B -> replicate "
                                "[B,B,B] uint8 RGB -> resize -> preprocess",
            "tensorflow_version": tf.__version__,
            "python_version": platform.python_version(),
        },
        "clean_metrics": clean_m,
        "r_only": {**r_m, **rel(r_m, p_r, r_pred),
                   "per_class": per_class(r_pred)},
        "g_only": {**g_m, **rel(g_m, p_g, g_pred),
                   "per_class": per_class(g_pred)},
        "b_only": {**b_m, **rel(b_m, p_b, b_pred),
                   "per_class": per_class(b_pred)},
        "deltas_vs_clean": {
            "r_only": {k: rel(r_m, p_r, r_pred)[k] for k in
                       ("accuracy_delta", "precision_delta",
                        "recall_delta", "f1_delta")},
            "g_only": {k: rel(g_m, p_g, g_pred)[k] for k in
                       ("accuracy_delta", "precision_delta",
                        "recall_delta", "f1_delta")},
            "b_only": {k: rel(b_m, p_b, b_pred)[k] for k in
                       ("accuracy_delta", "precision_delta",
                        "recall_delta", "f1_delta")},
        },
        "change_rates": {
            "r_only": round(float(np.mean(r_pred != c_pred)), 4),
            "g_only": round(float(np.mean(g_pred != c_pred)), 4),
            "b_only": round(float(np.mean(b_pred != c_pred)), 4),
        },
        "per_class_change_rates": {
            "r_only": per_class(r_pred),
            "g_only": per_class(g_pred),
            "b_only": per_class(b_pred),
        },
        "clean_fp_stability": fp_stability,
        "per_image": per_image,
        "limitations": [
            "Single-channel-replicated inputs are out-of-distribution "
            "relative to the natural RGB images used to train/pretrain the "
            "model.",
            "The experiment measures sensitivity to channel removal, not "
            "causal feature importance.",
            "A channel producing stronger performance does not prove that "
            "the model uses that channel as a shortcut.",
            "Channel ablation also changes luminance/contrast "
            "characteristics.",
            "MobileNetV2 was pretrained on natural RGB images, so "
            "degradation may partly reflect backbone distribution shift.",
            "Dataset-level R-G/R-B differences may cause asymmetric "
            "information loss independently of model shortcut behavior.",
            "Results apply to this test split and this trained model.",
            "The six-clean-FP analysis is descriptive because n=6 is small.",
        ],
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    print(f"[OK] Channel ablation saved to {OUTPUT_PATH}")
    print(f"Clean: {clean_m}")
    print(f"R-only: {r_m} change={artifact['change_rates']['r_only']} "
          f"|dp|={rel(r_m, p_r, r_pred)['avg_absolute_probability_change']}")
    print(f"G-only: {g_m} change={artifact['change_rates']['g_only']} "
          f"|dp|={rel(g_m, p_g, g_pred)['avg_absolute_probability_change']}")
    print(f"B-only: {b_m} change={artifact['change_rates']['b_only']} "
          f"|dp|={rel(b_m, p_b, b_pred)['avg_absolute_probability_change']}")
    for tag in ("r_only", "g_only", "b_only"):
        print(fp_stability[tag]["summary"])


if __name__ == "__main__":
    main()
