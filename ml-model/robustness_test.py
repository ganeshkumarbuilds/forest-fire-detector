"""
Forest Fire Detection - Controlled Robustness Test
Offline batch inference on deterministic test split with controlled perturbations.
PIL/numpy only. No HTTP. Fixed seed.
"""
import os
import io
import json
import numpy as np
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image, ImageEnhance, ImageFilter

IMG_SIZE = (224, 224)
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
FIRE_DIR = os.path.join(DATASET_DIR, "fire")
NO_FIRE_DIR = os.path.join(DATASET_DIR, "no_fire")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "fire_model.h5")
ROBUSTNESS_PATH = os.path.join(os.path.dirname(__file__), "robustness.json")

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15 / 0.85
THRESHOLD = 0.5
RANDOM_SEED = 42

def _stratify_or_none(labels):
    labels = np.asarray(labels)
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return None
    if np.min(np.bincount(labels.astype(int))) < 2:
        return None
    return labels

def load_filenames_and_labels():
    """Return sorted filenames with labels. Filenames include prefix fire/ or no_fire/."""
    fire_files = sorted([f for f in os.listdir(FIRE_DIR) if f.lower().endswith((".jpg",".jpeg",".png"))])
    no_fire_files = sorted([f for f in os.listdir(NO_FIRE_DIR) if f.lower().endswith((".jpg",".jpeg",".png"))])
    all_files = ["fire/"+f for f in fire_files] + ["no_fire/"+f for f in no_fire_files]
    all_labels = [1]*len(fire_files) + [0]*len(no_fire_files)
    return np.array(all_files), np.array(all_labels), len(fire_files), len(no_fire_files)

def resolve_path(prefixed):
    # prefixed like fire/xxx.jpg or no_fire/xxx.jpg
    if prefixed.startswith("fire/"):
        return os.path.join(FIRE_DIR, os.path.basename(prefixed))
    else:
        return os.path.join(NO_FIRE_DIR, os.path.basename(prefixed))

def load_and_preprocess_clean(prefixed):
    p = resolve_path(prefixed)
    img = load_img(p, target_size=IMG_SIZE)
    arr = img_to_array(img).astype("float32")
    return arr

def apply_perturbation(pil_img, kind, param):
    """Apply perturbation on PIL image before resize. Returns PIL image."""
    # pil_img is original size (e.g., 250x250) RGB
    if kind == "brightness":
        return ImageEnhance.Brightness(pil_img).enhance(param)
    elif kind == "contrast":
        return ImageEnhance.Contrast(pil_img).enhance(param)
    elif kind == "gaussian_blur":
        return pil_img.filter(ImageFilter.GaussianBlur(radius=param))
    elif kind == "gaussian_noise":
        # param is sigma
        arr = np.array(pil_img).astype("float32")
        # Use deterministic RNG per image? We use global seeded RNG but need reproducibility per run.
        # Caller will provide seeded RNG instance.
        # For this helper, expect param is tuple (sigma, rng)
        sigma, rng = param
        noise = rng.normal(0, sigma, arr.shape).astype("float32")
        arr = np.clip(arr + noise, 0, 255).astype("uint8")
        return Image.fromarray(arr)
    elif kind == "reduced_resolution":
        # param is downscale size e.g., 112
        small = pil_img.resize((param, param), Image.BILINEAR)
        # return small; caller will still resize to 224 as final step, so small will be upscaled
        # To make explicit, return small (we will resize to 224 later)
        return small
    elif kind == "jpeg_compression":
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=param)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    else:
        return pil_img

def pil_to_preprocessed(pil_img):
    # final resize to 224 and preprocess_input
    resized = pil_img.resize(IMG_SIZE, Image.BILINEAR)
    if resized.mode != "RGB":
        resized = resized.convert("RGB")
    arr = np.array(resized).astype("float32")
    # expand dims later via stacking, but here return single preprocessed array (224,224,3) preprocessed
    return preprocess_input(arr)

def compute_metrics(y_true, y_pred, probs):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return {
        "test_size": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)),4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)),4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)),4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)),4),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }

def main():
    print("Loading filenames...")
    all_files, all_labels, n_fire, n_no_fire = load_filenames_and_labels()
    print(f"Loaded {n_fire} fire, {n_no_fire} no_fire, total {len(all_files)}")

    # Replicate exact split
    X_dummy = np.arange(len(all_files))  # placeholder for split indices
    X_temp_idx, X_test_idx, y_temp, y_test, f_temp, f_test = train_test_split(
        X_dummy, all_labels, all_files, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=_stratify_or_none(all_labels)
    )
    X_train_idx, X_val_idx, y_train, y_val, f_train, f_val = train_test_split(
        X_temp_idx, y_temp, f_temp, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=_stratify_or_none(y_temp)
    )
    print(f"Train: {len(X_train_idx)}, Val: {len(X_val_idx)}, Test: {len(X_test_idx)}")
    print(f"Test fire: {int((y_test==1).sum())}, no_fire: {int((y_test==0).sum())}")
    assert len(X_test_idx)==275, f"Expected 275 test, got {len(X_test_idx)}"

    # Load model
    print(f"Loading model {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f"Model loaded {model.count_params():,} params")

    # Clean baseline: load clean arrays
    print("Preprocessing clean test set...")
    clean_arrays = []
    clean_pils = []  # also keep PIL for perturbation reference? Not needed, reload from disk
    for fname in f_test:
        arr = load_and_preprocess_clean(fname)  # this already is raw 0-255 before preprocess? Actually load_and_preprocess_clean returns raw array, we need to preprocess
        clean_arrays.append(arr)
    X_test_raw = np.array(clean_arrays)  # shape (275,224,224,3) 0-255
    X_test_p = preprocess_input(X_test_raw.astype("float32"))

    print("Running clean batch inference...")
    clean_probs = model.predict(X_test_p, batch_size=32, verbose=1).flatten()
    clean_pred = (clean_probs > THRESHOLD).astype(int)
    y_true = y_test.astype(int)

    clean_metrics = compute_metrics(y_true, clean_pred, clean_probs)
    print(f"Clean: {clean_metrics}")
    assert clean_metrics["tp"]+clean_metrics["tn"]+clean_metrics["fp"]+clean_metrics["fn"]==275

    # Identify clean FPs for stability tracking
    clean_fp_mask = (y_true==0) & (clean_pred==1)
    clean_fp_indices = np.where(clean_fp_mask)[0]
    clean_fp_infos = []
    for idx in clean_fp_indices:
        clean_fp_infos.append({
            "filename": str(f_test[idx]),
            "clean_prob_fire": round(float(clean_probs[idx]),4),
            "clean_prediction": int(clean_pred[idx]),
        })
    print(f"Clean FPs: {len(clean_fp_infos)} FN: {int(np.sum((y_true==1)&(clean_pred==0)))}")

    # Define perturbations
    perturbations_spec = [
        ("brightness", 0.70, {"factor": 0.70}),
        ("brightness", 0.85, {"factor": 0.85}),
        ("brightness", 1.15, {"factor": 1.15}),
        ("brightness", 1.30, {"factor": 1.30}),
        ("contrast", 0.70, {"factor": 0.70}),
        ("contrast", 1.30, {"factor": 1.30}),
        ("gaussian_blur", 1.5, {"radius": 1.5}),
        ("gaussian_blur", 3.0, {"radius": 3.0}),
        ("gaussian_noise", 10, {"sigma": 10}),
        ("gaussian_noise", 25, {"sigma": 25}),
        ("reduced_resolution", 112, {"downscale": 112}),
        ("reduced_resolution", 56, {"downscale": 56}),
        ("jpeg_compression", 60, {"quality": 60}),
        ("jpeg_compression", 30, {"quality": 30}),
    ]
    # Note: spec asked for 112/56 and 60/30 etc. We have 14 levels (4 brightness +2 contrast+2 blur+2 noise+2 resolution+2 jpeg=14)
    # Spec lists exactly 12? Brightness 4 + contrast2 + blur2 + noise2 + resolution2 + jpeg2 =14. We'll include all 14.

    perturbations_results = []

    for kind, level, params in perturbations_spec:
        print(f"\nPerturbation {kind} level {level} params {params}...")
        # For gaussian noise, need fixed seed RNG per perturbation level (deterministic)
        rng = np.random.RandomState(RANDOM_SEED) if kind=="gaussian_noise" else None

        perturbed_preprocessed = []
        for fname in f_test:
            p = resolve_path(fname)
            pil = Image.open(p).convert("RGB")
            # Apply perturbation before final resize/preprocess
            if kind == "gaussian_noise":
                pert = apply_perturbation(pil, kind, (params["sigma"], rng))
            elif kind in ("brightness","contrast"):
                pert = apply_perturbation(pil, kind, params["factor"])
            elif kind == "gaussian_blur":
                pert = apply_perturbation(pil, kind, params["radius"])
            elif kind == "reduced_resolution":
                pert = apply_perturbation(pil, kind, params["downscale"])
            elif kind == "jpeg_compression":
                pert = apply_perturbation(pil, kind, params["quality"])
            else:
                pert = pil
            # Final preprocess
            arr_p = pil_to_preprocessed(pert)
            perturbed_preprocessed.append(arr_p)

        X_pert = np.array(perturbed_preprocessed)  # already preprocessed (x/127.5-1)
        probs = model.predict(X_pert, batch_size=32, verbose=0).flatten()
        pred = (probs > THRESHOLD).astype(int)

        metrics = compute_metrics(y_true, pred, probs)
        assert metrics["tp"]+metrics["tn"]+metrics["fp"]+metrics["fn"]==275

        pred_change_rate = float(np.mean(pred != clean_pred))
        avg_abs_delta = float(np.mean(np.abs(probs - clean_probs)))

        # Deltas vs clean
        deltas = {
            "accuracy_delta": round(metrics["accuracy"] - clean_metrics["accuracy"],4),
            "precision_delta": round(metrics["precision"] - clean_metrics["precision"],4),
            "recall_delta": round(metrics["recall"] - clean_metrics["recall"],4),
            "f1_delta": round(metrics["f1"] - clean_metrics["f1"],4),
            "fp_delta": int(metrics["fp"] - clean_metrics["fp"]),
            "fn_delta": int(metrics["fn"] - clean_metrics["fn"]),
        }

        # Clean FP stability
        fp_stability = []
        changed = 0
        for info in clean_fp_infos:
            fname = info["filename"]
            idx = np.where(f_test == fname)[0][0]
            p_pert = float(probs[idx])
            pred_pert = int(pred[idx])
            changed_flag = pred_pert != info["clean_prediction"]
            if changed_flag:
                changed+=1
            fp_stability.append({
                "filename": fname,
                "clean_prob_fire": info["clean_prob_fire"],
                "clean_prediction": info["clean_prediction"],
                "perturbed_prob_fire": round(p_pert,4),
                "perturbed_prediction": pred_pert,
                "changed": bool(changed_flag),
            })
        # Summary wording handled in UI, but also store counts
        stability_summary = {
            "total_clean_fp": len(clean_fp_infos),
            "changed": int(changed),
            "unchanged": int(len(clean_fp_infos)-changed),
            "details": fp_stability
        }
        # Also handle 0 FN case for completeness
        clean_fn_mask = (y_true==1)&(clean_pred==0)
        fn_stability = []
        # Not required per spec but keep for completeness
        if np.sum(clean_fn_mask)>0:
            for idx in np.where(clean_fn_mask)[0]:
                fn_stability.append({
                    "filename": str(f_test[idx]),
                    "clean_prob_fire": round(float(clean_probs[idx]),4),
                    "clean_prediction": int(clean_pred[idx]),
                    "perturbed_prob_fire": round(float(probs[idx]),4),
                    "perturbed_prediction": int(pred[idx]),
                    "changed": bool(pred[idx]!=clean_pred[idx]),
                })

        perturbations_results.append({
            "name": kind,
            "level": str(level),
            "parameters": params,
            "metrics": metrics,
            "prediction_change_rate": round(float(pred_change_rate),4),
            "avg_absolute_probability_change": round(float(avg_abs_delta),4),
            "metric_deltas": deltas,
            "clean_fp_stability": stability_summary,
            "clean_fn_stability": fn_stability,
        })
        print(f"  -> acc {metrics['accuracy']} delta {deltas['accuracy_delta']} change_rate {pred_change_rate:.4f} avg_delta {avg_abs_delta:.4f} FP {metrics['fp']} FN {metrics['fn']}")

    # Build artifact
    artifact = {
        "metadata": {
            "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
            "model": MODEL_PATH,
            "input_size": [224,224,3],
            "threshold": THRESHOLD,
            "random_seed": RANDOM_SEED,
            "random_state": RANDOM_STATE,
            "batch_size": 32,
        },
        "dataset": {
            "total_images": int(len(all_files)),
            "fire_count": int(n_fire),
            "no_fire_count": int(n_no_fire),
        },
        "split": {
            "train_ratio": 0.70, "val_ratio": 0.15, "test_ratio": 0.15,
            "random_state": RANDOM_STATE, "stratified": True,
            "train_size": int(len(X_train_idx)), "val_size": int(len(X_val_idx)), "test_size": int(len(X_test_idx)),
        },
        "clean_metrics": clean_metrics,
        "perturbations": perturbations_results,
    }

    with open(ROBUSTNESS_PATH, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    print(f"\n[OK] Robustness results saved to {ROBUSTNESS_PATH}")

    # Summary
    print("\n=== Robustness Summary ===")
    print(f"Clean: acc {clean_metrics['accuracy']} prec {clean_metrics['precision']} rec {clean_metrics['recall']} f1 {clean_metrics['f1']} FP {clean_metrics['fp']} FN {clean_metrics['fn']}")
    for p in perturbations_results:
        print(f"{p['name']} {p['level']}: acc {p['metrics']['accuracy']} delta {p['metric_deltas']['accuracy_delta']} change {p['prediction_change_rate']} avgDelta {p['avg_absolute_probability_change']} FP {p['metrics']['fp']} FN {p['metrics']['fn']}")

if __name__ == "__main__":
    main()
