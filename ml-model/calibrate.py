"""
Feature 9: OFFLINE probability-calibration experiment only.
============================================================
- Reconstructs the exact 70/15/15 stratified split (random_state=42).
- Loads ml-model/fire_model.h5 (no retraining, no weight changes).
- Generates clean validation probabilities with identical preprocessing.
- Reuses untouched test probabilities from ml-model/error_analysis.json.
- Fits Platt (primary), temperature (secondary, from real pre-sigmoid
  logits), isotonic (exploratory) on VALIDATION only.
- Evaluates uncalibrated vs calibrated on frozen test set at threshold 0.5.
- Writes ml-model/calibration_analysis.json. No backend/frontend changes.

Run:
    venv\\Scripts\\python.exe ml-model/calibrate.py
"""
import json
import os
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img

IMG_SIZE = (224, 224)
HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(HERE, "dataset")
FIRE_DIR = os.path.join(DATASET_DIR, "fire")
NO_FIRE_DIR = os.path.join(DATASET_DIR, "no_fire")
MODEL_PATH = os.path.join(HERE, "fire_model.h5")
ERROR_ANALYSIS_PATH = os.path.join(HERE, "error_analysis.json")
OUTPUT_PATH = os.path.join(HERE, "calibration_analysis.json")

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15 / 0.85
THRESHOLD = 0.5
N_BINS = 10
CV_FOLDS = 5
BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 42
LOGIT_EPS = 1e-7

STATEMENT = ("The existing sigmoid output is treated as an uncalibrated model "
             "score until calibration is empirically evaluated.")


def _stratify_or_none(labels):
    labels = np.asarray(labels)
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return None
    if np.min(np.bincount(labels.astype(int))) < 2:
        return None
    return labels


def load_dataset():
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
    print(f"Loaded {n_fire} fire, {n_no_fire} no_fire images.")
    return np.array(images), np.array(labels), np.array(filenames)


def _clip(p):
    return np.clip(np.asarray(p, dtype=np.float64), LOGIT_EPS, 1.0 - LOGIT_EPS)


def _logit(p):
    pc = _clip(p)
    return np.log(pc / (1.0 - pc))


def _sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def brier(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def ece_score(y, p, n_bins=N_BINS):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if b == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        acc_b = float(y[mask].mean())
        conf_b = float(p[mask].mean())
        ece += (n_b / len(y)) * abs(acc_b - conf_b)
    return float(ece)


def reliability_bins(y, p, n_bins=N_BINS):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for b in range(n_bins):
        lo, hi = float(edges[b]), float(edges[b + 1])
        if b == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        n_b = int(mask.sum())
        if n_b == 0:
            bins.append({"bin": b, "lo": lo, "hi": hi, "count": 0,
                         "mean_predicted": None, "observed_rate": None,
                         "gap": None})
        else:
            mean_p = float(p[mask].mean())
            obs = float(y[mask].mean())
            bins.append({"bin": b, "lo": lo, "hi": hi, "count": n_b,
                         "mean_predicted": round(mean_p, 4),
                         "observed_rate": round(obs, 4),
                         "gap": round(abs(obs - mean_p), 4)})
    return bins


def classification_metrics(y, p, threshold=THRESHOLD):
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def full_metrics(y, p, n_bins=N_BINS):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    out = {
        "n": int(len(y)),
        "brier": round(brier(y, p), 4),
        "log_loss": round(float(log_loss(y, _clip(p), labels=[0, 1])), 4),
        "ece": round(ece_score(y, p, n_bins), 4),
        "mean_predicted": round(float(np.mean(p)), 4),
        "observed_rate": round(float(np.mean(y)), 4),
    }
    out.update(classification_metrics(y, p))
    return out


def bootstrap_ci(y, p, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    """95% percentile bootstrap CIs for Brier/ECE/log-loss on a fixed set."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    n = len(y)
    b = np.empty(reps)
    e = np.empty(reps)
    l = np.empty(reps)
    for r in range(reps):
        idx = rng.integers(0, n, n)
        b[r] = brier(y[idx], p[idx])
        e[r] = ece_score(y[idx], p[idx])
        l[r] = log_loss(y[idx], _clip(p[idx]), labels=[0, 1])
    def ci(a):
        return [round(float(np.percentile(a, 2.5)), 4),
                round(float(np.percentile(a, 97.5)), 4)]
    return {"brier_95ci": ci(b), "ece_95ci": ci(e), "log_loss_95ci": ci(l),
            "reps": reps, "seed": seed}


def oof_predict(builder, X, y, n_splits=CV_FOLDS, seed=RANDOM_STATE):
    """Out-of-fold calibrated probs on val (avoids in-sample optimism)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    oof = np.empty(len(y), dtype=np.float64)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        model = builder()
        model.fit(X[tr].reshape(-1, 1) if X.ndim == 1 else X[tr], y[tr])
        if hasattr(model, "predict_proba"):
            oof[te] = model.predict_proba(
                X[te].reshape(-1, 1) if X.ndim == 1 else X[te])[:, 1]
        else:
            oof[te] = model.predict(X[te])
    return np.clip(oof, LOGIT_EPS, 1.0 - LOGIT_EPS)


def main():
    print("Reconstructing exact dataset split (70/15/15, random_state=42)...")
    X, y, filenames = load_dataset()
    assert len(X) == 1832, f"Expected 1832 images, got {len(X)}"
    y = np.asarray(y)

    X_temp, X_test, y_temp, y_test, f_temp, f_test = train_test_split(
        X, y, filenames, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=_stratify_or_none(y))
    X_train, X_val, y_train, y_val, f_train, f_val = train_test_split(
        X_temp, y_temp, f_temp, test_size=VAL_SIZE, random_state=RANDOM_STATE,
        stratify=_stratify_or_none(y_temp))
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    assert len(X_train) == 1282 and len(X_val) == 275 and len(X_test) == 275, \
        "Split sizes do not match 1282/275/275"

    # --- Validation inference (clean, same preprocessing) ---
    print(f"Loading model from {MODEL_PATH} (no retraining)...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    X_val_p = preprocess_input(X_val.astype("float32"))
    print("Running batch inference on validation set...")
    p_val = model.predict(X_val_p, batch_size=32, verbose=1).flatten().astype(float)
    p_val = np.clip(p_val, LOGIT_EPS, 1.0 - LOGIT_EPS)
    y_val = y_val.astype(int)

    val_records = [{"filename": str(f), "true_label": int(t),
                    "uncalibrated_prob_fire": round(float(p), 4)}
                   for f, t, p in zip(f_val, y_val, p_val)]

    # --- Untouched test probabilities from error_analysis.json ---
    if not os.path.exists(ERROR_ANALYSIS_PATH):
        raise FileNotFoundError(f"{ERROR_ANALYSIS_PATH} missing")
    with open(ERROR_ANALYSIS_PATH, "r", encoding="utf-8") as fh:
        analysis = json.load(fh)
    per_image = analysis.get("per_image_results", [])
    assert len(per_image) == 275, f"Expected 275 test entries, got {len(per_image)}"
    test_map = {r["filename"]: r for r in per_image}
    recon_test = set(f_test.tolist())
    stored_test = set(test_map.keys())
    compatible = (recon_test == stored_test)
    print(f"Test split compatible with error_analysis.json: {compatible}")
    if not compatible:
        raise RuntimeError("Reconstructed test filenames do not match "
                           "error_analysis.json; refusing to recompute test "
                           "inference implicitly.")
    # Use stored probs directly (rounded to 4dp in artifact).
    order = [test_map[f] for f in f_test]
    y_test_check = np.array([r["true_label"] for r in order]).astype(int)
    assert np.array_equal(y_test_check, y_test.astype(int)), "Test label mismatch"
    p_test = np.array([r["prob_fire"] for r in order], dtype=float)
    p_test = np.clip(p_test, LOGIT_EPS, 1.0 - LOGIT_EPS)
    print("Using test probabilities directly from error_analysis.json "
          "(no test fitting, no recompute).")

    # --- Platt scaling (primary): LogisticRegression on validation logit ---
    # Representation: logit(clip(p)) as scalar feature. Documented here and
    # in artifact parameters. L2 C=1.0 (sklearn default) guards against
    # separation from extreme scores; solver=lbfgs, fixed seed behavior.
    z_val = _logit(p_val)
    z_test = _logit(p_test)  # inverse-sigmoid of stored test probs
    platt = LogisticRegression(C=1.0, solver="lbfgs")
    platt.fit(z_val.reshape(-1, 1), y_val)
    p_val_platt_in = np.clip(platt.predict_proba(z_val.reshape(-1, 1))[:, 1],
                             LOGIT_EPS, 1.0 - LOGIT_EPS)
    p_test_platt = np.clip(platt.predict_proba(z_test.reshape(-1, 1))[:, 1],
                           LOGIT_EPS, 1.0 - LOGIT_EPS)
    p_val_platt_oof = oof_predict(
        lambda: LogisticRegression(C=1.0, solver="lbfgs"), z_val, y_val)

    # --- Temperature scaling (secondary): real pre-sigmoid logits ---
    # Logits obtained cleanly via submodel to layers[-1].input (pre-sigmoid
    # Dense input). No weight modification; Dropout is identity at inference.
    temperature_info = {"available": False}
    p_val_temp_in = p_test_temp = p_val_temp_oof = None
    try:
        # Pre-sigmoid logits = final Dense linear part (Wx+b). The last layer
        # is Dense(1, sigmoid), so layers[-1].input is the 128-dim feature,
        # NOT the logit. Recover logits cleanly: submodel to layers[-2].output
        # (Dense(128, relu); Dropout is identity at inference), then apply
        # the frozen final kernel/bias. No weight modification.
        feat_model = tf.keras.models.Model(
            inputs=model.inputs, outputs=model.layers[-2].output)
        F_val = feat_model.predict(X_val_p, batch_size=32,
                                   verbose=0).astype(float)
        W, b = model.layers[-1].get_weights()
        z_val_direct = (F_val @ W + b).flatten().astype(float)
        # Sanity: sigmoid(direct logits) must match model probs closely.
        recon = _sigmoid(z_val_direct)
        max_abs_err = float(np.max(np.abs(recon - p_val)))
        print(f"Direct val logits recovered; max|sigmoid(z)-p|={max_abs_err:.2e}")
        if max_abs_err > 1e-4:
            raise RuntimeError("Logit recovery sanity check failed")
        from scipy.optimize import minimize_scalar
        def nll(T):
            T = float(T)
            return float(log_loss(y_val, _sigmoid(z_val_direct / T),
                                  labels=[0, 1]))
        res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded",
                              options={"xatol": 1e-4})
        T = float(res.x)
        p_val_temp_in = np.clip(_sigmoid(z_val_direct / T),
                                LOGIT_EPS, 1.0 - LOGIT_EPS)
        # Test logits: inverse-sigmoid of stored test probs (avoids test
        # recompute; 4dp rounding documented as limitation).
        z_test_inv = _logit(p_test)
        p_test_temp = np.clip(_sigmoid(z_test_inv / T),
                              LOGIT_EPS, 1.0 - LOGIT_EPS)
        # OOF on val for unbiased val estimate (fold-wise T).
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                              random_state=RANDOM_STATE)
        oof_t = np.empty(len(y_val), dtype=float)
        for tr, te in skf.split(z_val_direct, y_val):
            r = minimize_scalar(
                lambda tt: float(log_loss(
                    y_val[tr], _sigmoid(z_val_direct[tr] / float(tt)),
                    labels=[0, 1])),
                bounds=(0.05, 10.0), method="bounded",
                options={"xatol": 1e-4})
            oof_t[te] = _sigmoid(z_val_direct[te] / float(r.x))
        p_val_temp_oof = np.clip(oof_t, LOGIT_EPS, 1.0 - LOGIT_EPS)
        temperature_info = {
            "available": True,
            "method": "minimize val NLL over T in [0.05, 10] "
                      "(scipy minimize_scalar, bounded)",
            "temperature": round(T, 4),
            "logit_source": "submodel to layers[-2].output (Dense 128) plus "
                            "frozen final Dense kernel/bias (Wx+b); "
                            "no weight changes",
            "test_logit_source": "logit(clip(stored test prob_fire)); "
                                 "no test inference recompute",
            "sanity_max_abs_err_sigmoid_z_vs_p": round(max_abs_err, 6),
            "representation": "p_cal = sigmoid(z / T)",
        }
        print(f"Temperature T={T:.4f}")
    except Exception as e:
        print(f"Temperature scaling unavailable: {e}")
        temperature_info = {"available": False, "reason": str(e)}

    # --- Isotonic regression (exploratory) ---
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_val, y_val)
    p_val_iso_in = np.clip(iso.predict(p_val), LOGIT_EPS, 1.0 - LOGIT_EPS)
    p_test_iso = np.clip(iso.predict(p_test), LOGIT_EPS, 1.0 - LOGIT_EPS)
    p_val_iso_oof = oof_predict(
        lambda: IsotonicRegression(out_of_bounds="clip",
                                   y_min=0.0, y_max=1.0), p_val, y_val)

    # --- Metrics ---
    uncal_val = full_metrics(y_val, p_val)
    uncal_test = full_metrics(y_test.astype(int), p_test)
    platt_val_in = full_metrics(y_val, p_val_platt_in)
    platt_val_oof = full_metrics(y_val, p_val_platt_oof)
    platt_test = full_metrics(y_test.astype(int), p_test_platt)
    iso_val_in = full_metrics(y_val, p_val_iso_in)
    iso_val_oof = full_metrics(y_val, p_val_iso_oof)
    iso_test = full_metrics(y_test.astype(int), p_test_iso)

    ci = {"uncalibrated_test": bootstrap_ci(y_test.astype(int), p_test),
          "platt_test": bootstrap_ci(y_test.astype(int), p_test_platt),
          "isotonic_test": bootstrap_ci(y_test.astype(int), p_test_iso)}
    if p_test_temp is not None:
        temp_val_in = full_metrics(y_val, p_val_temp_in)
        temp_val_oof = full_metrics(y_val, p_val_temp_oof)
        temp_test = full_metrics(y_test.astype(int), p_test_temp)
        ci["temperature_test"] = bootstrap_ci(y_test.astype(int), p_test_temp)

    # Decisions changed at 0.5?
    def changed(a, b):
        a = ((np.asarray(a) > THRESHOLD).astype(int))
        b = ((np.asarray(b) > THRESHOLD).astype(int))
        return int((a != b).sum())
    decision_changes = {
        "platt_test_changed": changed(p_test, p_test_platt),
        "isotonic_test_changed": changed(p_test, p_test_iso),
    }
    if p_test_temp is not None:
        decision_changes["temperature_test_changed"] = changed(p_test,
                                                               p_test_temp)

    artifact = {
        "statement": STATEMENT,
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_size": 1832,
            "validation_size": int(len(y_val)),
            "test_size": int(len(y_test)),
            "random_state": RANDOM_STATE,
            "threshold": THRESHOLD,
            "n_bins": N_BINS,
            "model_path": MODEL_PATH,
            "test_source": "ml-model/error_analysis.json (stored prob_fire)",
            "test_split_compatible": bool(compatible),
        },
        "protocol": {
            "calibration_fit_split": "validation",
            "evaluation_split": "test",
            "test_used_for_fitting": False,
            "preprocessing": "RGB, resize 224x224, MobileNetV2 "
                             "preprocess_input (x/127.5 - 1), batch 32",
            "cv": f"{CV_FOLDS}-fold StratifiedKFold OOF on validation "
                  "for unbiased val estimates",
        },
        "validation_predictions": val_records,
        "uncalibrated": {
            "validation": {**uncal_val,
                           "ci_bootstrap": bootstrap_ci(y_val, p_val)},
            "test": {**uncal_test, "ci_bootstrap": ci["uncalibrated_test"]},
        },
        "platt": {
            "validation": {"in_sample": platt_val_in,
                           "cv_oof": platt_val_oof},
            "test": {**platt_test, "ci_bootstrap": ci["platt_test"]},
            "parameters": {
                "representation": "LogisticRegression on logit(clip(p)) "
                                  "scalar; p_cal = sigmoid(a*logit(p)+b)",
                "coef_": [round(float(c), 6) for c in platt.coef_.flatten()],
                "intercept_": [round(float(c), 6)
                               for c in np.atleast_1d(platt.intercept_)],
                "C": 1.0, "solver": "lbfgs",
                "fit_split": "validation only (n=275)",
                "test_used_for_fitting": False,
            },
        },
        "temperature": {
            **temperature_info,
            **({"validation": {
                "in_sample": temp_val_in, "cv_oof": temp_val_oof},
                "test": {**temp_test,
                         "ci_bootstrap": ci["temperature_test"]}}
               if p_test_temp is not None else {}),
        },
        "isotonic": {
            "available": True,
            "validation": {"in_sample": iso_val_in, "cv_oof": iso_val_oof},
            "test": {**iso_test, "ci_bootstrap": ci["isotonic_test"]},
            "note": "exploratory",
            "warning": "n=275 is small for isotonic; in-sample val gains "
                       "are optimistic — use cv_oof; do not select on test.",
            "fit_split": "validation only (n=275)",
            "test_used_for_fitting": False,
        },
        "decision_changes_at_0.5": decision_changes,
        "reliability": {
            "uncalibrated": reliability_bins(y_test.astype(int), p_test),
            "platt": reliability_bins(y_test.astype(int), p_test_platt),
            "isotonic": reliability_bins(y_test.astype(int), p_test_iso),
            **({"temperature": reliability_bins(y_test.astype(int),
                                                p_test_temp)}
               if p_test_temp is not None else {}),
            "validation_uncalibrated": reliability_bins(y_val, p_val),
            "validation_platt_oof": reliability_bins(y_val, p_val_platt_oof),
        },
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    print(f"\n[OK] Calibration analysis saved to {OUTPUT_PATH}")

    print("\n=== Uncalibrated test ===")
    print(uncal_test)
    print("=== Platt test ===")
    print(platt_test)
    if p_test_temp is not None:
        print("=== Temperature test ===")
        print(temp_test)
    print("=== Isotonic test ===")
    print(iso_test)
    print("Decision changes at 0.5:", decision_changes)


if __name__ == "__main__":
    main()
