"""Forest Fire Detection - Flask backend."""
import base64
import io
import json
import os
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)
CORS(app)  # allow React dev server to call the API

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ml-model", "fire_model.h5")
MODEL_PATH = os.path.abspath(MODEL_PATH)

# Flat JSON file storage for detection history (hackathon scope: no database).
# Created automatically on the first /predict call — no manual setup needed.
LOG_PATH = os.path.join(os.path.dirname(__file__), "detection_log.json")
LOG_PATH = os.path.abspath(LOG_PATH)

# Precomputed test-set metrics written by ml-model/train_model.py.
METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "ml-model", "eval_metrics.json")
METRICS_PATH = os.path.abspath(METRICS_PATH)

# Load once at startup, not per-request.
model = None
if os.path.exists(MODEL_PATH):
    print(f"Loading model from {MODEL_PATH} ...")
    model = load_model(MODEL_PATH)
    print("Model loaded.")
else:
    print(f"WARNING: model file not found at {MODEL_PATH}. "
          "Run ml-model/train_model.py first. /predict will return 503.")


def preprocess_image(file_storage):
    img = Image.open(file_storage.stream).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)  # same normalization as training
    return arr


def _read_log():
    """Read the detection log file safely. Returns [] if missing/empty/corrupt."""
    if not os.path.exists(LOG_PATH):
        return []
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return []
            data = json.loads(content)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARNING: could not read detection log at {LOG_PATH}: {e}")
        return []


def _append_log(entry):
    """Append one entry to the log file (read-modify-write). Never raises."""
    try:
        entries = _read_log()
        entries.append(entry)
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)
    except OSError as e:
        print(f"WARNING: could not write detection log at {LOG_PATH}: {e}")


def _find_last_conv_layer_name(m):
    """Find the last Conv2D layer by scanning in reverse (robust to naming).

    For this MobileNetV2 model that resolves to 'Conv_1'. Also searches
    nested submodels in case the architecture ever changes. Returns None
    if no convolutional layer exists.
    """
    def search(layers):
        for layer in reversed(layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
        for layer in reversed(layers):
            sub = getattr(layer, "layers", None)
            if sub:
                found = search(sub)
                if found:
                    return found
        return None

    try:
        return search(m.layers)
    except Exception:
        return None


_gradcam_model = None  # cached (conv_outputs, predictions) sub-model


def _get_gradcam_model():
    """Build (once) a sub-model exposing conv features + prediction. Never raises."""
    global _gradcam_model
    if _gradcam_model is not None:
        return _gradcam_model
    if model is None:
        return None
    try:
        layer_name = _find_last_conv_layer_name(model)
        if layer_name is None:
            print("WARNING: no Conv2D layer found; Grad-CAM disabled.")
            return None
        _gradcam_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(layer_name).output, model.output],
        )
        print(f"Grad-CAM target layer: {layer_name}")
    except Exception as e:
        print(f"WARNING: could not build Grad-CAM model: {e}")
        return None
    return _gradcam_model


def _apply_jet(heatmap):
    """Map a [0,1] (h, w) heatmap to JET RGB uint8. No extra dependencies."""
    x = np.clip(heatmap, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def _gradcam_base64(file_storage, x, fire_detected):
    """Compute Grad-CAM overlay as a base64 PNG string, or None on any failure.

    Never raises — callers treat None as 'no heatmap available'.
    """
    try:
        grad_model = _get_gradcam_model()
        if grad_model is None:
            return None
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(x, training=False)
            # Explain the predicted class: P(fire), or P(no fire) = 1 - output.
            target = preds[:, 0] if fire_detected else (1.0 - preds[:, 0])
        grads = tape.gradient(target, conv_out)
        if grads is None:
            return None
        weights = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
        fmaps = conv_out[0].numpy()
        cam = np.maximum(fmaps @ weights, 0.0)
        peak = cam.max()
        cam = cam / (peak + 1e-8) if peak > 0 else np.zeros_like(cam)
        # Upsample the low-res activation map to the 224x224 input size.
        cam_img = Image.fromarray((cam * 255.0).astype(np.uint8)).resize(
            (224, 224), Image.BILINEAR
        )
        cam = np.asarray(cam_img).astype("float32") / 255.0
        heat = _apply_jet(cam)
        # Re-read the original upload (rewind; preprocess consumed the stream).
        try:
            file_storage.stream.seek(0)
        except Exception:
            return None
        orig = Image.open(file_storage.stream).convert("RGB").resize((224, 224))
        orig_arr = np.asarray(orig).astype("float32")
        overlay = (0.4 * heat.astype("float32") + 0.6 * orig_arr).clip(0, 255)
        buf = io.BytesIO()
        Image.fromarray(overlay.astype(np.uint8)).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as e:
        print(f"WARNING: Grad-CAM failed: {e}")
        return None


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Train it first."}), 503

    if "file" not in request.files:
        return jsonify({"error": "No 'file' part in request."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    try:
        x = preprocess_image(file)
        prob = float(model.predict(x, verbose=0)[0][0])  # P(fire)
        fire_detected = prob >= 0.5
        confidence = round(prob if fire_detected else 1.0 - prob, 4)
        timestamp = datetime.now(timezone.utc).isoformat()
        # Persist to server-side log (must not change the response shape).
        _append_log({
            "filename": file.filename,
            "fire_detected": fire_detected,
            "confidence": confidence,
            "timestamp": timestamp,
        })
        result = {
            "fire_detected": fire_detected,
            "confidence": confidence,
            "timestamp": timestamp,
        }
        # Visual explainability (additive): omit the field if it fails.
        try:
            heatmap = _gradcam_base64(file, x, fire_detected)
            if heatmap:
                result["heatmap_image"] = heatmap
        except Exception as e:
            print(f"WARNING: Grad-CAM failed: {e}")
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Inference failed: {e}"}), 500


@app.get("/history")
def history():
    """Return the last 20 log entries, most recent first."""
    entries = _read_log()
    return jsonify(entries[::-1][:20])


@app.get("/stats")
def stats():
    """Return aggregate stats over ALL logged entries (not just last 20)."""
    entries = _read_log()
    total = len(entries)
    fire_count = sum(1 for e in entries if e.get("fire_detected"))
    if total:
        avg = sum(float(e.get("confidence", 0) or 0) for e in entries) / total
    else:
        avg = 0
    return jsonify({
        "total_analyzed": total,
        "fire_count": fire_count,
        "safe_count": total - fire_count,
        "average_confidence": round(avg, 4),
    })


@app.get("/model-info")
def model_info():
    """Return precomputed test-set metrics from the last training run."""
    if not os.path.exists(METRICS_PATH):
        return jsonify({"available": False})
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return jsonify({"available": False})
        return jsonify({"available": True, **data})
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARNING: could not read eval metrics at {METRICS_PATH}: {e}")
        return jsonify({"available": False})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
