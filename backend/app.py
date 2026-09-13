"""Forest Fire Detection - Flask backend."""
import base64
import csv
import io
import json
import os
import re
import smtplib
import time
import urllib.request
from datetime import datetime, timezone
from email.message import EmailMessage

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

try:
    from dotenv import load_dotenv
    load_dotenv()  # local dev: backend/.env or repo-root .env
except ImportError:
    pass

app = Flask(__name__)
CORS(app)  # allow React dev server to call the API

# Postgres (your own account) — optional, falls back to JSON files.
# Set DATABASE_URL in backend/.env to enable. Never raises.
try:
    import db as pgdb
    try:
        pgdb.init_db()
    except Exception as _e:
        print(f"WARNING: Postgres init skipped: {_e}")
except ImportError as _e:
    pgdb = None
    print(f"WARNING: Postgres driver missing, using JSON storage: {_e}")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ml-model", "fire_model.h5")
MODEL_PATH = os.path.abspath(MODEL_PATH)

# Flat JSON file storage for detection history (hackathon scope: no database).
# Created automatically on the first /predict call — no manual setup needed.
LOG_PATH = os.path.join(os.path.dirname(__file__), "detection_log.json")
LOG_PATH = os.path.abspath(LOG_PATH)

# Precomputed test-set metrics written by ml-model/train_model.py.
METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "ml-model", "eval_metrics.json")
METRICS_PATH = os.path.abspath(METRICS_PATH)

# Alert-email recipient storage (single address, overwritten on each POST).
ALERT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "alert_config.json")
ALERT_CONFIG_PATH = os.path.abspath(ALERT_CONFIG_PATH)

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Genuine worldwide active-fire data: NASA FIRMS satellite hotspots.
# Free MAP_KEY at https://firms.modaps.eosdis.nasa.gov/api/map_key/
# Set FIRMS_MAP_KEY in backend/.env (never commit it).
FIRMS_SOURCE = "VIIRS_SNPP_NRT"
FIRMS_DAY_RANGE = 1
FIRMS_CACHE_TTL_S = 30 * 60
FIRMS_MAX_POINTS = 2000
HOTSPOTS_CACHE_PATH = os.path.join(os.path.dirname(__file__), "hotspots_cache.json")
HOTSPOTS_CACHE_PATH = os.path.abspath(HOTSPOTS_CACHE_PATH)
_hotspots_mem = {"at": 0.0, "payload": None}

# Load once at startup, not per-request.
model = None
if os.path.exists(MODEL_PATH):
    print(f"Loading model from {MODEL_PATH} ...")
    model = load_model(MODEL_PATH)
    print("Model loaded.")
    # Warm up TF graph so first /predict is fast, not 20-30s compile.
    # Full Grad-CAM warm-up happens lazily on first request (needs model graph).
    try:
        _dummy = np.zeros((1, 224, 224, 3), dtype="float32")
        _dummy = preprocess_input(_dummy)
        model.predict(_dummy, verbose=0)
        print("Model warm-up done — first request will be fast.")
    except Exception as _we:
        print(f"WARNING: warm-up failed (first request will be slower): {_we}")
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
    """Postgres first (your account), JSON fallback. Never raises."""
    try:
        if pgdb is not None and pgdb.db_available():
            rows = pgdb.db_read_log()
            if rows is not None:
                return rows
    except Exception as e:
        print(f"WARNING: Postgres log read skipped: {e}")
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
    """Write to Postgres AND JSON mirror. Never raises."""
    try:
        if pgdb is not None and pgdb.db_available():
            pgdb.db_append_log(entry)
    except Exception as e:
        print(f"WARNING: Postgres log write skipped: {e}")
    try:
        entries = []
        if os.path.exists(LOG_PATH):
            try:
                with open(LOG_PATH, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        data = json.loads(content)
                        entries = data if isinstance(data, list) else []
            except (json.JSONDecodeError, OSError):
                entries = []
        entries.append(entry)
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)
    except OSError as e:
        print(f"WARNING: could not write detection log at {LOG_PATH}: {e}")


def _read_alert_email():
    """Return the configured alert email string, or None if not set."""
    try:
        if pgdb is not None and pgdb.db_available():
            email = pgdb.db_read_alert_email()
            if email:
                return email
    except Exception as e:
        print(f"WARNING: Postgres alert read skipped: {e}")
    if not os.path.exists(ALERT_CONFIG_PATH):
        return None
    try:
        with open(ALERT_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        email = (data.get("email") or "").strip() if isinstance(data, dict) else ""
        return email or None
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARNING: could not read alert config at {ALERT_CONFIG_PATH}: {e}")
        return None


def _write_alert_email(email):
    """Persist the alert email to Postgres + JSON mirror."""
    try:
        if pgdb is not None and pgdb.db_available():
            pgdb.db_write_alert_email(email)
    except Exception as e:
        print(f"WARNING: Postgres alert write skipped: {e}")
    with open(ALERT_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump({"email": email}, f, indent=2)


def _is_critical(fire_detected, confidence):
    """Mirror frontend getRiskLevel critical band: fire + conf >= 0.85."""
    try:
        return bool(fire_detected) and float(confidence or 0) >= 0.85
    except (TypeError, ValueError):
        return False


def _send_critical_alert_email(to_email, location, confidence, timestamp):
    """Send a critical-risk alert via Gmail SMTP. Never raises (logs only)."""
    try:
        sender = (os.environ.get("GMAIL_ADDRESS") or "").strip()
        app_password = (os.environ.get("GMAIL_APP_PASSWORD") or "").strip()
        if not sender or not app_password:
            print("WARNING: GMAIL_ADDRESS/GMAIL_APP_PASSWORD not set; skipping alert email.")
            return
        pct = f"{float(confidence or 0) * 100:.1f}%"
        msg = EmailMessage()
        msg["Subject"] = "🚨 Fire Alert - Critical Risk Detected"
        msg["From"] = sender
        msg["To"] = to_email
        msg.set_content(
            "Critical fire risk detected.\n\n"
            f"Location: {location or 'Uploaded Image'}\n"
            f"Confidence: {pct}\n"
            f"Timestamp: {timestamp}\n"
        )
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as server:
            server.starttls()
            server.login(sender, app_password)
            server.send_message(msg)
        print(f"Critical alert email sent to {to_email}.")
    except Exception as e:
        print(f"WARNING: failed to send critical alert email: {e}")


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
            heatmap = request.args.get("heatmap", "0") == "1"
            if heatmap:
                heatmap = _gradcam_base64(file, x, fire_detected)
                if heatmap:
                    result["heatmap_image"] = heatmap
        except Exception as e:
            print(f"WARNING: Grad-CAM failed: {e}")
        # Critical-risk email alert (never breaks the response).
        try:
            if _is_critical(fire_detected, confidence):
                recipient = _read_alert_email()
                if recipient:
                    location = (
                        request.form.get("location")
                        or request.form.get("camera")
                        or (file.filename or "").strip()
                        or "Uploaded Image"
                    )
                    _send_critical_alert_email(
                        recipient, location, confidence, timestamp
                    )
        except Exception as e:
            print(f"WARNING: critical alert hook failed: {e}")
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


@app.get("/configure-alert-email")
def get_alert_email():
    """Return the currently stored alert email (or null if not set)."""
    return jsonify({"email": _read_alert_email()})


@app.post("/configure-alert-email")
def set_alert_email():
    """Store the alert recipient, overwriting any previous value."""
    data = request.get_json(silent=True) or {}
    email = str(data.get("email") or "").strip()
    if not email or not EMAIL_RE.match(email):
        return jsonify({"error": "A valid 'email' is required."}), 400
    try:
        _write_alert_email(email)
    except OSError as e:
        return jsonify({"error": f"Could not save email: {e}"}), 500
    return jsonify({"email": email})


def _parse_firms_csv(text):
    """Parse FIRMS CSV into hotspot dicts. Skips malformed rows. Never raises."""
    hotspots = []
    try:
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            try:
                lat = float((row.get("latitude") or "").strip())
                lng = float((row.get("longitude") or "").strip())
            except (ValueError, AttributeError):
                continue
            if not (-90 <= lat <= 90 and -180 <= lng <= 180):
                continue
            try:
                brightness = float((row.get("brightness") or 0) or 0)
            except (ValueError, TypeError):
                brightness = 0.0
            try:
                frp = float((row.get("frp") or 0) or 0)
            except (ValueError, TypeError):
                frp = 0.0
            hotspots.append({
                "lat": round(lat, 4),
                "lng": round(lng, 4),
                "brightness": round(brightness, 1),
                "frp": round(frp, 1),
                "confidence": (row.get("confidence") or "").strip().lower(),
                "satellite": (row.get("satellite") or "").strip(),
                "acq_date": (row.get("acq_date") or "").strip(),
                "acq_time": (row.get("acq_time") or "").strip(),
                "daynight": (row.get("daynight") or "").strip(),
            })
    except Exception as e:
        print(f"WARNING: FIRMS CSV parse failed: {e}")
    return hotspots


def _fetch_firms_world():
    """Download 1-day global VIIRS hotspots from NASA FIRMS. May raise."""
    key = (os.environ.get("FIRMS_MAP_KEY") or "").strip()
    if not key:
        raise RuntimeError("no_key")
    url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
           f"{key}/{FIRMS_SOURCE}/world/{FIRMS_DAY_RANGE}")
    req = urllib.request.Request(url, headers={"User-Agent": "forest-fire-detector/1.0"})
    with urllib.request.urlopen(req, timeout=25) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    if raw.lstrip().startswith(("Invalid", "Error", "<")):
        raise RuntimeError(f"firms_rejected: {raw[:120]}")
    return _parse_firms_csv(raw)


def _get_hotspots_payload():
    """Return cached-or-fresh hotspot payload. Never raises; always a dict."""
    now = time.time()
    if _hotspots_mem["payload"] and now - _hotspots_mem["at"] < FIRMS_CACHE_TTL_S:
        return _hotspots_mem["payload"]
    try:
        all_points = _fetch_firms_world()
        top = sorted(all_points, key=lambda p: p["brightness"], reverse=True)[:FIRMS_MAX_POINTS]
        payload = {
            "available": True,
            "source": f"NASA FIRMS {FIRMS_SOURCE} (past 24h)",
            "count": len(top),
            "total": len(all_points),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "stale": False,
            "hotspots": top,
        }
        _hotspots_mem["at"] = now
        _hotspots_mem["payload"] = payload
        try:
            with open(HOTSPOTS_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(payload, f)
        except OSError as e:
            print(f"WARNING: could not write hotspots cache: {e}")
        try:
            if pgdb is not None and pgdb.db_available():
                pgdb.db_write_hotspots(payload)
        except Exception as e:
            print(f"WARNING: Postgres hotspots write skipped: {e}")
        return payload
    except Exception as e:
        print(f"WARNING: FIRMS fetch failed: {e}")
        if _hotspots_mem["payload"]:
            stale = dict(_hotspots_mem["payload"])
            stale["stale"] = True
            return stale
        try:
            if pgdb is not None and pgdb.db_available():
                cached = pgdb.db_read_hotspots()
                if isinstance(cached, dict) and cached.get("hotspots"):
                    cached["stale"] = True
                    return cached
        except Exception as ce:
            print(f"WARNING: Postgres hotspots read skipped: {ce}")
        try:
            if os.path.exists(HOTSPOTS_CACHE_PATH):
                with open(HOTSPOTS_CACHE_PATH, "r", encoding="utf-8") as f:
                    stale = json.load(f)
                if isinstance(stale, dict) and stale.get("hotspots"):
                    stale["stale"] = True
                    return stale
        except (json.JSONDecodeError, OSError) as ce:
            print(f"WARNING: could not read hotspots cache: {ce}")
        reason = "no_key" if "no_key" in str(e) else "unavailable"
        return {"available": False, "reason": reason, "hotspots": [], "count": 0}


@app.get("/hotspots")
def hotspots():
    """Genuine worldwide satellite fire detections (NASA FIRMS).

    Returns cached data when fresh; never invents points — if NASA is
    unreachable and nothing is cached, responds available:false.
    """
    try:
        return jsonify(_get_hotspots_payload())
    except Exception as e:
        print(f"WARNING: /hotspots failed: {e}")
        return jsonify({"available": False, "reason": "unavailable",
                        "hotspots": [], "count": 0})


@app.get("/db-health")
def db_health():
    """Check your Postgres account connection. Never fails the app."""
    try:
        if pgdb is None:
            return jsonify({"postgres": False, "mode": "json-fallback", "reason": "driver-missing"})
        if not pgdb.db_available():
            return jsonify({"postgres": False, "mode": "json-fallback", "reason": "DATABASE_URL-not-set"})
        ok = pgdb.init_db()
        rows = pgdb.db_read_log(limit=1)
        return jsonify({
            "postgres": bool(ok and rows is not None),
            "mode": "postgres" if (ok and rows is not None) else "json-fallback",
        })
    except Exception as e:
        return jsonify({"postgres": False, "mode": "json-fallback", "reason": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
