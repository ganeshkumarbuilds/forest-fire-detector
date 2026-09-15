"""Forest Fire Detection - Flask backend (TFLite inference)."""
import base64
import csv
import gc
import io
import json
import os
import re
import smtplib
import threading
import time
import urllib.request
from datetime import datetime, timezone
from email.message import EmailMessage

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

# ── TFLite interpreter (tiny, fast on weak CPU; no TensorFlow dependency) ──
# Render (linux): tflite-runtime. Local Windows dev: TF's built-in interpreter
# (tflite-runtime ships no Windows wheels). Both run the same .tflite bytes.
try:
    from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
    _TFLITE_BACKEND = "tflite-runtime"
except ImportError:  # local dev fallback (tf.lite.Interpreter attribute form)
    import tensorflow as _tf
    TFLiteInterpreter = _tf.lite.Interpreter
    _TFLITE_BACKEND = "tensorflow"

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__)
CORS(app)

try:
    import db as pgdb
    try:
        pgdb.init_db()
    except Exception as _e:
        print(f"WARNING: Postgres init skipped: {_e}")
except ImportError as _e:
    pgdb = None
    print(f"WARNING: Postgres driver missing, using JSON storage: {_e}")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "fire_model.tflite")
MODEL_PATH = os.path.abspath(MODEL_PATH)

FEAT_PATH = os.path.join(os.path.dirname(__file__), "fire_model_feat.tflite")
FEAT_PATH = os.path.abspath(FEAT_PATH)

HEAD_PATH = os.path.join(os.path.dirname(__file__), "fire_model_head.tflite")
HEAD_PATH = os.path.abspath(HEAD_PATH)

LOG_PATH = os.path.join(os.path.dirname(__file__), "detection_log.json")
LOG_PATH = os.path.abspath(LOG_PATH)

METRICS_PATH = os.path.join(os.path.dirname(__file__), "..", "ml-model", "eval_metrics.json")
METRICS_PATH = os.path.abspath(METRICS_PATH)

ALERT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "alert_config.json")
ALERT_CONFIG_PATH = os.path.abspath(ALERT_CONFIG_PATH)

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

FIRMS_SOURCE = "VIIRS_SNPP_NRT"
FIRMS_DAY_RANGE = 1
FIRMS_CACHE_TTL_S = 30 * 60
FIRMS_MAX_POINTS = 2000
HOTSPOTS_CACHE_PATH = os.path.join(os.path.dirname(__file__), "hotspots_cache.json")
HOTSPOTS_CACHE_PATH = os.path.abspath(HOTSPOTS_CACHE_PATH)
_hotspots_mem = {"at": 0.0, "payload": None}

# ── Inference lock ──
# Gunicorn runs with >1 thread so /health and /ready stay responsive while
# a /predict is computing. TFLite interpreters are not thread-safe for
# concurrent invoke(), so all interpreter calls are serialized here.
# RLock (reentrant): heatmap helpers take it again while the caller holds it.
# No logic change — only mutual exclusion around interpreter calls.
_infer_lock = threading.RLock()

# FD-CAM finite-difference step (validated: corr 0.997 vs GradientTape Grad-CAM).
_FD_EPS = 0.1


def _load_interpreter(path):
    interp = TFLiteInterpreter(model_path=path)
    interp.allocate_tensors()
    return interp


def _run_full(x):
    """P(fire) for a preprocessed (1,224,224,3) batch. Caller holds _infer_lock."""
    interp = model["full"]
    det = interp.get_input_details()[0]
    interp.set_tensor(det["index"], x.astype(np.float32))
    interp.invoke()
    out = interp.get_output_details()[0]
    return float(interp.get_tensor(out["index"])[0][0])


def _run_feat(x):
    """Last-conv feature map (7,7,1280) for a preprocessed batch. Caller holds lock."""
    interp = model["feat"]
    det = interp.get_input_details()[0]
    interp.set_tensor(det["index"], x.astype(np.float32))
    interp.invoke()
    out = interp.get_output_details()[0]
    return interp.get_tensor(out["index"])[0]


def _run_head(A):
    """P(fire) for a (1,7,7,1280) feature batch. Caller holds _infer_lock."""
    interp = model["head"]
    det = interp.get_input_details()[0]
    interp.set_tensor(det["index"], A.astype(np.float32))
    interp.invoke()
    out = interp.get_output_details()[0]
    return float(interp.get_tensor(out["index"])[0][0])


model = None
try:
    print(f"Loading TFLite models ({_TFLITE_BACKEND}) ...", flush=True)
    missing = [p for p in (MODEL_PATH, FEAT_PATH, HEAD_PATH) if not os.path.exists(p)]
    if missing:
        print(f"WARNING: model file(s) not found: {missing}", flush=True)
    else:
        _full = _load_interpreter(MODEL_PATH)
        _feat = _load_interpreter(FEAT_PATH)
        _head = _load_interpreter(HEAD_PATH)
        model = {"full": _full, "feat": _feat, "head": _head}
        print("TFLite models loaded at import.", flush=True)
        try:
            _dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
            _t0 = time.time()
            _run_full(_dummy)
            _run_head(_run_feat(_dummy)[None])
            print(f"TFLite warmed up at boot ({(time.time() - _t0) * 1000:.0f} ms)"
                  " -- first /predict will be fast.", flush=True)
        except Exception as _we:
            print(f"WARNING: boot warm-up failed (first request slower): {_we}", flush=True)
except Exception as e:
    import traceback
    traceback.print_exc()
    model = None
    print(f"Model load failed at import: {e}", flush=True)


def _mobilenet_v2_preprocess(arr):
    """Identical math to keras MobileNetV2 preprocess_input (mode 'tf'): x/127.5 - 1."""
    return (arr / 127.5 - 1.0).astype(np.float32)


def preprocess_image(file_storage):
    img = Image.open(file_storage.stream).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    arr = _mobilenet_v2_preprocess(arr)  # same normalization as training
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


def _fd_gradcam_weights(A, fire_detected):
    """Grad-CAM channel weights via finite differences (no gradients needed).

    w_k = (S(A + eps*e_k) - S(A)) / eps, where S is the explained class score:
    P(fire) when fire was detected, else P(no fire) = 1 - output.
    The head (GAP + Dense) is microscopic, so 1280 micro-forwards take ~1-2s.
    Validated: corr 0.997 vs GradientTape Grad-CAM at eps=0.1.
    Caller holds _infer_lock. Never raises (returns None on failure).
    """
    try:
        if model is None:
            return None
        A = np.asarray(A, dtype=np.float32)
        p = _run_head(A[None])
        s0 = p if fire_detected else 1.0 - p
        n_channels = A.shape[-1]
        weights = np.zeros(n_channels, dtype=np.float64)
        for k in range(n_channels):
            Ap = A.copy()
            Ap[:, :, k] += _FD_EPS
            sk = _run_head(Ap[None])
            sk = sk if fire_detected else 1.0 - sk
            weights[k] = (sk - s0) / _FD_EPS
        return weights, A.astype(np.float64)
    except Exception as e:
        print(f"WARNING: FD Grad-CAM weights failed: {e}")
        return None


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
        if model is None:
            return None
        fmaps = _run_feat(x)
        fd = _fd_gradcam_weights(fmaps, fire_detected)
        if fd is None:
            return None
        weights, fmaps = fd
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


@app.get("/ready")
def ready():
    """True once the TFLite models are loaded.

    With synchronous loading at import, models are always ready after import.
    Frontend polls this endpoint to check if /predict will succeed.
    """
    return jsonify({"ready": model is not None})


def _rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except Exception:
        pass
    return None


@app.get("/debug/model")
def debug_model():
    """Debug endpoint to check model status."""
    import os
    return jsonify({
        "model_loaded": model is not None,
        "engine": _TFLITE_BACKEND,
        "rss_mb": _rss_mb(),
        "cwd": os.getcwd(),
        "files_in_cwd": os.listdir(".")[:10],
    })


@app.get("/debug/infer")
def debug_infer():
    """Run a dummy inference server-side; returns elapsed ms. Isolates engine speed."""
    import time as _time
    if model is None:
        return jsonify({"ok": False, "error": "model not loaded"}), 503
    try:
        x = np.zeros((1, 224, 224, 3), dtype=np.float32)
        t0 = _time.time()
        with _infer_lock:
            prob = _run_full(x)
        dt = round((_time.time() - t0) * 1000)
        return jsonify({"ok": True, "elapsed_ms": dt, "rss_mb": _rss_mb(),
                        "engine": _TFLITE_BACKEND, "output": prob})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"ok": False, "error": str(e), "rss_mb": _rss_mb()}), 500


@app.post("/predict")
def predict():
    if model is None:
        # 503 (not a hang): server still warming up — frontend waits
        # for /ready and retries. Proxies never time this out.
        return jsonify({"error": "Server is warming up, please retry in a few seconds.",
                        "retry": True}), 503

    if "file" not in request.files:
        return jsonify({"error": "No 'file' part in request."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    try:
        x = preprocess_image(file)
        with _infer_lock:
            prob = _run_full(x)
        fire_detected = prob >= 0.5
        confidence = round(prob if fire_detected else 1.0 - prob, 4)
        timestamp = datetime.now(timezone.utc).isoformat()
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
        try:
            heatmap = request.args.get("heatmap", "0") == "1"
            if heatmap:
                with _infer_lock:
                    heatmap = _gradcam_base64(file, x, fire_detected)
                if heatmap:
                    result["heatmap_image"] = heatmap
        except Exception as e:
            print(f"WARNING: Grad-CAM failed: {e}")
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
        try:
            gc.collect()
        except Exception:
            pass
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Inference failed: {e}",
                        "detail": str(e)}), 500


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
    # Short timeout: /hotspots must never block a request near proxy limits.
    # Stale cache is served instantly while a background thread refreshes.
    with urllib.request.urlopen(req, timeout=12) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    if raw.lstrip().startswith(("Invalid", "Error", "<")):
        raise RuntimeError(f"firms_rejected: {raw[:120]}")
    return _parse_firms_csv(raw)


def _store_hotspots_payload(payload):
    """Persist a fresh payload to memory + disk + Postgres. Never raises."""
    import copy
    _hotspots_mem["at"] = time.time()
    _hotspots_mem["payload"] = copy.deepcopy(payload)
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


def _read_stale_hotspots():
    """Best-effort stale payload from memory, Postgres, or disk. May be None."""
    if _hotspots_mem["payload"]:
        return dict(_hotspots_mem["payload"])
    try:
        if pgdb is not None and pgdb.db_available():
            cached = pgdb.db_read_hotspots()
            if isinstance(cached, dict) and cached.get("hotspots"):
                return cached
    except Exception as ce:
        print(f"WARNING: Postgres hotspots read skipped: {ce}")
    try:
        if os.path.exists(HOTSPOTS_CACHE_PATH):
            with open(HOTSPOTS_CACHE_PATH, "r", encoding="utf-8") as f:
                stale = json.load(f)
            if isinstance(stale, dict) and stale.get("hotspots"):
                return stale
    except (json.JSONDecodeError, OSError) as ce:
        print(f"WARNING: could not read hotspots cache: {ce}")
    return None


_hotspots_refresh_lock = threading.Lock()
_hotspots_refreshing = False


def _refresh_hotspots_async():
    """Refresh NASA FIRMS data in a daemon thread; never blocks a request."""
    global _hotspots_refreshing
    with _hotspots_refresh_lock:
        if _hotspots_refreshing:
            return
        _hotspots_refreshing = True

    def _work():
        global _hotspots_refreshing
        try:
            all_points = _fetch_firms_world()
            top = sorted(all_points, key=lambda p: p["brightness"], reverse=True)[:FIRMS_MAX_POINTS]
            _store_hotspots_payload({
                "available": True,
                "source": f"NASA FIRMS {FIRMS_SOURCE} (past 24h)",
                "count": len(top),
                "total": len(all_points),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "stale": False,
                "hotspots": top,
            })
        except Exception as e:
            print(f"WARNING: background FIRMS refresh failed: {e}")
        finally:
            with _hotspots_refresh_lock:
                _hotspots_refreshing = False

    t = threading.Thread(target=_work, daemon=True)
    t.start()


def _get_hotspots_payload():
    """Return cached-or-fresh hotspot payload. Never raises; always a dict.

    Stale-while-revalidate: when the cache is expired but a stale copy
    exists, it is returned instantly (stale:true) while a background
    thread refreshes — so /hotspots can never hang a request near
    proxy timeouts even when NASA FIRMS is slow. Only when no cache
    exists at all do we fetch synchronously (bounded by short timeout).
    """
    now = time.time()
    if _hotspots_mem["payload"] and now - _hotspots_mem["at"] < FIRMS_CACHE_TTL_S:
        return _hotspots_mem["payload"]
    stale = _read_stale_hotspots()
    if stale is not None:
        stale["stale"] = True
        _refresh_hotspots_async()
        return stale
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
        _store_hotspots_payload(payload)
        return payload
    except Exception as e:
        print(f"WARNING: FIRMS fetch failed: {e}")
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
    finally:
        # Cleanup large payload memory
        try:
            gc.collect()
        except Exception:
            pass


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
