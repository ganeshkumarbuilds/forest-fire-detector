"""
Feature 8: Explainable Error Investigation for existing false positives.
========================================================================
OFFLINE script. Reads the six false positives from error_analysis.json
(do NOT hardcode filenames), generates an FD-CAM overlay for each FP
using the EXISTING FD-CAM implementation in backend/app.py, and persists
results to ml-model/error_explanations.json.

- Does not modify the model, does not retrain.
- Does not create a second Grad-CAM implementation: all FD-CAM math
  (feature extractor, classifier head, epsilon, preprocessing, heatmap
  generation) is imported from backend/app.py and called directly.
- Explanation corresponds to the predicted FIRE class (fire_detected=True).

Run:
    venv\\Scripts\\python.exe ml-model/explain_fps.py
"""
import base64
import io
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ERROR_ANALYSIS_PATH = os.path.join(HERE, "error_analysis.json")
OUTPUT_PATH = os.path.join(HERE, "error_explanations.json")
DATASET_DIR = os.path.join(HERE, "dataset")

# ── Reuse EXISTING FD-CAM implementation (no second implementation) ──
# Import backend/app.py so feature extractor, classifier head, epsilon,
# preprocessing and heatmap generation are exactly the production ones.
BACKEND_DIR = os.path.join(HERE, "..", "backend")
BACKEND_DIR = os.path.abspath(BACKEND_DIR)
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import app as backend_app  # noqa: E402


class _DiskFileStorage:
    """Minimal adapter so backend _gradcam_base64 can read a dataset file.

    backend_app._gradcam_base64 expects a Flask file_storage with
    `.stream` (seekable binary) and `.filename`. This wraps an on-disk
    image without changing any FD-CAM behavior.
    """

    def __init__(self, path, filename):
        self.filename = filename
        self.stream = open(path, "rb")  # noqa: SIM115 - closed explicitly

    def close(self):
        try:
            self.stream.close()
        except Exception:
            pass


def _original_image_b64(image_path):
    """Encode the original image (RGB, 224x224 PNG, base64) for the UI.

    This is plain image encoding, not saliency logic, so it does not
    constitute a second Grad-CAM implementation. Storing it avoids
    adding static-file handling to the backend.
    """
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main():
    if not os.path.exists(ERROR_ANALYSIS_PATH):
        print(f"ERROR: {ERROR_ANALYSIS_PATH} not found. Run ml-model/error_analysis.py first.")
        sys.exit(1)

    with open(ERROR_ANALYSIS_PATH, "r", encoding="utf-8") as f:
        analysis = json.load(f)

    # Read the false positives directly — never hardcode filenames.
    false_positives = analysis.get("false_positives", [])
    if not false_positives:
        print("ERROR: no false positives found in error_analysis.json")
        sys.exit(1)
    print(f"Found {len(false_positives)} false positive(s) in error_analysis.json")

    if backend_app.model is None:
        print("ERROR: backend TFLite model failed to load; cannot generate FD-CAM.")
        sys.exit(1)

    eval_meta = analysis.get("evaluation_metadata", {})
    explanations = []

    for entry in false_positives:
        filename = entry["filename"]  # e.g. "no_fire/nofire_0062.jpg"
        img_path = os.path.join(DATASET_DIR, filename)
        img_path = os.path.abspath(img_path)
        if not os.path.exists(img_path):
            print(f"ERROR: image not found under dataset dir: {img_path}")
            sys.exit(1)

        # Same preprocessing as backend (/predict): RGB, 224x224, x/127.5-1.
        pil = Image.open(img_path).convert("RGB").resize((224, 224))
        arr = np.array(pil).astype("float32")
        arr = np.expand_dims(arr, axis=0)
        x = backend_app._mobilenet_v2_preprocess(arr)

        storage = _DiskFileStorage(img_path, os.path.basename(filename))
        try:
            # Single locked section: one full-model forward (verification
            # only) + one FD-CAM pass. No duplicate inference beyond that.
            # Explanation targets the predicted FIRE class for these FPs.
            with backend_app._infer_lock:
                p_tflite = backend_app._run_full(x)
                heatmap_b64 = backend_app._gradcam_base64(
                    storage, x, True  # fire_detected=True → explain P(fire)
                )
        finally:
            storage.close()

        if not heatmap_b64:
            print(f"ERROR: FD-CAM generation failed for {filename}")
            sys.exit(1)

        # Verify TFLite recomputation matches the clean Keras evaluation.
        clean_prob = float(entry["prob_fire"])
        if abs(float(p_tflite) - clean_prob) > 0.01:
            print(
                f"WARNING: {filename} TFLite P(fire)={float(p_tflite):.4f} "
                f"differs from clean eval {clean_prob:.4f}"
            )

        original_b64 = _original_image_b64(img_path)

        explanations.append(
            {
                "filename": filename,
                "true_label": int(entry.get("true_label", 0)),
                "true_class": entry.get("true_class", "no_fire"),
                "predicted_label": int(entry.get("predicted_label", 1)),
                "predicted_class": entry.get("predicted_class", "fire"),
                # Persist the clean evaluation probability so the stored
                # value matches error_analysis.json exactly.
                "prob_fire": clean_prob,
                "quality": entry.get("quality", {}),
                "heatmap_image": heatmap_b64,
                "original_image": original_b64,
            }
        )
        print(
            f"[OK] {filename} P(fire)={clean_prob:.4f} "
            f"heatmap={len(heatmap_b64)} chars"
        )

    artifact = {
        "evaluation_timestamp": eval_meta.get("timestamp"),
        "explanation_method": "FD-CAM",
        "model": {
            "model_path": eval_meta.get("model_path"),
            "model_type": eval_meta.get("model_type"),
            "decision_threshold": eval_meta.get("decision_threshold", 0.5),
            "fd_eps": getattr(backend_app, "_FD_EPS", 0.1),
        },
        "input": {
            "input_shape": eval_meta.get("input_shape", [224, 224, 3]),
            "preprocessing": "MobileNetV2 tf mode (x/127.5 - 1), RGB, 224x224",
            "explained_class": "fire (predicted class for false positives)",
        },
        "count": len(explanations),
        "explanations": explanations,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    size = os.path.getsize(OUTPUT_PATH)
    print(f"\n[OK] Wrote {len(explanations)} FP explanations to {OUTPUT_PATH} ({size} bytes)")


if __name__ == "__main__":
    main()
