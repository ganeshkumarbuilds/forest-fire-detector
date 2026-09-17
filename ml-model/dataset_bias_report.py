"""
Feature 10: Descriptive Dataset Bias / Shortcut Analysis Report.
================================================================
OFFLINE research artifact only. No retraining, no weight changes,
no threshold change, no production-path change, no new ML model,
no causal claims.

Uses: ml-model/error_analysis.json, ml-model/robustness.json,
      ml-model/dataset/, PIL/numpy only.

Run:
    venv\\Scripts\\python.exe ml-model/dataset_bias_report.py
"""
import json
import os
import re
from datetime import datetime, timezone

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(HERE, "dataset")
FIRE_DIR = os.path.join(DATASET_DIR, "fire")
NO_FIRE_DIR = os.path.join(DATASET_DIR, "no_fire")
ERROR_ANALYSIS_PATH = os.path.join(HERE, "error_analysis.json")
ROBUSTNESS_PATH = os.path.join(HERE, "robustness.json")
OUTPUT_PATH = os.path.join(HERE, "dataset_bias_report.json")

IMG_EXTS = (".jpg", ".jpeg", ".png")
N_HIST_BINS = 10


def describe(values):
    a = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(a)),
        "mean": round(float(np.mean(a)), 4) if len(a) else None,
        "median": round(float(np.median(a)), 4) if len(a) else None,
        "std": round(float(np.std(a, ddof=1)), 4) if len(a) > 1 else 0.0,
        "min": round(float(np.min(a)), 4) if len(a) else None,
        "max": round(float(np.max(a)), 4) if len(a) else None,
    }


def cohens_d(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return None
    va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    pooled = np.sqrt((va + vb) / 2.0)
    if pooled == 0:
        return 0.0
    return round(float((np.mean(a) - np.mean(b)) / pooled), 4)


def color_stats(image_path):
    """RGB means, R-G, R-B, saturation. Numpy only, no ML model.

    Saturation per pixel = (max-min)/max (0 when max==0), averaged.
    Computed on the original image converted to RGB.
    """
    img = Image.open(image_path).convert("RGB")
    a = np.array(img, dtype=np.float32)
    r = float(a[:, :, 0].mean())
    g = float(a[:, :, 1].mean())
    b = float(a[:, :, 2].mean())
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1.0), 0.0)
    return {"r": r, "g": g, "b": b, "r_minus_g": r - g,
            "r_minus_b": r - b, "saturation": float(s.mean())}


def histogram_summary(fire_vals, nofire_vals, n_bins=N_HIST_BINS):
    f = np.asarray(fire_vals, dtype=np.float64)
    n = np.asarray(nofire_vals, dtype=np.float64)
    lo = float(min(f.min(), n.min()))
    hi = float(max(f.max(), n.max()))
    if hi == lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, n_bins + 1).tolist()
    fc, _ = np.histogram(f, bins=edges)
    nc, _ = np.histogram(n, bins=edges)
    return {"bin_edges": [round(float(e), 4) for e in edges],
            "fire_counts": [int(c) for c in fc],
            "no_fire_counts": [int(c) for c in nc],
            "n_bins": n_bins}


def prefix_of(fname):
    m = re.match(r"^([A-Za-z_]+?)(\d+)\.", fname)
    if m:
        return m.group(1)
    m = re.match(r"^([A-Za-z_]+)", fname)
    return m.group(1) if m else "other"


def main():
    with open(ERROR_ANALYSIS_PATH, "r", encoding="utf-8") as f:
        analysis = json.load(f)
    per_image = analysis.get("per_image_results", [])
    false_positives = analysis.get("false_positives", [])

    # --- Full dataset scan (counts, dimensions, aspect ratio) ---
    def scan(folder):
        files = sorted([f for f in os.listdir(folder)
                        if f.lower().endswith(IMG_EXTS)])
        sizes, aspects = [], []
        for fn in files:
            with Image.open(os.path.join(folder, fn)) as im:
                w, h = im.size
            sizes.append((w, h))
            aspects.append(w / h if h else 0)
        return files, sizes, aspects

    fire_files, fire_sizes, fire_aspects = scan(FIRE_DIR)
    nofire_files, nofire_sizes, nofire_aspects = scan(NO_FIRE_DIR)
    full_n = len(fire_files) + len(nofire_files)
    assert full_n == 1832, f"Expected 1832, got {full_n}"
    assert len(fire_files) == 928 and len(nofire_files) == 904
    assert len(per_image) == 275
    fire_test = [r for r in per_image if r["true_class"] == "fire"]
    nofire_test = [r for r in per_image if r["true_class"] == "no_fire"]
    assert len(fire_test) == 139 and len(nofire_test) == 136
    assert len(false_positives) == 6
    fp_names = {r["filename"] for r in false_positives}
    assert fp_names.issubset({r["filename"] for r in per_image})

    fire_dims = sorted({tuple(s) for s in fire_sizes})
    nofire_dims = sorted({tuple(s) for s in nofire_sizes})

    full_dataset = {
        "total_images": full_n,
        "fire_count": len(fire_files),
        "no_fire_count": len(nofire_files),
        "image_dimensions": {
            "fire_unique_sizes": [[int(w), int(h)] for w, h in fire_dims],
            "no_fire_unique_sizes": [[int(w), int(h)] for w, h in nofire_dims],
            "note": "All images 250x250 in both classes (no dimension difference).",
        },
        "aspect_ratio": {
            "fire": describe(fire_aspects),
            "no_fire": describe(nofire_aspects),
        },
    }

    # --- Test-set statistics ---
    # Brightness/contrast/sharpness reused from error_analysis.json quality
    # (same methodology as error_analysis.py / validate_image_quality()).
    metrics = {}
    color_acc = {"fire": [], "no_fire": []}
    for cls, items in (("fire", fire_test), ("no_fire", nofire_test)):
        br = [r["quality"]["brightness"] for r in items]
        co = [r["quality"]["contrast"] for r in items]
        sh = [r["quality"]["blur_score"] for r in items]
        cols = [color_stats(os.path.join(DATASET_DIR, r["filename"]))
                for r in items]
        color_acc[cls] = cols
        metrics[cls] = {
            "brightness": describe(br),
            "contrast": describe(co),
            "sharpness_laplacian_variance": describe(sh),
            "r_mean": describe([c["r"] for c in cols]),
            "g_mean": describe([c["g"] for c in cols]),
            "b_mean": describe([c["b"] for c in cols]),
            "r_minus_g": describe([c["r_minus_g"] for c in cols]),
            "r_minus_b": describe([c["r_minus_b"] for c in cols]),
            "saturation": describe([c["saturation"] for c in cols]),
        }

    effect_sizes = {}
    for key in ["brightness", "contrast", "sharpness_laplacian_variance",
                "r_minus_g", "r_minus_b", "saturation"]:
        if key == "brightness":
            a = [r["quality"]["brightness"] for r in fire_test]
            b = [r["quality"]["brightness"] for r in nofire_test]
        elif key == "contrast":
            a = [r["quality"]["contrast"] for r in fire_test]
            b = [r["quality"]["contrast"] for r in nofire_test]
        elif key == "sharpness_laplacian_variance":
            a = [r["quality"]["blur_score"] for r in fire_test]
            b = [r["quality"]["blur_score"] for r in nofire_test]
        elif key == "r_minus_g":
            a = [c["r_minus_g"] for c in color_acc["fire"]]
            b = [c["r_minus_g"] for c in color_acc["no_fire"]]
        elif key == "r_minus_b":
            a = [c["r_minus_b"] for c in color_acc["fire"]]
            b = [c["r_minus_b"] for c in color_acc["no_fire"]]
        else:
            a = [c["saturation"] for c in color_acc["fire"]]
            b = [c["saturation"] for c in color_acc["no_fire"]]
        effect_sizes[key] = {
            "cohens_d_fire_minus_nofire": cohens_d(a, b),
            "note": "Descriptive effect size only; a large value is not "
                    "proof of model bias.",
        }

    # --- False positives vs no_fire test population ---
    fp_br = [r["quality"]["brightness"] for r in false_positives]
    fp_co = [r["quality"]["contrast"] for r in false_positives]
    fp_sh = [r["quality"]["blur_score"] for r in false_positives]
    fp_cols = [color_stats(os.path.join(DATASET_DIR, r["filename"]))
               for r in false_positives]
    nf_br = [r["quality"]["brightness"] for r in nofire_test]
    nf_co = [r["quality"]["contrast"] for r in nofire_test]
    nf_sh = [r["quality"]["blur_score"] for r in nofire_test]
    nf_rmg = [c["r_minus_g"] for c in color_acc["no_fire"]]
    nf_rmb = [c["r_minus_b"] for c in color_acc["no_fire"]]
    nf_sat = [c["saturation"] for c in color_acc["no_fire"]]
    fp_rmg = [c["r_minus_g"] for c in fp_cols]
    fp_rmb = [c["r_minus_b"] for c in fp_cols]
    fp_sat = [c["saturation"] for c in fp_cols]
    false_positive_analysis = {
        "count": len(false_positives),
        "filenames": sorted(fp_names),
        "brightness": {**describe(fp_br), "cohens_d_vs_nofire_test":
                       cohens_d(fp_br, nf_br)},
        "contrast": {**describe(fp_co), "cohens_d_vs_nofire_test":
                     cohens_d(fp_co, nf_co)},
        "sharpness_laplacian_variance": {
            **describe(fp_sh), "cohens_d_vs_nofire_test": cohens_d(fp_sh, nf_sh)},
        "r_minus_g": {**describe(fp_rmg), "cohens_d_vs_nofire_test":
                      cohens_d(fp_rmg, nf_rmg)},
        "r_minus_b": {**describe(fp_rmb), "cohens_d_vs_nofire_test":
                      cohens_d(fp_rmb, nf_rmb)},
        "saturation": {**describe(fp_sat), "cohens_d_vs_nofire_test":
                       cohens_d(fp_sat, nf_sat)},
        "reference_nofire_test_means": {
            "brightness": round(float(np.mean(nf_br)), 4),
            "contrast": round(float(np.mean(nf_co)), 4),
            "sharpness": round(float(np.mean(nf_sh)), 4),
        },
        "warning": "n=6 is too small for statistical generalization. No bias "
                   "rule is defined from these cases and no measured "
                   "characteristic is claimed to have caused the errors.",
    }

    # --- Filename / source-pattern audit ---
    def audit(files):
        counts = {}
        for fn in files:
            p = prefix_of(fn)
            counts[p] = counts.get(p, 0) + 1
        return counts
    fire_prefix = audit(fire_files)
    nofire_prefix = audit(nofire_files)
    filename_audit = {
        "directory_structure": {"fire": "ml-model/dataset/fire/",
                               "no_fire": "ml-model/dataset/no_fire/"},
        "extensions": {"fire": [".jpg"], "no_fire": [".jpg"]},
        "counts_by_prefix": {"fire": fire_prefix, "no_fire": nofire_prefix},
        "prefixes_across_both_classes": sorted(
            set(fire_prefix) & set(nofire_prefix)),
        "note": "Directory encodes the label by construction and is not a "
                "model input. Shared prefixes (e.g. abc) occur in both "
                "classes. Undocumented source identities are not inferred "
                "from filename prefixes.",
    }

    # --- Robustness summary (read only, no recompute) ---
    with open(ROBUSTNESS_PATH, "r", encoding="utf-8") as f:
        rob = json.load(f)
    pert_summary = []
    for p in rob.get("perturbations", []):
        pert_summary.append({
            "name": p.get("name"),
            "level": str(p.get("level")),
            "accuracy": p.get("metrics", {}).get("accuracy"),
            "accuracy_delta": (p.get("metric_deltas", {}) or {}).get(
                "accuracy_delta"),
            "prediction_change_rate": p.get("prediction_change_rate"),
            "avg_absolute_probability_change": p.get(
                "avg_absolute_probability_change"),
            "fp": (p.get("metrics", {}) or {}).get("fp"),
            "fn": (p.get("metrics", {}) or {}).get("fn"),
            "clean_fp_changed": ((p.get("clean_fp_stability", {}) or {})
                                 .get("changed")),
        })
    robustness_summary = {
        "label": "Synthetic perturbation experiments only; not real-world "
                 "evidence.",
        "clean_metrics": rob.get("clean_metrics"),
        "perturbations": pert_summary,
    }

    # --- Visualization data (histogram summaries, no new dependency) ---
    histograms = {}
    pairs = {
        "brightness": ([r["quality"]["brightness"] for r in fire_test],
                       [r["quality"]["brightness"] for r in nofire_test]),
        "contrast": ([r["quality"]["contrast"] for r in fire_test],
                     [r["quality"]["contrast"] for r in nofire_test]),
        "sharpness": ([r["quality"]["blur_score"] for r in fire_test],
                      [r["quality"]["blur_score"] for r in nofire_test]),
        "saturation": ([c["saturation"] for c in color_acc["fire"]],
                       [c["saturation"] for c in color_acc["no_fire"]]),
        "r_minus_g": ([c["r_minus_g"] for c in color_acc["fire"]],
                      [c["r_minus_g"] for c in color_acc["no_fire"]]),
        "r_minus_b": ([c["r_minus_b"] for c in color_acc["fire"]],
                      [c["r_minus_b"] for c in color_acc["no_fire"]]),
    }
    for k, (a, b) in pairs.items():
        histograms[k + "_by_class"] = histogram_summary(a, b)

    report = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_size": full_n,
            "fire_count": len(fire_files),
            "no_fire_count": len(nofire_files),
            "test_size": len(per_image),
            "fire_test": len(fire_test),
            "no_fire_test": len(nofire_test),
            "false_positive_count": len(false_positives),
            "random_state": 42,
            "threshold": 0.5,
            "sources": ["ml-model/error_analysis.json",
                        "ml-model/robustness.json",
                        "ml-model/dataset/"],
            "methodology": "Brightness=mean RGB; contrast=std of luminance "
                           "(0.299R+0.587G+0.114B); sharpness=Laplacian "
                           "variance (kernel [[0,1,0],[1,-4,1],[0,1,0]], "
                           "reflect pad), consistent with error_analysis.py "
                           "and validate_image_quality(); color stats via "
                           "PIL/numpy only, no ML model.",
        },
        "full_dataset": full_dataset,
        "test_set": {"class_statistics": metrics,
                     "effect_sizes": effect_sizes},
        "false_positive_analysis": false_positive_analysis,
        "filename_audit": filename_audit,
        "robustness_summary": robustness_summary,
        "histograms": histograms,
        "limitations": [
            "These are observed associations, not causal explanations.",
            "Fire naturally has visual characteristics such as orange/red "
            "coloration, so class differences do not automatically indicate "
            "shortcuts.",
            "Dataset source/geographic/device/day-night metadata is "
            "unavailable.",
            "The six false positives are too small a sample for statistical "
            "generalization.",
            "The held-out test set is being used descriptively, not for "
            "fitting any rule.",
            "Synthetic perturbations do not establish real-world robustness "
            "or OOD performance.",
            "No claim should be made that the model relies on a particular "
            "feature without additional controlled experiments.",
        ],
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[OK] Dataset bias report saved to {OUTPUT_PATH}")
    print(f"Full: {full_n} (fire {len(fire_files)}, no_fire {len(nofire_files)})")
    print(f"Test: {len(per_image)} (fire {len(fire_test)}, "
          f"no_fire {len(nofire_test)}), FPs {len(false_positives)}")
    for k, v in effect_sizes.items():
        print(f"  d({k}) = {v['cohens_d_fire_minus_nofire']}")


if __name__ == "__main__":
    main()
