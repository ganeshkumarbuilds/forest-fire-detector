"""
Forest Fire Detection - Model Training Script
=============================================
Transfer learning with MobileNetV2 (ImageNet, frozen base) for binary
fire / no_fire classification.

WHERE TO PUT REAL DATA (Kaggle wildfire dataset):
-------------------------------------------------
Drop real images here:

  ml-model/dataset/fire/      <- fire images (.jpg / .png)
  ml-model/dataset/no_fire/   <- no-fire / forest images (.jpg / .png)

Example Kaggle source: search "wildfire dataset" or "forest fire images"
(e.g. https://www.kaggle.com/datasets). Download, unzip, and copy the
class folders into the paths above, then re-run:
    python train_model.py

If the folders are empty/missing, the script generates dummy random
images so it doesn't crash (useful for pipeline testing only — accuracy
will be ~50% on dummy data).
"""

import os
import json
from datetime import datetime, timezone
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score, recall_score,
                             f1_score)

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array,
)

IMG_SIZE = (224, 224)
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
FIRE_DIR = os.path.join(DATASET_DIR, "fire")
NO_FIRE_DIR = os.path.join(DATASET_DIR, "no_fire")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "fire_model.h5")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "eval_metrics.json")
EPOCHS = 10
BATCH_SIZE = 32


def load_dataset():
    """Load images as RAW 0-255 float32 arrays (no normalization here).

    Normalization (preprocess_input) is applied later: baked into the
    training ImageDataGenerator via preprocessing_function, and applied
    directly to the val/test arrays — so augmentation happens in pixel
    space and val/test stay clean.
    """
    images, labels = [], []

    def load_folder(folder, label):
        if not os.path.isdir(folder):
            return 0
        count = 0
        for fname in os.listdir(folder):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            try:
                img = load_img(os.path.join(folder, fname), target_size=IMG_SIZE)
                arr = img_to_array(img).astype("float32")  # raw 0-255
                images.append(arr)
                labels.append(label)
                count += 1
            except Exception as e:
                print(f"Skipping {fname}: {e}")
        return count

    n_fire = load_folder(FIRE_DIR, 1)
    n_no_fire = load_folder(NO_FIRE_DIR, 0)
    print(f"Loaded {n_fire} fire images, {n_no_fire} no_fire images.")
    return np.array(images), np.array(labels)


def make_dummy_data(num_per_class=20):
    print("Dataset empty or missing — generating dummy random images "
          "so the script doesn't crash. Replace with real Kaggle data!")
    rng = np.random.default_rng(42)
    # Raw 0-255 pixels (normalization happens downstream, same as real data).
    dummy = rng.integers(0, 256, size=(num_per_class * 2, 224, 224, 3)).astype("float32")
    labels = np.array([1] * num_per_class + [0] * num_per_class)
    return dummy, labels


def build_model():
    base = MobileNetV2(weights="imagenet", include_top=False,
                       input_shape=(224, 224, 3))
    base.trainable = False  # freeze base layers

    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base.input, outputs=output)
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def _stratify_or_none(labels):
    """Return labels for stratify=, or None when stratification is unsafe.

    sklearn needs >= 2 samples of every class to stratify a split —
    fall back to an unstratified split for degenerate tiny datasets.
    """
    labels = np.asarray(labels)
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return None
    if np.min(np.bincount(labels.astype(int))) < 2:
        return None
    return labels


def main():
    X, y = load_dataset()
    if len(X) == 0:
        X, y = make_dummy_data()
    y = np.asarray(y)

    n_fire = int((y == 1).sum())
    n_no_fire = int((y == 0).sum())

    # 1. Small-dataset guard: flag unreliable/overfit-prone metrics.
    warning = None
    if n_fire < 100 or n_no_fire < 100:
        warning = "small dataset (<100 images) — metrics may be unreliable/overfit."
        print(f"WARNING: {warning}")

    # 2. Stratified 70/15/15 train/val/test split (class balance preserved
    # in all three sets). Second split takes 15/85 of the temp remainder
    # so val ends up ~15% of the full dataset.
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42,
        stratify=_stratify_or_none(y),
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15 / 0.85, random_state=42,
        stratify=_stratify_or_none(y_temp),
    )
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    model = build_model()
    model.summary()

    # 3. Augmentation (rotation, flip, zoom, brightness) for TRAIN only —
    # val/test stay clean. preprocessing_function applies the same
    # MobileNetV2 normalization training always used, after augmenting
    # in raw pixel space.
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=(0.8, 1.2),
        preprocessing_function=preprocess_input,
    )
    train_gen = train_datagen.flow(
        X_train, y_train, batch_size=BATCH_SIZE, shuffle=True, seed=42
    )
    X_val_p = preprocess_input(X_val.astype("float32"))
    X_test_p = preprocess_input(X_test.astype("float32"))

    model.fit(train_gen,
              validation_data=(X_val_p, y_val),
              epochs=EPOCHS)

    loss, acc = model.evaluate(X_test_p, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f} (loss: {loss:.4f})")

    # === Real evaluation: classification report + confusion matrix ===
    y_pred = (model.predict(X_test_p) > 0.5).astype(int).flatten()
    y_true = y_test.astype(int)

    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=["No Fire", "Fire"]))

    print("\n=== Confusion Matrix ===")
    print("(rows = actual, columns = predicted; order: [No Fire, Fire])")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(cm)
    # ===================================================================

    # === Persist evaluation metrics for the /model-info dashboard endpoint ===
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "test_size": int(len(y_true)),
        "dataset_size": {"fire": n_fire, "no_fire": n_no_fire},
        "warning": warning,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nEvaluation metrics saved to {METRICS_PATH}")

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()