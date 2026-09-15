"""Convert fire_model.h5 to TFLite (float32, no quantization) for fast CPU serving.

Produces three files in backend/:
  fire_model.tflite        full model:      image -> P(fire)      (for /predict)
  fire_model_feat.tflite   feature part:    image -> last-conv map (for Grad-CAM)
  fire_model_head.tflite   head part:       last-conv map -> P(fire) (for Grad-CAM)

Float32 (no quantization) keeps outputs numerically identical to the .h5 model.
Run:  venv\\Scripts\\python.exe ml-model/convert_to_tflite.py
"""
import os

import tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "fire_model.h5")
OUT_DIR = os.path.join(HERE, "..", "backend")
OUT_DIR = os.path.abspath(OUT_DIR)


def convert(model, name):
    # from_keras_model hits an MLIR 'missing attribute value' error on this
    # Keras-3 model; round-trip through SavedModel first (same graph, converts).
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        model.export(tmp) if hasattr(model, "export") else model.save(tmp, save_format="tf")
        converter = tf.lite.TFLiteConverter.from_saved_model(tmp)
        converter.optimizations = []  # float32: exact, no quantization drift
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        tflite_bytes = converter.convert()
    path = os.path.join(OUT_DIR, name)
    with open(path, "wb") as f:
        f.write(tflite_bytes)
    print(f"wrote {path} ({len(tflite_bytes) / 1e6:.1f} MB)")


def find_last_conv(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    for layer in reversed(model.layers):
        sub = getattr(layer, "layers", None)
        if sub:
            for s in reversed(sub):
                if isinstance(s, tf.keras.layers.Conv2D):
                    return s.name
    raise RuntimeError("no Conv2D layer found")


def main():
    print(f"loading {SRC} ...")
    model = tf.keras.models.load_model(SRC, compile=False)
    print(f"loaded: {model.count_params():,} params")

    convert(model, "fire_model.tflite")

    conv_name = find_last_conv(model)
    print(f"last conv layer: {conv_name}")
    conv_layer = model.get_layer(conv_name)

    feat = tf.keras.models.Model(inputs=model.inputs, outputs=conv_layer.output)
    convert(feat, "fire_model_feat.tflite")

    head_in = tf.keras.layers.Input(shape=conv_layer.output.shape[1:])
    x = head_in
    started = False
    for layer in model.layers:
        if not started:
            if layer is conv_layer or layer.name == conv_name:
                started = True
            continue
        x = layer(x)
    head = tf.keras.models.Model(inputs=head_in, outputs=x)
    convert(head, "fire_model_head.tflite")
    print("done.")


if __name__ == "__main__":
    main()
