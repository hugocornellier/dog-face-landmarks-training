"""Re-export a trained landmark model with a fully static inference graph.

The shipped float16 file carries 7 PACK / 5 SHAPE / 5 STRIDED_SLICE ops whose
only job is to compute the output shapes of the 4 TRANSPOSE_CONV layers at run
time. Keras emits those because the functional model keeps a dynamic batch
dimension, and Conv2DTranspose then builds its output shape from tf.shape().
The consequence is a dynamic-sized tensor in the graph, which makes TFLite log

  Attempting to use a delegate that only supports static-sized tensors with a
  graph that has dynamic-sized tensors (tensor#421 ...)

and drop part of the graph from the XNNPACK delegate.

Converting from a concrete function whose input signature pins the batch to 1
lets the converter constant-fold all of that away. Weights are untouched, so
this is a graph-shape change only -- the script verifies that by comparing
against the original file over the whole val set.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

REPO = Path(__file__).resolve().parent.parent


def load_model(keras_path: Path) -> tf.keras.Model:
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    import train_dog_face_landmarks as T  # registers SoftArgmax2D / WarmupSchedule
    _ = T
    return tf.keras.models.load_model(keras_path, compile=False)


def export_static_fp16(model: tf.keras.Model, out_path: Path, img_size: int) -> None:
    @tf.function(input_signature=[
        tf.TensorSpec([1, img_size, img_size, 3], tf.float32, name="crop")
    ])
    def serve(x):
        return {"landmarks_xy": model(x, training=False)}

    concrete = serve.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    blob = converter.convert()
    out_path.write_bytes(blob)
    print(f"Saved {out_path} ({len(blob)/1024/1024:.2f} MB)")


def export_static_int8(model: tf.keras.Model, out_path: Path, img_size: int) -> None:
    """Dynamic-range int8: int8 weights, float32 activations, static shapes.

    Same concrete-function path as the fp16 export so the two are comparable on
    latency rather than differing by graph shape as well as weight precision.
    """
    @tf.function(input_signature=[
        tf.TensorSpec([1, img_size, img_size, 3], tf.float32, name="crop")
    ])
    def serve(x):
        return {"landmarks_xy": model(x, training=False)}

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [serve.get_concrete_function()], model
    )
    # DEFAULT with no supported_types override means dynamic-range int8 weights.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    blob = converter.convert()
    out_path.write_bytes(blob)
    print(f"Saved {out_path} ({len(blob)/1024/1024:.2f} MB)")


def op_histogram(path: Path) -> dict[str, int]:
    import collections
    from tensorflow.lite.python import schema_py_generated as schema
    buf = open(path, "rb").read()
    m = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(bytearray(buf), 0))
    names = {v: k for k, v in schema.BuiltinOperator.__dict__.items()
             if isinstance(v, int)}
    codes = [oc.builtinCode if oc.builtinCode else 0 for oc in m.operatorCodes]
    cnt = collections.Counter()
    for op in m.subgraphs[0].operators:
        cnt[names.get(codes[op.opcodeIndex], "?")] += 1
    return dict(cnt)


def transpose_conv_versions(path: Path) -> list[int]:
    """Opcode versions of every TRANSPOSE_CONV in a .tflite file."""
    from tensorflow.lite.python import schema_py_generated as schema

    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(path.read_bytes(), 0))
    return [oc.version for oc in model.operatorCodes
            if (oc.builtinCode if oc.builtinCode else oc.deprecatedBuiltinCode) == 67]


def ensure_gpu_compatible(path: Path) -> None:
    """Leave every TRANSPOSE_CONV at version 3 or lower, or fail.

    The converter fuses the ReLU after each Conv2DTranspose into the op, which makes it
    version 4. LiteRT's GPU accelerator accepts at most version 3, so it leaves the whole
    deconv head on the CPU. The static conversion above does not change this: every
    landmark export so far came out at version 4, MobileNetV3-Large and EfficientNetV2-S,
    cat and dog. The shipped landmark files are version 3 only because
    unfuse_transpose_conv_relu.py was run on them as a separate step, which is easy to
    miss, so it runs here on every export. A graph with nothing above version 3 is left
    byte-for-byte unchanged. See "READ THIS BEFORE EXPORTING ANY MODEL" in
    LANDMARK_DETECTION_REPORT.md.
    """
    if max(transpose_conv_versions(path), default=0) > 3:
        from unfuse_transpose_conv_relu import unfuse

        unfuse(path, path)
    versions = transpose_conv_versions(path)
    if max(versions, default=0) > 3:
        raise SystemExit(f"TRANSPOSE_CONV is still version {max(versions)} in {path}")
    print("TRANSPOSE_CONV versions:", versions)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keras", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--int8", action="store_true",
                    help="dynamic-range int8 weights instead of float16")
    args = ap.parse_args()

    model = load_model(args.keras)
    if args.int8:
        export_static_int8(model, args.out, args.img_size)
    else:
        export_static_fp16(model, args.out, args.img_size)
    ensure_gpu_compatible(args.out)

    hist = op_histogram(args.out)
    print("ops:", sum(hist.values()))
    for k in sorted(hist, key=lambda k: -hist[k]):
        print(f"  {k:22s} {hist[k]}")


if __name__ == "__main__":
    main()
