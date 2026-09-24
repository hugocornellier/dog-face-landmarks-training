"""Time one .tflite on one Python LiteRT runtime, in a fresh process; print one JSON line.

The Python-side counterpart of bench_litert_macos.py: ai-edge-litert (LiteRT's own pip
package, with its Interpreter and the CompiledModel API) and TensorFlow's bundled
tf.lite.Interpreter. Like every Python number in this repo, it ranks runtimes and exports
against each other; it is not what the Flutter packages see. The 2026-09-24 entry under
"READ THIS BEFORE EXPORTING ANY MODEL" in LANDMARK_DETECTION_REPORT.md came from this
script, run over a matrix by bench_litert_python_matrix.py.

Every runtime is timed on the same unit of work: write the input, run, read the output.
Runtimes:
  tf_interp            TensorFlow's bundled tf.lite.Interpreter (XNNPACK by default)
  lrt_interp           ai_edge_litert Interpreter (XNNPACK by default)
  lrt_interp_builtin   ai_edge_litert Interpreter with default delegates off (built-in kernels)
  cm_cpu               CompiledModel, CPU accelerator (XNNPACK kernel mode)
  cm_cpu_builtin       CompiledModel, CPU accelerator, built-in kernel mode
  cm_gpu               CompiledModel, GPU with CPU fallback (Metal on macOS)

CompiledModel always gets an explicit thread count: left at its default (num_threads=0),
ai-edge-litert 2.2.0 runs the CPU single-threaded.
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--runtime", required=True)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--save-output", default="")
args = ap.parse_args()

# Keep native runtime chatter out of the result line.
devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(devnull, 2)

result = {"model": args.model, "runtime": args.runtime, "threads": args.threads}


def interp_runner(interp):
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    x = np.random.default_rng(0).random(inp["shape"], dtype=np.float32)

    def step():
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        return interp.get_tensor(out["index"]).reshape(-1)

    return step


def cm_runner(model):
    from ai_edge_litert.interpreter import Interpreter

    probe = Interpreter(model_path=args.model)
    shape = probe.get_input_details()[0]["shape"]
    n_out = int(np.prod(probe.get_output_details()[0]["shape"]))
    del probe
    ins, outs = model.create_input_buffers(0), model.create_output_buffers(0)
    x = np.random.default_rng(0).random(shape, dtype=np.float32)
    result["fully_accelerated"] = bool(model.is_fully_accelerated())

    def step():
        ins[0].write(x)
        model.run_by_index(0, ins, outs)
        return outs[0].read(n_out, np.float32)

    return step


try:
    rt = args.runtime
    if rt == "tf_interp":
        import tensorflow as tf
        step = interp_runner(tf.lite.Interpreter(model_path=args.model, num_threads=args.threads))
    elif rt in ("lrt_interp", "lrt_interp_builtin"):
        from ai_edge_litert.interpreter import Interpreter, OpResolverType
        kw = {"experimental_op_resolver_type": OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES} if rt.endswith("builtin") else {}
        step = interp_runner(Interpreter(model_path=args.model, num_threads=args.threads, **kw))
    elif rt.startswith("cm_"):
        from ai_edge_litert.compiled_model import CompiledModel
        from ai_edge_litert.cpu_kernel_mode import CpuKernelMode
        from ai_edge_litert.cpu_options import CpuOptions
        from ai_edge_litert.hardware_accelerator import HardwareAccelerator
        from ai_edge_litert.options import Options
        if rt == "cm_gpu":
            opts = Options(HardwareAccelerator.GPU | HardwareAccelerator.CPU,
                           cpu_options=CpuOptions(num_threads=args.threads))
        else:
            mode = CpuKernelMode.BUILTIN if rt == "cm_cpu_builtin" else CpuKernelMode.XNNPACK
            opts = Options(HardwareAccelerator.CPU, cpu_options=CpuOptions(num_threads=args.threads, kernel_mode=mode))
        step = cm_runner(CompiledModel.from_file(args.model, options=opts))
    else:
        raise SystemExit(f"unknown runtime {rt}")

    first = step()
    for _ in range(args.warmup):
        step()
    times = []
    for _ in range(args.iters):
        t = time.perf_counter()
        step()
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    result.update(status="ok", median_ms=round(statistics.median(times), 2),
                  p10_ms=round(times[len(times) // 10], 2), p90_ms=round(times[(len(times) * 9) // 10 - 1], 2))
    if args.save_output:
        np.save(args.save_output, first)
except Exception as e:  # e.g. CompiledModel on a dynamic-shape graph
    result.update(status="FAIL", error=f"{type(e).__name__}: {str(e)[:160]}")

print(json.dumps(result), flush=True)
