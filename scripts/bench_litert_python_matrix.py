"""Run bench_litert_python.py over a model x runtime x thread-count matrix; write one JSON file.

Each configuration runs in its own process, one at a time, so thread pools and caches never
overlap. Every result row carries the model file's sha256, so a number can always be tied to
the exact bytes it measured.

--models is a JSON object mapping a row name to {"path", "family", "species", "export"}.
For the 2026-09-24 run in LANDMARK_DETECTION_REPORT.md those were, per species: the
EfficientNetV2-S file published on Hugging Face until that day (dynamic batch), its
reexport_static.py export, the dynamic MobileNetV3-Large landmark model the packages shipped
before 2026-08-12 (dog_detection 464e188, cat_detection 90ccec8), and the static one they
ship now.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CPU_RUNTIMES = ["tf_interp", "lrt_interp", "lrt_interp_builtin", "cm_cpu", "cm_cpu_builtin"]
THREADS = [4, 16]
GPU_THREADS = 4  # CPU threads for the ops the GPU leaves behind


def sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, required=True, help="JSON: name -> {path, family, species, export}")
    ap.add_argument("--out", type=Path, required=True, help="results JSON, rewritten after every run")
    ap.add_argument("--save-outputs", type=Path, help="directory for each run's first output (.npy)")
    args = ap.parse_args()

    models = json.loads(args.models.read_text())
    hashes = {name: sha256(m["path"]) for name, m in models.items()}
    configs = [(name, rt, th) for name in models for rt in CPU_RUNTIMES for th in THREADS]
    configs += [(name, "cm_gpu", GPU_THREADS) for name in models]
    if args.save_outputs:
        args.save_outputs.mkdir(parents=True, exist_ok=True)

    results = []
    for i, (name, rt, th) in enumerate(configs, 1):
        m = models[name]
        cmd = [sys.executable, str(HERE / "bench_litert_python.py"), "--model", m["path"],
               "--runtime", rt, "--threads", str(th)]
        if args.save_outputs:
            cmd += ["--save-output", str(args.save_outputs / f"{name}__{rt}__{th}.npy")]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        line = (proc.stdout.strip().splitlines() or ["{}"])[-1]
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            r = {"status": "CRASH", "error": (proc.stdout + proc.stderr)[-200:]}
        r.update(name=name, family=m["family"], species=m["species"], export=m["export"],
                 runtime=rt, threads=th, sha256=hashes[name])
        results.append(r)
        print(f"[{i:3}/{len(configs)}] {name:28} {rt:19} t={th:<2} "
              + (f"{r['median_ms']:8.2f} ms  (p10 {r['p10_ms']}, p90 {r['p90_ms']})" if r.get("status") == "ok"
                 else f"{r.get('status')}: {r.get('error', '')[:90]}"), flush=True)
        args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
