"""Kaggle/Linux CUDA setup. All probes run in child processes to avoid stale DLLs.

Build flags follow https://github.com/lightgbm-org/LightGBM/blob/main/python-package/README.rst
"""
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

CUDA_LIGHTGBM_VERSION = "4.7.0"


def _probe(command, env, log_path):
    result = subprocess.run(command, text=True, capture_output=True, env=env)
    output = (result.stdout or "") + (result.stderr or "")
    Path(log_path).write_text(f"returncode={result.returncode}\n{output}", encoding="utf-8")
    if output:
        print(output, end="", flush=True)
    return result, output


def _failed(result, log_path):
    signal = f"native crash (signal {-result.returncode})" if result.returncode < 0 else f"exit {result.returncode}"
    raise RuntimeError(f"LightGBM backend test failed: {signal}; no CPU fallback was applied. "
        f"Full diagnostic log: {log_path}. Share this log before trying another rebuild.")


def ensure_lightgbm_backend(script, device="cuda", gpu_id=0, build_if_missing=True):
    if device not in ("cpu", "cuda"):
        raise ValueError("Choose cpu or cuda")
    command = [sys.executable, "-X", "faulthandler", "-u", str(script), "check-device", "--lgb-device", device,
               "--lgb-gpu-id", str(gpu_id), "--threads", "2"]
    folder = Path(script).resolve().parent
    env = os.environ.copy()
    env["PYTHONFAULTHANDLER"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    if device == "cuda":
        if platform.system() != "Linux":
            raise RuntimeError("Use LightGBM CUDA in the Kaggle Linux GPU session. On this computer use LGB_DEVICE='cpu'.")
        smi = shutil.which("nvidia-smi")
        if not smi:
            raise RuntimeError("Enable a GPU in Kaggle Settings > Accelerator, then rerun this cell.")
        subprocess.run([smi, "--query-gpu=index,name", "--format=csv,noheader"], check=True)
    log_path = folder / "gpu_preflight_before.log"
    probe, failure = _probe(command, env, log_path)
    if probe.returncode == 0:
        return
    # One bounded repair for absent CUDA support or a native crash; keep other errors visible.
    not_compiled = "CUDA Tree Learner was not enabled in this build" in failure
    native_crash = probe.returncode in (-11, -6)
    if device != "cuda" or not build_if_missing or not (not_compiled or native_crash):
        _failed(probe, log_path)
    attempt = folder / "gpu_cuda_repair.json"
    repair = dict(version=CUDA_LIGHTGBM_VERSION, architecture="native", gpu_id=gpu_id)
    if attempt.exists() and json.loads(attempt.read_text(encoding="utf-8")) == repair:
        _failed(probe, log_path)
    cuda_bin = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin"
    env["PATH"] = str(cuda_bin) + os.pathsep + env.get("PATH", "")
    if not shutil.which("nvcc", path=env["PATH"]):
        raise RuntimeError("CUDA compiler nvcc is unavailable. Use a Kaggle GPU runtime with CUDA Toolkit.")
    env.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", "2")
    version = CUDA_LIGHTGBM_VERSION
    build = [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps",
        "--no-cache-dir", "--no-binary=lightgbm", "--config-settings=cmake.define.USE_CUDA=ON",
        "--config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=native",
        f"lightgbm=={version}"]
    print(f"Building LightGBM {version} with CUDA. Enable Kaggle Internet; this first setup can take several minutes.", flush=True)
    subprocess.run(build, check=True, env=env)
    attempt.write_text(json.dumps(repair), encoding="utf-8")
    # A fresh interpreter loads the newly installed native library.
    log_path = folder / "gpu_preflight_after.log"
    probe, _ = _probe(command, env, log_path)
    if probe.returncode:
        _failed(probe, log_path)
