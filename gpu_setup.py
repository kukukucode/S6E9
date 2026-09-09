"""Kaggle/Linux CUDA setup. All probes run in child processes to avoid stale DLLs.

Build flags follow https://github.com/lightgbm-org/LightGBM/blob/main/python-package/README.rst
"""
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def ensure_lightgbm_backend(script, device="cuda", gpu_id=0, build_if_missing=True):
    if device not in ("cpu", "cuda"):
        raise ValueError("Choose cpu or cuda")
    command = [sys.executable, str(script), "check-device", "--lgb-device", device,
               "--lgb-gpu-id", str(gpu_id), "--threads", "2"]
    if device == "cuda":
        if platform.system() != "Linux":
            raise RuntimeError("Use LightGBM CUDA in the Kaggle Linux GPU session. On this computer use LGB_DEVICE='cpu'.")
        smi = shutil.which("nvidia-smi")
        if not smi:
            raise RuntimeError("Enable a GPU in Kaggle Settings > Accelerator, then rerun this cell.")
        subprocess.run([smi, "--query-gpu=index,name", "--format=csv,noheader"], check=True)
    probe = subprocess.run(command, text=True, capture_output=True)
    if probe.returncode == 0:
        print(probe.stdout, end="")
        return
    failure = probe.stdout + probe.stderr
    # Rebuilding only repairs an absent CUDA build, not arbitrary errors or OOM.
    not_compiled = "CUDA Tree Learner was not enabled in this build" in failure
    if device != "cuda" or not build_if_missing or not not_compiled:
        raise RuntimeError("LightGBM backend test failed; no CPU fallback was applied.\n" + failure)
    cuda_bin = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin"
    env = os.environ.copy()
    env["PATH"] = str(cuda_bin) + os.pathsep + env.get("PATH", "")
    if not shutil.which("nvcc", path=env["PATH"]):
        raise RuntimeError("CUDA compiler nvcc is unavailable. Use a Kaggle GPU runtime with CUDA Toolkit.")
    env.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", "2")
    version = importlib.metadata.version("lightgbm")
    build = [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps",
        "--no-cache-dir", "--no-binary=lightgbm", "--config-settings=cmake.define.USE_CUDA=ON",
        f"lightgbm=={version}"]
    print(f"Building LightGBM {version} with CUDA. Enable Kaggle Internet; this first setup can take several minutes.", flush=True)
    subprocess.run(build, check=True, env=env)
    # A fresh interpreter loads the newly installed native library.
    subprocess.run(command, check=True, env=env)
