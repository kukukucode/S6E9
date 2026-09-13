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


def detect_gpu_ids(max_gpus=2):
    """Return physical selectors for child CUDA_VISIBLE_DEVICES, preserving restrictions."""
    if max_gpus < 1:
        raise ValueError('max_gpus must be positive')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:
        selectors = []
        for token in visible.split(','):
            token = token.strip()
            if token.isdigit() or token.startswith(('GPU-', 'MIG-')):
                if token in selectors:
                    break
                selectors.append(token)
            else:
                # CUDA stops enumeration at the first invalid token, including -1.
                break
    else:
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                check=True, text=True, capture_output=True)
            selectors = [s.strip() for s in result.stdout.splitlines() if s.strip().startswith('GPU-')]
        except (FileNotFoundError, subprocess.CalledProcessError):
            selectors = []
    if not selectors:
        raise RuntimeError('No visible NVIDIA GPU. Enable Kaggle Settings > Accelerator and check CUDA_VISIBLE_DEVICES.')
    if any(s.startswith('MIG-') for s in selectors):
        return selectors[:1]
    return selectors[:max_gpus]


def require_t4_pair(selectors=None):
    """Require two visible T4 devices and return their physical CUDA selectors."""
    selectors = detect_gpu_ids(2) if selectors is None else list(map(str, selectors))
    if len(selectors) != 2 or len(set(selectors)) != 2:
        raise RuntimeError('This Notebook requires exactly two visible T4 GPUs. Select GPU T4 x2 in Kaggle.')
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name', '--format=csv,noheader'],
            check=True, text=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError('Cannot inspect the required T4 x2 accelerator with nvidia-smi.') from exc
    inventory = [tuple(part.strip() for part in line.split(',', 2))
                 for line in result.stdout.splitlines() if line.strip()]
    names = []
    for selector in selectors:
        matches = [name for index, uuid, name in inventory
                   if selector == index or uuid.startswith(selector) or selector.startswith(uuid)]
        if len(matches) != 1:
            raise RuntimeError(f'Cannot resolve visible GPU selector {selector!r}.')
        names.append(matches[0])
    if any('T4' not in name.upper() for name in names):
        raise RuntimeError(f'This Notebook requires T4 x2; detected {names}.')
    return selectors


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


def ensure_lightgbm_backend(script, device="cuda", gpu_id=0, build_if_missing=True, visible_devices=None):
    if device not in ("cpu", "cuda"):
        raise ValueError("Choose cpu or cuda")
    command = [sys.executable, "-X", "faulthandler", "-u", str(script), "check-device", "--lgb-device", device,
               "--lgb-gpu-id", str(gpu_id), "--threads", "2"]
    folder = Path(script).resolve().parent
    env = os.environ.copy()
    if visible_devices is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(visible_devices)
    env["PYTHONFAULTHANDLER"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    if device == "cuda":
        if platform.system() != "Linux":
            raise RuntimeError("Use LightGBM CUDA in the Kaggle Linux GPU session. On this computer use LGB_DEVICE='cpu'.")
        smi = shutil.which("nvidia-smi")
        if not smi:
            raise RuntimeError("Enable a GPU in Kaggle Settings > Accelerator, then rerun this cell.")
        subprocess.run([smi, "--query-gpu=index,name", "--format=csv,noheader"], check=True, env=env)
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
