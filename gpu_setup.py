"""Kaggle/Linux CUDA setup. All probes run in child processes to avoid stale DLLs.

Build flags follow https://github.com/lightgbm-org/LightGBM/blob/main/python-package/README.rst
"""
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

CUDA_LIGHTGBM_VERSION = "4.7.0"
WHEEL_FOLDER = "lightgbm_cuda_wheels"


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


def _digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def _wheel_marker(wheel):
    return dict(version=CUDA_LIGHTGBM_VERSION, architecture='native',
        machine=platform.machine(), python=[sys.version_info.major, sys.version_info.minor],
        wheel=Path(wheel).name, sha256=_digest(wheel))


def _cached_cuda_wheels(folder):
    markers = [folder / WHEEL_FOLDER / 'lightgbm_cuda_wheel.json']
    kaggle_inputs = Path('/kaggle/input')
    if kaggle_inputs.exists():
        markers.extend(kaggle_inputs.glob(f'*/{WHEEL_FOLDER}/lightgbm_cuda_wheel.json'))
    expected = dict(version=CUDA_LIGHTGBM_VERSION, architecture='native',
        machine=platform.machine(), python=[sys.version_info.major, sys.version_info.minor])
    for marker_path in markers:
        if not marker_path.exists():
            continue
        try:
            marker = json.loads(marker_path.read_text(encoding='utf-8'))
            wheel = marker_path.parent / marker['wheel']
        except (KeyError, ValueError, OSError):
            continue
        if all(marker.get(name) == value for name, value in expected.items()) \
                and wheel.is_file() and marker.get('sha256') == _digest(wheel):
            yield wheel


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
    for wheel in _cached_cuda_wheels(folder):
        print(f'Reusing cached CUDA LightGBM wheel: {wheel}', flush=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--force-reinstall', '--no-deps', str(wheel)],
            check=True, env=env)
        cached_log = folder / 'gpu_preflight_cached.log'
        probe, failure = _probe(command, env, cached_log)
        log_path = cached_log
        if probe.returncode == 0:
            return
        not_compiled = "CUDA Tree Learner was not enabled in this build" in failure
        native_crash = probe.returncode in (-11, -6)
        if not (not_compiled or native_crash):
            _failed(probe, cached_log)
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
    wheel_folder = folder / WHEEL_FOLDER
    wheel_folder.mkdir(parents=True, exist_ok=True)
    build = [sys.executable, "-m", "pip", "wheel", "--no-deps",
        "--no-cache-dir", "--no-binary=lightgbm", "--config-settings=cmake.define.USE_CUDA=ON",
        "--config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=native",
        "--wheel-dir", str(wheel_folder), f"lightgbm=={version}"]
    print(f"Building LightGBM {version} with CUDA. Enable Kaggle Internet; this first setup can take several minutes.", flush=True)
    subprocess.run(build, check=True, env=env)
    wheels = sorted(wheel_folder.glob(f'lightgbm-{version}-*.whl'), key=lambda path: path.stat().st_mtime)
    if not wheels:
        raise RuntimeError(f'CUDA LightGBM wheel was not created in {wheel_folder}.')
    wheel = wheels[-1]
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--force-reinstall', '--no-deps', str(wheel)],
        check=True, env=env)
    (wheel_folder / 'lightgbm_cuda_wheel.json').write_text(
        json.dumps(_wheel_marker(wheel), sort_keys=True), encoding='utf-8')
    attempt.write_text(json.dumps(repair), encoding="utf-8")
    # A fresh interpreter loads the newly installed native library.
    log_path = folder / "gpu_preflight_after.log"
    probe, _ = _probe(command, env, log_path)
    if probe.returncode:
        _failed(probe, log_path)
