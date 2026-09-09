import ast
from argparse import Namespace
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import lightgbm as lgb
import prototype as p
import gpu_setup as g


def test_cuda_configuration_keeps_bins_and_excludes_cpu_only_flags(monkeypatch):
    captured = {}
    class Model:
        best_iteration_ = 3
        def __init__(self, **kwargs):
            captured.update(kwargs)
        def fit(self, *args, **kwargs):
            return self
    monkeypatch.setattr(lgb, 'LGBMClassifier', Model)
    x = pd.DataFrame({'a': np.arange(20)})
    cfg = dict(max_rounds=4, early_stopping=2, threads=2, lgb_device='cuda', lgb_gpu_id=1)
    params = dict(p.defaults('lgb'), max_bin=511)
    p.fit_model('lgb', params, x, np.tile([0, 1], 10), None, 2026, cfg, 4)
    assert captured['device_type'] == 'cuda' and captured['gpu_device_id'] == 1
    assert captured['max_bin'] == 511 and captured['num_gpu'] == 1
    assert 'deterministic' not in captured and 'force_col_wise' not in captured


def test_cpu_backend_has_identical_predictions_to_measured_version():
    # Actual model comparison against the byte-frozen v3 experiment source.
    import importlib.util
    path = Path(__file__).parent / 'runs/signals_v3/prototype_snapshot.py'
    spec = importlib.util.spec_from_file_location('measured_v3', path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    rng = np.random.default_rng(4)
    x = pd.DataFrame(rng.normal(size=(800, 4)))
    x.columns = ['a', 'b', 'c', 'd']
    x['cat'] = pd.Categorical(rng.integers(0, 3, len(x)))
    y = (x.a + rng.normal(size=len(x)) > 0).to_numpy().astype(int)
    cfg = dict(max_rounds=30, early_stopping=4, threads=2, lgb_device='cpu')
    a, _ = old.fit_model('lgb', old.defaults('lgb'), x, y, None, 2026, cfg, 30)
    b, _ = p.fit_model('lgb', p.defaults('lgb'), x, y, None, 2026, cfg, 30)
    np.testing.assert_array_equal(p.predict(a, x, 'lgb'), p.predict(b, x, 'lgb'))


def test_cpu_preflight_is_real_fit():
    p.check_device(Namespace(lgb_device='cpu', lgb_gpu_id=0, threads=2))


def test_gpu_setup_builds_only_missing_cuda_support(monkeypatch, tmp_path):
    monkeypatch.setattr(g.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(g.shutil, 'which', lambda name, **kwargs: '/usr/bin/' + name)
    monkeypatch.setattr(g.importlib.metadata, 'version', lambda _: '4.7.0')
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if kwargs.get('capture_output') and len(calls) == 2:
            return SimpleNamespace(returncode=1, stdout='', stderr='CUDA Tree Learner was not enabled in this build.')
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    monkeypatch.setattr(g.subprocess, 'run', run)
    g.ensure_lightgbm_backend(tmp_path / 'prototype.py')
    build = next(c for c in calls if 'pip' in c)
    assert '--no-deps' in build and 'lightgbm==4.7.0' in build
    assert '--config-settings=cmake.define.USE_CUDA=ON' in build
    assert calls[-1][calls[-1].index('--lgb-device') + 1] == 'cuda'


@pytest.mark.parametrize('failure', ['out of memory', 'invalid device ordinal'])
def test_gpu_setup_does_not_rebuild_or_fall_back_for_unrelated_failure(monkeypatch, failure, tmp_path):
    monkeypatch.setattr(g.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(g.shutil, 'which', lambda name, **kwargs: '/usr/bin/' + name)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1, stdout='', stderr=failure)
    monkeypatch.setattr(g.subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='no CPU fallback'):
        g.ensure_lightgbm_backend(tmp_path / 'prototype.py')
    assert all('pip' not in c for c in calls)


def test_notebook_embeds_both_sources_and_passes_gpu_options():
    root = Path(__file__).parent
    nb = json.loads((root / 'S6E9_Prototype.ipynb').read_text(encoding='utf8'))
    code = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
    embedded = next(c for c in code if c.startswith('SCRIPT ='))
    literals = [ast.literal_eval(n.value.args[0]) for n in ast.parse(embedded).body
                if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == 'write_text']
    assert literals == [(root / name).read_text(encoding='utf8') for name in ('prototype.py', 'gpu_setup.py')]
    search = next(c for c in code if c.startswith("execute('search'"))
    assert "'--lgb-device', LGB_DEVICE" in search and "'--lgb-gpu-id'" in search


def test_gpu_disabled_reports_accelerator_before_any_install(monkeypatch, tmp_path):
    monkeypatch.setattr(g.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(g.shutil, 'which', lambda *args, **kwargs: None)
    def unexpected(*args, **kwargs):
        raise AssertionError('No GPU: do not start installation or training')
    monkeypatch.setattr(g.subprocess, 'run', unexpected)
    with pytest.raises(RuntimeError, match='Accelerator'):
        g.ensure_lightgbm_backend(tmp_path / 'prototype.py')


def test_working_cuda_build_is_not_reinstalled(monkeypatch, tmp_path):
    monkeypatch.setattr(g.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(g.shutil, 'which', lambda name, **kwargs: '/usr/bin/' + name)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout='backend check passed: cuda', stderr='')
    monkeypatch.setattr(g.subprocess, 'run', run)
    g.ensure_lightgbm_backend(tmp_path / 'prototype.py')
    assert len(calls) == 2 and all('pip' not in c for c in calls)


@pytest.mark.parametrize('repaired', [True, False])
def test_sigsegv_repair_is_bounded_and_preserves_crash_logs(monkeypatch, tmp_path, repaired):
    monkeypatch.setattr(g.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(g.shutil, 'which', lambda name, **kwargs: '/usr/bin/' + name)
    calls = []
    probes = []
    def run(command, **kwargs):
        calls.append(command)
        if kwargs.get('capture_output'):
            probes.append(command)
            code = 0 if repaired and len(probes) > 1 else -11
            return SimpleNamespace(returncode=code, stdout='CHECK_STAGE: fit begin\n', stderr='native stack trace\n')
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    monkeypatch.setattr(g.subprocess, 'run', run)
    script = tmp_path / 'prototype.py'
    if repaired:
        g.ensure_lightgbm_backend(script)
    else:
        with pytest.raises(RuntimeError, match='signal 11'):
            g.ensure_lightgbm_backend(script)
        # A repeated Run All must not spend another build on the same failed repair.
        with pytest.raises(RuntimeError, match='signal 11'):
            g.ensure_lightgbm_backend(script)
    builds = [c for c in calls if 'pip' in c]
    assert len(builds) == 1
    assert 'lightgbm==4.7.0' in builds[0]
    assert '--config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=native' in builds[0]
    assert 'native stack trace' in (tmp_path / 'gpu_preflight_after.log').read_text()
    assert 'faulthandler' in probes[0]
