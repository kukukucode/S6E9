"""Small-data and isolated-process tests; no competition training or GPU required."""
from argparse import Namespace
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import diversity as d
import gpu_setup as g


def test_micro_blend_keeps_baseline_and_convex_probabilities():
    grid = list(d.weight_grid(6))
    np.testing.assert_array_equal(grid[0], [1, 0, 0, 0, 0, 0])
    for weights in grid:
        assert np.isclose(weights.sum(), 1) and (weights >= 0).all()
        assert weights[0] >= .9 - 1e-12
        assert np.count_nonzero(weights[1:]) <= 2


def test_changed_frozen_weights_cannot_be_audited_again(tmp_path):
    d.p.write_json(tmp_path / 'config.json', {'seed': 2026})
    frozen = dict(config_sha256=d.p.digest(tmp_path / 'config.json'), weights=[1.0])
    d.p.write_json(tmp_path / 'frozen.json', frozen)
    d.p.write_json(tmp_path / 'SEALED_OPENED.json', {'frozen_sha256': d.p.digest(tmp_path / 'frozen.json')})
    assert d.frozen_config(tmp_path) == frozen
    d.p.write_json(tmp_path / 'frozen.json', dict(frozen, weights=[.9, .1]))
    with pytest.raises(ValueError, match='after holdout'):
        d.frozen_config(tmp_path)


def test_features_reused_across_model_families_but_not_seeds(tmp_path, monkeypatch):
    data, run = tmp_path / 'data', tmp_path / 'run'
    data.mkdir()
    run.mkdir()
    n = 500
    rng = np.random.default_rng(32)
    frame = pd.DataFrame({'id': np.arange(n), 'Age': rng.integers(18, 70, n),
        'Annual_Income_USD': rng.normal(70000, 10000, n), 'City_Type': rng.choice(['Urban', 'Rural'], n)})
    frame[d.p.TARGET] = np.tile(['No', 'Yes'], n // 2)
    frame.iloc[:400].to_csv(data / 'train.csv', index=False)
    frame.iloc[400:].drop(columns=d.p.TARGET).to_csv(data / 'test.csv', index=False)
    pd.DataFrame({'id': np.arange(400, 500), d.p.TARGET: .5}).to_csv(data / 'sample_submission.csv', index=False)
    binary = run / 'data.joblib'
    loaded = d.p.load_data(data)
    y = loaded[0][d.p.TARGET].to_numpy(dtype=int)
    d.joblib.dump((loaded[0], loaded[1], loaded[3], loaded[4], y, d.p.split_plan(y)), binary, compress=0)
    cfg = dict(seed=2026, hashes={f.name: d.p.digest(f) for f in data.iterdir()},
        sources={'implementation': 'fixture'}, versions={'fixture': '1'}, max_rounds=3,
        early_stopping=1, _data_cache=str(binary))
    calls, fits = [], []
    make = d.p.make_features
    def counted(variant, seed):
        calls.append((variant, seed))
        return make(variant, seed)
    def fit(family, params, x, y, valid, seed, backend, rounds):
        fits.append((x.copy(), valid[0].copy()))
        return None, 3
    monkeypatch.setattr(d.p, 'make_features', counted)
    monkeypatch.setattr(d.p, 'fit_model', fit)
    monkeypatch.setattr(d.p, 'predict', lambda model, x, family: np.full(len(x), .5))
    for i, (family, seed) in enumerate([('lgb', 2026), ('xgb', 2026), ('lgb', 42)]):
        if seed != 2026:
            split = d.p.split_plan(y, seed)
            seed_binary = run / f'data_{seed}.joblib'
            d.joblib.dump((loaded[0], loaded[1], loaded[3], loaded[4], y, split), seed_binary, compress=0)
        else:
            seed_binary = binary
        job = dict(config=dict(cfg, seed=seed, _data_cache=str(seed_binary)), candidate=dict(name=str(i), family=family,
            variant='multiscale_dual', params={}, device='cpu'), run=str(run),
            phase='dev', folds=[0], threads=1, outputs={'0': str(run / f'pred_{i}.npy')})
        path = run / 'job.json'
        d.p.write_json(path, job)
        d.worker(path)
    assert calls == [('multiscale_dual', 2026), ('multiscale_dual', 42)]
    pd.testing.assert_frame_equal(fits[0][0], fits[1][0])
    pd.testing.assert_frame_equal(fits[0][1], fits[1][1])
    path = run / 'pred_0.npy'
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='Corrupt prediction cache'):
        d.checked_prediction(path)


@pytest.mark.parametrize('visible, expected', [
    ('2,1', ['2', '1']), ('0,2,-1,1', ['0', '2']),
    ('GPU-aaa,GPU-bbb', ['GPU-aaa', 'GPU-bbb']),
    ('MIG-GPU-aaa/1/2', ['MIG-GPU-aaa/1/2'])])
def test_gpu_detection_respects_visibility_and_order(monkeypatch, visible, expected):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible)
    def forbidden(*args, **kwargs):
        raise AssertionError('Do not enumerate GPUs outside a visibility restriction')
    monkeypatch.setattr(g.subprocess, 'run', forbidden)
    assert g.detect_gpu_ids() == expected


@pytest.mark.parametrize('visible', ['', '-1'])
def test_disabled_visibility_does_not_enable_a_physical_gpu(monkeypatch, visible):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible)
    with pytest.raises(RuntimeError, match='No visible'):
        g.detect_gpu_ids()


@pytest.mark.parametrize('count', [1, 2, 3])
def test_one_or_two_gpus_are_detected_using_uuids(monkeypatch, count):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.setattr(g.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout='\n'.join(f'GPU-{i}' for i in range(count))))
    assert g.detect_gpu_ids() == [f'GPU-{i}' for i in range(min(count, 2))]


def test_t4_pair_is_required_and_validated(monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    outputs = [
        'GPU-a\nGPU-b\n',
        '0, GPU-a, Tesla T4\n1, GPU-b, Tesla T4\n',
    ]
    monkeypatch.setattr(g.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout=outputs.pop(0)))
    assert g.require_t4_pair() == ['GPU-a', 'GPU-b']
    monkeypatch.setattr(g.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout='0, GPU-a, Tesla T4\n'))
    with pytest.raises(RuntimeError, match='exactly two'):
        g.require_t4_pair(['GPU-a'])
    monkeypatch.setattr(g.subprocess, 'run', lambda *a, **kw: SimpleNamespace(
        stdout='0, GPU-a, Tesla T4\n1, GPU-b, NVIDIA P100\n'))
    with pytest.raises(RuntimeError, match='requires T4 x2'):
        g.require_t4_pair(['GPU-a', 'GPU-b'])


def test_uncompressed_feature_cache_preserves_values_and_categories(tmp_path):
    x = pd.DataFrame({'value': np.array([.1, np.nan, .3], dtype=np.float32),
        'cat': pd.Categorical(['a', 'b', 'a'], categories=['a', 'b', 'unseen'])})
    path = tmp_path / 'cache.joblib'
    d.joblib.dump((x, x), path, compress=0)
    train, valid = d.joblib.load(path)
    pd.testing.assert_frame_equal(train, x)
    pd.testing.assert_frame_equal(valid, x)


def test_two_gpu_folds_are_isolated_and_complete_predictions_are_reused(tmp_path, monkeypatch):
    y = np.tile([0, 1], 50)
    frame = pd.DataFrame({d.p.TARGET: y})
    monkeypatch.setattr(d.p, 'load_data', lambda _: (frame, None, None, None, None))
    _, _, cv, _ = d.p.split_plan(y)
    calls, active = [], []
    class Process:
        def __init__(self, command, stdout, stderr, env):
            self.code = None
            active.append(self)
            assert len([p for p in active if p.code is None]) <= 2
            job = d.p.read_json(command[-1])
            calls.append((job['folds'], env['CUDA_VISIBLE_DEVICES'], job['threads']))
            for fold in job['folds']:
                output = Path(job['outputs'][str(fold)])
                np.save(output, np.full(len(cv[fold][1]), .5), allow_pickle=False)
                d.p.write_json(output.with_suffix('.json'), dict(sha256=d.p.digest(output), rounds=3))
        def wait(self):
            self.code = 0
            return 0
        def poll(self):
            return self.code
        def terminate(self):
            self.code = -15
    monkeypatch.setattr(d.subprocess, 'Popen', Process)
    args = Namespace(run=str(tmp_path), data='unused', gpu_ids=['GPU-a', 'GPU-b'], threads=4, seed=2026,
        _data=(None, None, None, None, None, y, d.p.split_plan(y)))
    candidate = dict(d.anchor(), device='cuda', params=dict(d.anchor()['params'], max_bin=255))
    first = d.folds(args, {}, candidate, 'dev', [0, 1, 2, 3])
    second = d.folds(args, {}, candidate, 'dev', [0, 1, 2, 3])
    assert calls == [([0, 2], 'GPU-a', 2), ([1, 3], 'GPU-b', 2)]
    for fold in range(4):
        np.testing.assert_array_equal(first[fold][0], second[fold][0])
    calls.clear()
    cpu = d.anchor()
    d.folds(args, {}, cpu, 'dev', [0, 1, 2, 3])
    assert calls == [([0, 1, 2, 3], '', 4)]


def test_domain_can_be_primary_but_original_anchor_remains_audit_baseline(tmp_path, monkeypatch):
    y = np.tile([0, 1], 100)
    split = d.p.split_plan(y)
    dev, sealed, _, _ = split
    d.p.write_json(tmp_path / 'config.json', {'fixture': True})
    folder = tmp_path / 'candidates'
    folder.mkdir()
    for name, values in [('main', np.random.default_rng(1).uniform(.1, .9, len(y))),
                         ('domain', .1 + .8 * y)]:
        oof = values.copy()
        oof[sealed] = np.nan
        np.save(folder / f'{name}.npy', oof)
        record = dict(d.anchor(), name=name, variant='multiscale_domain' if name == 'domain' else 'multiscale_dual',
            dev_auc=float(d.roc_auc_score(y[dev], oof[dev])), rounds=[4] * 4, fixed_rounds=4,
            oof_sha256=d.p.digest(folder / f'{name}.npy'))
        d.p.write_json(folder / f'{name}.json', record)
    monkeypatch.setattr(d, 'context', lambda args: (tmp_path, {}, None, None, None, None, None, y, split))
    d.freeze(Namespace(domain_compare=True, auc_window=.0004, max_corr=.999))
    frozen = d.p.read_json(tmp_path / 'frozen.json')
    assert frozen['candidates'][0]['name'] == 'domain' and frozen['weights'] == [1.0]
    assert frozen['baseline']['name'] == 'main'


def test_gpu_preflight_uses_same_isolated_selector_as_worker(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-a,GPU-b')
    monkeypatch.setattr(g.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(g.shutil, 'which', lambda *a, **kw: '/usr/bin/nvidia-smi')
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout='', stderr='')
    monkeypatch.setattr(g.subprocess, 'run', run)
    g.ensure_lightgbm_backend(tmp_path / 'prototype.py', 'cuda', 0, False, visible_devices='GPU-b')
    command, options = calls[-1]
    assert options['env']['CUDA_VISIBLE_DEVICES'] == 'GPU-b'
    assert command[command.index('--lgb-gpu-id') + 1] == '0'
    assert g.os.environ['CUDA_VISIBLE_DEVICES'] == 'GPU-a,GPU-b'
