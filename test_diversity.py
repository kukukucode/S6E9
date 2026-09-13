"""Tests supplied for user-side execution; v5 tests have not been run by Codex."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import diversity as d


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
    cfg = dict(seed=2026, hashes={f.name: d.p.digest(f) for f in data.iterdir()},
        sources={'implementation': 'fixture'}, versions={'fixture': '1'}, max_rounds=3, early_stopping=1)
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
        job = dict(config=dict(cfg, seed=seed), candidate=dict(name=str(i), family=family,
            variant='multiscale_dual', params={}, device='cpu'), data=str(data), run=str(run),
            phase='dev', fold=0, threads=1, output=str(run / f'pred_{i}.npz'))
        path = run / 'job.json'
        d.p.write_json(path, job)
        d.worker(path)
    assert calls == [('multiscale_dual', 2026), ('multiscale_dual', 42)]
    pd.testing.assert_frame_equal(fits[0][0], fits[1][0])
    pd.testing.assert_frame_equal(fits[0][1], fits[1][1])
    path = run / 'pred_0.npz'
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='Corrupt prediction cache'):
        d.checked_npz(path)
