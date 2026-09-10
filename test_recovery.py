from argparse import Namespace
import sys
import numpy as np
import pandas as pd
import optuna
import pytest
import prototype as p


@pytest.fixture
def stopped_run(tmp_path, monkeypatch):
    data, source = tmp_path / 'data', tmp_path / 'old'
    data.mkdir()
    rng = np.random.default_rng(21)
    frame = pd.DataFrame({'id': np.arange(700), 'Age': rng.integers(18, 75, 700),
        'Annual_Income_USD': rng.normal(70000, 10000, 700),
        'Daily_Commute_km': rng.uniform(5, 80, 700)})
    frame[p.TARGET] = np.where(frame.Age + rng.normal(0, 12, 700) < 42, 'Yes', 'No')
    frame.iloc[:600].to_csv(data / 'train.csv', index=False)
    frame.iloc[600:].drop(columns=p.TARGET).to_csv(data / 'test.csv', index=False)
    pd.DataFrame({'id': np.arange(600, 700), p.TARGET: .5}).to_csv(data / 'sample_submission.csv', index=False)
    common = ['--data', str(data), '--profile', 'signals', '--families', 'lgb',
        '--max-rounds', '8', '--early-stopping', '3', '--threads', '2', '--lgb-device', 'cuda']
    # Synthetic OOF isolates database crash/recovery behavior from GPU availability.
    def synthetic_cv(x, y, cv, candidate, cfg, seeds):
        result = np.full(len(y), np.nan)
        scores = []
        for _, iv in cv:
            result[iv] = 1 / (1 + np.exp((x.Age.iloc[iv].to_numpy() - 42) / 12))
            scores.append(float(p.roc_auc_score(y[iv], result[iv])))
        return result, [4] * len(cv), scores
    with monkeypatch.context() as m:
        m.setattr(p, 'run_cv', synthetic_cv)
        m.setattr(sys, 'argv', ['prototype.py', 'search', '--run', str(source), '--trials', '5', *common])
        p.main()
    study = optuna.load_study(study_name='lgb', storage=f"sqlite:///{(source / 'optuna.db').as_posix()}")
    interrupted = study.ask()
    interrupted.suggest_categorical('variant', ['raw', 'frequency', 'interaction', 'artifact', 'target', *p.SIGNAL_VARIANTS])
    params = p.suggest(interrupted, 'lgb')
    assert params['max_bin'] == 511
    cfg = p.read_json(source / 'config.json')
    cfg['code_sha256'] = next(iter(p.RECOVERABLE_CUDA_CODE_HASHES))
    cfg.pop('lgb_cuda_bin_limit')
    p.write_json(source / 'config.json', cfg)
    return data, source, common


def test_recovery_retains_five_trials_and_only_trains_interrupted_candidate(stopped_run, tmp_path, monkeypatch):
    data, source, common = stopped_run
    before = {f.name: p.digest(f) for f in source.iterdir() if f.is_file()}
    destination = tmp_path / 'new'
    args = Namespace(data=str(data), run=str(destination), source_run=str(source))
    p.recover_run(args)
    p.recover_run(args)  # Idempotent recovery must not overwrite later work.
    assert {f.name: p.digest(f) for f in source.iterdir() if f.is_file()} == before
    manifest = p.read_json(destination / 'recovery.json')
    assert manifest['copied_trials'] == list(range(5)) and manifest['retried_trials'] == [5]
    for number in range(5):
        assert p.digest(destination / f'lgb_{number}_oof.npy') == p.digest(source / f'lgb_{number}_oof.npy')
    calls = []
    actual_cv = p.run_cv
    def observed_cv(x, y, cv, candidate, cfg, seeds):
        assert candidate['params']['max_bin'] == 511
        calls.append(candidate['trial'])
        return actual_cv(x, y, cv, candidate, cfg, seeds)
    monkeypatch.setattr(p, 'run_cv', observed_cv)
    monkeypatch.setattr(sys, 'argv', ['prototype.py', 'search', '--run', str(destination), '--trials', '6', *common])
    p.main()
    p.main()
    assert calls == [6]
    record = p.read_json(destination / 'lgb_6.json')
    assert record['actual_device'] == 'cpu' and record['params']['max_bin'] == 511
    assert {f.name: p.digest(f) for f in source.iterdir() if f.is_file()} == before


@pytest.mark.parametrize('damage', ['source_code', 'oof', 'frozen'])
def test_recovery_rejects_unverifiable_or_frozen_results(stopped_run, tmp_path, damage):
    data, source, _ = stopped_run
    if damage == 'source_code':
        cfg = p.read_json(source / 'config.json')
        p.write_json(source / 'config.json', dict(cfg, code_sha256='unknown'))
    elif damage == 'oof':
        np.save(source / 'lgb_0_oof.npy', np.zeros(600))
    else:
        p.write_json(source / 'frozen.json', {})
    with pytest.raises(ValueError):
        p.recover_run(Namespace(data=str(data), run=str(tmp_path / 'new'), source_run=str(source)))


def test_single_model_blend_keeps_predictions_and_avoids_repeated_auc(monkeypatch):
    original = p.roc_auc_score
    calls = []
    def counted(y, pred):
        calls.append(1)
        return original(y, pred)
    monkeypatch.setattr(p, 'roc_auc_score', counted)
    y, matrix = np.array([0, 1, 0, 1]), np.array([[.2], [.3], [.3], [.7]])
    weights, auc = p.blend_weights(y, matrix)
    np.testing.assert_array_equal(matrix @ weights, matrix[:, 0])
    assert len(calls) == 1 and auc == original(y, matrix[:, 0])
