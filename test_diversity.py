"""Small-data and isolated-process tests; no competition training or GPU required."""
from argparse import Namespace
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import diversity as d
import gpu_setup as g
import s6e9.freeze as freezing
import s6e9.tuning as tuning


def test_micro_blend_keeps_baseline_and_convex_probabilities():
    grid = list(d.weight_grid(6))
    np.testing.assert_array_equal(grid[0], [1, 0, 0, 0, 0, 0])
    for weights in grid:
        assert np.isclose(weights.sum(), 1) and (weights >= 0).all()
        assert weights[0] >= .9 - 1e-12
        assert np.count_nonzero(weights[1:]) <= 2


def test_wide_blend_remains_bounded_and_crossfit_gate_needs_three_wins():
    grid = list(d.weight_grid(5, .30))
    assert any(np.isclose(weights[0], .70) for weights in grid)
    assert all(weights[0] >= .70 - 1e-12 and np.isclose(weights.sum(), 1) for weights in grid)
    base = dict(auc=.8, fold_auc=[.70, .70, .70, .70])
    accepted, wins = d.improvement_gate(base, dict(auc=.801, fold_auc=[.71, .71, .71, .69]))
    assert accepted and wins == 3
    accepted, wins = d.improvement_gate(base, dict(auc=.801, fold_auc=[.71, .71, .69, .69]))
    assert not accepted and wins == 2


def test_fast_auc_and_combined_grid_match_reference_paths():
    rng = np.random.default_rng(19)
    y = np.tile([0, 1], 100)
    values = np.round(rng.uniform(.05, .95, (len(y), 4)), 2)
    rows = np.arange(len(y))
    assert d.binary_auc(y, values[:, 0]) == pytest.approx(d.roc_auc_score(y, values[:, 0]), abs=1e-15)
    narrow, wide = d.best_blends(y, values, rows)
    reference_narrow = freezing.best_blend(y, values, rows, .10)
    reference_wide = freezing.best_blend(y, values, rows, .30)
    assert narrow[0] == pytest.approx(reference_narrow[0], abs=1e-15)
    assert wide[0] == pytest.approx(reference_wide[0], abs=1e-15)
    np.testing.assert_array_equal(narrow[1], reference_narrow[1])
    np.testing.assert_array_equal(wide[1], reference_wide[1])


def test_seed_ensemble_averages_cached_member_predictions(monkeypatch):
    calls = []
    def fake(args, cfg, candidate, phase, requested):
        seed = candidate.get('model_seed', cfg['seed'])
        calls.append(seed)
        return {fold: (np.full(3, seed / 10000), {'rounds': seed % 10 + 1}) for fold in requested}
    monkeypatch.setattr(tuning, 'single_seed_folds', fake)
    candidate = dict(d.domain_anchor(), model_seeds=[2026, 42, 3407])
    result = d.folds(None, {'seed': 2026}, candidate, 'dev', [0])
    assert calls == [2026, 42, 3407]
    np.testing.assert_allclose(result[0][0], np.mean([2026, 42, 3407]) / 10000)
    assert result[0][1]['model_seeds'] == [2026, 42, 3407]


def test_rank_blend_is_fold_local_and_requires_three_fold_wins():
    matrix = np.array([[.1, .9], [.2, .8], [.8, .2], [.9, .1]])
    cv = [(None, np.array([0, 1])), (None, np.array([2, 3]))]
    ranked = d.rank_oof_matrix(matrix, cv)
    np.testing.assert_allclose(ranked, [[.25, .75], [.75, .25], [.25, .75], [.75, .25]])
    allowed, wins = d.rank_blend_allowed(.80, .81, [.70, .71, .72, .73], [.71, .72, .73, .72])
    assert allowed and wins == 3
    allowed, wins = d.rank_blend_allowed(.80, .81, [.70, .71, .72, .73], [.71, .72, .71, .72])
    assert not allowed and wins == 2


def test_lexicographic_rank_changes_only_primary_ties():
    primary = np.array([.1, .2, .2, .2, .3, .4, .4])
    secondary = np.array([.5, .1, .9, .5, .2, .3, .3])
    ranked = d.lexicographic_rank(primary, secondary)
    assert ranked[0] < min(ranked[1:4]) < max(ranked[1:4]) < ranked[4] < ranked[5]
    assert ranked[1] < ranked[3] < ranked[2]
    assert ranked[5] == ranked[6]
    stats = d.fold_tie_statistics(primary, [(None, np.arange(len(primary)))])
    assert stats['tied_rows'] == 5 and stats['tie_groups'] == 2 and stats['largest_tie'] == 3
    global_oof = d.lexicographic_oof(np.array([.1, .4, .2, .3]), np.arange(4),
                                    [(None, np.array([0, 1])), (None, np.array([2, 3]))])
    assert global_oof[0] < global_oof[2] < global_oof[3] < global_oof[1]
    allowed, wins = d.tie_break_allowed(.8, .8000001, [.7, .7, .7, .7],
                                        [.7000001, .7000001, .7000001, .6999999])
    assert allowed and wins == 3


def test_domain_candidate_is_cuda_safe_and_gpu_primary_needs_three_wins():
    candidate = d.domain_anchor()
    assert candidate['device'] == 'cuda'
    assert candidate['params']['max_bin'] <= d.p.LGB_CUDA_BIN_LIMIT
    y = np.tile([0, 1], 8)
    cv = [(None, np.arange(start, start + 4)) for start in range(0, 16, 4)]
    reference = np.array([.4, .6, .7, .3] * 4)
    stronger = reference.copy()
    for fold in range(3):
        iv = cv[fold][1]
        stronger[iv] = np.array([.1, .9, .2, .8])
    gate = d.gpu_primary_gate(y, stronger, reference, np.arange(16), cv)
    assert gate['allowed'] and gate['fold_wins'] == 3
    stronger[cv[2][1]] = reference[cv[2][1]]
    gate = d.gpu_primary_gate(y, stronger, reference, np.arange(16), cv)
    assert not gate['allowed'] and gate['fold_wins'] == 2


def test_fixed_realmlp_candidate_uses_raw_features_and_gpu():
    candidate = d.realmlp_anchor()
    assert candidate == dict(name='realmlp_fixed', lane='realmlp', family='realmlp',
                             variant='raw', params={'n_epochs': 128}, device='cuda')


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


def test_xgb_cuda_worker_validates_serialized_backend(tmp_path, monkeypatch):
    y = np.tile([0, 1], 50)
    train = pd.DataFrame({'feature': np.arange(len(y)), d.p.TARGET: y})
    test = pd.DataFrame({'feature': [101, 102]})
    cache = tmp_path / 'data.joblib'
    d.joblib.dump((train, test, None, ['feature'], y, d.p.split_plan(y)), cache, compress=0)
    config = dict(seed=2026, hashes={'data': 'fixture'}, sources={'code': 'fixture'},
                  versions={'xgboost': 'fixture'}, max_rounds=3, early_stopping=1,
                  _data_cache=str(cache))
    output = tmp_path / 'prediction.npy'
    job = dict(config=config, candidate=dict(name='xgb_gpu', family='xgb', variant='raw',
        params={}, device='cuda'), run=str(tmp_path), phase='dev', folds=[0], threads=1,
        outputs={'0': str(output)})
    job_path = tmp_path / 'job.json'
    d.p.write_json(job_path, job)
    class Features:
        def fit_transform(self, x, labels):
            return x
        def transform(self, x):
            return x
    class Booster:
        def save_config(self):
            return '{"learner":{"generic_param":{"device":"cuda:0"}}}'
    class Model:
        def get_booster(self):
            return Booster()
    backends = []
    monkeypatch.setattr(d.p, 'make_features', lambda variant, seed: Features())
    monkeypatch.setattr(d.p, 'fit_model', lambda family, params, x, labels, valid, seed, backend, rounds:
                        (backends.append(backend) or Model(), 3))
    monkeypatch.setattr(d.p, 'predict', lambda model, x, family: np.full(len(x), .5))
    d.worker(job_path)
    assert backends[0]['xgb_device'] == 'cuda:0'
    prediction, metadata = d.checked_prediction(output)
    assert prediction.shape[0] > 0 and metadata['device'] == 'cuda'


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
    calls, processes = [], []
    def complete(job_path, env):
        job = d.p.read_json(job_path)
        calls.append((job['folds'], env['CUDA_VISIBLE_DEVICES'], job['threads']))
        for fold in job['folds']:
            output = Path(job['outputs'][str(fold)])
            np.save(output, np.full(len(cv[fold][1]), .5), allow_pickle=False)
            d.p.write_json(output.with_suffix('.json'), dict(sha256=d.p.digest(output), rounds=3))
        if 'status' in job:
            d.p.write_json(job['status'], {'ok': True})
    class Process:
        def __init__(self, command, stdout, stderr, env, stdin=None, text=None, bufsize=None):
            self.code = None
            self.command = command
            processes.append(self)
            if 'worker-loop' in command:
                owner = self
                class Input:
                    def write(self, line):
                        complete(line.strip(), env)
                    def flush(self):
                        pass
                    def close(self):
                        owner.code = 0
                self.stdin = Input()
            else:
                self.stdin = None
                complete(command[-1], env)
        def wait(self, timeout=None):
            self.code = 0
            return 0
        def poll(self):
            return self.code
        def terminate(self):
            self.code = -15
    d.shutdown_gpu_workers()
    monkeypatch.setattr(d.subprocess, 'Popen', Process)
    args = Namespace(run=str(tmp_path), data='unused', gpu_ids=['GPU-a', 'GPU-b'], threads=4, seed=2026,
        _data=(None, None, None, None, None, y, d.p.split_plan(y)))
    candidate = dict(d.anchor(), device='cuda', params=dict(d.anchor()['params'], max_bin=255))
    try:
        first = d.folds(args, {}, candidate, 'dev', [0, 1, 2, 3])
        second = d.folds(args, {}, candidate, 'dev', [0, 1, 2, 3])
        other = dict(candidate, name='other', params=dict(candidate['params'], max_bin=127))
        d.folds(args, {}, other, 'dev', [0, 1, 2, 3])
        assert calls == [([0, 2], 'GPU-a', 2), ([1, 3], 'GPU-b', 2),
                         ([0, 2], 'GPU-a', 2), ([1, 3], 'GPU-b', 2)]
        assert len(processes) == 2
        assert all(Path(process.command[4]).name == 'diversity.py' for process in processes)
        for fold in range(4):
            np.testing.assert_array_equal(first[fold][0], second[fold][0])
    finally:
        d.shutdown_gpu_workers()
    calls.clear()
    cpu = d.anchor()
    d.folds(args, {}, cpu, 'dev', [0, 1, 2, 3])
    assert calls == [([0, 1, 2, 3], '', 4)]


def test_screen_evaluation_reuses_predictions_when_promoted(tmp_path, monkeypatch):
    y = np.tile([0, 1], 100)
    split = d.p.split_plan(y)
    dev, sealed, cv, _ = split
    args = Namespace(run=str(tmp_path), _data=(None, None, None, None, None, y, split))
    calls = []
    def fake_folds(args, cfg, candidate, phase, requested):
        calls.append(list(requested))
        return {fold: (.1 + .8 * y[cv[fold][1]], {'rounds': 10 + fold}) for fold in requested}
    monkeypatch.setattr(tuning, 'folds', fake_folds)
    candidate = d.anchor()
    assert d.evaluate(args, {}, candidate, [0, 1], 'screen') == 1
    screen = d.p.read_json(tmp_path / 'candidates' / 'main.json')
    assert screen['stage'] == 'screen' and screen['screen_folds'] == [0, 1]
    assert screen['dev_auc'] is None
    partial = np.load(tmp_path / 'candidates' / 'main.npy')
    assert np.isfinite(partial[np.concatenate([cv[0][1], cv[1][1]])]).all()
    assert np.isnan(partial[np.concatenate([cv[2][1], cv[3][1], sealed])]).all()
    assert d.evaluate(args, {}, candidate, [0, 1, 2, 3], 'full') == 1
    full = d.p.read_json(tmp_path / 'candidates' / 'main.json')
    assert full['stage'] == 'full' and full['screen_auc'] == 1 and full['dev_auc'] == 1
    assert calls == [[0, 1], [0, 1, 2, 3]]


def test_search_screens_every_trial_and_promotes_only_top_three(tmp_path, monkeypatch):
    args = Namespace(run=str(tmp_path), seed=2026, lgb_trials=4, xgb_trials=0,
        screen_folds=2, promote_trials=3, domain_compare=False, cat_compare=False)
    monkeypatch.setattr(tuning, 'context', lambda args: (tmp_path, {}))
    calls = []
    def fake_evaluate(args, cfg, candidate, requested=None, stage='full'):
        requested = [0, 1, 2, 3] if requested is None else requested
        calls.append((candidate['name'], list(requested), stage))
        score = .5 if candidate['name'] == 'main' else .5 + int(candidate['name'].split('_')[1]) / 100
        path = tmp_path / 'candidates' / f"{candidate['name']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        previous = d.p.read_json(path) if path.exists() else {}
        d.p.write_json(path, dict(candidate, stage=stage,
            screen_auc=score if stage == 'screen' else previous.get('screen_auc'),
            screen_fold_auc={'0': score, '1': score} if stage == 'screen' else previous.get('screen_fold_auc'),
            full_fold_auc={str(i): score for i in range(4)} if stage == 'full' else None,
            dev_auc=score if stage == 'full' else None))
        return score
    monkeypatch.setattr(tuning, 'evaluate', fake_evaluate)
    d.search(args)
    assert [name for name, folds, stage in calls if stage == 'screen'] == [f'lgb_{i}' for i in range(4)]
    assert {name for name, folds, stage in calls if stage == 'full' and name.startswith('lgb_')} == {
        'lgb_1', 'lgb_2', 'lgb_3'}
    assert all(folds == [0, 1] for _, folds, stage in calls if stage == 'screen')
    assert d.p.read_json(tmp_path / 'promoted.json')['lanes']['lgb'] == ['lgb_3', 'lgb_2', 'lgb_1']
    diagnostics = d.p.read_json(tmp_path / 'screen_promotion_diagnostics.json')['lanes']['lgb']
    assert diagnostics['promoted_count'] == 3 and diagnostics['rank_correlation'] == pytest.approx(1)


def test_search_evaluates_one_fixed_catboost_candidate(tmp_path, monkeypatch):
    args = Namespace(run=str(tmp_path), seed=2026, lgb_trials=0, xgb_trials=0,
        screen_folds=2, promote_trials=3, domain_compare=False, cat_compare=True)
    monkeypatch.setattr(tuning, 'context', lambda args: (tmp_path, {}))
    calls = []
    def fake_evaluate(args, cfg, candidate, requested=None, stage='full'):
        requested = [0, 1, 2, 3] if requested is None else requested
        calls.append((candidate['name'], list(requested), stage, candidate['device']))
        path = tmp_path / 'candidates' / f"{candidate['name']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        previous = d.p.read_json(path) if path.exists() else {}
        score = .6
        d.p.write_json(path, dict(candidate, stage=stage,
            screen_auc=score if stage == 'screen' else previous.get('screen_auc'),
            screen_fold_auc={'0': score, '1': score} if stage == 'screen' else previous.get('screen_fold_auc'),
            full_fold_auc={str(i): score for i in range(4)} if stage == 'full' else None,
            dev_auc=score if stage == 'full' else None))
        return score
    monkeypatch.setattr(tuning, 'evaluate', fake_evaluate)
    d.search(args)
    assert [call for call in calls if call[0] == 'cat_fixed'] == [
        ('cat_fixed', [0, 1], 'screen', 'cuda'),
        ('cat_fixed', [0, 1, 2, 3], 'full', 'cuda')]
    assert d.p.read_json(tmp_path / 'promoted.json')['lanes']['cat'] == ['cat_fixed']


def test_freeze_ignores_old_full_candidates_outside_promotion_manifest(tmp_path, monkeypatch):
    y = np.tile([0, 1], 100)
    split = d.p.split_plan(y)
    dev, _, _, _ = split
    folder = tmp_path / 'candidates'
    folder.mkdir()
    for name in ('lgb_0', 'lgb_1'):
        d.p.write_json(folder / f'{name}.json', {'stage': 'full'})
    d.p.write_json(tmp_path / 'promoted.json', {'lanes': {'lgb': ['lgb_1']}})
    trials = [SimpleNamespace(number=i, state=d.optuna.trial.TrialState.COMPLETE) for i in range(2)]
    monkeypatch.setattr(d.optuna, 'load_study', lambda **kwargs: SimpleNamespace(trials=trials))
    monkeypatch.setattr(freezing, 'context', lambda args: (tmp_path, {}, None, None, None, None, None, y, split))
    loaded = []
    def fake_load(run, name, labels, development, sealed):
        loaded.append(name)
        prediction = np.random.default_rng(len(loaded)).uniform(.1, .9, len(labels))
        prediction[sealed] = np.nan
        return ({'name': name, 'dev_auc': float(d.roc_auc_score(labels[development], prediction[development]))},
                prediction)
    monkeypatch.setattr(freezing, 'load_candidate', fake_load)
    monkeypatch.setattr(d.p, 'digest', lambda path: 'fixture')
    d.freeze(Namespace(domain_compare=False, auc_window=1, max_corr=1))
    assert loaded == ['main', 'lgb_1']


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
        template = d.domain_anchor() if name == 'domain' else d.anchor()
        record = dict(template,
            dev_auc=float(d.roc_auc_score(y[dev], oof[dev])), rounds=[4] * 4, fixed_rounds=4,
            oof_sha256=d.p.digest(folder / f'{name}.npy'))
        d.p.write_json(folder / f'{name}.json', record)
    monkeypatch.setattr(freezing, 'context', lambda args: (tmp_path, {}, None, None, None, None, None, y, split))
    d.freeze(Namespace(domain_compare=True, auc_window=.0004, max_corr=.999))
    frozen = d.p.read_json(tmp_path / 'frozen.json')
    assert frozen['candidates'][0]['name'] == 'domain' and frozen['weights'] == [1.0]
    assert frozen['baseline']['name'] == 'main'


def test_runtime_profile_sums_worker_metadata(tmp_path):
    folder = tmp_path / 'cache' / 'predictions'
    folder.mkdir(parents=True)
    for index, hit in enumerate((False, True)):
        d.p.write_json(folder / f'{index}.json', dict(candidate='lgb_0', family='lgb', device='cuda',
            phase='dev', fold=index, feature_cache_hit=hit,
            prepare_seconds=1.25, fit_predict_seconds=2.5))
    d.write_runtime_profile(tmp_path)
    group = d.p.read_json(tmp_path / 'runtime_profile.json')['groups']['cuda:lgb:dev']
    assert group == dict(jobs=2, feature_cache_hits=1, prepare_seconds=2.5, fit_predict_seconds=5.0)


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
