import json
import sys

import numpy as np
import pandas as pd

import prototype as p


def test_encoding_trials_then_three_seed_finalists_and_submission(tmp_path, monkeypatch):
    data, run = tmp_path / 'data', tmp_path / 'run'
    data.mkdir()
    rng = np.random.default_rng(42)
    n = 800
    frame = pd.DataFrame({'id': np.arange(n), 'Age': rng.integers(18, 75, n),
        'Annual_Income_USD': rng.normal(70000, 10000, n),
        'Daily_Commute_km': rng.uniform(5, 80, n),
        'City_Type': rng.choice(['Urban', 'Rural', 'Suburban'], n)})
    frame[p.TARGET] = np.where(frame.Age + rng.normal(0, 12, n) < 42, 'Yes', 'No')
    frame.iloc[:600].to_csv(data / 'train.csv', index=False)
    frame.iloc[600:].drop(columns=p.TARGET).to_csv(data / 'test.csv', index=False)
    pd.DataFrame({'id': np.arange(799, 599, -1), p.TARGET: .5}).to_csv(data / 'sample_submission.csv', index=False)
    common = ['--data', str(data), '--run', str(run), '--profile', 'legacy', '--families', 'lgb', '--trials', '3',
        '--max-rounds', '8', '--early-stopping', '3', '--threads', '2',
        '--seeds', '2026', '42', '3407', '--holdout-already-reviewed']
    calls = []
    original_cv = p.run_cv

    def observed_cv(x, y, cv, candidate, cfg, seeds):
        calls.append((candidate['variant'], list(seeds)))
        return original_cv(x, y, cv, candidate, cfg, seeds)

    monkeypatch.setattr(p, 'run_cv', observed_cv)

    def execute(command):
        monkeypatch.setattr(sys, 'argv', ['prototype.py', command, *common])
        p.main()

    execute('search')
    assert calls == [('raw', [2026]), ('frequency', [2026]), ('target', [2026])]
    execute('search')
    assert len(calls) == 3  # Same total trial budget resumes without repeating training.
    y = frame.iloc[:600][p.TARGET].eq('Yes').astype(int).to_numpy()
    dev, sealed, _, _ = p.split_plan(y)
    for trial in range(3):
        oof = np.load(run / f'lgb_{trial}_oof.npy')
        assert np.isnan(oof[sealed]).all() and np.isfinite(oof[dev]).all()
    execute('freeze')
    assert len(calls) == 5  # Best family candidate plus the fixed baseline.
    assert all(seeds == [2026, 42, 3407] for _, seeds in calls[3:])
    frozen = json.loads((run / 'frozen.json').read_text())
    assert 'not a fresh independent evaluation' in frozen['warning']
    execute('audit')
    report = json.loads((run / 'sealed_report.json').read_text())
    assert report['holdout_previously_reviewed'] is True
    assert report['public_lb'] is None
    execute('finalize')
    submission = pd.read_csv(run / 'submission.csv')
    assert submission.id.tolist() == list(range(799, 599, -1))
    assert submission[p.TARGET].between(0, 1).all()
    before = (run / 'submission.csv').read_bytes()
    def unexpected_fit(*args, **kwargs):
        raise AssertionError('Finalization must reuse completed fits')
    monkeypatch.setattr(p, 'fit_model', unexpected_fit)
    execute('finalize')
    assert (run / 'submission.csv').read_bytes() == before
    assert {f.name for f in run.glob('*.csv')} == {'submission.csv', 'final_oof.csv'}
