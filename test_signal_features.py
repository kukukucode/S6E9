import numpy as np
import pandas as pd
import pytest
import prototype as p


def dataset(n=120):
    rng = np.random.default_rng(3)
    x = pd.DataFrame({'Annual_Income_USD': rng.integers(30000, 170000, n) + .1234,
        'Daily_Commute_km': rng.uniform(0, 100, n), 'City_Type': rng.choice(['A', 'B'], n),
        'Number_of_Cars_Owned': rng.integers(0, 3, n)})
    return x, np.tile([0, 1], n // 2)


@pytest.mark.parametrize('variant', p.SIGNAL_VARIANTS)
def test_numeric_te_excludes_own_label_and_keeps_validation_schema(variant):
    x, y = dataset()
    fe = p.make_features(variant)
    a = fe.fit_transform(x, y)
    te_cols = [c for c in a if '_te' in c]
    assert te_cols and any(c.startswith('k0_te') for c in te_cols)
    for row in (0, 30, 70):
        flipped = y.copy()
        flipped[row] = 1 - flipped[row]
        b = p.make_features(variant).fit_transform(x, flipped)
        np.testing.assert_array_equal(a.loc[row, te_cols], b.loc[row, te_cols])
    unseen = pd.DataFrame({'Annual_Income_USD': [999999.1234, np.nan],
        'Daily_Commute_km': [987.6, np.nan], 'City_Type': ['UNKNOWN', None],
        'Number_of_Cars_Owned': [99, 0]})
    valid = fe.transform(unseen)
    assert list(valid) == list(a)
    assert valid.loc[0, 'k0_freq'] == 0
    np.testing.assert_allclose(valid.filter(regex='^k0_te').iloc[0], y.mean())
    assert np.isfinite(valid[te_cols].to_numpy()).all()
    cfg = dict(max_rounds=8, early_stopping=3, threads=2, xgb_device='cpu')
    model, _ = p.fit_model('lgb', p.defaults('lgb'), a, y, None, 42, cfg, 8)
    assert np.isfinite(p.predict(model, valid, 'lgb')).all()


def test_digit_keys_are_built_before_float32_conversion():
    x, _ = dataset()
    x.loc[0, 'Annual_Income_USD'] = 123456.1234
    _, digits = p.SignalFeatures('digits_te').keys(x)
    expected = np.floor_divide(np.float64(123456.1234), .001) % 10
    assert digits.loc[0, 'd0_1'] == expected


def test_cached_predictions_equal_uncached_and_invalidate_on_input_changes(tmp_path, monkeypatch):
    x, y = dataset()
    candidate = dict(family='lgb', variant='multiscale_dual', params=p.defaults('lgb'))
    cfg = dict(max_rounds=8, early_stopping=3, threads=2, xgb_device='cpu')
    baseline, baseline_rounds = p.prediction_batch(x.iloc[:80], y[:80], x.iloc[80:], candidate, cfg, [2026, 42], y[80:])
    cfg['_cache_dir'] = str(tmp_path)
    original_fit = p.fit_model
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original_fit(*args, **kwargs)
    monkeypatch.setattr(p, 'fit_model', counted)
    # Search seed first; finalist evaluation must add only the second seed.
    p.prediction_batch(x.iloc[:80], y[:80], x.iloc[80:], candidate, cfg, [2026], y[80:])
    cached, cached_rounds = p.prediction_batch(x.iloc[:80], y[:80], x.iloc[80:], candidate, cfg, [2026, 42], y[80:])
    assert len(calls) == 2
    np.testing.assert_array_equal(cached, baseline)
    assert cached_rounds == baseline_rounds
    reordered, _ = p.prediction_batch(x.iloc[:80], y[:80], x.iloc[80:].iloc[::-1], candidate, cfg, [2026], y[80:][::-1])
    assert len(calls) == 3
    changed_y = y[:80].copy()
    changed_y[0] = 1 - changed_y[0]
    p.prediction_batch(x.iloc[:80], changed_y, x.iloc[80:], candidate, cfg, [2026], y[80:])
    assert len(calls) == 4


def test_constant_key_is_selected_only_from_training_rows():
    x, y = dataset()
    x['constant'] = 1
    fe = p.make_features('multiscale_dual')
    a = fe.fit_transform(x, y)
    changed = x.copy()
    changed['constant'] = np.arange(len(x))
    b = fe.transform(changed)
    assert list(a) == list(b)
    assert 'k4' not in fe.keep
