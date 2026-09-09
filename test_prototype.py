import numpy as np
import pandas as pd
import pytest
from prototype import Features, split_plan, fit_model, predict, defaults, blend_weights


def test_sealed_is_absent_from_all_development_training_and_validation():
    y = np.tile([0, 1], 100)
    dev, sealed, cv, outer = split_plan(y)
    assert len(dev) == 160 and len(sealed) == 40
    seen = []
    for tr, va in cv:
        assert not set(tr) & set(sealed)
        assert not set(va) & set(sealed)
        assert not set(tr) & set(va)
        seen.extend(va)
    assert sorted(seen) == list(dev)


def test_target_encoding_excludes_own_label_even_for_unknown_prior():
    x = pd.DataFrame({"category": [f"unique_{i}" for i in range(40)], "value": np.arange(40)})
    y = np.tile([0, 1], 20)
    original = Features("target").fit_transform(x, y)
    for row in (0, 11, 30):
        flipped = y.copy()
        flipped[row] = 1 - flipped[row]
        changed = Features("target").fit_transform(x, flipped)
        assert original.loc[row, "f0_te"] == changed.loc[row, "f0_te"]


def test_validation_cannot_change_category_or_frequency_maps():
    x = pd.DataFrame({"category": ["a", "b", "a", None], "value": [1, 2, 3, 4]})
    fe = Features("frequency")
    xt = fe.fit_transform(x, [0, 1, 0, 1])
    unseen = fe.transform(pd.DataFrame({"category": ["new"], "value": [1000]}))
    assert unseen.loc[0, "f0"] == 0
    assert unseen.loc[0, "f0_freq"] == 0
    assert xt.loc[0, "f0_freq"] == 0.5
    assert "new" not in fe.maps["f0"]


@pytest.mark.parametrize("family", ["xgb", "lgb", "cat"])
@pytest.mark.parametrize("variant", ["raw", "frequency", "interaction", "artifact", "target"])
def test_model_paths_handle_missing_and_unseen_categories(family, variant):
    rng = np.random.default_rng(42)
    x = pd.DataFrame({"City": rng.choice(["a", "b", None], 100), "Income": rng.normal(size=100)})
    y = (x.Income.to_numpy() + rng.normal(size=100) > 0).astype(int)
    x.loc[0, "Income"] = np.nan
    fe = Features(variant)
    xt = fe.fit_transform(x.iloc[:80], y[:80])
    xv_raw = x.iloc[80:].copy()
    xv_raw.loc[80, "City"] = "never_seen"
    xv = fe.transform(xv_raw)
    cfg = dict(max_rounds=10, early_stopping=3, threads=2, xgb_device="cpu")
    model, rounds = fit_model(family, defaults(family), xt, y[:80], (xv, y[80:]), 42, cfg)
    p = predict(model, xv, family)
    assert len(p) == 20 and np.isfinite(p).all() and 1 <= rounds <= 10
    fixed, _ = fit_model(family, defaults(family), xt, y[:80], None, 42, cfg, rounds)
    assert len(predict(fixed, xv, family)) == 20


def test_blending_rejects_useless_candidate_and_stays_convex():
    y = np.array([0, 1, 0, 1])
    matrix = np.array([[.1, .9], [.9, .1], [.2, .8], [.8, .2]])
    weights, auc = blend_weights(y, matrix)
    assert np.all(weights >= 0) and np.isclose(weights.sum(), 1)
    assert weights[0] == 1 and auc == 1
