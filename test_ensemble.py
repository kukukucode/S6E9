import json

import numpy as np
import pandas as pd
import pytest

import ensemble as e


def write_run(root, name, ids, test_ids, y, folds, oof, test, hashes=None, shuffle=False):
    folder = root / name / "s6e9_run"
    folder.mkdir(parents=True)
    order = np.random.default_rng(len(name)).permutation(len(ids)) if shuffle else np.arange(len(ids))
    frame = pd.DataFrame({"id": ids[order], e.TARGET: y[order], "fold": folds[order],
        "prediction": oof[order], "probability_prediction": oof[order],
        "rank_prediction": oof[order]})
    frame.to_csv(folder / "final_oof.csv", index=False)
    submission = pd.DataFrame({"id": test_ids, e.TARGET: test})
    for filename in ("submission.csv", "submission_probability.csv", "submission_rank.csv"):
        submission.to_csv(folder / filename, index=False)
    report = {"submission_sha256": e.digest(folder / "submission.csv"),
        "cpu_submission_variants": {
            mode: {"submission_sha256": e.digest(folder / f"submission_{mode}.csv")}
            for mode in ("probability", "rank")}}
    (folder / "final_report.json").write_text(json.dumps(report))
    (folder / "config.json").write_text(json.dumps({"hashes": hashes or {"train.csv": "same"}}))
    return folder


@pytest.fixture
def aligned_outputs(tmp_path):
    rng = np.random.default_rng(8)
    n, m = 2500, 300
    folds = np.arange(n) % 5
    latent = rng.normal(size=n)
    y = (latent + rng.normal(scale=.7, size=n) > 0).astype(int)
    def sigmoid(value):
        return 1 / (1 + np.exp(-value))
    first = sigmoid(latent + rng.normal(scale=1.2, size=n))
    second = sigmoid(latent + rng.normal(scale=1.2, size=n))
    test_latent = rng.normal(size=m)
    test_first = sigmoid(test_latent + rng.normal(scale=1.2, size=m))
    test_second = sigmoid(test_latent + rng.normal(scale=1.2, size=m))
    ids, test_ids = np.arange(10000, 10000 + n), np.arange(90000, 90000 + m)
    write_run(tmp_path, "run-a", ids, test_ids, y, folds, first, test_first)
    write_run(tmp_path, "run-b", ids, test_ids[::-1], y, folds, second,
              test_second[::-1], shuffle=True)
    return tmp_path


def test_load_candidates_aligns_ids_and_removes_duplicate_variants(aligned_outputs):
    bundle = e.load_candidates(aligned_outputs)
    assert len(bundle["candidates"]) == 2
    assert [candidate["variant"] for candidate in bundle["candidates"]] == ["chosen", "chosen"]
    np.testing.assert_array_equal(bundle["test_ids"], np.arange(90000, 90300))
    assert set(bundle["folds"]) == set(range(5))


def test_cpu_ensemble_crossfit_gate_creates_submission_for_stable_complementarity(
        aligned_outputs, tmp_path):
    bundle = e.load_candidates(aligned_outputs)
    output = tmp_path / "output"
    report = e.build_ensemble(bundle, output)
    assert report["accepted"] is True
    assert report["crossfit"][report["chosen_mode"]]["fold_wins"] >= 4
    assert len(report["final_blend"]["candidates"]) == 2
    submission = pd.read_csv(output / "submission_ensemble.csv")
    assert submission[e.TARGET].between(0, 1).all()
    assert submission.id.tolist() == list(range(90000, 90300))
    assert report["submission_sha256"] == e.digest(output / "submission_ensemble.csv")


def test_no_submission_is_written_when_crossfit_does_not_improve(tmp_path):
    rng = np.random.default_rng(4)
    n, m = 1000, 50
    y = np.arange(n) % 2
    strong = np.clip(.1 + .8 * y + rng.normal(0, .02, n), 0, 1)
    weak = rng.uniform(0, 1, n)
    bundle = dict(id_col="id", test_ids=np.arange(m), labels=y, folds=np.arange(n) % 5,
        data_hashes={"train.csv": "same"}, candidates=[
            dict(name="strong", oof=strong, test=rng.uniform(0, 1, m)),
            dict(name="weak", oof=weak, test=rng.uniform(0, 1, m))])
    report = e.build_ensemble(bundle, tmp_path / "output")
    assert report["accepted"] is False
    assert not (tmp_path / "output" / "submission_ensemble.csv").exists()


def test_mismatched_data_hashes_stop_before_blending(tmp_path):
    y = np.arange(100) % 2
    folds = np.arange(100) % 5
    ids, test_ids = np.arange(100), np.arange(10)
    prediction, test = np.linspace(.1, .9, 100), np.linspace(.2, .8, 10)
    write_run(tmp_path, "first", ids, test_ids, y, folds, prediction, test,
              hashes={"train.csv": "first"})
    write_run(tmp_path, "second", ids, test_ids, y, folds, prediction[::-1], test[::-1],
              hashes={"train.csv": "second"})
    with pytest.raises(ValueError, match="hash mismatch"):
        e.load_candidates(tmp_path)


def test_requires_two_completed_outputs(tmp_path):
    with pytest.raises(FileNotFoundError, match="at least two"):
        e.discover_runs(tmp_path)
