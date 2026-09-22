"""CPU-only cross-fitted ensemble over completed S6E9 Notebook outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


TARGET = "Will_Buy_EV"
VARIANTS = (
    ("chosen", "prediction", "submission.csv", None),
    ("probability", "probability_prediction", "submission_probability.csv", "probability"),
    ("rank", "rank_prediction", "submission_rank.csv", "rank"),
)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def binary_auc(y, prediction):
    y = np.asarray(y)
    prediction = np.asarray(prediction, dtype=float)
    if y.ndim != 1 or prediction.shape != y.shape or not np.isfinite(prediction).all():
        raise ValueError("AUC inputs must be aligned finite vectors")
    positives = int(np.count_nonzero(y == 1))
    negatives = int(np.count_nonzero(y == 0))
    if positives + negatives != len(y) or not positives or not negatives:
        raise ValueError("AUC labels must contain both binary classes")
    order = np.argsort(prediction, kind="quicksort")
    ordered_prediction, ordered_y = prediction[order], y[order]
    starts = np.r_[0, 1 + np.flatnonzero(ordered_prediction[1:] != ordered_prediction[:-1])]
    ends = np.r_[starts[1:], len(y)]
    counts = np.add.reduceat(ordered_y, starts)
    rank_sum = np.sum(counts * ((starts + 1 + ends) * .5))
    return float((rank_sum - positives * (positives + 1) * .5) / (positives * negatives))


def _validated_frame(path, columns):
    frame = pd.read_csv(path)
    missing = set(columns) - set(frame)
    if missing or frame[list(columns)].isna().any().any():
        raise ValueError(f"Invalid {path}: missing={sorted(missing)}")
    return frame


def discover_runs(input_root):
    root = Path(input_root)
    runs = []
    for oof_path in sorted(root.rglob("final_oof.csv")) if root.is_dir() else []:
        folder = oof_path.parent
        required = [folder / name for name in ("config.json", "final_report.json", "submission.csv")]
        if all(path.is_file() for path in required):
            runs.append(folder)
    if len(runs) < 2:
        raise FileNotFoundError(
            "Add at least two completed 03_finalize/monolithic Notebook Outputs as Input; "
            f"found {len(runs)} valid run folder(s).")
    return runs


def load_candidates(input_root):
    runs = discover_runs(input_root)
    base_ids = base_test_ids = test_output_ids = labels = folds = data_hashes = id_col = None
    candidates, seen = [], set()
    for folder in runs:
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        report = json.loads((folder / "final_report.json").read_text(encoding="utf-8"))
        hashes = config.get("hashes")
        if not isinstance(hashes, dict):
            raise ValueError(f"Missing data hashes: {folder}")
        submission = pd.read_csv(folder / "submission.csv")
        ids = [name for name in submission if name != TARGET]
        if len(ids) != 1:
            raise ValueError(f"Expected one submission ID column: {folder}")
        current_id = ids[0]
        oof = _validated_frame(folder / "final_oof.csv", [current_id, TARGET, "fold", "prediction"])
        if oof[current_id].duplicated().any() or submission[current_id].duplicated().any():
            raise ValueError(f"Duplicate IDs: {folder}")
        oof.index = oof[current_id].astype(str)
        submission.index = submission[current_id].astype(str)
        if base_ids is None:
            id_col = current_id
            base_ids = oof.index.to_numpy(copy=True)
            base_test_ids = submission.index.to_numpy(copy=True)
            test_output_ids = submission[current_id].to_numpy(copy=True)
            labels = oof.loc[base_ids, TARGET].to_numpy(dtype=int)
            folds = oof.loc[base_ids, "fold"].to_numpy(dtype=int)
            data_hashes = hashes
            if set(np.unique(labels)) != {0, 1} or set(np.unique(folds)) != set(range(5)):
                raise ValueError("Expected binary labels and exactly five final OOF folds")
        else:
            if current_id != id_col or hashes != data_hashes:
                raise ValueError(f"Data schema/hash mismatch: {folder}")
            if set(oof.index) != set(base_ids) or set(submission.index) != set(base_test_ids):
                raise ValueError(f"Train/test ID mismatch: {folder}")
            if not np.array_equal(oof.loc[base_ids, TARGET].to_numpy(dtype=int), labels):
                raise ValueError(f"Target mismatch: {folder}")
            if not np.array_equal(oof.loc[base_ids, "fold"].to_numpy(dtype=int), folds):
                raise ValueError(f"Fold mismatch: {folder}")
        source = folder.relative_to(Path(input_root)).as_posix()
        for variant, oof_column, submission_name, report_key in VARIANTS:
            submission_path = folder / submission_name
            if oof_column not in oof or not submission_path.is_file():
                continue
            variant_submission = pd.read_csv(submission_path)
            if set(variant_submission) != {id_col, TARGET} or variant_submission[id_col].duplicated().any():
                raise ValueError(f"Invalid submission schema: {submission_path}")
            variant_submission.index = variant_submission[id_col].astype(str)
            if set(variant_submission.index) != set(base_test_ids):
                raise ValueError(f"Submission ID mismatch: {submission_path}")
            expected_hash = (report.get("submission_sha256") if report_key is None else
                report.get("cpu_submission_variants", {}).get(report_key, {}).get("submission_sha256"))
            if expected_hash is not None and digest(submission_path) != expected_hash:
                raise ValueError(f"Submission hash mismatch: {submission_path}")
            train_prediction = oof.loc[base_ids, oof_column].to_numpy(dtype=float)
            test_prediction = variant_submission.loc[base_test_ids, TARGET].to_numpy(dtype=float)
            if (not np.isfinite(train_prediction).all() or not np.isfinite(test_prediction).all()
                    or ((train_prediction < 0) | (train_prediction > 1)).any()
                    or ((test_prediction < 0) | (test_prediction > 1)).any()):
                raise ValueError(f"Invalid predictions: {folder}/{variant}")
            identity = hashlib.sha256(train_prediction.tobytes() + test_prediction.tobytes()).hexdigest()
            if identity in seen:
                continue
            seen.add(identity)
            candidates.append(dict(name=f"{source}:{variant}", oof=train_prediction,
                                   test=test_prediction, source=source, variant=variant))
    if len(candidates) < 2:
        raise ValueError("Fewer than two unique aligned prediction candidates were found")
    return dict(id_col=id_col, train_ids=base_ids, test_ids=test_output_ids,
                labels=labels, folds=folds, data_hashes=data_hashes, candidates=candidates)


def rank_matrix_by_fold(matrix, folds):
    result = np.empty_like(matrix, dtype=float)
    for fold in sorted(np.unique(folds)):
        rows = folds == fold
        for column in range(matrix.shape[1]):
            result[rows, column] = (rankdata(matrix[rows, column], method="average") - .5) / rows.sum()
    return result


def rank_matrix(matrix):
    result = np.empty_like(matrix, dtype=float)
    for column in range(matrix.shape[1]):
        result[:, column] = (rankdata(matrix[:, column], method="average") - .5) / len(matrix)
    return result


def best_single(y, values, rows):
    scores = [binary_auc(y[rows], values[rows, column]) for column in range(values.shape[1])]
    index = int(np.argmax(scores))
    return float(scores[index]), index


def best_pair(y, values, rows, shortlist=6, weights=(.25, .5, .75)):
    single_scores = np.array([binary_auc(y[rows], values[rows, column])
                              for column in range(values.shape[1])])
    order = np.argsort(-single_scores, kind="stable")[:min(shortlist, values.shape[1])]
    best = dict(score=float(single_scores[order[0]]), indices=[int(order[0])], weights=[1.])
    for position, left in enumerate(order):
        for right in order[position + 1:]:
            for weight in weights:
                prediction = weight * values[rows, left] + (1 - weight) * values[rows, right]
                score = binary_auc(y[rows], prediction)
                if score > best["score"] + 1e-12:
                    best = dict(score=score, indices=[int(left), int(right)],
                                weights=[float(weight), float(1 - weight)])
    return best


def refine_pair(y, values, rows, choice):
    if len(choice["indices"]) != 2:
        return choice
    left, right = choice["indices"]
    best = dict(choice)
    for step in range(1, 20):
        weight = step / 20
        prediction = weight * values[rows, left] + (1 - weight) * values[rows, right]
        score = binary_auc(y[rows], prediction)
        if score > best["score"] + 1e-12:
            best = dict(score=score, indices=[left, right], weights=[weight, 1 - weight])
    return best


def crossfit_mode(y, values, folds, min_fold_wins=4):
    baseline_prediction = np.empty(len(y), dtype=float)
    blend_prediction = np.empty(len(y), dtype=float)
    selections = []
    for fold in sorted(np.unique(folds)):
        training, validation = folds != fold, folds == fold
        _, baseline = best_single(y, values, training)
        blend = best_pair(y, values, training)
        baseline_prediction[validation] = values[validation, baseline]
        blend_prediction[validation] = sum(weight * values[validation, index]
            for index, weight in zip(blend["indices"], blend["weights"]))
        selections.append(dict(fold=int(fold), baseline=int(baseline),
                               indices=blend["indices"], weights=blend["weights"]))
    baseline_auc = binary_auc(y, baseline_prediction)
    blend_auc = binary_auc(y, blend_prediction)
    baseline_folds, blend_folds = [], []
    for fold in sorted(np.unique(folds)):
        rows = folds == fold
        baseline_folds.append(binary_auc(y[rows], baseline_prediction[rows]))
        blend_folds.append(binary_auc(y[rows], blend_prediction[rows]))
    wins = sum(right > left + 1e-6 for left, right in zip(baseline_folds, blend_folds))
    return dict(allowed=blend_auc > baseline_auc + 1e-6 and wins >= min_fold_wins,
        baseline_auc=baseline_auc, blend_auc=blend_auc, delta=blend_auc - baseline_auc,
        fold_wins=wins, required_fold_wins=min_fold_wins,
        baseline_fold_auc=baseline_folds, blend_fold_auc=blend_folds, selections=selections)


def build_ensemble(bundle, output_dir, min_fold_wins=4):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    y, folds = bundle["labels"], bundle["folds"]
    raw_oof = np.column_stack([candidate["oof"] for candidate in bundle["candidates"]])
    raw_test = np.column_stack([candidate["test"] for candidate in bundle["candidates"]])
    modes = {
        "probability": (raw_oof, raw_test),
        "rank": (rank_matrix_by_fold(raw_oof, folds), rank_matrix(raw_test)),
    }
    diagnostics = {mode: crossfit_mode(y, values[0], folds, min_fold_wins)
                   for mode, values in modes.items()}
    names = [candidate["name"] for candidate in bundle["candidates"]]
    for result in diagnostics.values():
        for selection in result["selections"]:
            selection["baseline_candidate"] = names[selection["baseline"]]
            selection["candidates"] = [names[index] for index in selection["indices"]]
    allowed = [mode for mode, result in diagnostics.items() if result["allowed"]]
    chosen_mode = max(allowed, key=lambda mode: diagnostics[mode]["blend_auc"]) if allowed else None
    report = dict(accepted=False, chosen_mode=chosen_mode,
        candidates=[dict(name=name, auc=binary_auc(y, raw_oof[:, index]),
            fold_auc=[binary_auc(y[folds == fold], raw_oof[folds == fold, index])
                      for fold in sorted(np.unique(folds))])
            for index, name in enumerate(names)],
        data_hashes=bundle["data_hashes"], crossfit=diagnostics,
        note="OOF model/blend selection remains selection-biased; Public LB was not used.")
    submission_path = output / "submission_ensemble.csv"
    submission_path.unlink(missing_ok=True)
    if chosen_mode is not None:
        oof_values, test_values = modes[chosen_mode]
        rows = np.ones(len(y), dtype=bool)
        _, global_baseline = best_single(y, oof_values, rows)
        blend = refine_pair(y, oof_values, rows, best_pair(y, oof_values, rows))
        global_baseline_auc = binary_auc(y, oof_values[:, global_baseline])
        accepted = (len(blend["indices"]) == 2
                    and blend["score"] > global_baseline_auc + 1e-6)
        report.update(accepted=accepted, global_baseline=dict(
            candidate=bundle["candidates"][global_baseline]["name"], auc=global_baseline_auc),
            final_blend=dict(auc=blend["score"], indices=blend["indices"], weights=blend["weights"],
                candidates=[bundle["candidates"][index]["name"] for index in blend["indices"]]))
        if accepted:
            prediction = sum(weight * test_values[:, index]
                for index, weight in zip(blend["indices"], blend["weights"]))
            pd.DataFrame({bundle["id_col"]: bundle["test_ids"], TARGET: prediction}).to_csv(
                submission_path, index=False)
            report["submission_sha256"] = digest(submission_path)
    write_json(output / "ensemble_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--output", default="/kaggle/working/s6e9_ensemble_v1")
    parser.add_argument("--min-fold-wins", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.min_fold_wins <= 5:
        parser.error("min-fold-wins must be 1..5")
    bundle = load_candidates(args.input_root)
    report = build_ensemble(bundle, args.output, args.min_fold_wins)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["accepted"]:
        print("No ensemble submission was created because the cross-fit/full-OOF gates did not pass.")


if __name__ == "__main__":
    main()
