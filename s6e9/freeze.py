"""Development-only candidate, seed, blend, and tie-break selection."""
import itertools
from pathlib import Path

import numpy as np
import optuna
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

import prototype as p
from .tuning import context, evaluate


def load_candidate(run, name, y, dev, sealed):
    path = run / 'candidates' / name
    record = p.read_json(path.with_suffix('.json'))
    if p.digest(path.with_suffix('.npy')) != record['oof_sha256']:
        raise ValueError(f'Changed OOF: {name}')
    oof = np.load(path.with_suffix('.npy'), allow_pickle=False)
    if oof.shape != y.shape or not np.isnan(oof[sealed]).all() or not np.isfinite(oof[dev]).all():
        raise ValueError(f'Invalid OOF or holdout contamination: {name}')
    if not np.isclose(roc_auc_score(y[dev], oof[dev]), record['dev_auc'], atol=1e-12, rtol=0):
        raise ValueError(f'OOF score mismatch: {name}')
    return record, oof


def weight_grid(size, max_secondary=.10):
    if not 0 <= max_secondary <= .30:
        raise ValueError('Secondary blend cap must be between 0 and 0.30')
    unit = np.zeros(size)
    unit[0] = 1
    yield unit.copy()
    fractions = [fraction for fraction in (.01, .02, .03, .05, .07, .10, .15, .20, .25, .30)
                 if fraction <= max_secondary + 1e-12]
    for i in range(1, size):
        for fraction in fractions:
            w = unit.copy()
            w[0], w[i] = 1 - fraction, fraction
            yield w
    for i, j in itertools.combinations(range(1, size), 2):
        for a in fractions:
            for b in fractions:
                if a + b > max_secondary + 1e-12:
                    continue
                w = unit.copy()
                w[0], w[i], w[j] = 1 - a - b, a, b
                yield w


def rank_columns(matrix):
    ranked = np.empty(matrix.shape, dtype=float)
    for column in range(matrix.shape[1]):
        ranked[:, column] = (rankdata(matrix[:, column], method='average') - .5) / len(matrix)
    return ranked


def rank_oof_matrix(matrix, cv):
    ranked = np.full(matrix.shape, np.nan, dtype=float)
    for _, iv in cv:
        ranked[iv] = rank_columns(matrix[iv])
    return ranked


def blend_values(matrix, weights, mode):
    if mode not in ('probability', 'rank'):
        raise ValueError(f'Unknown blend mode: {mode}')
    values = rank_columns(matrix) if mode == 'rank' else matrix
    return values @ np.asarray(weights)


def fold_auc_scores(y, prediction, cv):
    return [float(roc_auc_score(y[iv], prediction[iv])) for _, iv in cv]


def best_blend(y, values, rows, max_secondary):
    weights = next(weight_grid(values.shape[1], max_secondary))
    score = float(roc_auc_score(y[rows], values[rows] @ weights))
    for candidate in weight_grid(values.shape[1], max_secondary):
        candidate_score = float(roc_auc_score(y[rows], values[rows] @ candidate))
        if candidate_score > score + 1e-6:
            score, weights = candidate_score, candidate.copy()
    return score, weights


def crossfit_blend(y, values, cv, max_secondary):
    prediction = np.full(len(y), np.nan)
    weights = []
    for fold, (_, iv) in enumerate(cv):
        training = np.concatenate([other_iv for other_fold, (_, other_iv) in enumerate(cv)
                                   if other_fold != fold])
        _, selected = best_blend(y, values, training, max_secondary)
        prediction[iv] = values[iv] @ selected
        weights.append(selected.tolist())
    development = np.concatenate([iv for _, iv in cv])
    return dict(auc=float(roc_auc_score(y[development], prediction[development])),
                fold_auc=fold_auc_scores(y, prediction, cv), fold_weights=weights,
                prediction=prediction)


def improvement_gate(base, candidate):
    wins = sum(right > left + 1e-6 for left, right in zip(base['fold_auc'], candidate['fold_auc']))
    return candidate['auc'] > base['auc'] + 1e-6 and wins >= 3, wins


def rank_blend_allowed(probability_auc, rank_auc, probability_folds, rank_folds):
    wins = sum(rank > probability + 1e-6
               for probability, rank in zip(probability_folds, rank_folds))
    return rank_auc > probability_auc + 1e-6 and wins >= 3, wins


def tie_break_allowed(base_auc, tie_auc, base_folds, tie_folds):
    wins = sum(tie > base for base, tie in zip(base_folds, tie_folds))
    return tie_auc > base_auc and wins >= 3, wins


def lexicographic_rank(primary, secondary):
    primary, secondary = np.asarray(primary), np.asarray(secondary)
    if primary.ndim != 1 or primary.shape != secondary.shape or not len(primary):
        raise ValueError('Tie-breaking inputs must be non-empty aligned vectors')
    if not np.isfinite(primary).all() or not np.isfinite(secondary).all():
        raise ValueError('Tie-breaking inputs must be finite')
    order = np.lexsort((secondary, primary))
    ordered_primary, ordered_secondary = primary[order], secondary[order]
    starts = np.r_[0, 1 + np.flatnonzero(
        (ordered_primary[1:] != ordered_primary[:-1]) |
        (ordered_secondary[1:] != ordered_secondary[:-1]))]
    ends = np.r_[starts[1:], len(primary)]
    average_ranks = (starts + 1 + ends) / 2
    ranked = np.empty(len(primary), dtype=float)
    ranked[order] = np.repeat(average_ranks, ends - starts)
    return (ranked - .5) / len(primary)


def lexicographic_oof(primary, secondary, cv):
    result = np.full(len(primary), np.nan, dtype=float)
    development = np.concatenate([iv for _, iv in cv])
    result[development] = lexicographic_rank(primary[development], secondary[development])
    return result


def fold_tie_statistics(primary, cv):
    development = np.concatenate([iv for _, iv in cv])
    counts = np.unique(primary[development], return_counts=True)[1]
    tied = counts[counts > 1]
    per_fold = []
    for fold, (_, iv) in enumerate(cv):
        counts = np.unique(primary[iv], return_counts=True)[1]
        fold_ties = counts[counts > 1]
        item = dict(fold=fold, rows=len(iv), tied_rows=int(fold_ties.sum()),
            tie_groups=int(len(fold_ties)), largest_tie=int(fold_ties.max()) if len(fold_ties) else 1)
        per_fold.append(item)
    return dict(rows=len(development), tied_rows=int(tied.sum()),
        tied_fraction=float(tied.sum() / len(development)), tie_groups=int(len(tied)),
        largest_tie=int(tied.max()) if len(tied) else 1, per_fold=per_fold)


def gpu_primary_gate(y, candidate, reference, dev, cv):
    candidate_auc = float(roc_auc_score(y[dev], candidate[dev]))
    reference_auc = float(roc_auc_score(y[dev], reference[dev]))
    candidate_folds = fold_auc_scores(y, candidate, cv)
    reference_folds = fold_auc_scores(y, reference, cv)
    wins = sum(left > right + 1e-6 for left, right in zip(candidate_folds, reference_folds))
    return dict(allowed=candidate_auc > reference_auc + 1e-6 and wins >= 3,
        auc=candidate_auc, baseline_auc=reference_auc, fold_wins=wins,
        fold_auc=candidate_folds, baseline_fold_auc=reference_folds)


def freeze(args):
    run, cfg, tr, te, sub, id_col, cols, y, (dev, sealed, cv, outer) = context(args)
    if (run / 'frozen.json').exists():
        print('Already frozen; selection retained.', flush=True)
        return
    if (run / 'SEALED_OPENED.json').exists():
        raise RuntimeError('Holdout already opened; cannot select candidates again.')
    baseline = load_candidate(run, 'main', y, dev, sealed)
    selected = [baseline]
    tie_pool = [baseline]
    if args.domain_compare:
        domain = load_candidate(run, 'domain', y, dev, sealed)
        selected.append(domain)
        tie_pool.append(domain)
    manifest_path = run / 'promoted.json'
    manifest = p.read_json(manifest_path) if manifest_path.exists() else {'lanes': {}}
    # Only the latest explicit promotion set can be selected.
    for lane, limit in [('lgb', 3), ('xgb', 2), ('cat', 1), ('realmlp', 1)]:
        if not any((run / 'candidates').glob(f'{lane}_*.json')):
            continue
        if lane not in manifest.get('lanes', {}):
            raise ValueError(f'Missing promoted.json entry for {lane}; rerun search before freeze.')
        names = manifest['lanes'][lane]
        if lane in ('cat', 'realmlp'):
            complete = set(names)
        else:
            study = optuna.load_study(study_name=lane,
                storage=f"sqlite:///{(run / 'optuna.db').resolve().as_posix()}")
            complete = {f'{lane}_{trial.number}' for trial in study.trials
                        if trial.state == optuna.trial.TrialState.COMPLETE}
        invalid = [name for name in names if name not in complete
                   or not (run / 'candidates' / f'{name}.json').exists()
                   or p.read_json(run / 'candidates' / f'{name}.json').get('stage') != 'full']
        if invalid:
            raise ValueError(f'Invalid promoted candidates for {lane}: {invalid}')
        records = [load_candidate(run, name, y, dev, sealed) for name in names]
        if not records:
            continue
        records.sort(key=lambda pair: pair[0]['dev_auc'], reverse=True)
        tie_pool.extend(records)
        kept = 0
        for record, oof in records:
            if record['dev_auc'] < records[0][0]['dev_auc'] - args.auc_window or kept >= limit:
                continue
            # Check both Pearson and rank correlation; retain weaker-but-different families.
            redundant = any(np.corrcoef(oof[dev], other[dev])[0, 1] > args.max_corr and
                np.corrcoef(rankdata(oof[dev]), rankdata(other[dev]))[0, 1] > args.max_corr
                for _, other in selected)
            gpu_replacement = (record.get('device') == 'cuda' and
                gpu_primary_gate(y, oof, baseline[1], dev, cv)['allowed'])
            if not redundant or gpu_replacement:
                selected.append((record, oof))
                kept += 1
    gates = {record['name']: gpu_primary_gate(y, oof, baseline[1], dev, cv)
             for record, oof in selected if record.get('device') == 'cuda'}
    eligible = [(record, oof) for record, oof in selected
                if gates.get(record['name'], {}).get('allowed')]
    primary = max(eligible, key=lambda pair: pair[0]['dev_auc']) if eligible else baseline
    selected = [primary] + [pair for pair in selected if pair[0]['name'] != primary[0]['name']]
    p.write_json(run / 'gpu_primary_selection.json', dict(cpu_baseline=baseline[0]['name'],
        chosen_primary=primary[0]['name'], required_fold_wins=3, candidates=gates))
    seed_diagnostic = dict(enabled=False, accepted=False, reason='Primary is not a GPU model or one seed was requested.')
    final_seeds = list(getattr(args, 'final_seeds', [cfg.get('seed', 2026)]))
    if primary[0].get('device') == 'cuda' and len(final_seeds) > 1:
        averaged_candidate = dict(primary[0], name=primary[0]['name'] + '_seedavg',
                                  model_seeds=final_seeds)
        evaluate(args, cfg, averaged_candidate, requested=list(range(4)), stage='full')
        averaged = load_candidate(run, averaged_candidate['name'], y, dev, sealed)
        gate = gpu_primary_gate(y, averaged[1], primary[1], dev, cv)
        seed_diagnostic = dict(enabled=True, accepted=gate['allowed'], seeds=final_seeds,
            base_candidate=primary[0]['name'], averaged_candidate=averaged[0]['name'], **gate)
        if gate['allowed']:
            selected[0] = averaged
            primary = averaged
            tie_pool.append(averaged)
    p.write_json(run / 'seed_average_comparison.json', seed_diagnostic)
    matrix = np.column_stack([oof for _, oof in selected])
    rank_matrix = rank_oof_matrix(matrix, cv)
    best = {}
    for mode, values in [('probability', matrix), ('rank', rank_matrix)]:
        narrow_cv = crossfit_blend(y, values, cv, .10)
        wide_cv = crossfit_blend(y, values, cv, .30)
        wide_allowed, wide_wins = improvement_gate(narrow_cv, wide_cv)
        cap = .30 if wide_allowed else .10
        score, weights = best_blend(y, values, dev, cap)
        winner = dict(auc=score, weights=weights,
            fold_auc=fold_auc_scores(y, values @ weights, cv), cap=cap,
            crossfit_auc=(wide_cv if wide_allowed else narrow_cv)['auc'],
            crossfit_fold_auc=(wide_cv if wide_allowed else narrow_cv)['fold_auc'],
            crossfit_fold_weights=(wide_cv if wide_allowed else narrow_cv)['fold_weights'],
            wide_allowed=wide_allowed, wide_fold_wins=wide_wins,
            narrow_crossfit_auc=narrow_cv['auc'], narrow_crossfit_fold_auc=narrow_cv['fold_auc'],
            wide_crossfit_auc=wide_cv['auc'], wide_crossfit_fold_auc=wide_cv['fold_auc'])
        best[mode] = winner
    allowed, rank_wins = rank_blend_allowed(best['probability']['crossfit_auc'],
        best['rank']['crossfit_auc'], best['probability']['crossfit_fold_auc'],
        best['rank']['crossfit_fold_auc'])
    mode = 'rank' if allowed else 'probability'
    winner = best[mode]
    blend_matrix = rank_matrix if mode == 'rank' else matrix
    base_oof = blend_matrix @ winner['weights']
    tie_trials = []
    for record, oof in tie_pool:
        prediction = lexicographic_oof(base_oof, oof, cv)
        score = float(roc_auc_score(y[dev], prediction[dev]))
        scores = fold_auc_scores(y, prediction, cv)
        accepted, wins = tie_break_allowed(winner['auc'], score, winner['fold_auc'], scores)
        tie_trials.append(dict(candidate=record, auc=score, fold_auc=scores,
                               fold_wins=wins, accepted=accepted))
    accepted = [trial for trial in tie_trials if trial['accepted']]
    tie_winner = max(accepted, key=lambda trial: trial['auc']) if accepted else None
    tie_breaker = None if tie_winner is None else dict(candidate=tie_winner['candidate'],
        auc=tie_winner['auc'], fold_auc=tie_winner['fold_auc'], fold_wins=tie_winner['fold_wins'])
    p.write_json(run / 'tie_break_comparison.json', dict(
        base_auc=winner['auc'], base_fold_auc=winner['fold_auc'],
        tie_statistics=fold_tie_statistics(base_oof, cv), required_fold_wins=3,
        candidates=[dict(name=trial['candidate']['name'], auc=trial['auc'],
            fold_auc=trial['fold_auc'], fold_wins=trial['fold_wins'], accepted=trial['accepted'])
            for trial in tie_trials],
        chosen=None if tie_winner is None else tie_winner['candidate']['name']))
    p.write_json(run / 'blend_comparison.json', dict(
        candidates=[c['name'] for c, _ in selected],
        probability=dict(auc=best['probability']['auc'], weights=best['probability']['weights'].tolist(),
                         fold_auc=best['probability']['fold_auc'], cap=best['probability']['cap'],
                         crossfit_auc=best['probability']['crossfit_auc'],
                         crossfit_fold_auc=best['probability']['crossfit_fold_auc'],
                         crossfit_fold_weights=best['probability']['crossfit_fold_weights'],
                         wide_allowed=best['probability']['wide_allowed'],
                         wide_fold_wins=best['probability']['wide_fold_wins'],
                         narrow_crossfit_auc=best['probability']['narrow_crossfit_auc'],
                         wide_crossfit_auc=best['probability']['wide_crossfit_auc']),
        rank=dict(auc=best['rank']['auc'], weights=best['rank']['weights'].tolist(),
                  fold_auc=best['rank']['fold_auc'], cap=best['rank']['cap'],
                  crossfit_auc=best['rank']['crossfit_auc'],
                  crossfit_fold_auc=best['rank']['crossfit_fold_auc'],
                  crossfit_fold_weights=best['rank']['crossfit_fold_weights'],
                  wide_allowed=best['rank']['wide_allowed'],
                  wide_fold_wins=best['rank']['wide_fold_wins'],
                  narrow_crossfit_auc=best['rank']['narrow_crossfit_auc'],
                  wide_crossfit_auc=best['rank']['wide_crossfit_auc']),
        rank_fold_wins=rank_wins, rank_required_fold_wins=3, chosen_mode=mode))
    p.write_json(run / 'candidate_correlation.json', dict(names=[c['name'] for c, _ in selected],
        pearson=np.nan_to_num(np.atleast_2d(np.corrcoef(matrix[dev], rowvar=False)), nan=1).tolist()))
    chosen = [(c, float(w)) for (c, _), w in zip(selected, winner['weights']) if w > 0]
    final_auc = tie_winner['auc'] if tie_winner is not None else winner['auc']
    final_fold_auc = tie_winner['fold_auc'] if tie_winner is not None else winner['fold_auc']
    p.write_json(run / 'frozen.json', dict(candidates=[c for c, _ in chosen], weights=[w for _, w in chosen],
        mode=mode, tie_breaker=tie_breaker, development_selection_auc=final_auc,
        development_fold_auc=final_fold_auc, rank_fold_wins=rank_wins,
        primary=primary[0]['name'], baseline=baseline[0],
        config_sha256=p.digest(run / 'config.json'),
        note='Weights selected only on development OOF. Selection-biased; no Public LB tuning.'))
    tie_name = None if tie_winner is None else tie_winner['candidate']['name']
    print(f'Frozen {mode} blend; tie_breaker={tie_name}; dev AUC={final_auc:.7f}; '
          f'weights={[w for _, w in chosen]}', flush=True)


def frozen_config(run):
    frozen = p.read_json(run / 'frozen.json')
    if frozen['config_sha256'] != p.digest(run / 'config.json'):
        raise ValueError('Frozen configuration changed')
    mark = run / 'SEALED_OPENED.json'
    if mark.exists() and p.read_json(mark)['frozen_sha256'] != p.digest(run / 'frozen.json'):
        raise ValueError('Frozen selection changed after holdout was opened')
    return frozen
