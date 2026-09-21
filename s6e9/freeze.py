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
    return [binary_auc(y[iv], prediction[iv]) for _, iv in cv]


def binary_auc(y, prediction):
    """Exact binary ROC-AUC with much less per-call validation overhead."""
    y = np.asarray(y)
    prediction = np.asarray(prediction, dtype=float)
    if y.ndim != 1 or prediction.shape != y.shape or not np.isfinite(prediction).all():
        raise ValueError('AUC inputs must be aligned finite vectors')
    positives = int(np.count_nonzero(y == 1))
    negatives = int(np.count_nonzero(y == 0))
    if positives + negatives != len(y) or not positives or not negatives:
        raise ValueError('AUC labels must contain both binary classes')
    order = np.argsort(prediction, kind='quicksort')
    ordered_prediction = prediction[order]
    ordered_y = y[order]
    starts = np.r_[0, 1 + np.flatnonzero(ordered_prediction[1:] != ordered_prediction[:-1])]
    ends = np.r_[starts[1:], len(y)]
    positive_counts = np.add.reduceat(ordered_y, starts)
    positive_rank_sum = np.sum(positive_counts * ((starts + 1 + ends) * .5))
    return float((positive_rank_sum - positives * (positives + 1) * .5) /
                 (positives * negatives))


def best_blend(y, values, rows, max_secondary):
    weights = next(weight_grid(values.shape[1], max_secondary))
    score = binary_auc(y[rows], values[rows] @ weights)
    for candidate in weight_grid(values.shape[1], max_secondary):
        candidate_score = binary_auc(y[rows], values[rows] @ candidate)
        if candidate_score > score + 1e-6:
            score, weights = candidate_score, candidate.copy()
    score, weights = refine_weights(y, values, rows, weights, max_secondary)
    return score, weights


def refine_weights(y, values, rows, weights, max_secondary, step=.01, max_steps=8):
    """Hill-climb on a 1% simplex lattice without an exponential full grid."""
    weights = np.asarray(weights, dtype=float).copy()
    score = binary_auc(y[rows], values[rows] @ weights)
    for _ in range(max_steps):
        best_score, best_weights = score, weights
        for donor in range(len(weights)):
            if weights[donor] < step - 1e-12:
                continue
            for receiver in range(len(weights)):
                if receiver == donor:
                    continue
                candidate = weights.copy()
                candidate[donor] -= step
                candidate[receiver] += step
                if 1 - candidate[0] > max_secondary + 1e-12:
                    continue
                candidate_score = binary_auc(y[rows], values[rows] @ candidate)
                if candidate_score > best_score + 1e-6:
                    best_score, best_weights = candidate_score, candidate
        if best_score <= score + 1e-6:
            break
        score, weights = best_score, best_weights
    weights[np.abs(weights) < 1e-12] = 0
    return score, weights


def _coarse_blends(y, values, rows):
    """Select the 10% and 30% caps in one pass over the larger grid."""
    first = next(weight_grid(values.shape[1], .30))
    first_score = binary_auc(y[rows], values[rows] @ first)
    narrow_score, narrow_weights = first_score, first.copy()
    wide_score, wide_weights = first_score, first.copy()
    for candidate in weight_grid(values.shape[1], .30):
        candidate_score = binary_auc(y[rows], values[rows] @ candidate)
        secondary = 1 - candidate[0]
        if secondary <= .10 + 1e-12 and candidate_score > narrow_score + 1e-6:
            narrow_score, narrow_weights = candidate_score, candidate.copy()
        if candidate_score > wide_score + 1e-6:
            wide_score, wide_weights = candidate_score, candidate.copy()
    return (narrow_score, narrow_weights), (wide_score, wide_weights)


def best_blends(y, values, rows):
    narrow, wide = _coarse_blends(y, values, rows)
    return (refine_weights(y, values, rows, narrow[1], .10),
            refine_weights(y, values, rows, wide[1], .30))


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


def crossfit_blends(y, values, cv):
    names = ('coarse_narrow', 'coarse_wide', 'narrow', 'wide')
    predictions = {name: np.full(len(y), np.nan) for name in names}
    weights = {name: [] for name in names}
    for fold, (_, iv) in enumerate(cv):
        training = np.concatenate([other_iv for other_fold, (_, other_iv) in enumerate(cv)
                                   if other_fold != fold])
        coarse_narrow, coarse_wide = _coarse_blends(y, values, training)
        narrow = refine_weights(y, values, training, coarse_narrow[1], .10)
        wide = refine_weights(y, values, training, coarse_wide[1], .30)
        choices = [('coarse_narrow', coarse_narrow), ('coarse_wide', coarse_wide),
                   ('narrow', narrow), ('wide', wide)]
        for name, (_, selected) in choices:
            predictions[name][iv] = values[iv] @ selected
            weights[name].append(selected.tolist())
    development = np.concatenate([iv for _, iv in cv])
    return {name: dict(auc=binary_auc(y[development], prediction[development]),
        fold_auc=fold_auc_scores(y, prediction, cv), fold_weights=weights[name],
        prediction=prediction) for name, prediction in predictions.items()}


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
    cat_names = manifest.get('lanes', {}).get('cat', [])
    if not cat_names and any((run / 'candidates').glob('cat_*.json')):
        raise ValueError('Missing promoted.json entry for cat; rerun search before freeze.')
    if cat_names:
        invalid = [name for name in cat_names
                   if not (run / 'candidates' / f'{name}.json').exists()
                   or p.read_json(run / 'candidates' / f'{name}.json').get('stage') != 'full']
        if invalid or cat_names[0] != 'cat_fixed' or len(cat_names) > 2:
            raise ValueError(f'Invalid promoted candidates for cat: {invalid or cat_names}')
        fixed = load_candidate(run, cat_names[0], y, dev, sealed)
        chosen = fixed
        cat_gate = None
        if len(cat_names) == 2:
            challenger = load_candidate(run, cat_names[1], y, dev, sealed)
            cat_gate = gpu_primary_gate(y, challenger[1], fixed[1], dev, cv)
            if cat_gate['allowed']:
                chosen = challenger
        p.write_json(run / 'cat_candidate_comparison.json', dict(
            fixed=fixed[0]['name'], challenger=None if len(cat_names) == 1 else cat_names[1],
            chosen=chosen[0]['name'], required_fold_wins=3, gate=cat_gate))
        redundant = any(np.corrcoef(chosen[1][dev], other[dev])[0, 1] > args.max_corr and
            np.corrcoef(rankdata(chosen[1][dev]), rankdata(other[dev]))[0, 1] > args.max_corr
            for _, other in selected)
        gpu_replacement = gpu_primary_gate(y, chosen[1], baseline[1], dev, cv)['allowed']
        retained = not redundant or gpu_replacement
        cat_seed = dict(enabled=False, accepted=False, reason=(
            'The tuned CatBoost candidate did not pass the fixed-control gate.'
            if chosen[0]['name'] == 'cat_fixed' else
            'The tuned CatBoost candidate was redundant with the retained pool.'
            if not retained else 'Only one seed was requested.'))
        final_seeds = list(getattr(args, 'final_seeds', [cfg.get('seed', 2026)]))
        if chosen[0]['name'] != 'cat_fixed' and retained and len(final_seeds) > 1:
            averaged_candidate = {key: chosen[0][key]
                for key in ('name', 'lane', 'family', 'variant', 'params', 'device')}
            averaged_candidate.update(name=chosen[0]['name'] + '_seedavg', model_seeds=final_seeds)
            evaluate(args, cfg, averaged_candidate, requested=list(range(4)), stage='full')
            averaged = load_candidate(run, averaged_candidate['name'], y, dev, sealed)
            seed_gate = gpu_primary_gate(y, averaged[1], chosen[1], dev, cv)
            cat_seed = dict(enabled=True, accepted=seed_gate['allowed'], seeds=final_seeds,
                base_candidate=chosen[0]['name'], averaged_candidate=averaged[0]['name'], **seed_gate)
            if seed_gate['allowed']:
                chosen = averaged
        p.write_json(run / 'cat_seed_average_comparison.json', cat_seed)
        if retained:
            selected.append(chosen)
        tie_pool.append(fixed)
        if chosen[0]['name'] != fixed[0]['name']:
            tie_pool.append(chosen)
    # Only the latest explicit promotion set can be selected.
    for lane, limit in [('lgb', 3), ('xgb', 2), ('realmlp', 1)]:
        if not any((run / 'candidates').glob(f'{lane}_*.json')):
            continue
        if lane not in manifest.get('lanes', {}):
            raise ValueError(f'Missing promoted.json entry for {lane}; rerun search before freeze.')
        names = manifest['lanes'][lane]
        if lane == 'realmlp':
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
    if (primary[0].get('device') == 'cuda' and primary[0].get('family') != 'cat'
            and len(final_seeds) > 1):
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
        crossfit = crossfit_blends(y, values, cv)
        narrow_refined, narrow_refine_wins = improvement_gate(
            crossfit['coarse_narrow'], crossfit['narrow'])
        wide_refined, wide_refine_wins = improvement_gate(
            crossfit['coarse_wide'], crossfit['wide'])
        narrow_cv = crossfit['narrow'] if narrow_refined else crossfit['coarse_narrow']
        wide_cv = crossfit['wide'] if wide_refined else crossfit['coarse_wide']
        wide_allowed, wide_wins = improvement_gate(narrow_cv, wide_cv)
        cap = .30 if wide_allowed else .10
        refine = wide_refined if wide_allowed else narrow_refined
        if refine:
            score, weights = best_blend(y, values, dev, cap)
        else:
            chosen = _coarse_blends(y, values, dev)[1 if wide_allowed else 0]
            score, weights = chosen
        winner = dict(auc=score, weights=weights,
            fold_auc=fold_auc_scores(y, values @ weights, cv), cap=cap,
            crossfit_auc=(wide_cv if wide_allowed else narrow_cv)['auc'],
            crossfit_fold_auc=(wide_cv if wide_allowed else narrow_cv)['fold_auc'],
            crossfit_fold_weights=(wide_cv if wide_allowed else narrow_cv)['fold_weights'],
            wide_allowed=wide_allowed, wide_fold_wins=wide_wins,
            one_percent_refinement=refine,
            one_percent_fold_wins=wide_refine_wins if wide_allowed else narrow_refine_wins,
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
                         one_percent_refinement=best['probability']['one_percent_refinement'],
                         one_percent_fold_wins=best['probability']['one_percent_fold_wins'],
                         narrow_crossfit_auc=best['probability']['narrow_crossfit_auc'],
                         wide_crossfit_auc=best['probability']['wide_crossfit_auc']),
        rank=dict(auc=best['rank']['auc'], weights=best['rank']['weights'].tolist(),
                  fold_auc=best['rank']['fold_auc'], cap=best['rank']['cap'],
                  crossfit_auc=best['rank']['crossfit_auc'],
                  crossfit_fold_auc=best['rank']['crossfit_fold_auc'],
                  crossfit_fold_weights=best['rank']['crossfit_fold_weights'],
                  wide_allowed=best['rank']['wide_allowed'],
                  wide_fold_wins=best['rank']['wide_fold_wins'],
                  one_percent_refinement=best['rank']['one_percent_refinement'],
                  one_percent_fold_wins=best['rank']['one_percent_fold_wins'],
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
