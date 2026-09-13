"""Cached fold execution and development-only model selection."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import joblib
import numpy as np
import optuna
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

import prototype as p
from gpu_setup import require_t4_pair


def key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def context(args):
    run = Path(args.run)
    settings = dict(seed=args.seed, threads=args.threads, gpu_ids=args.gpu_ids,
        parallel_folds=2, max_rounds=args.max_rounds,
        early_stopping=args.early_stopping, auc_window=args.auc_window, max_corr=args.max_corr,
        domain_compare=args.domain_compare,
        hashes={f: p.digest(Path(args.data) / f) for f in ('train.csv', 'test.csv', 'sample_submission.csv')},
        sources={f: p.digest(Path(__file__).with_name(f)) for f in ('prototype.py', 'diversity.py', 'gpu_setup.py')},
        versions={name: importlib.metadata.version(name) for name in
            ('numpy', 'pandas', 'scikit-learn', 'lightgbm', 'xgboost', 'catboost', 'optuna', 'joblib', 'scipy')})
    path = run / 'config.json'
    if path.exists():
        if p.read_json(path) != settings:
            raise ValueError('Data, implementation or settings changed. Choose a new v5 RUN; old results are retained.')
    elif args.command == 'search':
        p.write_json(path, settings)
    else:
        raise ValueError('Run search first with the same settings.')
    tr, te, sub, id_col, cols = p.load_data(args.data)
    y = tr[p.TARGET].to_numpy(dtype=int)
    split = p.split_plan(y, args.seed)
    data_path = run / 'cache' / 'data' / (key(dict(hashes=settings['hashes'], seed=args.seed,
        versions=settings['versions'], source=settings['sources']['prototype.py'])) + '.joblib')
    meta_path = data_path.with_suffix('.json')
    if data_path.exists() and meta_path.exists():
        if p.read_json(meta_path)['sha256'] != p.digest(data_path):
            raise ValueError(f'Corrupt worker data cache: {data_path}')
    else:
        data_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = data_path.with_suffix('.tmp')
        joblib.dump((tr, te, id_col, cols, y, split), temporary, compress=0)
        temporary.replace(data_path)
        p.write_json(meta_path, dict(sha256=p.digest(data_path), bytes=data_path.stat().st_size))
    args._data = (tr, te, sub, id_col, cols, y, split)
    return run, dict(settings, _data_cache=str(data_path.resolve())), tr, te, sub, id_col, cols, y, split


def anchor():
    params = dict(p.defaults('lgb'), max_depth=5, num_leaves=31, min_child_samples=20,
        colsample_bytree=.5, reg_alpha=.07, reg_lambda=2., max_bin=511)
    return dict(name='main', lane='main', family='lgb', variant='multiscale_dual',
                params=params, device='cpu')


def suggest(trial, lane):
    if lane == 'lgb':
        depth = trial.suggest_int('max_depth', 4, 7)
        params = dict(max_depth=depth, num_leaves=min(2 ** depth, trial.suggest_int('num_leaves', 20, 64)),
            min_child_samples=trial.suggest_int('min_child_samples', 10, 150),
            reg_alpha=trial.suggest_float('reg_alpha', 1e-4, 2, log=True),
            reg_lambda=trial.suggest_float('reg_lambda', .2, 10, log=True),
            subsample=trial.suggest_float('subsample', .7, 1), subsample_freq=1,
            colsample_bytree=trial.suggest_float('colsample_bytree', .35, .8),
            learning_rate=trial.suggest_float('learning_rate', .015, .06, log=True),
            max_bin=trial.suggest_categorical('max_bin', [63, 127, 255]))
    elif lane == 'xgb':
        params = dict(max_depth=trial.suggest_int('max_depth', 4, 8),
            min_child_weight=trial.suggest_float('min_child_weight', 2, 30, log=True),
            gamma=trial.suggest_float('gamma', 0, 2),
            reg_alpha=trial.suggest_float('reg_alpha', 1e-4, 3, log=True),
            reg_lambda=trial.suggest_float('reg_lambda', .5, 15, log=True),
            subsample=trial.suggest_float('subsample', .7, .95),
            colsample_bytree=trial.suggest_float('colsample_bytree', .4, .9),
            learning_rate=trial.suggest_float('learning_rate', .02, .06, log=True))
    else:
        raise ValueError(f'Unknown search lane: {lane}')
    return dict(name=f'{lane}_{trial.number}', lane=lane, family=lane,
        variant='multiscale_dual', params=params, device='cuda')


def checked_prediction(path):
    meta = p.read_json(path.with_suffix('.json'))
    if p.digest(path) != meta['sha256']:
        raise ValueError(f'Corrupt prediction cache: {path}')
    pred = np.load(path, allow_pickle=False)
    if not np.isfinite(pred).all() or not ((pred >= 0) & (pred <= 1)).all():
        raise ValueError(f'Invalid probabilities: {path}')
    return pred, meta


def worker(path):
    """Load binary data once and process all folds assigned to this device."""
    job = p.read_json(path)
    cfg, candidate = job['config'], job['candidate']
    tr, te, _, cols, y, (_, _, cv, outer) = joblib.load(cfg['_data_cache'])
    phase = job['phase']
    for fold in job['folds']:
        started = time.perf_counter()
        if phase == 'dev':
            it, iv = cv[fold]
            xp = tr[cols].iloc[iv]
        else:
            it, iv = np.flatnonzero(outer != fold), np.flatnonzero(outer == fold)
            xp = pd.concat([tr[cols].iloc[iv], te[cols]], ignore_index=True)
        xt = tr[cols].iloc[it]
        identity = dict(hashes=cfg['hashes'], sources=cfg['sources'], versions=cfg['versions'],
            seed=cfg['seed'], variant=candidate['variant'], phase=phase, fold=fold)
        cache = Path(job['run']) / 'cache' / 'features' / (key(identity) + '.joblib')
        meta_path = cache.with_suffix('.json')
        cache_hit = cache.exists() and meta_path.exists()
        if cache_hit:
            if p.read_json(meta_path)['sha256'] != p.digest(cache):
                raise ValueError(f'Corrupt feature cache: {cache}')
            train_x, pred_x = joblib.load(cache)
            print(f'Reused features: {candidate["variant"]}/{phase}/{fold}', flush=True)
        else:
            fe = p.make_features(candidate['variant'], cfg['seed'])
            train_x, pred_x = fe.fit_transform(xt, y[it]), fe.transform(xp)
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache.with_suffix('.tmp')
            joblib.dump((train_x, pred_x), temporary, compress=0)
            temporary.replace(cache)
            p.write_json(meta_path, dict(sha256=p.digest(cache), compression=0, bytes=cache.stat().st_size))
        valid = (pred_x, y[iv]) if phase == 'dev' else None
        fixed = candidate.get('fixed_rounds') if phase != 'dev' else None
        backend = dict(max_rounds=cfg['max_rounds'], early_stopping=cfg['early_stopping'],
            threads=job['threads'], lgb_device=candidate['device'], lgb_gpu_id=0,
            xgb_device='cuda:0' if candidate['device'] == 'cuda' else 'cpu')
        prepared_at = time.perf_counter()
        model, rounds = p.fit_model(candidate['family'], candidate['params'], train_x, y[it],
            valid, cfg['seed'], backend, fixed)
        if candidate['family'] == 'xgb' and candidate['device'] == 'cuda':
            actual = json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
            if not actual.startswith('cuda'):
                raise RuntimeError(f'XGBoost did not use the requested GPU: {actual}')
        prediction = p.predict(model, pred_x, candidate['family'])
        output = Path(job['outputs'][str(fold)])
        temporary = output.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.save(stream, prediction, allow_pickle=False)
        temporary.replace(output)
        p.write_json(output.with_suffix('.json'), dict(sha256=p.digest(output), rounds=max(1, int(rounds)),
            validation_rows=len(iv), test_rows=0 if phase == 'dev' else len(te), feature_cache_hit=cache_hit,
            prepare_seconds=prepared_at-started, fit_predict_seconds=time.perf_counter()-prepared_at,
            feature_cache_bytes=cache.stat().st_size))


def folds(args, cfg, candidate, phase, requested):
    run = Path(args.run)
    cache = run / 'cache' / 'predictions'
    cache.mkdir(parents=True, exist_ok=True)
    jobs = run / 'cache' / 'jobs'
    jobs.mkdir(parents=True, exist_ok=True)
    results = {}
    y, (_, _, cv, _) = args._data[5], args._data[6]
    spec = {k: candidate[k] for k in ('family', 'variant', 'params', 'device')}
    if phase != 'dev':
        spec['fixed_rounds'] = candidate['fixed_rounds']
    threads = args.threads if candidate['device'] == 'cpu' else max(1, args.threads // 2)
    pending, outputs = [], {}
    for fold in requested:
        tag = key(dict(config=cfg, candidate=spec, phase=phase, fold=fold, threads=threads))
        output = cache / f'{tag}.npy'
        outputs[fold] = output
        if output.exists() and output.with_suffix('.json').exists():
            results[fold] = checked_prediction(output)
            print(f'Reused predictions: {candidate["name"]}/{phase}/fold {fold}', flush=True)
        else:
            pending.append(fold)
    groups = ([pending] if candidate['device'] == 'cpu' else
              [[fold for fold in pending if fold % 2 == slot] for slot in range(2)])
    processes = []
    try:
        for slot, assigned in enumerate(groups):
            if not assigned:
                continue
            selector = '' if candidate['device'] == 'cpu' else str(args.gpu_ids[slot])
            tag = key(dict(candidate=spec, phase=phase, folds=assigned, selector=selector, threads=threads))
            job = jobs / f'{tag}.json'
            p.write_json(job, dict(config=cfg, candidate=candidate, phase=phase, folds=assigned,
                threads=threads, run=str(run.resolve()), outputs={str(f): str(outputs[f].resolve()) for f in assigned}))
            env = dict(os.environ, OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads),
                PYTHONFAULTHANDLER='1', PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES=selector)
            log_path = jobs / f'{tag}.log'
            device_label = 'CPU' if candidate['device'] == 'cpu' else f'GPU {selector}'
            print(f'Starting {candidate["name"]}/{phase}/folds {assigned} on {device_label}; log={log_path}', flush=True)
            log = log_path.open('w', encoding='utf8')
            try:
                proc = subprocess.Popen([sys.executable, '-X', 'faulthandler', '-u', str(Path(__file__).resolve()),
                    'worker', '--job', str(job.resolve())], stdout=log, stderr=subprocess.STDOUT, env=env)
            except BaseException:
                log.close()
                raise
            processes.append((assigned, proc, log, log_path))
        for assigned, proc, log, log_path in processes:
            code = proc.wait()
            log.close()
            if code:
                tail = log_path.read_text(encoding='utf8', errors='replace')[-5000:]
                raise RuntimeError(f'{candidate["name"]}/{phase}/folds {assigned} failed ({code}). Log: {log_path}\n{tail}')
            for fold in assigned:
                results[fold] = checked_prediction(outputs[fold])
    finally:
        for _, proc, log, _ in processes:
            if proc.poll() is None:
                proc.terminate()
                proc.wait()
            log.close()
    for fold in requested:
        if phase == 'dev':
            score = float(roc_auc_score(y[cv[fold][1]], results[fold][0]))
            print(f'{candidate["name"]} fold={fold} AUC={score:.7f}', flush=True)
    return results


def evaluate(args, cfg, candidate):
    result = folds(args, cfg, candidate, 'dev', list(range(4)))
    y, (dev, _, cv, _) = args._data[5], args._data[6]
    oof = np.full(len(y), np.nan)
    for fold, (_, iv) in enumerate(cv):
        oof[iv] = result[fold][0]
    rounds = [result[f][1]['rounds'] for f in range(4)]
    record = dict(candidate, dev_auc=float(roc_auc_score(y[dev], oof[dev])),
        fixed_rounds=int(np.median(rounds)), rounds=rounds)
    path = Path(args.run) / 'candidates' / candidate['name']
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_suffix('.npy'), oof)
    record['oof_sha256'] = p.digest(path.with_suffix('.npy'))
    p.write_json(path.with_suffix('.json'), record)
    return record['dev_auc']


def search(args):
    run, cfg, *_ = context(args)
    if (run / 'frozen.json').exists():
        print('Candidates already frozen; search skipped.', flush=True)
        return
    if (run / 'SEALED_OPENED.json').exists() or (run / 'sealed_report.json').exists():
        raise RuntimeError('Holdout already opened; search is disabled for this run.')
    if not (run / 'candidates/main.json').exists():
        evaluate(args, cfg, anchor())
    if args.domain_compare and not (run / 'candidates/domain.json').exists():
        evaluate(args, cfg, dict(anchor(), name='domain', lane='domain', variant='multiscale_domain'))
    for lane, budget in [('lgb', args.lgb_trials), ('xgb', args.xgb_trials)]:
        if budget == 0:
            continue
        study = optuna.create_study(study_name=lane, direction='maximize',
            storage=f"sqlite:///{(run / 'optuna.db').resolve().as_posix()}", load_if_exists=True)
        # Parent interruption leaves RUNNING; a failed child is marked FAIL by Optuna.
        # Both cases retry exact parameters so completed fold predictions remain reusable.
        requeued = {t.user_attrs.get('retry_of') for t in study.trials}
        for old in study.trials:
            interrupted = old.state == optuna.trial.TrialState.RUNNING
            failed = old.state == optuna.trial.TrialState.FAIL and old.user_attrs.get('retry_pending')
            if (interrupted or failed) and old.number not in requeued:
                retry = dict(old.system_attrs.get('fixed_params', {}), **old.params)
                if interrupted:
                    study.tell(old.number, state=optuna.trial.TrialState.FAIL)
                study.enqueue_trial(retry, user_attrs={'retry_of': old.number})
        study.sampler = optuna.samplers.TPESampler(seed=args.seed + len(study.trials))
        done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
        def objective(trial):
            try:
                return evaluate(args, cfg, suggest(trial, lane))
            except BaseException:
                trial.set_user_attr('retry_pending', True)
                raise
        study.optimize(objective, n_trials=max(0, budget - done))
    print('Search complete. No holdout labels used.', flush=True)


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


def weight_grid(size):
    unit = np.zeros(size)
    unit[0] = 1
    yield unit.copy()
    for i in range(1, size):
        for fraction in (.01, .02, .03, .05, .07, .10):
            w = unit.copy()
            w[0], w[i] = 1 - fraction, fraction
            yield w
    for i, j in itertools.combinations(range(1, size), 2):
        for a, b in ((.01, .01), (.02, .02), (.03, .02), (.02, .03), (.03, .03), (.05, .05)):
            w = unit.copy()
            w[0], w[i], w[j] = 1 - a - b, a, b
            yield w


def freeze(args):
    run, cfg, tr, te, sub, id_col, cols, y, (dev, sealed, cv, outer) = context(args)
    if (run / 'frozen.json').exists():
        print('Already frozen; selection retained.', flush=True)
        return
    if (run / 'SEALED_OPENED.json').exists():
        raise RuntimeError('Holdout already opened; cannot select candidates again.')
    baseline = load_candidate(run, 'main', y, dev, sealed)
    selected = [baseline]
    if args.domain_compare:
        selected.append(load_candidate(run, 'domain', y, dev, sealed))
        selected.sort(key=lambda pair: pair[0]['dev_auc'], reverse=True)
    # Only COMPLETE study trials qualify; never include interrupted or pruned artifacts.
    for lane, limit in [('lgb', 3), ('xgb', 2)]:
        if not any((run / 'candidates').glob(f'{lane}_*.json')):
            continue
        study = optuna.load_study(study_name=lane, storage=f"sqlite:///{(run / 'optuna.db').resolve().as_posix()}")
        records = [load_candidate(run, f'{lane}_{t.number}', y, dev, sealed)
                   for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        records.sort(key=lambda pair: pair[0]['dev_auc'], reverse=True)
        kept = 0
        for record, oof in records:
            if record['dev_auc'] < records[0][0]['dev_auc'] - args.auc_window or kept >= limit:
                continue
            # Check both Pearson and rank correlation; retain weaker-but-different families.
            redundant = any(np.corrcoef(oof[dev], other[dev])[0, 1] > args.max_corr and
                np.corrcoef(rankdata(oof[dev]), rankdata(other[dev]))[0, 1] > args.max_corr
                for _, other in selected)
            if not redundant:
                selected.append((record, oof))
                kept += 1
    matrix = np.column_stack([oof for _, oof in selected])
    best = dict(auc=float(roc_auc_score(y[dev], matrix[dev, 0])),
                weights=next(weight_grid(len(selected))))
    comparisons = []
    for weights in weight_grid(len(selected)):
        score = float(roc_auc_score(y[dev], matrix[dev] @ weights))
        comparisons.append(dict(weights=weights.tolist(), auc=score))
        if score > best['auc'] + 1e-6:
            best = dict(auc=score, weights=weights.copy())
    p.write_json(run / 'blend_comparison.json', comparisons)
    p.write_json(run / 'candidate_correlation.json', dict(names=[c['name'] for c, _ in selected],
        pearson=np.nan_to_num(np.atleast_2d(np.corrcoef(matrix[dev], rowvar=False)), nan=1).tolist()))
    chosen = [(c, float(w)) for (c, _), w in zip(selected, best['weights']) if w > 0]
    p.write_json(run / 'frozen.json', dict(candidates=[c for c, _ in chosen], weights=[w for _, w in chosen],
        mode='probability', development_selection_auc=best['auc'], baseline=baseline[0],
        config_sha256=p.digest(run / 'config.json'),
        note='Weights selected only on development OOF. Selection-biased; no Public LB tuning.'))
    print(f'Frozen probability blend; dev AUC={best["auc"]:.7f}; weights={[w for _, w in chosen]}', flush=True)


def frozen_config(run):
    frozen = p.read_json(run / 'frozen.json')
    if frozen['config_sha256'] != p.digest(run / 'config.json'):
        raise ValueError('Frozen configuration changed')
    mark = run / 'SEALED_OPENED.json'
    if mark.exists() and p.read_json(mark)['frozen_sha256'] != p.digest(run / 'frozen.json'):
        raise ValueError('Frozen selection changed after holdout was opened')
    return frozen


def audit(args):
    run, cfg, tr, te, sub, id_col, cols, y, (dev, sealed, cv, outer) = context(args)
    frozen = frozen_config(run)
    if (run / 'sealed_report.json').exists():
        print('Existing holdout report retained.', flush=True)
        return
    p.write_json(run / 'SEALED_OPENED.json', dict(frozen_sha256=p.digest(run / 'frozen.json')))
    values = [folds(args, cfg, c, 'final', [4])[4][0][:len(sealed)] for c in frozen['candidates']]
    pred = np.column_stack(values) @ np.array(frozen['weights'])
    baseline = folds(args, cfg, frozen['baseline'], 'final', [4])[4][0][:len(sealed)]
    p.write_json(run / 'sealed_report.json', dict(sealed_auc=float(roc_auc_score(y[sealed], pred)),
        baseline_auc=float(roc_auc_score(y[sealed], baseline)), public_lb=None,
        note='Previously reviewed holdout split: reference only, not a fresh independent evaluation. Do not retune.'))
    print(p.read_json(run / 'sealed_report.json'), flush=True)


def finalize(args):
    run, cfg, tr, te, sub, id_col, cols, y, (dev, sealed, cv, outer) = context(args)
    frozen = frozen_config(run)
    if not (run / 'sealed_report.json').exists():
        raise ValueError('Run audit after freeze first.')
    predictions = [folds(args, cfg, c, 'final', list(range(5))) for c in frozen['candidates']]
    oof, test_folds = np.empty(len(y)), []
    for fold in range(5):
        iv = np.flatnonzero(outer == fold)
        vm = np.column_stack([r[fold][0][:len(iv)] for r in predictions])
        tm = np.column_stack([r[fold][0][len(iv):] for r in predictions])
        oof[iv] = vm @ np.array(frozen['weights'])
        test_folds.append(tm @ np.array(frozen['weights']))
    test = np.mean(test_folds, axis=0)
    sub[p.TARGET] = sub[id_col].map(pd.Series(test, index=te[id_col]))
    if sub[p.TARGET].isna().any() or not sub[p.TARGET].between(0, 1).all():
        raise ValueError('Invalid submission probabilities or ID alignment')
    sub.to_csv(run / 'submission.csv', index=False)
    result = pd.DataFrame({id_col: tr[id_col], p.TARGET: y, 'fold': outer, 'prediction': oof})
    for c, model_predictions in zip(frozen['candidates'], predictions):
        raw_oof = np.empty(len(y))
        for fold in range(5):
            iv = np.flatnonzero(outer == fold)
            raw_oof[iv] = model_predictions[fold][0][:len(iv)]
        result[c['name']] = raw_oof
    result.to_csv(run / 'final_oof.csv', index=False)
    p.write_json(run / 'final_report.json', dict(final_oof_auc=float(roc_auc_score(y, oof)),
        public_lb=None, submission_sha256=p.digest(run / 'submission.csv'),
        note='OOF includes candidate/weight selection bias. Public LB remains unmeasured.'))
    print(f'Created {run / "submission.csv"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['search', 'freeze', 'audit', 'finalize', 'worker'])
    parser.add_argument('--job')
    parser.add_argument('--data', default='/kaggle/input/competitions/playground-series-s6e9')
    parser.add_argument('--run', default='/kaggle/working/s6e9_diversity_v5_2')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--threads', type=int, default=4, help='Total CPU budget across concurrent folds')
    parser.add_argument('--gpu-ids', nargs='+', default=None, help='Two physical T4 IDs/UUIDs; default auto-detect')
    parser.add_argument('--domain-compare', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--max-rounds', type=int, default=3500)
    parser.add_argument('--early-stopping', type=int, default=120)
    parser.add_argument('--lgb-trials', type=int, default=12)
    parser.add_argument('--xgb-trials', type=int, default=6)
    parser.add_argument('--auc-window', type=float, default=.0004)
    parser.add_argument('--max-corr', type=float, default=.999)
    args = parser.parse_args()
    if args.command == 'worker':
        if not args.job:
            parser.error('worker requires --job')
        worker(args.job)
        return
    if min(args.threads, args.max_rounds, args.early_stopping) < 1 or min(args.lgb_trials, args.xgb_trials) < 0:
        parser.error('Invalid budgets')
    if not 0 < args.max_corr <= 1 or args.auc_window < 0 or args.auc_window > 1:
        parser.error('Invalid candidate filtering thresholds')
    args.gpu_ids = require_t4_pair(args.gpu_ids)
    globals()[args.command](args)


if __name__ == '__main__':
    main()
