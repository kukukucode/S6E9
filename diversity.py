"""Cached fold execution and development-only model selection."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

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
        domain_compare=args.domain_compare, cat_compare=args.cat_compare, screen_folds=args.screen_folds,
        promote_trials=args.promote_trials,
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


def cat_anchor():
    return dict(name='cat_fixed', lane='cat', family='cat', variant='multiscale_dual',
                params=p.defaults('cat'), device='cuda')


def domain_anchor():
    params = dict(anchor()['params'], max_bin=p.LGB_CUDA_BIN_LIMIT)
    return dict(name='domain', lane='domain', family='lgb', variant='multiscale_domain',
                params=params, device='cuda')


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


def worker(path, data_state=None):
    """Load binary data once and process all folds assigned to this device."""
    job = p.read_json(path)
    cfg, candidate = job['config'], job['candidate']
    if data_state is None:
        data = joblib.load(cfg['_data_cache'])
    else:
        cache_path = str(Path(cfg['_data_cache']).resolve())
        if data_state.get('path') != cache_path:
            data_state.clear()
            data_state.update(path=cache_path, value=joblib.load(cache_path))
        data = data_state['value']
    tr, te, _, cols, y, (_, _, cv, outer) = data
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
            xgb_device='cuda:0' if candidate['device'] == 'cuda' else 'cpu',
            cat_device=candidate['device'], cat_gpu_id=0)
        prepared_at = time.perf_counter()
        model, rounds = p.fit_model(candidate['family'], candidate['params'], train_x, y[it],
            valid, cfg['seed'], backend, fixed)
        if candidate['family'] == 'xgb' and candidate['device'] == 'cuda':
            actual = json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
            if not actual.startswith('cuda'):
                raise RuntimeError(f'XGBoost did not use the requested GPU: {actual}')
        if candidate['family'] == 'cat' and candidate['device'] == 'cuda':
            actual = str(model.get_param('task_type') or '').upper()
            if actual != 'GPU':
                raise RuntimeError(f'CatBoost did not use the requested GPU: {actual}')
        prediction = p.predict(model, pred_x, candidate['family'])
        output = Path(job['outputs'][str(fold)])
        temporary = output.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.save(stream, prediction, allow_pickle=False)
        temporary.replace(output)
        p.write_json(output.with_suffix('.json'), dict(sha256=p.digest(output), rounds=max(1, int(rounds)),
            candidate=candidate['name'], family=candidate['family'], device=candidate['device'],
            phase=phase, fold=fold,
            validation_rows=len(iv), test_rows=0 if phase == 'dev' else len(te), feature_cache_hit=cache_hit,
            prepare_seconds=prepared_at-started, fit_predict_seconds=time.perf_counter()-prepared_at,
            feature_cache_bytes=cache.stat().st_size))


def worker_loop():
    """Keep one interpreter and binary data cache alive for one isolated GPU."""
    data_state = {}
    for line in sys.stdin:
        job_path = line.strip()
        if not job_path:
            continue
        status = None
        try:
            job = p.read_json(job_path)
            status = Path(job['status'])
            worker(job_path, data_state)
            p.write_json(status, {'ok': True})
        except BaseException:
            error = traceback.format_exc()
            print(error, file=sys.stderr, flush=True)
            if status is not None:
                p.write_json(status, {'ok': False, 'error': error})
        finally:
            gc.collect()


class GpuWorkerPool:
    """One persistent child per T4; jobs remain isolated by CUDA visibility."""
    def __init__(self):
        self.processes = {}

    def _start(self, slot, selector, threads, log_path):
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads),
            PYTHONFAULTHANDLER='1', PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES=str(selector))
        log = log_path.open('a', encoding='utf8')
        try:
            proc = subprocess.Popen([sys.executable, '-X', 'faulthandler', '-u', str(Path(__file__).resolve()),
                'worker-loop'], stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                env=env, text=True, bufsize=1)
        except BaseException:
            log.close()
            raise
        self.processes[slot] = dict(selector=str(selector), threads=threads, proc=proc,
            log=log, log_path=log_path)
        return self.processes[slot]

    def submit(self, slot, selector, threads, job_path, status_path, log_path):
        item = self.processes.get(slot)
        if item is not None and (item['selector'] != str(selector) or item['threads'] != threads
                                 or item['proc'].poll() is not None):
            self._stop(item)
            self.processes.pop(slot, None)
            item = None
        if item is None:
            item = self._start(slot, selector, threads, log_path)
        status_path.unlink(missing_ok=True)
        item['proc'].stdin.write(str(Path(job_path).resolve()) + '\n')
        item['proc'].stdin.flush()
        return dict(item, status_path=status_path)

    @staticmethod
    def _tail(path):
        return path.read_text(encoding='utf8', errors='replace')[-5000:] if path.exists() else ''

    def wait(self, tasks):
        pending = list(tasks)
        while pending:
            for task in pending[:]:
                if task['status_path'].exists():
                    status = p.read_json(task['status_path'])
                    if not status.get('ok'):
                        raise RuntimeError(f'GPU worker failed. Log: {task["log_path"]}\n'
                            f'{status.get("error", self._tail(task["log_path"]))}')
                    pending.remove(task)
                elif task['proc'].poll() is not None:
                    raise RuntimeError(f'GPU worker exited ({task["proc"].poll()}). Log: '
                        f'{task["log_path"]}\n{self._tail(task["log_path"])}')
            if pending:
                time.sleep(.05)

    @staticmethod
    def _stop(item):
        proc = item['proc']
        try:
            if proc.poll() is None:
                proc.stdin.close()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    proc.wait()
        finally:
            item['log'].close()

    def shutdown(self):
        for item in list(self.processes.values()):
            self._stop(item)
        self.processes.clear()


_GPU_POOL = None


def gpu_pool():
    global _GPU_POOL
    if _GPU_POOL is None:
        _GPU_POOL = GpuWorkerPool()
    return _GPU_POOL


def shutdown_gpu_workers():
    global _GPU_POOL
    if _GPU_POOL is not None:
        _GPU_POOL.shutdown()
        _GPU_POOL = None


def write_runtime_profile(run):
    groups = {}
    for path in (Path(run) / 'cache' / 'predictions').glob('*.json'):
        record = p.read_json(path)
        if 'prepare_seconds' not in record or 'fit_predict_seconds' not in record:
            continue
        name = f"{record.get('device', 'unknown')}:{record.get('family', 'unknown')}:{record.get('phase', 'unknown')}"
        group = groups.setdefault(name, dict(jobs=0, feature_cache_hits=0,
            prepare_seconds=0., fit_predict_seconds=0.))
        group['jobs'] += 1
        group['feature_cache_hits'] += int(record.get('feature_cache_hit', False))
        group['prepare_seconds'] += float(record['prepare_seconds'])
        group['fit_predict_seconds'] += float(record['fit_predict_seconds'])
    for group in groups.values():
        group['prepare_seconds'] = round(group['prepare_seconds'], 3)
        group['fit_predict_seconds'] = round(group['fit_predict_seconds'], 3)
    p.write_json(Path(run) / 'runtime_profile.json', dict(
        note='Summed worker time; parallel jobs overlap in wall-clock time.', groups=groups))


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
    processes, tasks = [], []
    try:
        for slot, assigned in enumerate(groups):
            if not assigned:
                continue
            selector = '' if candidate['device'] == 'cpu' else str(args.gpu_ids[slot])
            tag = key(dict(candidate=spec, phase=phase, folds=assigned, selector=selector, threads=threads))
            job = jobs / f'{tag}.json'
            status_path = jobs / f'{tag}.status.json'
            p.write_json(job, dict(config=cfg, candidate=candidate, phase=phase, folds=assigned,
                threads=threads, run=str(run.resolve()), status=str(status_path.resolve()),
                outputs={str(f): str(outputs[f].resolve()) for f in assigned}))
            env = dict(os.environ, OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads),
                PYTHONFAULTHANDLER='1', PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES=selector)
            log_path = jobs / (f'gpu_worker_{slot}.log' if candidate['device'] == 'cuda' else f'{tag}.log')
            device_label = 'CPU' if candidate['device'] == 'cpu' else f'GPU {selector}'
            print(f'Starting {candidate["name"]}/{phase}/folds {assigned} on {device_label}; log={log_path}', flush=True)
            if candidate['device'] == 'cuda':
                tasks.append((assigned, gpu_pool().submit(slot, selector, threads, job, status_path, log_path)))
            else:
                log = log_path.open('w', encoding='utf8')
                try:
                    proc = subprocess.Popen([sys.executable, '-X', 'faulthandler', '-u', str(Path(__file__).resolve()),
                        'worker', '--job', str(job.resolve())], stdout=log, stderr=subprocess.STDOUT, env=env)
                except BaseException:
                    log.close()
                    raise
                processes.append((assigned, proc, log, log_path))
        if tasks:
            gpu_pool().wait([task for _, task in tasks])
            for assigned, _ in tasks:
                for fold in assigned:
                    results[fold] = checked_prediction(outputs[fold])
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


def evaluate(args, cfg, candidate, requested=None, stage='full'):
    requested = list(range(4)) if requested is None else sorted(set(requested))
    if not requested or any(fold not in range(4) for fold in requested):
        raise ValueError('Development folds must be a non-empty subset of 0..3')
    if stage not in ('screen', 'full'):
        raise ValueError('Evaluation stage must be screen or full')
    result = folds(args, cfg, candidate, 'dev', requested)
    y, (dev, _, cv, _) = args._data[5], args._data[6]
    oof = np.full(len(y), np.nan)
    for fold in requested:
        oof[cv[fold][1]] = result[fold][0]
    scored = np.concatenate([cv[fold][1] for fold in requested])
    score = float(roc_auc_score(y[scored], oof[scored]))
    fold_auc = {str(fold): float(roc_auc_score(y[cv[fold][1]], oof[cv[fold][1]]))
                for fold in requested}
    rounds = [result[fold][1]['rounds'] for fold in requested]
    path = Path(args.run) / 'candidates' / candidate['name']
    previous = p.read_json(path.with_suffix('.json')) if path.with_suffix('.json').exists() else {}
    record = dict(candidate, stage=stage,
        screen_folds=requested if stage == 'screen' else previous.get('screen_folds'),
        screen_auc=score if stage == 'screen' else previous.get('screen_auc'),
        screen_fold_auc=fold_auc if stage == 'screen' else previous.get('screen_fold_auc'),
        dev_auc=score if stage == 'full' else None,
        full_fold_auc=fold_auc if stage == 'full' else None,
        fixed_rounds=int(np.median(rounds)), rounds=rounds)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_suffix('.npy'), oof)
    record['oof_sha256'] = p.digest(path.with_suffix('.npy'))
    p.write_json(path.with_suffix('.json'), record)
    return score


def write_screen_diagnostics(run):
    """Measure whether two-fold screening preserves the promoted order."""
    manifest_path = run / 'promoted.json'
    if not manifest_path.exists():
        return
    manifest = p.read_json(manifest_path)
    lanes = {}
    for lane, names in manifest.get('lanes', {}).items():
        records = []
        for name in names:
            path = run / 'candidates' / f'{name}.json'
            if not path.exists():
                continue
            record = p.read_json(path)
            if record.get('screen_auc') is None or record.get('dev_auc') is None:
                continue
            records.append(record)
        if not records:
            continue
        screen_values = np.array([record['screen_auc'] for record in records], dtype=float)
        full_values = np.array([record['dev_auc'] for record in records], dtype=float)
        screen_ranks = rankdata(-screen_values, method='average')
        full_ranks = rankdata(-full_values, method='average')
        correlation = None
        if len(records) > 1 and np.std(screen_ranks) > 0 and np.std(full_ranks) > 0:
            correlation = float(np.corrcoef(screen_ranks, full_ranks)[0, 1])
        entries = []
        for record, screen_rank, full_rank in zip(records, screen_ranks, full_ranks):
            entries.append(dict(name=record['name'], screen_auc=record['screen_auc'],
                full_auc=record['dev_auc'], delta_auc=record['dev_auc'] - record['screen_auc'],
                screen_rank=float(screen_rank), full_rank=float(full_rank),
                screen_fold_auc=record.get('screen_fold_auc'),
                full_fold_auc=record.get('full_fold_auc')))
        lanes[lane] = dict(promoted_count=len(records), rank_correlation=correlation,
                           candidates=entries)
    p.write_json(run / 'screen_promotion_diagnostics.json', dict(
        screen_folds=manifest.get('screen_folds'),
        note='Diagnostic only; screening folds and promotion counts are unchanged.', lanes=lanes))


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
        evaluate(args, cfg, domain_anchor())
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
                return evaluate(args, cfg, suggest(trial, lane),
                    requested=list(range(args.screen_folds)), stage='screen')
            except BaseException:
                trial.set_user_attr('retry_pending', True)
                raise
        study.optimize(objective, n_trials=max(0, budget - done))
        complete = sorted((trial for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None),
            key=lambda trial: trial.value, reverse=True)
        promoted = complete[:min(args.promote_trials, len(complete))]
        for trial in promoted:
            path = run / 'candidates' / f'{lane}_{trial.number}.json'
            record = p.read_json(path)
            if record.get('stage') == 'full':
                continue
            candidate = {name: record[name]
                         for name in ('name', 'lane', 'family', 'variant', 'params', 'device')}
            print(f'Promoting {candidate["name"]} to full 4-fold CV', flush=True)
            evaluate(args, cfg, candidate, requested=list(range(4)), stage='full')
        manifest_path = run / 'promoted.json'
        manifest = p.read_json(manifest_path) if manifest_path.exists() else {
            'screen_folds': args.screen_folds, 'promote_trials': args.promote_trials, 'lanes': {}}
        manifest['lanes'][lane] = [f'{lane}_{trial.number}' for trial in promoted]
        p.write_json(manifest_path, manifest)
    if args.cat_compare:
        candidate = cat_anchor()
        path = run / 'candidates' / 'cat_fixed.json'
        if not path.exists():
            evaluate(args, cfg, candidate, requested=list(range(args.screen_folds)), stage='screen')
        record = p.read_json(path)
        if record.get('stage') != 'full':
            print('Promoting cat_fixed to full 4-fold CV', flush=True)
            evaluate(args, cfg, candidate, requested=list(range(4)), stage='full')
        manifest_path = run / 'promoted.json'
        manifest = p.read_json(manifest_path) if manifest_path.exists() else {
            'screen_folds': args.screen_folds, 'promote_trials': args.promote_trials, 'lanes': {}}
        manifest['lanes']['cat'] = ['cat_fixed']
        p.write_json(manifest_path, manifest)
    write_screen_diagnostics(run)
    write_runtime_profile(run)
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


def rank_blend_allowed(probability_auc, rank_auc, probability_folds, rank_folds):
    wins = sum(rank > probability + 1e-6
               for probability, rank in zip(probability_folds, rank_folds))
    return rank_auc > probability_auc + 1e-6 and wins >= 3, wins


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
    if args.domain_compare:
        selected.append(load_candidate(run, 'domain', y, dev, sealed))
    manifest_path = run / 'promoted.json'
    manifest = p.read_json(manifest_path) if manifest_path.exists() else {'lanes': {}}
    # Only the latest explicit promotion set can be selected.
    for lane, limit in [('lgb', 3), ('xgb', 2), ('cat', 1)]:
        if not any((run / 'candidates').glob(f'{lane}_*.json')):
            continue
        if lane not in manifest.get('lanes', {}):
            raise ValueError(f'Missing promoted.json entry for {lane}; rerun search before freeze.')
        names = manifest['lanes'][lane]
        if lane == 'cat':
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
    matrix = np.column_stack([oof for _, oof in selected])
    rank_matrix = rank_oof_matrix(matrix, cv)
    best = {}
    for mode, values in [('probability', matrix), ('rank', rank_matrix)]:
        winner = dict(auc=float(roc_auc_score(y[dev], values[dev, 0])),
                      weights=next(weight_grid(len(selected))))
        for weights in weight_grid(len(selected)):
            score = float(roc_auc_score(y[dev], values[dev] @ weights))
            if score > winner['auc'] + 1e-6:
                winner = dict(auc=score, weights=weights.copy())
        winner['fold_auc'] = fold_auc_scores(y, values @ winner['weights'], cv)
        best[mode] = winner
    allowed, rank_wins = rank_blend_allowed(best['probability']['auc'], best['rank']['auc'],
        best['probability']['fold_auc'], best['rank']['fold_auc'])
    mode = 'rank' if allowed else 'probability'
    winner = best[mode]
    p.write_json(run / 'blend_comparison.json', dict(
        candidates=[c['name'] for c, _ in selected],
        probability=dict(auc=best['probability']['auc'], weights=best['probability']['weights'].tolist(),
                         fold_auc=best['probability']['fold_auc']),
        rank=dict(auc=best['rank']['auc'], weights=best['rank']['weights'].tolist(),
                  fold_auc=best['rank']['fold_auc']),
        rank_fold_wins=rank_wins, rank_required_fold_wins=3, chosen_mode=mode))
    p.write_json(run / 'candidate_correlation.json', dict(names=[c['name'] for c, _ in selected],
        pearson=np.nan_to_num(np.atleast_2d(np.corrcoef(matrix[dev], rowvar=False)), nan=1).tolist()))
    chosen = [(c, float(w)) for (c, _), w in zip(selected, winner['weights']) if w > 0]
    p.write_json(run / 'frozen.json', dict(candidates=[c for c, _ in chosen], weights=[w for _, w in chosen],
        mode=mode, development_selection_auc=winner['auc'], development_fold_auc=winner['fold_auc'],
        rank_fold_wins=rank_wins, primary=primary[0]['name'], baseline=baseline[0],
        config_sha256=p.digest(run / 'config.json'),
        note='Weights selected only on development OOF. Selection-biased; no Public LB tuning.'))
    print(f'Frozen {mode} blend; dev AUC={winner["auc"]:.7f}; weights={[w for _, w in chosen]}', flush=True)


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
    pred = blend_values(np.column_stack(values), frozen['weights'], frozen['mode'])
    baseline = folds(args, cfg, frozen['baseline'], 'final', [4])[4][0][:len(sealed)]
    p.write_json(run / 'sealed_report.json', dict(sealed_auc=float(roc_auc_score(y[sealed], pred)),
        baseline_auc=float(roc_auc_score(y[sealed], baseline)), public_lb=None,
        note='Previously reviewed holdout split: reference only, not a fresh independent evaluation. Do not retune.'))
    write_runtime_profile(run)
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
        oof[iv] = blend_values(vm, frozen['weights'], frozen['mode'])
        test_folds.append(blend_values(tm, frozen['weights'], frozen['mode']))
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
    write_runtime_profile(run)
    print(f'Created {run / "submission.csv"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['search', 'freeze', 'audit', 'finalize', 'worker', 'worker-loop'])
    parser.add_argument('--job')
    parser.add_argument('--data', default='/kaggle/input/competitions/playground-series-s6e9')
    parser.add_argument('--run', default='/kaggle/working/s6e9_diversity_v5_6')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--threads', type=int, default=4, help='Total CPU budget across concurrent folds')
    parser.add_argument('--gpu-ids', nargs='+', default=None, help='Two physical T4 IDs/UUIDs; default auto-detect')
    parser.add_argument('--domain-compare', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--cat-compare', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--screen-folds', type=int, default=2)
    parser.add_argument('--promote-trials', type=int, default=3)
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
    if args.command == 'worker-loop':
        worker_loop()
        return
    if min(args.threads, args.max_rounds, args.early_stopping) < 1 or min(args.lgb_trials, args.xgb_trials) < 0:
        parser.error('Invalid budgets')
    if not 0 < args.max_corr <= 1 or args.auc_window < 0 or args.auc_window > 1:
        parser.error('Invalid candidate filtering thresholds')
    if not 1 <= args.screen_folds <= 4 or args.promote_trials < 1:
        parser.error('screen-folds must be 1..4 and promote-trials must be positive')
    args.gpu_ids = require_t4_pair(args.gpu_ids)
    try:
        globals()[args.command](args)
    finally:
        shutdown_gpu_workers()


if __name__ == '__main__':
    main()
