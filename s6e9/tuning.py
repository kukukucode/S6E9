"""Candidate tuning and cached CPU/T4 fold execution."""
from __future__ import annotations

import gc
import importlib.metadata
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
from sklearn.metrics import roc_auc_score

import prototype as p
from .artifact import (checked_prediction, key, read_json, write_json,
                       write_runtime_profile, write_screen_diagnostics)


def worker_entrypoint():
    return Path(__file__).resolve().parent.parent / 'diversity.py'


def context(args):
    run = Path(args.run)
    source_root = Path(__file__).resolve().parent.parent
    source_files = [source_root / name for name in ('prototype.py', 'diversity.py', 'gpu_setup.py')]
    source_files.extend(sorted((source_root / 's6e9').glob('*.py')))
    settings = dict(seed=args.seed, threads=args.threads, gpu_ids=args.gpu_ids,
        parallel_folds=2, max_rounds=args.max_rounds,
        early_stopping=args.early_stopping, auc_window=args.auc_window, max_corr=args.max_corr,
        domain_compare=args.domain_compare, cat_compare=args.cat_compare,
        realmlp_compare=args.realmlp_compare, final_seeds=args.final_seeds,
        screen_folds=args.screen_folds, promote_trials=args.promote_trials,
        hashes={f: p.digest(Path(args.data) / f) for f in ('train.csv', 'test.csv', 'sample_submission.csv')},
        sources={path.relative_to(source_root).as_posix(): p.digest(path) for path in source_files},
        versions={name: importlib.metadata.version(name) for name in
            ('numpy', 'pandas', 'scikit-learn', 'lightgbm', 'xgboost', 'catboost', 'optuna',
             'joblib', 'scipy') + (('pytabkit',) if args.realmlp_compare else ())})
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


def realmlp_anchor():
    return dict(name='realmlp_fixed', lane='realmlp', family='realmlp', variant='raw',
                params=p.defaults('realmlp'), device='cuda')


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
        model_seed = candidate.get('model_seed', cfg['seed'])
        backend = dict(max_rounds=cfg['max_rounds'], early_stopping=cfg['early_stopping'],
            threads=job['threads'], lgb_device=candidate['device'], lgb_gpu_id=0,
            xgb_device='cuda:0' if candidate['device'] == 'cuda' else 'cpu',
            cat_device=candidate['device'], cat_gpu_id=0,
            realmlp_device=candidate['device'])
        prepared_at = time.perf_counter()
        model, rounds = p.fit_model(candidate['family'], candidate['params'], train_x, y[it],
            valid, model_seed, backend, fixed)
        if candidate['family'] == 'xgb' and candidate['device'] == 'cuda':
            actual = json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
            if not actual.startswith('cuda'):
                raise RuntimeError(f'XGBoost did not use the requested GPU: {actual}')
        if candidate['family'] == 'cat' and candidate['device'] == 'cuda':
            actual = str(model.get_param('task_type') or '').upper()
            if actual != 'GPU':
                raise RuntimeError(f'CatBoost did not use the requested GPU: {actual}')
        if candidate['family'] == 'realmlp' and candidate['device'] == 'cuda':
            import torch
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError('RealMLP worker does not see its isolated T4.')
        prediction = p.predict(model, pred_x, candidate['family'])
        output = Path(job['outputs'][str(fold)])
        temporary = output.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.save(stream, prediction, allow_pickle=False)
        temporary.replace(output)
        p.write_json(output.with_suffix('.json'), dict(sha256=p.digest(output), rounds=max(1, int(rounds)),
            candidate=candidate['name'], family=candidate['family'], device=candidate['device'],
            phase=phase, fold=fold, model_seed=model_seed,
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
            proc = subprocess.Popen([sys.executable, '-X', 'faulthandler', '-u', str(worker_entrypoint()),
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


def single_seed_folds(args, cfg, candidate, phase, requested):
    run = Path(args.run)
    cache = run / 'cache' / 'predictions'
    cache.mkdir(parents=True, exist_ok=True)
    jobs = run / 'cache' / 'jobs'
    jobs.mkdir(parents=True, exist_ok=True)
    results = {}
    y, (_, _, cv, _) = args._data[5], args._data[6]
    spec = {k: candidate[k] for k in ('family', 'variant', 'params', 'device')}
    if 'model_seed' in candidate:
        spec['model_seed'] = candidate['model_seed']
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
                    proc = subprocess.Popen([sys.executable, '-X', 'faulthandler', '-u', str(worker_entrypoint()),
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


def folds(args, cfg, candidate, phase, requested):
    seeds = candidate.get('model_seeds')
    if not seeds:
        return single_seed_folds(args, cfg, candidate, phase, requested)
    members = []
    for seed in seeds:
        member = {key: value for key, value in candidate.items() if key != 'model_seeds'}
        if seed != cfg['seed']:
            member['model_seed'] = seed
        members.append(single_seed_folds(args, cfg, member, phase, requested))
    averaged = {}
    for fold in requested:
        predictions = [result[fold][0] for result in members]
        metadata = dict(members[0][fold][1])
        metadata.update(rounds=int(np.median([result[fold][1]['rounds'] for result in members])),
                        model_seeds=list(seeds), ensemble_size=len(seeds))
        averaged[fold] = np.mean(predictions, axis=0), metadata
    return averaged


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
    if getattr(args, 'realmlp_compare', False):
        candidate = realmlp_anchor()
        path = run / 'candidates' / 'realmlp_fixed.json'
        if not path.exists():
            evaluate(args, cfg, candidate, requested=list(range(args.screen_folds)), stage='screen')
        record = p.read_json(path)
        if record.get('stage') != 'full':
            print('Promoting realmlp_fixed to full 4-fold CV', flush=True)
            evaluate(args, cfg, candidate, requested=list(range(4)), stage='full')
        manifest_path = run / 'promoted.json'
        manifest = p.read_json(manifest_path) if manifest_path.exists() else {
            'screen_folds': args.screen_folds, 'promote_trials': args.promote_trials, 'lanes': {}}
        manifest['lanes']['realmlp'] = ['realmlp_fixed']
        p.write_json(manifest_path, manifest)
    write_screen_diagnostics(run)
    write_runtime_profile(run)
    print('Search complete. No holdout labels used.', flush=True)
