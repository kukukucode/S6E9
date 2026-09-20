"""Artifact serialization and integrity helpers."""
import hashlib
import json

import numpy as np
from scipy.stats import rankdata
from pathlib import Path


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()



def key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def checked_prediction(path):
    meta = read_json(path.with_suffix('.json'))
    if digest(path) != meta['sha256']:
        raise ValueError(f'Corrupt prediction cache: {path}')
    pred = np.load(path, allow_pickle=False)
    if not np.isfinite(pred).all() or not ((pred >= 0) & (pred <= 1)).all():
        raise ValueError(f'Invalid probabilities: {path}')
    return pred, meta


def write_runtime_profile(run):
    groups = {}
    for path in (Path(run) / 'cache' / 'predictions').glob('*.json'):
        record = read_json(path)
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
    write_json(Path(run) / 'runtime_profile.json', dict(
        note='Summed worker time; parallel jobs overlap in wall-clock time.', groups=groups))


def write_screen_diagnostics(run):
    """Measure whether two-fold screening preserves the promoted order."""
    manifest_path = run / 'promoted.json'
    if not manifest_path.exists():
        return
    manifest = read_json(manifest_path)
    lanes = {}
    for lane, names in manifest.get('lanes', {}).items():
        records = []
        for name in names:
            path = run / 'candidates' / f'{name}.json'
            if not path.exists():
                continue
            record = read_json(path)
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
    write_json(run / 'screen_promotion_diagnostics.json', dict(
        screen_folds=manifest.get('screen_folds'),
        note='Diagnostic only; screening folds and promotion counts are unchanged.', lanes=lanes))
