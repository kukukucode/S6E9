"""Final five-fold OOF and submission artifact generation."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import prototype as p
from .artifact import write_runtime_profile
from .freeze import blend_values, frozen_config, lexicographic_rank
from .tuning import context, folds


def finalize(args):
    run, cfg, tr, te, sub, id_col, cols, y, (dev, sealed, cv, outer) = context(args)
    frozen = frozen_config(run)
    if not (run / 'sealed_report.json').exists():
        raise ValueError('Run audit after freeze first.')
    candidates = list(frozen['candidates'])
    tie = frozen.get('tie_breaker')
    if tie is not None and tie['candidate']['name'] not in {c['name'] for c in candidates}:
        candidates.append(tie['candidate'])
    predictions = {c['name']: folds(args, cfg, c, 'final', list(range(5))) for c in candidates}
    oof, test_folds, test_secondary_folds = np.empty(len(y)), [], []
    oof_secondary = np.empty(len(y)) if tie is not None else None
    for fold in range(5):
        iv = np.flatnonzero(outer == fold)
        vm = np.column_stack([predictions[c['name']][fold][0][:len(iv)] for c in frozen['candidates']])
        tm = np.column_stack([predictions[c['name']][fold][0][len(iv):] for c in frozen['candidates']])
        valid_blend = blend_values(vm, frozen['weights'], frozen['mode'])
        test_blend = blend_values(tm, frozen['weights'], frozen['mode'])
        if tie is not None:
            secondary = predictions[tie['candidate']['name']][fold][0]
            oof_secondary[iv] = secondary[:len(iv)]
            test_secondary_folds.append(secondary[len(iv):])
        oof[iv] = valid_blend
        test_folds.append(test_blend)
    test = np.mean(test_folds, axis=0)
    if tie is not None:
        oof = lexicographic_rank(oof, oof_secondary)
        test = lexicographic_rank(test, np.mean(test_secondary_folds, axis=0))
    sub[p.TARGET] = sub[id_col].map(pd.Series(test, index=te[id_col]))
    if sub[p.TARGET].isna().any() or not sub[p.TARGET].between(0, 1).all():
        raise ValueError('Invalid submission probabilities or ID alignment')
    sub.to_csv(run / 'submission.csv', index=False)
    result = pd.DataFrame({id_col: tr[id_col], p.TARGET: y, 'fold': outer, 'prediction': oof})
    for c in candidates:
        model_predictions = predictions[c['name']]
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
