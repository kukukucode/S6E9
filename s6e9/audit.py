"""Sealed holdout audit after selection is frozen."""
import numpy as np
from sklearn.metrics import roc_auc_score

import prototype as p
from .artifact import write_runtime_profile
from .freeze import blend_values, frozen_config, lexicographic_rank
from .tuning import context, folds


def audit(args):
    run, cfg, tr, te, sub, id_col, cols, y, (dev, sealed, cv, outer) = context(args)
    frozen = frozen_config(run)
    if (run / 'sealed_report.json').exists():
        print('Existing holdout report retained.', flush=True)
        return
    p.write_json(run / 'SEALED_OPENED.json', dict(frozen_sha256=p.digest(run / 'frozen.json')))
    values = {c['name']: folds(args, cfg, c, 'final', [4])[4][0][:len(sealed)]
              for c in frozen['candidates']}
    pred = blend_values(np.column_stack([values[c['name']] for c in frozen['candidates']]),
                        frozen['weights'], frozen['mode'])
    tie = frozen.get('tie_breaker')
    if tie is not None:
        candidate = tie['candidate']
        secondary = values.get(candidate['name'])
        if secondary is None:
            secondary = folds(args, cfg, candidate, 'final', [4])[4][0][:len(sealed)]
        pred = lexicographic_rank(pred, secondary)
    baseline = folds(args, cfg, frozen['baseline'], 'final', [4])[4][0][:len(sealed)]
    p.write_json(run / 'sealed_report.json', dict(sealed_auc=float(roc_auc_score(y[sealed], pred)),
        baseline_auc=float(roc_auc_score(y[sealed], baseline)), public_lb=None,
        note='Previously reviewed holdout split: reference only, not a fresh independent evaluation. Do not retune.'))
    write_runtime_profile(run)
    print(p.read_json(run / 'sealed_report.json'), flush=True)
