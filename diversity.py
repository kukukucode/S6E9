"""Compatibility CLI for the package-based T4 x2 pipeline."""
import argparse
import subprocess

import joblib
import numpy as np
import optuna
from sklearn.metrics import roc_auc_score

import prototype as p
from gpu_setup import require_t4_pair
from s6e9.artifact import (checked_prediction, key, write_runtime_profile,
                           write_screen_diagnostics)
from s6e9.audit import audit
from s6e9.finalize import finalize
from s6e9.freeze import (blend_values, crossfit_blend, fold_tie_statistics, freeze,
                         frozen_config, gpu_primary_gate, improvement_gate,
                         lexicographic_oof, lexicographic_rank, load_candidate,
                         rank_blend_allowed, rank_columns, rank_oof_matrix,
                         tie_break_allowed, weight_grid)
from s6e9.tuning import (anchor, cat_anchor, context, domain_anchor, evaluate, folds,
                         gpu_pool, realmlp_anchor, search, shutdown_gpu_workers,
                         single_seed_folds, suggest, worker, worker_loop)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['search', 'freeze', 'audit', 'finalize', 'worker', 'worker-loop'])
    parser.add_argument('--job')
    parser.add_argument('--data', default='/kaggle/input/competitions/playground-series-s6e9')
    parser.add_argument('--run', default='/kaggle/working/s6e9_diversity_v6_0')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--threads', type=int, default=4, help='Total CPU budget across concurrent folds')
    parser.add_argument('--gpu-ids', nargs='+', default=None, help='Two physical T4 IDs/UUIDs; default auto-detect')
    parser.add_argument('--domain-compare', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--cat-compare', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--realmlp-compare', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--final-seeds', nargs='+', type=int, default=[2026, 42, 3407],
                        help='Average these seeds only when the selected primary GPU model passes the OOF gate')
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
    if not args.final_seeds or len(set(args.final_seeds)) != len(args.final_seeds):
        parser.error('final-seeds must be a non-empty unique list')
    args.gpu_ids = require_t4_pair(args.gpu_ids)
    try:
        globals()[args.command](args)
    finally:
        shutdown_gpu_workers()


if __name__ == '__main__':
    main()
