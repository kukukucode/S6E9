"""Leak-free development and sealed split definitions."""
import numpy as np
from sklearn.model_selection import StratifiedKFold

from .config import SEED


def split_plan(y, seed=SEED):
    """Sealed fold never enters any development fold's training indices."""
    outer = np.full(len(y), -1, dtype=int)
    for fold, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)):
        outer[va] = fold
    dev, sealed = np.flatnonzero(outer != 4), np.flatnonzero(outer == 4)
    cv = [(dev[tr], dev[va]) for tr, va in StratifiedKFold(4, shuffle=True, random_state=seed + 1).split(dev, y[dev])]
    return dev, sealed, cv, outer
