"""Competition data loading and schema validation."""
from pathlib import Path
import pandas as pd

from .config import TARGET


def load_data(data):
    data = Path(data)
    required = ["train.csv", "test.csv", "sample_submission.csv"]
    if not all((data / f).is_file() for f in required):
        raise FileNotFoundError(f"Place {required} in {data}; no artificial fallback is used.")
    tr, te, sub = [pd.read_csv(data / f) for f in required]
    if TARGET not in tr or TARGET in te or TARGET not in sub:
        raise ValueError(f"Expected binary target {TARGET}")
    ids = [c for c in sub if c != TARGET]
    if len(ids) != 1:
        raise ValueError("Expected one submission ID column")
    id_col = ids[0]
    if set(tr[TARGET].unique()) == {"No", "Yes"}:
        tr[TARGET] = tr[TARGET].map({"No": 0, "Yes": 1})
    if not set(tr[TARGET].unique()) == {0, 1}:
        raise ValueError("Target must contain exactly 0 and 1, with no missing labels")
    for frame in (tr, te, sub):
        if id_col not in frame or frame[id_col].isna().any() or frame[id_col].duplicated().any():
            raise ValueError("Missing or duplicate IDs")
    if set(te[id_col]) != set(sub[id_col]) or len(te) != len(sub):
        raise ValueError("Test/submission IDs differ")
    if set(tr[id_col]) & set(te[id_col]):
        raise ValueError("Train/test IDs overlap")
    features = [c for c in tr if c not in (id_col, TARGET)]
    if set(features) != set(te.columns) - {id_col}:
        raise ValueError("Train/test feature schemas differ")
    return tr, te, sub, id_col, features
