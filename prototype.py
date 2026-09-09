"""S6E9: development-only search, frozen sealed audit, final 5-fold ensemble."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

TARGET = "Will_Buy_EV"
SEED = 2026


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


def split_plan(y, seed=SEED):
    """Sealed fold never enters any development fold's training indices."""
    outer = np.full(len(y), -1, dtype=int)
    for fold, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)):
        outer[va] = fold
    dev, sealed = np.flatnonzero(outer != 4), np.flatnonzero(outer == 4)
    cv = [(dev[tr], dev[va]) for tr, va in StratifiedKFold(4, shuffle=True, random_state=seed + 1).split(dev, y[dev])]
    return dev, sealed, cv, outer


def augment(x, variant):
    x = x.copy().reset_index(drop=True)
    # Stable internal names work with all three libraries, including LightGBM.
    original = list(x)
    x.columns = [f"f{i}" for i in range(len(original))]
    if variant in ("interaction", "artifact"):
        numeric = [c for c in x if pd.api.types.is_numeric_dtype(x[c])]
        for c in numeric:
            values = x[c].astype(float)
            if variant == "interaction":
                x[c + "_log"] = np.sign(values) * np.log1p(np.abs(values))
            else:
                x[c + "_fraction"] = values - np.floor(values)
                x[c + "_rounded"] = values.round(1)
        lookup = {re.sub(r"[^a-z0-9]", "", c.lower()): f"f{i}" for i, c in enumerate(original)}
        pairs = [("Annual_Income", "Vehicle_Cost"), ("Monthly_Income", "Vehicle_Cost"),
                 ("Daily_Commute_Distance", "Charging_Stations_Nearby"),
                 ("Daily_Usage_km", "Battery_Range_km")]
        for a, b in pairs:
            ca, cb = [lookup.get(re.sub(r"[^a-z0-9]", "", c.lower())) for c in (a, b)]
            if ca in numeric and cb in numeric:
                x[f"{ca}_over_{cb}"] = x[ca] / (x[cb].abs() + 1)
        cats = [c for c in x if not pd.api.types.is_numeric_dtype(x[c])]
        # A bounded generic interaction set; variant must earn its place in dev CV.
        for a, b in zip(cats[:4], cats[1:5]):
            x[f"{a}_cross_{b}"] = x[a].astype("string").fillna("<NA>") + "|" + x[b].astype("string").fillna("<NA>")
        # S6E9 domain features: deterministic, no target or validation statistics.
        names = {c: f"f{i}" for i, c in enumerate(original)}
        home, work = names.get("Charging_Stations_Near_Home"), names.get("Charging_Stations_Near_Work")
        commute = names.get("Daily_Commute_km")
        if home is not None and work is not None:
            x["stations_total"] = x[home] + x[work]
            x["stations_gap"] = x[home] - x[work]
            if commute is not None:
                x["commute_per_station"] = x[commute] / (1 + x["stations_total"])
        anxiety = names.get("Range_Anxiety_Level")
        if anxiety is not None:
            x["anxiety_ordinal"] = x[anxiety].map({"Low": 0, "Medium": 1, "High": 2})
        income, cars = names.get("Annual_Income_USD"), names.get("Number_of_Cars_Owned")
        if income is not None and cars is not None:
            x["income_per_car"] = x[income] / (1 + x[cars])
    return x


class Features:
    """All learned mappings fit on this fit partition only, including frequency/TE."""
    def __init__(self, variant="raw", seed=SEED):
        self.variant, self.seed = variant, seed

    @staticmethod
    def tokens(s):
        return s.astype("string").fillna("<MISSING>")

    def te_map(self, s, y):
        stats = pd.DataFrame({"key": s.to_numpy(), "y": np.asarray(y)}).groupby("key")["y"].agg(["sum", "count"])
        prior = float(np.mean(y))
        return (stats["sum"] + 20 * prior) / (stats["count"] + 20), prior

    def fit_transform(self, x, y):
        z = augment(x, self.variant)
        self.cats = [c for c in z if not pd.api.types.is_numeric_dtype(z[c])]
        self.maps = {c: {v: i + 1 for i, v in enumerate(sorted(self.tokens(z[c]).unique()))} for c in self.cats}
        self.freq_cols = list(z) if self.variant in ("frequency", "artifact") else []
        self.freq = {c: self.tokens(z[c]).value_counts(normalize=True) for c in self.freq_cols}
        self.te = {c: self.te_map(self.tokens(z[c]), y) for c in self.cats} if self.variant == "target" else {}
        result = self._transform_augmented(z, include_te=False)
        # KFold assignment does not depend on labels. Both map AND prior exclude each row.
        if self.te:
            splits = list(KFold(4, shuffle=True, random_state=self.seed).split(z))
            for c in self.cats:
                tokens = self.tokens(z[c])
                values = np.empty(len(z), dtype=np.float32)
                for tr, va in splits:
                    mapping, prior = self.te_map(tokens.iloc[tr], np.asarray(y)[tr])
                    values[va] = tokens.iloc[va].map(mapping).fillna(prior)
                result[c + "_te"] = values
        return result

    def transform(self, x):
        z = augment(x, self.variant)
        return self._transform_augmented(z)

    def _transform_augmented(self, z, include_te=True):
        out = z.copy()
        for c in self.cats:
            codes = self.tokens(z[c]).map(self.maps[c]).fillna(0).astype(int)
            out[c] = pd.Categorical(codes, categories=range(len(self.maps[c]) + 1))
        for c in self.freq_cols:
            out[c + "_freq"] = self.tokens(z[c]).map(self.freq[c]).fillna(0).astype(np.float32)
        for c, (mapping, prior) in (self.te.items() if include_te else []):
            out[c + "_te"] = self.tokens(z[c]).map(mapping).fillna(prior).astype(np.float32)
        for c in out:
            if c not in self.cats:
                out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan).astype(np.float32)
        return out


SIGNAL_VARIANTS = ("numeric_te", "digits_te", "multiscale_te", "multiscale_dual")


class SignalFeatures:
    """Numeric/digit keys with fit-only frequency and label-independent cross-fit TE.

    Inspired by Naji's S6E9 feature experiments; implementation is independent.
    https://www.kaggle.com/code/najiama/pure-lgbm-model-cv-0-94606-lb-0-94637
    No reference submission, test labels, or external labels are used.
    """
    def __init__(self, variant, seed=SEED):
        self.variant, self.seed = variant, seed
        self.smoothing = (10., 100.) if variant == "multiscale_dual" else (20.,)

    def keys(self, x):
        # Build keys before float32 conversion: income digits can be lost on downcast.
        x = x.reset_index(drop=True)
        keys = {f"k{i}": Features.tokens(x[c]) for i, c in enumerate(x)}
        numeric = {}
        if self.variant != "numeric_te":
            for i, c in enumerate(x):
                if not pd.api.types.is_numeric_dtype(x[c]) or c == "Number_of_Cars_Owned":
                    continue
                values = pd.to_numeric(x[c], errors="coerce").to_numpy(dtype=np.float64)
                for power in range(-4, 4):
                    name = f"d{i}_{power + 4}"
                    with np.errstate(invalid="ignore"):
                        digit = np.floor_divide(values, 10. ** power) % 10
                    numeric[name] = digit.astype(np.float32)
                    keys[name] = Features.tokens(pd.Series(digit))
        if self.variant in ("multiscale_te", "multiscale_dual"):
            scales = {"Annual_Income_USD": (1., 100., 1000.), "Daily_Commute_km": (1., 5., 10.)}
            for column, widths in scales.items():
                if column not in x:
                    continue
                values = pd.to_numeric(x[column], errors="coerce").to_numpy(dtype=np.float64)
                for j, width in enumerate(widths):
                    name = f"b{list(x).index(column)}_{j}"
                    bins = np.floor(values / width)
                    numeric[name] = bins.astype(np.float32)
                    keys[name] = Features.tokens(pd.Series(bins))
        return pd.DataFrame(keys), pd.DataFrame(numeric, index=x.index)

    def fit_transform(self, x, y):
        y = np.asarray(y, dtype=np.float64)
        self.base = Features("raw", self.seed)
        base = self.base.fit_transform(x, y)
        keys, numeric = self.keys(x)
        # Fit-only pruning: validation/test values do not select columns.
        self.keep = []
        seen = set()
        for c in keys:
            if keys[c].nunique(dropna=False) <= 1:
                continue
            signature = hashlib.sha256(pd.util.hash_pandas_object(keys[c], index=False).values.tobytes()).hexdigest()
            if signature not in seen:
                seen.add(signature)
                self.keep.append(c)
        self.numeric_keep = [c for c in numeric if c in self.keep]
        self.maps = {}
        splits = list(KFold(4, shuffle=True, random_state=self.seed).split(y))
        extras = {c: numeric[c].to_numpy() for c in self.numeric_keep}
        for c in self.keep:
            codes, levels = pd.factorize(keys[c], sort=True)
            size = len(levels)
            counts = np.bincount(codes, minlength=size).astype(float)
            sums = np.bincount(codes, weights=y, minlength=size)
            prior = float(y.mean())
            frequency = counts / len(y)
            full = [(sums + strength * prior) / (counts + strength) for strength in self.smoothing]
            self.maps[c] = (pd.Index(levels), frequency, full, prior)
            extras[c + "_freq"] = frequency[codes].astype(np.float32)
            oof = np.empty((len(y), len(self.smoothing)), dtype=np.float32)
            for it, iv in splits:
                inner_count = np.bincount(codes[it], minlength=size)
                inner_sum = np.bincount(codes[it], weights=y[it], minlength=size)
                inner_prior = float(y[it].mean())
                for j, strength in enumerate(self.smoothing):
                    mapping = (inner_sum + strength * inner_prior) / (inner_count + strength)
                    oof[iv, j] = mapping[codes[iv]]
            for j, strength in enumerate(self.smoothing):
                extras[f"{c}_te{int(strength)}"] = oof[:, j]
        return pd.concat([base, pd.DataFrame(extras, index=base.index)], axis=1)

    def transform(self, x):
        base = self.base.transform(x)
        keys, numeric = self.keys(x)
        extras = {c: numeric[c].to_numpy() for c in self.numeric_keep}
        for c in self.keep:
            levels, frequency, full, prior = self.maps[c]
            codes = levels.get_indexer(keys[c])
            known = codes >= 0
            freq = np.zeros(len(x), dtype=np.float32)
            freq[known] = frequency[codes[known]]
            extras[c + "_freq"] = freq
            for strength, mapping in zip(self.smoothing, full):
                values = np.full(len(x), prior, dtype=np.float32)
                values[known] = mapping[codes[known]]
                extras[f"{c}_te{int(strength)}"] = values
        return pd.concat([base, pd.DataFrame(extras, index=base.index)], axis=1)


def make_features(variant, seed=SEED):
    return SignalFeatures(variant, seed) if variant in SIGNAL_VARIANTS else Features(variant, seed)


def model_frame(x, family):
    if family != "cat":
        return x
    x = x.copy()
    for c in x.select_dtypes(include="category"):
        x[c] = x[c].astype(int).astype(str)
    return x


def fit_model(family, params, x, y, valid, seed, cfg, rounds=None):
    n = int(rounds or cfg["max_rounds"])
    x = model_frame(x, family)
    valid = None if valid is None else (model_frame(valid[0], family), valid[1])
    stop = cfg["early_stopping"]
    if family == "xgb":
        from xgboost import XGBClassifier
        model = XGBClassifier(**params, n_estimators=n, objective="binary:logistic", eval_metric="auc",
            tree_method="hist", enable_categorical=True, device=cfg["xgb_device"],
            n_jobs=cfg["threads"], random_state=seed, early_stopping_rounds=stop if valid else None)
        model.fit(x, y, eval_set=[valid] if valid else None, verbose=False)
        best = model.best_iteration + 1 if valid else n
    elif family == "lgb":
        import lightgbm as lgb
        device = cfg.get("lgb_device", "cpu")
        backend = (dict(deterministic=True, force_col_wise=True) if device == "cpu" else
                   dict(device_type="cuda", gpu_device_id=cfg.get("lgb_gpu_id", 0), num_gpu=1))
        if device not in ("cpu", "cuda"):
            raise ValueError("LightGBM device must be cpu or cuda")
        model = lgb.LGBMClassifier(**params, n_estimators=n, objective="binary", metric="auc",
            n_jobs=cfg["threads"], random_state=seed, verbosity=-1, **backend)
        model.fit(x, y, eval_set=[valid] if valid else None,
            callbacks=[lgb.early_stopping(stop, verbose=False)] if valid else [])
        best = model.best_iteration_ if valid else n
    else:
        from catboost import CatBoostClassifier
        cats = list(x.select_dtypes(include=["object", "string"]))
        model = CatBoostClassifier(**params, iterations=n, loss_function="Logloss", eval_metric="AUC",
            thread_count=cfg["threads"], random_seed=seed, allow_writing_files=False, verbose=False)
        model.fit(x, y, cat_features=cats, eval_set=valid, use_best_model=bool(valid),
            early_stopping_rounds=stop if valid else None, verbose=False)
        best = model.get_best_iteration() + 1 if valid else n
    return model, max(1, int(best))


def predict(model, x, family):
    p = model.predict_proba(model_frame(x, family))[:, 1]
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid probabilities")
    return p


def defaults(family):
    if family == "xgb":
        return dict(max_depth=5, min_child_weight=8., gamma=0.1, reg_alpha=0.01,
            reg_lambda=8., subsample=0.85, colsample_bytree=0.85, learning_rate=0.04)
    if family == "lgb":
        return dict(max_depth=6, num_leaves=31, min_child_samples=80, reg_alpha=0.01,
            reg_lambda=8., subsample=0.85, subsample_freq=1, colsample_bytree=0.85, learning_rate=0.04, max_bin=255)
    return dict(depth=6, learning_rate=0.04, l2_leaf_reg=8., random_strength=1.,
        bootstrap_type="Bayesian", bagging_temperature=1.)


def suggest(trial, family):
    p = defaults(family)
    p["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
    if family in ("xgb", "lgb"):
        p.update(max_depth=trial.suggest_int("max_depth", 3, 9),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 30, log=True),
            subsample=trial.suggest_float("subsample", 0.65, 1),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1))
        if family == "xgb":
            p.update(min_child_weight=trial.suggest_float("min_child_weight", 1, 30, log=True),
                     gamma=trial.suggest_float("gamma", 0, 5))
        else:
            p.update(num_leaves=min(2 ** p["max_depth"], trial.suggest_int("num_leaves", 15, 127)),
                     min_child_samples=trial.suggest_int("min_child_samples", 10, 400),
                     max_bin=trial.suggest_categorical("max_bin", [255, 511, 1023]))
    else:
        p.update(depth=trial.suggest_int("depth", 4, 8), l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1, 30, log=True),
            random_strength=trial.suggest_float("random_strength", 0.01, 5, log=True),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0, 5))
    return p


def context(args, create=False):
    tr, te, sub, id_col, features = load_data(args.data)
    run = Path(args.run)
    cfg_path = run / "config.json"
    hashes = {name: digest(Path(args.data) / name) for name in ("train.csv", "test.csv", "sample_submission.csv")}
    code_hash = digest(__file__)
    if create and not cfg_path.exists():
        cfg = dict(seed=SEED, max_rounds=args.max_rounds, early_stopping=args.early_stopping,
            threads=args.threads, xgb_device=args.xgb_device, seeds=args.seeds,
            lgb_device=args.lgb_device, lgb_gpu_id=args.lgb_gpu_id,
            holdout_previously_reviewed=args.holdout_already_reviewed, profile=args.profile,
            hashes=hashes, code_sha256=code_hash, families=args.families,
            versions={p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "catboost", "optuna")},
            python=platform.python_version())
        write_json(cfg_path, cfg)
    cfg = read_json(cfg_path)
    if cfg["hashes"] != hashes or cfg["code_sha256"] != code_hash:
        raise ValueError("Data/code changed. Use a fresh run directory; do not reuse a revealed sealed holdout for tuning.")
    if create:
        for key in ("max_rounds", "early_stopping", "threads", "xgb_device", "lgb_device", "lgb_gpu_id", "seeds", "families", "profile"):
            if getattr(args, key) != cfg[key]:
                raise ValueError(f"Resume configuration changed: {key}")
        if args.holdout_already_reviewed != cfg.get("holdout_previously_reviewed", False):
            raise ValueError("Resume configuration changed: holdout review status")
    y = tr[TARGET].to_numpy(dtype=int)
    dev, sealed, cv, outer = split_plan(y, cfg["seed"])
    cfg = dict(cfg, _cache_dir=str(run / "cache"))
    return tr, te, sub, id_col, features, run, cfg, y, dev, sealed, cv, outer


def frame_digest(x):
    h = hashlib.sha256()
    h.update(repr([(str(c), str(t)) for c, t in x.dtypes.items()]).encode())
    h.update(pd.util.hash_pandas_object(x, index=False).to_numpy().tobytes())
    return h.hexdigest()


def prediction_batch(x, y, xp, candidate, cfg, seeds, valid_y=None, model_dir=None):
    """Checkpoint each exact fit. Trial numbers and selection metadata are not model inputs."""
    family, variant = candidate["family"], candidate["variant"]
    fixed = candidate.get("fixed_rounds") if valid_y is None else None
    cache_dir = Path(cfg["_cache_dir"]) if cfg.get("_cache_dir") else None
    signature = None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        signature = dict(x=frame_digest(x), xp=frame_digest(xp),
            y=hashlib.sha256(np.asarray(y, dtype=np.int64).tobytes()).hexdigest(),
            valid_y=None if valid_y is None else hashlib.sha256(np.asarray(valid_y, dtype=np.int64).tobytes()).hexdigest(),
            family=family, variant=variant, params=candidate["params"], fixed_rounds=fixed,
            config={k: v for k, v in cfg.items() if not k.startswith("_") and k not in ("seeds", "families", "profile")},
            implementation=digest(__file__))
    preds, rounds = [], []
    prepared = None
    seed_dependent = variant == "target" or variant in SIGNAL_VARIANTS
    for seed in seeds:
        path = meta_path = None
        if signature is not None:
            key = hashlib.sha256(json.dumps(dict(signature, model_seed=seed), sort_keys=True).encode()).hexdigest()
            path, meta_path = cache_dir / f"{key}.npz", cache_dir / f"{key}.json"
        model_path = Path(model_dir) / f"seed_{seed}.joblib" if model_dir is not None else None
        if path is not None and path.exists() and meta_path.exists() and (model_path is None or model_path.exists()):
            meta = read_json(meta_path)
            if meta["sha256"] != digest(path):
                raise ValueError(f"Corrupt prediction cache: {path}")
            with np.load(path, allow_pickle=False) as saved:
                values = saved["prediction"]
            if len(values) != len(xp) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
                raise ValueError(f"Invalid prediction cache: {path}")
            preds.append(values)
            rounds.append(int(meta["rounds"]))
            print(f"  Reused {family}/{variant} seed={seed}", flush=True)
            continue
        if prepared is None or seed_dependent:
            fe = make_features(variant, seed)
            xt, xv = fe.fit_transform(x, y), fe.transform(xp)
            prepared = (fe, xt, xv)
        else:
            fe, xt, xv = prepared
        model, n = fit_model(family, candidate["params"], xt, y,
            None if valid_y is None else (xv, valid_y), seed, cfg, fixed)
        values = predict(model, xv, family)
        if model_path is not None:
            import joblib
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"features": fe, "model": model, "family": family}, model_path)
        if path is not None:
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, prediction=values)
            temporary.replace(path)
            write_json(meta_path, {"rounds": n, "sha256": digest(path)})
        preds.append(values)
        rounds.append(n)
    return np.mean(preds, axis=0), rounds


def run_cv(x, y, cv, candidate, cfg, seeds):
    oof = np.full(len(y), np.nan)
    rounds, scores = [], []
    family, variant, params = [candidate[k] for k in ("family", "variant", "params")]
    for fold, (it, iv) in enumerate(cv):
        oof[iv], fold_rounds = prediction_batch(x.iloc[it], y[it], x.iloc[iv], candidate, cfg, seeds, valid_y=y[iv])
        rounds.extend(fold_rounds)
        scores.append(float(roc_auc_score(y[iv], oof[iv])))
        print(f"  {family}/{variant} fold={fold} AUC={scores[-1]:.6f}", flush=True)
    return oof, rounds, scores


def search(args):
    tr, te, sub, id_col, cols, run, cfg, y, dev, sealed, cv, outer = context(args, True)
    if (run / "frozen.json").exists() or (run / "SEALED_OPENED.json").exists():
        raise RuntimeError("This experiment is frozen; search is disabled.")
    pd.DataFrame({id_col: tr[id_col], "outer_fold": outer}).to_csv(run / "folds.csv", index=False)
    import optuna
    for family in cfg["families"]:
        study = optuna.create_study(study_name=family, storage=f"sqlite:///{(run / 'optuna.db').resolve().as_posix()}",
            load_if_exists=True, direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg["seed"]))
        # Test encodings at the same baseline parameters before the wider search.
        if not study.trials:
            variants = ("raw", *SIGNAL_VARIANTS, "multiscale_dual") if cfg["profile"] == "signals" else ("raw", "frequency", "target", "interaction")
            for index, variant in enumerate(variants):
                params = defaults(family)
                if cfg["profile"] == "signals" and index == 5 and family == "lgb":
                    params.update(max_depth=5, num_leaves=31, min_child_samples=20,
                        colsample_bytree=0.5, reg_alpha=0.07, reg_lambda=2., max_bin=511)
                study.enqueue_trial({"variant": variant, **{k: v for k, v in params.items() if k not in ("subsample_freq", "bootstrap_type")}})
        study.sampler = optuna.samplers.TPESampler(seed=cfg["seed"] + len(study.trials))
        def objective(trial):
            started = time.time()
            variant = trial.suggest_categorical("variant", ["raw", "frequency", "interaction", "artifact", "target", *SIGNAL_VARIANTS])
            candidate = dict(family=family, variant=variant, params=suggest(trial, family), trial=trial.number)
            print(f"Starting {family} trial={trial.number}, features={variant}; search seed={cfg['seed']}", flush=True)
            oof, rounds, scores = run_cv(tr[cols], y, cv, candidate, cfg, [cfg["seed"]])
            name = f"{family}_{trial.number}"
            np.save(run / f"{name}_oof.npy", oof)
            candidate.update(rounds=rounds, fold_auc=scores, dev_auc=float(roc_auc_score(y[dev], oof[dev])), seconds=time.time() - started)
            write_json(run / f"{name}.json", candidate)
            trial.set_user_attr("artifact", name)
            return candidate["dev_auc"]
        # trials is a TOTAL completed-trial budget per family, making resume idempotent.
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        study.optimize(objective, n_trials=max(0, args.trials - completed))
        study.trials_dataframe().to_csv(run / f"{family}_trials.csv", index=False)
    print("Search complete. Sealed scores have NOT been computed.", flush=True)


def blend_weights(y, matrix):
    # Small greedy convex search: 20 steps, no unconstrained optimizer.
    total = np.zeros(len(y))
    counts = np.zeros(matrix.shape[1])
    best_score, best = -1, None
    for step in range(20):
        scores = [roc_auc_score(y, (total + matrix[:, j]) / (step + 1)) for j in range(matrix.shape[1])]
        j = int(np.argmax(scores))
        counts[j] += 1
        total += matrix[:, j]
        if scores[j] > best_score:
            best_score, best = float(scores[j]), counts.copy() / (step + 1)
    return best, best_score


def freeze(args):
    tr, te, sub, id_col, cols, run, cfg, y, dev, sealed, cv, outer = context(args)
    if (run / "frozen.json").exists():
        print("Already frozen; no changes made.")
        return
    if (run / "SEALED_OPENED.json").exists():
        raise RuntimeError("Sealed fold already opened")
    candidates = []
    for family in cfg["families"]:
        records = [read_json(p) for p in run.glob(f"{family}_[0-9]*.json")]
        if not records:
            raise RuntimeError(f"Run search for {family} first")
        candidates.append(max(records, key=lambda c: c["dev_auc"]))
    # A fixed, reproducible comparator, independent of the search winners.
    baseline = dict(family="lgb", variant="raw", params=defaults("lgb"), trial=-1)
    all_candidates = candidates + [baseline]
    matrix = []
    for i, c in enumerate(all_candidates):
        cached = run / f"{c['family']}_{c['trial']}_oof.npy"
        if c["trial"] == -1:
            record_path = run / "lgb_0.json"
            if record_path.exists():
                record = read_json(record_path)
                if record["variant"] == "raw" and record["params"] == c["params"]:
                    cached = run / "lgb_0_oof.npy"
                    c["rounds"] = record["rounds"]
                    c["fold_auc"] = record["fold_auc"]
        if cfg["seeds"] == [cfg["seed"]] and cached.exists():
            oof, rounds, scores = np.load(cached), c["rounds"], c["fold_auc"]
        else:
            oof, rounds, scores = run_cv(tr[cols], y, cv, c, cfg, cfg["seeds"])
        c["fixed_rounds"] = int(np.median(rounds))
        c["seed_dev_auc"] = float(roc_auc_score(y[dev], oof[dev]))
        c["seed_fold_auc"] = scores
        matrix.append(oof[dev])
        np.save(run / f"frozen_candidate_{i}_dev_oof.npy", oof)
    matrix = np.column_stack(matrix)
    weights, score = blend_weights(y[dev], matrix[:, :-1])
    pd.DataFrame(matrix, columns=[f"candidate_{i}" for i in range(len(all_candidates))]).corr().to_csv(run / "dev_prediction_correlation.csv")
    report = pd.DataFrame({id_col: tr[id_col].iloc[dev].to_numpy(), TARGET: y[dev]})
    for i in range(matrix.shape[1]):
        report[f"candidate_{i}"] = matrix[:, i]
    report["blend"] = matrix[:, :-1] @ weights
    report.to_csv(run / "dev_oof.csv", index=False)
    write_json(run / "frozen.json", dict(candidates=candidates, baseline=baseline,
        weights=weights.tolist(), development_selection_auc=score,
        warning=("Development AUC is selection-biased. This holdout was reviewed previously; its audit is not a fresh independent evaluation."
                 if cfg.get("holdout_previously_reviewed", False) else
                 "Development AUC is selection-biased; sealed audit is the independent check."),
        config_sha256=digest(run / "config.json")))
    print(f"Frozen dev selection AUC={score:.6f}; weights={weights}. Next: audit.", flush=True)


def fitted_predictions(x, y, xpred, candidate, cfg, model_dir=None):
    return prediction_batch(x, y, xpred, candidate, cfg, cfg["seeds"], model_dir=model_dir)[0]


def paired_bootstrap(y, p, baseline, n=300):
    rng = np.random.default_rng(SEED)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    aucs, deltas = [], []
    for _ in range(n):
        ix = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        a = roc_auc_score(y[ix], p[ix])
        aucs.append(a)
        deltas.append(a - roc_auc_score(y[ix], baseline[ix]))
    return dict(auc_95_ci=np.quantile(aucs, [0.025, 0.975]).tolist(),
                delta_95_ci=np.quantile(deltas, [0.025, 0.975]).tolist(), bootstrap_replicates=n)


def audit(args):
    tr, te, sub, id_col, cols, run, cfg, y, dev, sealed, cv, outer = context(args)
    frozen = read_json(run / "frozen.json")
    if frozen["config_sha256"] != digest(run / "config.json"):
        raise ValueError("Frozen configuration mismatch")
    mark = run / "SEALED_OPENED.json"
    if mark.exists() and read_json(mark)["frozen_sha256"] != digest(run / "frozen.json"):
        raise ValueError("Frozen candidates changed after audit started")
    if (run / "sealed_report.json").exists():
        print(json.dumps(read_json(run / "sealed_report.json"), indent=2))
        return
    write_json(mark, {"frozen_sha256": digest(run / "frozen.json"), "opened_at": time.time()})
    # No sealed labels are supplied to fitting, encoding, early stopping or weight selection.
    both = pd.concat([tr[cols].iloc[sealed], te[cols]], ignore_index=True)
    preds = [fitted_predictions(tr[cols].iloc[dev], y[dev], both, c, cfg)[:len(sealed)] for c in frozen["candidates"]]
    p = np.column_stack(preds) @ np.asarray(frozen["weights"])
    baseline = fitted_predictions(tr[cols].iloc[dev], y[dev], both, frozen["baseline"], cfg)[:len(sealed)]
    report = dict(sealed_auc=float(roc_auc_score(y[sealed], p)),
        baseline_sealed_auc=float(roc_auc_score(y[sealed], baseline)), public_lb=None,
        public_lb_target=0.94635, sealed_rows=len(sealed),
        holdout_previously_reviewed=cfg.get("holdout_previously_reviewed", False),
        note=("Public LB is unmeasured. This holdout split was reviewed previously; this is not a fresh independent evaluation. Do not tune using this result."
              if cfg.get("holdout_previously_reviewed", False) else
              "Public LB is unmeasured. Do not tune using this sealed result."))
    report["delta_vs_baseline"] = report["sealed_auc"] - report["baseline_sealed_auc"]
    report.update(paired_bootstrap(y[sealed], p, baseline))
    pd.DataFrame({id_col: tr[id_col].iloc[sealed].to_numpy(), TARGET: y[sealed], "prediction": p, "baseline": baseline}).to_csv(run / "sealed_predictions.csv", index=False)
    write_json(run / "sealed_report.json", report)
    print(json.dumps(report, indent=2), flush=True)


def finalize(args):
    tr, te, sub, id_col, cols, run, cfg, y, dev, sealed, cv, outer = context(args)
    frozen = read_json(run / "frozen.json")
    if not (run / "sealed_report.json").exists():
        raise RuntimeError("Run the frozen sealed audit first")
    if read_json(run / "SEALED_OPENED.json")["frozen_sha256"] != digest(run / "frozen.json"):
        raise ValueError("Candidates changed after audit")
    all_oof, all_test = [], []
    for ci, (c, weight) in enumerate(zip(frozen["candidates"], frozen["weights"])):
        if weight == 0:
            continue
        oof, test_preds = np.empty(len(y)), []
        for fold in range(5):
            it, iv = np.flatnonzero(outer != fold), np.flatnonzero(outer == fold)
            # Fixed rounds from development CV: no final-fold early stopping or retuning.
            both = pd.concat([tr[cols].iloc[iv], te[cols]], ignore_index=True)
            p = fitted_predictions(tr[cols].iloc[it], y[it], both, c, cfg,
                run / "models" / f"candidate_{ci}_fold_{fold}" if args.save_models else None)
            oof[iv] = p[:len(iv)]
            test_preds.append(p[len(iv):])
            print(f"Final candidate={ci}, fold={fold} done", flush=True)
        test_pred = np.mean(test_preds, axis=0)
        all_oof.append(oof * weight)
        all_test.append(test_pred * weight)
        np.savez_compressed(run / f"final_candidate_{ci}.npz", oof=oof, test=test_pred)
    poof, ptest = np.sum(all_oof, axis=0), np.sum(all_test, axis=0)
    # Align to sample_submission by ID, never assume its row order equals test order.
    mapped = pd.Series(ptest, index=te[id_col])
    sub[TARGET] = sub[id_col].map(mapped)
    if sub[TARGET].isna().any() or not sub[TARGET].between(0, 1).all():
        raise ValueError("Invalid submission")
    sub.to_csv(run / "submission.csv", index=False)
    pd.DataFrame({id_col: tr[id_col], TARGET: y, "fold": outer, "prediction": poof}).to_csv(run / "final_oof.csv", index=False)
    write_json(run / "final_report.json", dict(final_oof_auc=float(roc_auc_score(y, poof)), public_lb=None,
        note="Final OOF reuses development-selected configurations and is NOT an independent score estimate.",
        submission_rows=len(sub), submission_sha256=digest(run / "submission.csv")))
    print(f"Created {run / 'submission.csv'}; Public LB remains unmeasured.", flush=True)


def diagnose(args):
    tr, te, sub, id_col, cols = load_data(args.data)
    y = tr[TARGET].to_numpy(dtype=int)
    dev, _, _, _ = split_plan(y)
    a = tr.iloc[dev][cols].sample(min(30000, len(dev)), random_state=SEED)
    b = te[cols].sample(min(30000, len(te)), random_state=SEED)
    x = pd.concat([a, b], ignore_index=True)
    labels = np.r_[np.zeros(len(a), int), np.ones(len(b), int)]
    it, iv = train_test_split(np.arange(len(x)), test_size=0.3, stratify=labels, random_state=SEED)
    fe = Features("raw")
    xt = fe.fit_transform(x.iloc[it], labels[it])
    xv = fe.transform(x.iloc[iv])
    cfg = dict(max_rounds=300, early_stopping=40, threads=args.threads, xgb_device="cpu")
    # Fixed complexity to avoid optimizing/reporting the same AV holdout.
    model, _ = fit_model("lgb", defaults("lgb"), xt, labels[it], None, SEED, cfg, 300)
    p = predict(model, xv, "lgb")
    report = dict(adversarial_auc=float(roc_auc_score(labels[iv], p)), id_excluded=True,
        sealed_excluded=True, train_rows=len(tr), test_rows=len(te), features=cols,
        development_positive_rate=float(y[dev].mean()),
        development_duplicate_feature_rows=int(tr.iloc[dev][cols].duplicated().sum()),
        test_duplicate_feature_rows=int(te[cols].duplicated().sum()),
        note="AV near 0.5 means this classifier found little shift; it does not prove identical distributions.")
    Path(args.run).mkdir(parents=True, exist_ok=True)
    write_json(Path(args.run) / "diagnostics.json", report)
    pd.DataFrame({"feature": cols, "importance": model.feature_importances_}).sort_values("importance", ascending=False).to_csv(Path(args.run) / "adversarial_importance.csv", index=False)
    pd.DataFrame({"dev_missing": tr.iloc[dev][cols].isna().mean(), "test_missing": te[cols].isna().mean(),
        "dev_unique": tr.iloc[dev][cols].nunique(), "test_unique": te[cols].nunique()}).to_csv(Path(args.run) / "schema_diagnostics.csv")
    print(json.dumps(report, indent=2), flush=True)


def check_device(args):
    """Small real fit before a long run; never silently substitute CPU for CUDA."""
    print(f"LightGBM {importlib.metadata.version('lightgbm')}; Python {platform.python_version()}; "
          f"device={args.lgb_device}; gpu_id={args.lgb_gpu_id}", flush=True)
    rng = np.random.default_rng(SEED)
    x = pd.DataFrame(rng.normal(size=(4096, 5)), columns=[f"f{i}" for i in range(5)])
    x["category"] = pd.Categorical(rng.integers(0, 4, len(x)))
    y = (x.f0.to_numpy() + x.f1.to_numpy() + rng.normal(size=len(x)) > 0).astype(int)
    params = defaults("lgb")
    params.update(max_bin=511, min_child_samples=20)
    cfg = dict(max_rounds=4, early_stopping=2, threads=args.threads,
               lgb_device=args.lgb_device, lgb_gpu_id=args.lgb_gpu_id)
    print("CHECK_STAGE: fit begin (max_bin=511, categories, bagging, validation)", flush=True)
    model, _ = fit_model("lgb", params, x.iloc[:3072], y[:3072], (x.iloc[3072:], y[3072:]), SEED, cfg)
    print("CHECK_STAGE: fit completed; predict begin", flush=True)
    predict(model, x.iloc[3072:], "lgb")
    print("CHECK_STAGE: predict completed", flush=True)
    actual = model.booster_.params.get("device_type", "cpu")
    if actual != args.lgb_device:
        raise RuntimeError(f"Requested {args.lgb_device}, but got {actual}")
    print(f"LightGBM backend check passed: {actual}, max_bin=511, categorical features enabled", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check-device", "diagnose", "search", "freeze", "audit", "finalize"])
    parser.add_argument("--data", default="/kaggle/input/competitions/playground-series-s6e9")
    parser.add_argument("--run", default="runs/signals_v4")
    parser.add_argument("--profile", choices=["signals", "legacy"], default="signals")
    parser.add_argument("--trials", type=int, default=6, help="Total completed trials per family; increase to resume")
    parser.add_argument("--families", nargs="+", choices=["xgb", "cat", "lgb"], default=["lgb"])
    parser.add_argument("--max-rounds", type=int, default=3500)
    parser.add_argument("--early-stopping", type=int, default=120)
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--xgb-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--lgb-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--lgb-gpu-id", type=int, default=0, help="Single CUDA device used by LightGBM")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026],
                        help="Seeds used after model selection; search always uses the single split seed")
    parser.add_argument("--holdout-already-reviewed", action="store_true",
                        help="Mark a reused holdout as previously reviewed, not fresh independent evidence")
    parser.add_argument("--save-models", action="store_true")
    args = parser.parse_args()
    if min(args.trials, args.max_rounds, args.early_stopping, args.threads) < 1:
        parser.error("Numeric budgets must be positive")
    if args.lgb_gpu_id < 0:
        parser.error("GPU device ID must be nonnegative")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.families)) != len(args.families):
        parser.error("Duplicate seeds/families")
    globals()[args.command.replace("-", "_")](args)


if __name__ == "__main__":
    main()
