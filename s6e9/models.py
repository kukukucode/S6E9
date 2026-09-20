"""Model construction, fitting, backend selection, and prediction."""
import numpy as np

from .config import LGB_CUDA_BIN_LIMIT


def model_frame(x, family):
    if family not in ("cat", "realmlp"):
        return x
    x = x.copy()
    if family == "cat":
        for c in x.select_dtypes(include="category"):
            x[c] = x[c].astype(int).astype(str)
    return x


def realmlp_stop_epoch(model, fallback):
    values = getattr(model, "fit_params_", {}).get("stop_epoch", {})
    if isinstance(values, dict):
        values = list(values.values())
    values = np.asarray(values if np.size(values) else [fallback], dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    return max(1, int(np.median(values))) if len(values) else max(1, int(fallback))


def lgb_effective_device(params, cfg):
    requested = cfg.get("lgb_device", "cpu")
    if requested == "cuda" and params.get("max_bin", 255) > LGB_CUDA_BIN_LIMIT:
        return "cpu"
    return requested


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
        device = lgb_effective_device(params, cfg)
        if device != cfg.get("lgb_device", "cpu"):
            print(f"  LightGBM max_bin={params['max_bin']}: using CPU to avoid the observed CUDA crash; bins unchanged", flush=True)
        backend = (dict(deterministic=True, force_col_wise=True) if device == "cpu" else
                   dict(device_type="cuda", gpu_device_id=cfg.get("lgb_gpu_id", 0), num_gpu=1))
        if device not in ("cpu", "cuda"):
            raise ValueError("LightGBM device must be cpu or cuda")
        model = lgb.LGBMClassifier(**params, n_estimators=n, objective="binary", metric="auc",
            n_jobs=cfg["threads"], random_state=seed, verbosity=-1, **backend)
        model.fit(x, y, eval_set=[valid] if valid else None,
            callbacks=[lgb.early_stopping(stop, verbose=False)] if valid else [])
        best = model.best_iteration_ if valid else n
    elif family == "cat":
        from catboost import CatBoostClassifier
        cats = list(x.select_dtypes(include=["object", "string"]))
        cat_device = cfg.get("cat_device", "cpu")
        if cat_device not in ("cpu", "cuda"):
            raise ValueError("CatBoost device must be cpu or cuda")
        backend = (dict(task_type="GPU", devices=str(cfg.get("cat_gpu_id", 0)))
                   if cat_device == "cuda" else dict(task_type="CPU"))
        model = CatBoostClassifier(**params, iterations=n, loss_function="Logloss", eval_metric="AUC",
            thread_count=cfg["threads"], random_seed=seed, allow_writing_files=False, verbose=False, **backend)
        model.fit(x, y, cat_features=cats, eval_set=valid, use_best_model=bool(valid),
            early_stopping_rounds=stop if valid else None, verbose=False)
        best = model.get_best_iteration() + 1 if valid else n
    elif family == "realmlp":
        from pytabkit import RealMLP_TD_Classifier
        options = dict(params)
        epochs = int(options.pop("n_epochs", min(n, 256))) if rounds is None else n
        options.update(device="cuda:0" if cfg.get("realmlp_device") == "cuda" else "cpu",
            random_state=seed, n_threads=cfg["threads"], verbosity=0,
            val_metric_name="1-auc_ovr", use_ls=False)
        if valid is None:
            options.update(stop_epoch=epochs, val_fraction=0.0)
        else:
            options["n_epochs"] = epochs
        model = RealMLP_TD_Classifier(**options)
        if valid is None:
            model.fit(x, y)
            best = epochs
        else:
            model.fit(x, y, valid[0], valid[1])
            best = realmlp_stop_epoch(model, epochs)
    else:
        raise ValueError(f"Unknown model family: {family}")
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
    if family == "cat":
        return dict(depth=6, learning_rate=0.04, l2_leaf_reg=8., random_strength=1.,
            bootstrap_type="Bayesian", bagging_temperature=1.)
    if family == "realmlp":
        return dict(n_epochs=128)
    raise ValueError(f"Unknown model family: {family}")
