"""Feature construction fitted only on each training partition."""
import hashlib
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .config import SEED


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


SIGNAL_VARIANTS = ("numeric_te", "digits_te", "multiscale_te", "multiscale_dual")


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


def domain_features(x):
    """Deterministic EV interactions; no labels or fitted statistics."""
    x = x.reset_index(drop=True)
    values = {}
    def num(name):
        return pd.to_numeric(x[name], errors='coerce').astype(np.float64)
    home, work = 'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work'
    if home in x and work in x:
        total = num(home) + num(work)
        values['domain_stations_total'] = total
        values['domain_stations_gap'] = num(home) - num(work)
        if 'Daily_Commute_km' in x:
            values['domain_commute_per_station'] = num('Daily_Commute_km') / (1 + total)
    if 'Annual_Income_USD' in x and 'Number_of_Cars_Owned' in x:
        values['domain_income_per_car'] = num('Annual_Income_USD') / (1 + num('Number_of_Cars_Owned'))
    anxiety = None
    if 'Range_Anxiety_Level' in x:
        anxiety = x['Range_Anxiety_Level'].astype('string').str.strip().str.lower().map(
            {'low': 0., 'medium': 1., 'high': 2.})
        values['domain_anxiety_ordinal'] = anxiety
    subsidy = None
    if 'Subsidy_Available' in x:
        subsidy = x['Subsidy_Available'].astype('string').str.strip().str.lower().map(
            {'no': 0., 'yes': 1.})
        values['source_subsidy'] = subsidy
    concern = num('Environmental_Concern_Level') if 'Environmental_Concern_Level' in x else None
    income = num('Annual_Income_USD') / 100000 if 'Annual_Income_USD' in x else None
    if income is not None and subsidy is not None:
        values['source_income_x_subsidy'] = income * subsidy
    if concern is not None and subsidy is not None:
        values['source_concern_x_subsidy'] = concern * subsidy
    if income is not None and concern is not None:
        values['source_income_x_concern'] = income * concern
    if anxiety is not None and subsidy is not None:
        values['source_anxiety_x_subsidy'] = anxiety * subsidy
    if all(value is not None for value in (income, concern, subsidy, anxiety)):
        anxiety_penalty = anxiety.map({0.: 0., 1.: 1., 2.: 3.})
        values['source_formula_score'] = (1.2 * income + .6 * concern + 2 * subsidy
                                          - anxiety_penalty)
    return pd.DataFrame(values, index=x.index).replace([np.inf, -np.inf], np.nan)


class DomainSignalFeatures(SignalFeatures):
    """Keep every multiscale_dual feature unchanged; add domain values, frequency and TE."""
    def __init__(self, seed=SEED):
        super().__init__('multiscale_dual', seed)

    def keys(self, x):
        keys, numeric = super().keys(x)
        for name, values in domain_features(x).items():
            numeric[name] = values.astype(np.float32)
            # Continuous source-formula values are useful to trees directly;
            # their near-unique frequency/TE maps only add memory and noise.
            if name not in {'source_income_x_subsidy', 'source_income_x_concern',
                            'source_formula_score'}:
                keys[name] = Features.tokens(values)
        return keys, numeric


def make_features(variant, seed=SEED):
    if variant == 'multiscale_domain':
        return DomainSignalFeatures(seed)
    return SignalFeatures(variant, seed) if variant in SIGNAL_VARIANTS else Features(variant, seed)
