#!/usr/bin/env python3
"""KB-EWS: knowledge-based early warning of entrepreneurial failure.

The decision time is a digital venture's first funding round. Predictors use
only information available then. Total capital raised, the number of later
rounds, and the date of the last round are excluded, because those quantities
keep changing until a venture closes and would leak the outcome.

Data: Crunchbase Data Export of 4 December 2015, CC-BY-NC, redistributed as
CSV by Peter Tripp (github.com/notpeter/crunchbase-data, commit be8711fd).
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache"
OUT = ROOT / "outputs"
COMMIT = "be8711fd4ee6b81110d243eb887ccc6bc8513601"
SOURCES = {
    "companies.csv": f"https://raw.githubusercontent.com/notpeter/crunchbase-data/{COMMIT}/companies.csv",
    "rounds.csv": f"https://raw.githubusercontent.com/notpeter/crunchbase-data/{COMMIT}/rounds.csv",
}
SEED = 7
TRAIN_YEARS = range(2006, 2011)
VAL_YEARS = range(2011, 2012)
TEST_YEARS = range(2012, 2014)
MAX_YEARS_TO_FUNDING = 10
ECOSYSTEM_WINDOW = 3

# Whole-category matches against the December 2015 Crunchbase category list.
DIGITAL = frozenset(
    {
        "software",
        "mobile",
        "e-commerce",
        "curated web",
        "enterprise software",
        "saas",
        "analytics",
        "cloud computing",
        "cloud data services",
        "cloud infrastructure",
        "apps",
        "social media",
        "social network media",
        "internet",
        "messaging",
        "search",
        "web hosting",
        "web development",
        "information technology",
        "developer apis",
        "developer tools",
        "big data",
        "payments",
        "marketplaces",
        "digital media",
        "cyber security",
        "cybersecurity",
        "artificial intelligence",
        "machine learning",
        "application platforms",
        "advertising platforms",
        "mobile commerce",
        "bitcoin",
        "paas",
        "iaas",
        "file sharing",
        "peer-to-peer",
        "virtualization",
        "predictive analytics",
        "business analytics",
        "data integration",
        "games",
        "video games",
        "online games",
        "online shopping",
    }
)

BANNED_FEATURES = ("funding_total_usd", "funding_rounds", "last_funding_at", "status")


@dataclass(frozen=True)
class Rule:
    code: str
    name: str
    antecedents: tuple[str, ...]
    cf: float
    mechanism: str


COLORS = {
    "Knowledge base": "#6B4C9A",
    "Logistic regression": "#1B9E77",
    "Decision tree": "#7570B3",
    "Gradient boosting": "#1F4E79",
    "Stacked fusion": "#B45309",
}

RULES = (
    Rule("R1", "Resource strain", ("thin_seed", "slow_funding"), 0.55, "Scaling a digital model without slack"),
    Rule("R2", "Unfunded scope", ("thin_seed", "broad_scope"), 0.45, "Category scope ahead of first-round capital"),
    Rule("R3", "Shallow digital base", ("thin_seed", "shallow_digital"), 0.35, "Thin capital and a peripheral digital position"),
    Rule("R4", "Isolated entry", ("thin_ecosystem", "slow_funding"), 0.30, "Slow funding in a thin regional pipeline"),
    Rule("R5", "Unfocused entry", ("broad_scope", "shallow_digital"), 0.25, "Broad scope without a digital core"),
    Rule("R6", "Resourced focus", ("adequate_seed", "narrow_scope", "deep_digital"), -0.50, "Capital, focus, and digital depth aligned"),
    Rule("R7", "Embedded start", ("dense_ecosystem", "timely_funding"), -0.30, "Timely funding inside an active ecosystem"),
)

def combine_cf(left: float, right: float) -> float:
    """Shortliffe-Buchanan certainty-factor combination."""
    a = float(np.clip(left, -0.99, 0.99))
    b = float(np.clip(right, -0.99, 0.99))
    if a >= 0 and b >= 0:
        return a + b * (1 - a)
    if a <= 0 and b <= 0:
        return a + b * (1 + a)
    return (a + b) / max(1e-6, 1 - min(abs(a), abs(b)))


def combine_arrays(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = np.clip(left, -0.99, 0.99)
    b = np.clip(right, -0.99, 0.99)
    out = np.empty_like(a, dtype=float)
    pos = (a >= 0) & (b >= 0)
    neg = (a <= 0) & (b <= 0)
    mix = ~(pos | neg)
    out[pos] = a[pos] + b[pos] * (1 - a[pos])
    out[neg] = a[neg] + b[neg] * (1 + a[neg])
    denom = np.maximum(1e-6, 1 - np.minimum(np.abs(a[mix]), np.abs(b[mix])))
    out[mix] = (a[mix] + b[mix]) / denom
    return np.clip(out, -0.99, 0.99)


def run_checks() -> None:
    assert abs(combine_cf(0.5, 0.4) - 0.7) < 1e-9
    assert abs(combine_cf(-0.2, -0.3) - (-0.44)) < 1e-9
    assert abs(combine_cf(0.0, 0.3) - 0.3) < 1e-9
    mixed = combine_arrays(np.array([0.6, -0.5]), np.array([-0.2, -0.2]))
    assert mixed.shape == (2,)
    assert np.all(np.abs(mixed) <= 0.99)
    counts = trailing_ecosystem(
        pd.Series(["A", "A", "A", "B", "A"]),
        pd.Series([2010, 2011, 2012, 2012, 2012]),
        window=3,
    )
    assert counts.tolist() == [0, 1, 2, 0, 2]


def download() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        path = CACHE / name
        if path.exists() and path.stat().st_size > 1_000_000:
            continue
        request = urllib.request.Request(url, headers={"User-Agent": "kbews-research"})
        with urllib.request.urlopen(request, timeout=120) as response:
            path.write_bytes(response.read())
        if path.stat().st_size < 1_000_000:
            raise SystemExit(f"Download looks incomplete: {path}")


def categories_of(value: object) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [part.strip().lower() for part in value.split("|") if part.strip()]


def is_digital(value: object) -> bool:
    return any(cat in DIGITAL for cat in categories_of(value))


def digital_depth(value: object) -> float:
    cats = categories_of(value)
    if not cats:
        return 0.0
    return sum(cat in DIGITAL for cat in cats) / len(cats)


def map_round(value: object) -> str:
    text = str(value or "").strip().lower()
    if text == "seed":
        return "seed"
    if text == "angel":
        return "angel"
    if text == "venture":
        return "venture"
    return "other"


def trailing_ecosystem(region: pd.Series, year: pd.Series, window: int = ECOSYSTEM_WINDOW) -> np.ndarray:
    """Count other ventures in the same region funded in [year-window, year-1]."""
    regions = region.fillna("unknown").astype(str).to_numpy()
    years = year.to_numpy()
    counts = np.zeros(len(years), dtype=int)
    buckets: dict[str, list[int]] = {}
    for pos, name in enumerate(regions):
        buckets.setdefault(name, []).append(pos)
    for positions in buckets.values():
        idx = np.asarray(positions)
        group_years = years[idx]
        order = np.argsort(group_years, kind="mergesort")
        ordered_idx = idx[order]
        ordered_years = group_years[order]
        for pos, current in enumerate(ordered_years):
            left = int(np.searchsorted(ordered_years, current - window, side="left"))
            right = int(np.searchsorted(ordered_years, current, side="left"))
            counts[ordered_idx[pos]] = int(right - left)
    return counts


def build_table() -> tuple[pd.DataFrame, dict]:
    companies = pd.read_csv(CACHE / "companies.csv", low_memory=False)
    rounds = pd.read_csv(CACHE / "rounds.csv", low_memory=False)
    flow = {"companies": int(len(companies)), "rounds": int(len(rounds))}

    rounds = rounds.dropna(subset=["company_permalink", "funded_at"]).copy()
    rounds["funded_at"] = pd.to_datetime(rounds["funded_at"], errors="coerce")
    rounds["raised_amount_usd"] = pd.to_numeric(rounds["raised_amount_usd"], errors="coerce")
    rounds = rounds.dropna(subset=["funded_at"])
    rounds = rounds[rounds["raised_amount_usd"] > 0]
    rounds = rounds.sort_values(["company_permalink", "funded_at", "raised_amount_usd"])
    first = rounds.groupby("company_permalink", as_index=False).head(1)
    first = first.rename(
        columns={
            "company_permalink": "permalink",
            "funded_at": "first_round_at",
            "raised_amount_usd": "first_round_usd",
            "funding_round_type": "round_type_raw",
        }
    )
    keep = [
        "permalink",
        "first_round_at",
        "first_round_usd",
        "round_type_raw",
    ]
    table = companies.merge(first[keep], on="permalink", how="inner")
    flow["with_priced_first_round"] = int(len(table))

    table["digital"] = table["category_list"].map(is_digital)
    table = table[table["digital"]].copy()
    flow["digital"] = int(len(table))

    table["founded_at"] = pd.to_datetime(table["founded_at"], errors="coerce")
    table["first_year"] = table["first_round_at"].dt.year
    table["days_to_funding"] = (table["first_round_at"] - table["founded_at"]).dt.days
    table = table.dropna(subset=["founded_at", "days_to_funding", "first_year"])
    table = table[table["days_to_funding"] >= -30]
    table.loc[table["days_to_funding"] < 0, "days_to_funding"] = 0
    table = table[table["days_to_funding"] <= MAX_YEARS_TO_FUNDING * 365]
    flow["dated_new_ventures"] = int(len(table))

    # Ecosystem includes still-operating digital ventures. The warning label does not.
    table["ecosystem"] = trailing_ecosystem(table["region"], table["first_year"].astype(int))
    table["n_categories"] = table["category_list"].map(lambda v: len(categories_of(v))).astype(float)
    table["digital_depth"] = table["category_list"].map(digital_depth).astype(float)
    table["log_first_usd"] = np.log(table["first_round_usd"].astype(float))
    table["log1p_days"] = np.log1p(table["days_to_funding"].astype(float))
    table["log1p_ecosystem"] = np.log1p(table["ecosystem"].astype(float))
    table["round_type"] = table["round_type_raw"].map(map_round)
    table["country"] = table["country_code"].fillna("missing").astype(str)
    table["failure"] = np.where(
        table["status"].str.lower() == "closed",
        1,
        np.where(table["status"].str.lower().isin(["acquired", "ipo"]), 0, np.nan),
    )
    flow["resolved"] = int(table["failure"].notna().sum())
    labeled = table[table["failure"].notna() & table["first_year"].between(2006, 2013)].copy()
    labeled["failure"] = labeled["failure"].astype(int)
    flow["analysis_sample"] = int(len(labeled))
    flow["failure_rate"] = float(labeled["failure"].mean())
    return labeled.reset_index(drop=True), flow


def split_years(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    parts = {
        "train": frame[frame["first_year"].isin(TRAIN_YEARS)].copy(),
        "val": frame[frame["first_year"].isin(VAL_YEARS)].copy(),
        "test": frame[frame["first_year"].isin(TEST_YEARS)].copy(),
    }
    for name, part in parts.items():
        if part["failure"].nunique() < 2 or len(part) < 80:
            raise SystemExit(f"{name} split is too small or single-class: n={len(part)}")
    if parts["train"]["first_year"].max() >= parts["val"]["first_year"].min():
        raise SystemExit("Train years are not strictly before validation.")
    if parts["val"]["first_year"].max() >= parts["test"]["first_year"].min():
        raise SystemExit("Validation years are not strictly before the test years.")
    return parts


def _band(series: pd.Series, low_q: float, high_q: float) -> tuple[float, float]:
    lo = float(series.quantile(low_q))
    hi = float(series.quantile(high_q))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def fit_cuts(train: pd.DataFrame) -> dict[str, tuple[float, float]]:
    return {
        "log_first_usd": _band(train["log_first_usd"], 0.25, 0.60),
        "log1p_days": _band(train["log1p_days"], 0.40, 0.75),
        "n_categories": _band(train["n_categories"], 0.40, 0.75),
        "log1p_ecosystem": _band(train["log1p_ecosystem"], 0.25, 0.60),
        "shallow_digital": (0.40, 0.80),
        "deep_digital": (0.50, 0.90),
    }


def low_membership(values: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    lo, hi = band
    return np.clip((hi - values) / (hi - lo), 0, 1)


def high_membership(values: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    lo, hi = band
    return np.clip((values - lo) / (hi - lo), 0, 1)


def memberships(frame: pd.DataFrame, cuts: dict[str, tuple[float, float]]) -> pd.DataFrame:
    thin_seed = low_membership(frame["log_first_usd"].to_numpy(), cuts["log_first_usd"])
    slow = high_membership(frame["log1p_days"].to_numpy(), cuts["log1p_days"])
    broad = high_membership(frame["n_categories"].to_numpy(), cuts["n_categories"])
    thin_eco = low_membership(frame["log1p_ecosystem"].to_numpy(), cuts["log1p_ecosystem"])
    shallow = low_membership(frame["digital_depth"].to_numpy(), cuts["shallow_digital"])
    deep = high_membership(frame["digital_depth"].to_numpy(), cuts["deep_digital"])
    return pd.DataFrame(
        {
            "thin_seed": thin_seed,
            "adequate_seed": 1 - thin_seed,
            "slow_funding": slow,
            "timely_funding": 1 - slow,
            "broad_scope": broad,
            "narrow_scope": 1 - broad,
            "thin_ecosystem": thin_eco,
            "dense_ecosystem": 1 - thin_eco,
            "shallow_digital": shallow,
            "deep_digital": deep,
        },
        index=frame.index,
    )


def infer(
    frame: pd.DataFrame,
    cuts: dict[str, tuple[float, float]],
    rules: tuple[Rule, ...] | list[Rule] = RULES,
) -> tuple[np.ndarray, pd.DataFrame]:
    degrees = memberships(frame, cuts)
    strength = pd.DataFrame(index=frame.index)
    cf = np.zeros(len(frame), dtype=float)
    for rule in rules:
        antecedent = np.minimum.reduce([degrees[name].to_numpy() for name in rule.antecedents])
        strength[rule.code] = antecedent
        cf = combine_arrays(cf, rule.cf * antecedent)
    return cf, strength


def revise_knowledge_base(
    train: pd.DataFrame, cuts: dict[str, tuple[float, float]], margin: float = 0.02
) -> tuple[list[Rule], list[dict]]:
    """Drop rules whose training-period direction contradicts the stated mechanism.

    Positive rules must be stronger among later closures. Negative rules must be
    stronger among later exits. The test years are not read here.
    """
    _, strength = infer(train, cuts, RULES)
    strength = strength.reset_index(drop=True)
    labels = train["failure"].to_numpy()
    kept: list[Rule] = []
    audit: list[dict] = []
    for rule in RULES:
        values = strength[rule.code].to_numpy()
        fail = float(values[labels == 1].mean())
        exit_mean = float(values[labels == 0].mean())
        keep = (fail >= exit_mean + margin) if rule.cf > 0 else (exit_mean >= fail + margin)
        audit.append(
            {
                "code": rule.code,
                "name": rule.name,
                "cf": rule.cf,
                "train_mean_failure": fail,
                "train_mean_exit": exit_mean,
                "retained": bool(keep),
            }
        )
        if keep:
            kept.append(rule)
    if len(kept) < 2:
        raise SystemExit("Training revision retained fewer than two rules.")
    return kept, audit


def design_matrix(parts: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], list[str]]:
    train = parts["train"]
    top_countries = train["country"].value_counts().head(8).index.tolist()
    columns: list[str] | None = None
    matrices: dict[str, pd.DataFrame] = {}
    numeric = ["log_first_usd", "log1p_days", "n_categories", "digital_depth", "log1p_ecosystem"]
    for name, part in parts.items():
        country = np.where(part["country"].isin(top_countries), part["country"], "OTHER")
        dummies = pd.get_dummies(
            pd.DataFrame({"round_type": part["round_type"].to_numpy(), "country": country}),
            columns=["round_type", "country"],
            dtype=float,
        )
        # Reference levels: venture rounds and the United States, when present.
        for reference in ("round_type_venture", "country_USA"):
            if reference in dummies.columns:
                dummies = dummies.drop(columns=reference)
        matrix = pd.concat(
            [part[numeric].reset_index(drop=True), dummies.reset_index(drop=True)],
            axis=1,
        )
        if columns is None:
            columns = matrix.columns.tolist()
        matrices[name] = matrix.reindex(columns=columns, fill_value=0.0)
    banned = [col for col in columns or [] if any(flag in col for flag in BANNED_FEATURES)]
    if banned:
        raise SystemExit(f"Leaking columns in the design matrix: {banned}")
    return matrices, columns or []


def fit_platt(scores: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(C=1.0, solver="lbfgs")
    model.fit(scores.reshape(-1, 1), labels)
    return model


def platt_predict(model: LogisticRegression, scores: np.ndarray) -> np.ndarray:
    return model.predict_proba(scores.reshape(-1, 1))[:, 1]


def logit(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def topk_capture(labels: np.ndarray, scores: np.ndarray, rate: float) -> dict[str, float]:
    """Precision and recall when the highest-scoring share of cases is reviewed."""
    n_alert = max(1, int(round(rate * len(labels))))
    chosen = np.argpartition(scores, -n_alert)[-n_alert:]
    hits = float(labels[chosen].sum())
    return {
        "alert_rate": n_alert / len(labels),
        "n_alert": n_alert,
        "precision": hits / n_alert,
        "recall": hits / float(labels.sum()) if labels.sum() else 0.0,
    }


def bootstrap_precision(labels: np.ndarray, scores: np.ndarray, rate: float, draws: int = 1000) -> dict[str, float]:
    point = topk_capture(labels, scores, rate)
    rng = np.random.default_rng(SEED + int(rate * 100))
    estimates = []
    n = len(labels)
    for _ in range(draws):
        idx = rng.integers(0, n, n)
        if labels[idx].sum() == 0:
            continue
        estimates.append(topk_capture(labels[idx], scores[idx], rate)["precision"])
    arr = np.asarray(estimates, dtype=float)
    return {
        **point,
        "precision_ci_low": float(np.percentile(arr, 2.5)),
        "precision_ci_high": float(np.percentile(arr, 97.5)),
    }


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def bootstrap_auc(labels: np.ndarray, scores: np.ndarray, draws: int = 1000) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    estimates = []
    n = len(labels)
    for _ in range(draws):
        idx = rng.integers(0, n, n)
        if len(np.unique(labels[idx])) < 2:
            continue
        estimates.append(roc_auc_score(labels[idx], scores[idx]))
    arr = np.asarray(estimates, dtype=float)
    return {
        "auc": safe_auc(labels, scores),
        "ci_low": float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
    }


def paired_contrast(
    labels: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    metric: str,
    draws: int = 2000,
    rate: float = 0.30,
) -> dict[str, float]:
    """Two-sided paired bootstrap test of left minus right on the same ventures."""
    if metric == "roc_auc":
        point = safe_auc(labels, left) - safe_auc(labels, right)
    elif metric == "precision_at_30":
        point = topk_capture(labels, left, rate)["precision"] - topk_capture(labels, right, rate)["precision"]
    else:
        raise ValueError(metric)
    rng = np.random.default_rng(SEED + 11 + (0 if metric == "roc_auc" else 17))
    diffs = []
    n = len(labels)
    for _ in range(draws):
        idx = rng.integers(0, n, n)
        if len(np.unique(labels[idx])) < 2:
            continue
        if metric == "roc_auc":
            diffs.append(roc_auc_score(labels[idx], left[idx]) - roc_auc_score(labels[idx], right[idx]))
        else:
            diffs.append(
                topk_capture(labels[idx], left[idx], rate)["precision"]
                - topk_capture(labels[idx], right[idx], rate)["precision"]
            )
    arr = np.asarray(diffs, dtype=float)
    # Two-sided p from the share of resamples on each side of zero.
    share_low = float(np.mean(arr <= 0))
    share_high = float(np.mean(arr >= 0))
    p_value = min(1.0, 2.0 * min(share_low, share_high))
    return {
        "difference": float(point),
        "ci_low": float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
        "p_value": p_value,
        "draws": int(len(arr)),
    }


def paired_table(fitted: dict) -> list[dict]:
    y = fitted["y"]["test"]
    issued = fitted["calibrated"]["Gradient boosting"]["test"]
    rows = []
    for other in ("Knowledge base", "Logistic regression", "Decision tree", "Stacked fusion"):
        rival = fitted["calibrated"][other]["test"]
        for metric in ("roc_auc", "precision_at_30"):
            contrast = paired_contrast(y, issued, rival, metric)
            rows.append({"comparison": f"Gradient boosting vs {other}", "metric": metric, **contrast})
    return rows


def budget_profiles(fitted: dict, rates: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)) -> dict:
    y = fitted["y"]["test"]
    profiles = {"base_rate": float(y.mean()), "rates": list(rates), "systems": {}}
    for name, probs in fitted["calibrated"].items():
        profiles["systems"][name] = [
            {"rate": rate, "precision": topk_capture(y, probs["test"], rate)["precision"], "recall": topk_capture(y, probs["test"], rate)["recall"]}
            for rate in rates
        ]
    return profiles


def margin_sensitivity(parts: dict[str, pd.DataFrame], cuts: dict) -> list[dict]:
    """Knowledge-base sensitivity to the retention margin. The booster is not refit."""
    y_val = parts["val"]["failure"].to_numpy()
    y_test = parts["test"]["failure"].to_numpy()
    rows = []
    for margin in (0.0, 0.02, 0.05):
        kept, _audit = revise_knowledge_base(parts["train"], cuts, margin=margin)
        cf_val, _ = infer(parts["val"], cuts, kept)
        cf_test, _ = infer(parts["test"], cuts, kept)
        mapper = fit_platt(cf_val, y_val)
        probs = platt_predict(mapper, cf_test)
        rows.append(
            {
                "margin": margin,
                "n_rules": len(kept),
                "rules": [rule.code for rule in kept],
                "roc_auc": safe_auc(y_test, probs),
                "precision_at_30": topk_capture(y_test, probs, 0.30)["precision"],
            }
        )
    return rows


def bootstrap_difference(labels: np.ndarray, left: np.ndarray, right: np.ndarray, draws: int = 1000) -> dict[str, float]:
    rng = np.random.default_rng(SEED + 1)
    diffs = []
    n = len(labels)
    for _ in range(draws):
        idx = rng.integers(0, n, n)
        if len(np.unique(labels[idx])) < 2:
            continue
        diffs.append(roc_auc_score(labels[idx], left[idx]) - roc_auc_score(labels[idx], right[idx]))
    arr = np.asarray(diffs, dtype=float)
    point = safe_auc(labels, left) - safe_auc(labels, right)
    return {
        "difference": float(point),
        "ci_low": float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
    }


def fit_models(
    parts: dict[str, pd.DataFrame],
    matrices: dict[str, pd.DataFrame],
    cuts: dict,
    rules: list[Rule],
    columns: list[str],
) -> dict:
    y = {name: part["failure"].to_numpy() for name, part in parts.items()}
    x = {name: matrices[name].to_numpy(dtype=float) for name in parts}
    cf = {}
    strength = {}
    full_strength = {}
    for name, part in parts.items():
        cf[name], strength[name] = infer(part, cuts, rules)
        _, full_strength[name] = infer(part, cuts, RULES)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x["train"])
    logistic = LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=500)
    logistic.fit(x_train, y["train"])
    tree = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=40,
        class_weight="balanced",
        random_state=SEED,
    )
    tree.fit(x["train"], y["train"])
    boosting = HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.06,
        max_iter=400,
        min_samples_leaf=40,
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        class_weight="balanced",
        random_state=SEED,
    )
    boosting.fit(x["train"], y["train"])

    raw = {
        "Knowledge base": cf,
        "Logistic regression": {
            "train": logistic.predict_proba(x_train)[:, 1],
            "val": logistic.predict_proba(scaler.transform(x["val"]))[:, 1],
            "test": logistic.predict_proba(scaler.transform(x["test"]))[:, 1],
        },
        "Decision tree": {
            name: tree.predict_proba(x[name])[:, 1] for name in ("train", "val", "test")
        },
        "Gradient boosting": {
            name: boosting.predict_proba(x[name])[:, 1] for name in ("train", "val", "test")
        },
    }
    # Certainty factors are scores in [-1, 1], not probabilities. Platt maps them.
    calibrated = {}
    platt = {}
    for name in ("Knowledge base", "Logistic regression", "Decision tree", "Gradient boosting"):
        platt[name] = fit_platt(raw[name]["val"], y["val"])
        calibrated[name] = {
            split: platt_predict(platt[name], raw[name][split]) for split in ("val", "test")
        }

    fusion = LogisticRegression(C=1.0, solver="lbfgs")
    fusion_val = np.column_stack([logit(raw["Gradient boosting"]["val"]), cf["val"]])
    fusion.fit(fusion_val, y["val"])
    fusion_test = np.column_stack([logit(raw["Gradient boosting"]["test"]), cf["test"]])
    calibrated["Stacked fusion"] = {
        "val": fusion.predict_proba(fusion_val)[:, 1],
        "test": fusion.predict_proba(fusion_test)[:, 1],
    }
    return {
        "y": y,
        "calibrated": calibrated,
        "cf_test": cf["test"],
        "strength_test": full_strength["test"],
        "retained_codes": [rule.code for rule in rules],
        "boosting_iterations": int(boosting.n_iter_),
        "fusion_intercept": float(fusion.intercept_[0]),
        "fusion_coef_logit_hgb": float(fusion.coef_[0, 0]),
        "fusion_coef_cf": float(fusion.coef_[0, 1]),
        "tree_depth": int(tree.get_depth()),
        "logistic_coefficients": [
            {"feature": feature, "standardized_coef": float(coef)}
            for feature, coef in sorted(zip(columns, logistic.coef_[0]), key=lambda item: -abs(item[1]))
        ],
    }


def evaluate(fitted: dict) -> list[dict]:
    rows = []
    y_test = fitted["y"]["test"]
    for name in ("Knowledge base", "Logistic regression", "Decision tree", "Gradient boosting", "Stacked fusion"):
        test_prob = fitted["calibrated"][name]["test"]
        auc = bootstrap_auc(y_test, test_prob)
        at_20 = bootstrap_precision(y_test, test_prob, 0.20)
        at_30 = bootstrap_precision(y_test, test_prob, 0.30)
        rows.append(
            {
                "system": name,
                "roc_auc": auc["auc"],
                "roc_auc_ci_low": auc["ci_low"],
                "roc_auc_ci_high": auc["ci_high"],
                "pr_auc": float(average_precision_score(y_test, test_prob)),
                "brier": float(brier_score_loss(y_test, test_prob)),
                "precision_at_20": at_20["precision"],
                "precision_at_20_ci_low": at_20["precision_ci_low"],
                "precision_at_20_ci_high": at_20["precision_ci_high"],
                "recall_at_20": at_20["recall"],
                "precision_at_30": at_30["precision"],
                "precision_at_30_ci_low": at_30["precision_ci_low"],
                "precision_at_30_ci_high": at_30["precision_ci_high"],
                "recall_at_30": at_30["recall"],
                "alerts_at_30": at_30["n_alert"],
            }
        )
    return rows


def sample_block(parts: dict[str, pd.DataFrame]) -> dict:
    block = {}
    for name, part in parts.items():
        block[name] = {
            "n": int(len(part)),
            "failures": int(part["failure"].sum()),
            "failure_rate": float(part["failure"].mean()),
            "year_min": int(part["first_year"].min()),
            "year_max": int(part["first_year"].max()),
            "median_first_round_usd": float(part["first_round_usd"].median()),
            "median_days_to_funding": float(part["days_to_funding"].median()),
            "median_categories": float(part["n_categories"].median()),
            "median_digital_depth": float(part["digital_depth"].median()),
            "median_ecosystem": float(part["ecosystem"].median()),
        }
    return block


def write_warning_cards(test: pd.DataFrame, fitted: dict, path: Path) -> dict:
    """One card per held-out venture: issued probability, fired rules, review flag.

    later_closed is the outcome used to audit the queue. It is not a model input
    and would not be known when the card is issued.
    """
    frame = test.reset_index(drop=True)
    strength = fitted["strength_test"].reset_index(drop=True)
    prob = np.asarray(fitted["calibrated"]["Gradient boosting"]["test"], dtype=float)
    labels = np.asarray(fitted["y"]["test"])
    if not (len(frame) == len(strength) == len(prob) == len(labels)):
        raise SystemExit("Warning-card rows are not aligned with the test cohort.")
    retained = [rule for rule in RULES if rule.code in set(fitted["retained_codes"])]
    n_alert = max(1, int(round(0.30 * len(frame))))
    chosen = np.argpartition(prob, -n_alert)[-n_alert:]
    in_queue = np.zeros(len(frame), dtype=bool)
    in_queue[chosen] = True
    order = np.lexsort((frame["permalink"].astype(str).to_numpy(), -prob))
    rank = np.empty(len(frame), dtype=int)
    rank[order] = np.arange(1, len(frame) + 1)
    fired_text = []
    for i in range(len(frame)):
        parts_fired = []
        for rule in retained:
            value = float(strength.at[i, rule.code])
            if value >= 0.5:
                parts_fired.append(f"{rule.code} {rule.name} ({value:.2f})")
        fired_text.append("; ".join(parts_fired))
    cards = pd.DataFrame(
        {
            "permalink": frame["permalink"].astype(str),
            "name": frame["name"].astype(str),
            "first_year": frame["first_year"].astype(int),
            "closure_probability": np.round(prob, 6),
            "queue_rank": rank,
            "in_review_queue": in_queue.astype(int),
            "fired_rules": fired_text,
            "later_closed": labels.astype(int),
        }
    )
    cards = cards.sort_values(["queue_rank", "permalink"], kind="mergesort").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    cards.to_csv(path, index=False)
    queue = cards[cards["in_review_queue"] == 1]
    issued = topk_capture(labels, prob, 0.30)
    summary = {
        "file": "outputs/warning_cards.csv",
        "n_cards": int(len(cards)),
        "n_review": int(len(queue)),
        "queue_precision": float(queue["later_closed"].mean()) if len(queue) else None,
        "cards_with_a_fired_rule": int((cards["fired_rules"] != "").sum()),
    }
    if summary["n_review"] != int(issued["n_alert"]) or abs(summary["queue_precision"] - issued["precision"]) > 1e-12:
        raise SystemExit("Warning-card queue does not match the 30% budget used in Table 1.")
    return summary


def rule_firing(labels: np.ndarray, strength: pd.DataFrame, retained: list[str]) -> list[dict]:
    rows = []
    failed = labels == 1
    survived = ~failed
    strength = strength.reset_index(drop=True)
    for rule in RULES:
        values = strength[rule.code].to_numpy()
        fired = values >= 0.5
        rows.append(
            {
                "code": rule.code,
                "name": rule.name,
                "cf": rule.cf,
                "antecedents": " AND ".join(rule.antecedents),
                "mechanism": rule.mechanism,
                "retained": rule.code in retained,
                "fire_rate_failure": float(fired[failed].mean()) if failed.any() else None,
                "fire_rate_exit": float(fired[survived].mean()) if survived.any() else None,
                "mean_strength_failure": float(values[failed].mean()) if failed.any() else None,
                "mean_strength_exit": float(values[survived].mean()) if survived.any() else None,
            }
        )
    return rows


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def _curve_order() -> list[str]:
    return ["Knowledge base", "Decision tree", "Logistic regression", "Stacked fusion", "Gradient boosting"]


def draw_roc(path: Path, fitted: dict, rows: list[dict]) -> None:
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(6.4, 5.3))
    y = fitted["y"]["test"]
    by_name = {row["system"]: row for row in rows}
    for name in _curve_order():
        scores = fitted["calibrated"][name]["test"]
        fpr, tpr, _ = roc_curve(y, scores)
        issued = name == "Gradient boosting"
        ax.plot(
            fpr,
            tpr,
            color=COLORS[name],
            lw=2.4 if issued else 1.35,
            ls="--" if name == "Stacked fusion" else "-",
            label=f"{name}  {by_name[name]['roc_auc']:.3f}",
            zorder=3 if issued else 2,
        )
    ax.plot([0, 1], [0, 1], color="#94A3B8", lw=0.8, ls=":")
    style_axes(ax)
    ax.set_xlabel("False-alarm rate")
    ax.set_ylabel("Closure recall")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.set_title("Held-out ROC, first rounds in 2012–2013", loc="left", fontsize=11, pad=8)
    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=400, facecolor="white")
    plt.close(fig)


def draw_budget(path: Path, profiles: dict) -> None:
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "axes.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    for name in _curve_order():
        series = profiles["systems"][name]
        xs = [100 * item["rate"] for item in series]
        ys = [item["precision"] for item in series]
        issued = name == "Gradient boosting"
        ax.plot(
            xs,
            ys,
            color=COLORS[name],
            lw=2.4 if issued else 1.35,
            ls="--" if name == "Stacked fusion" else "-",
            marker="o",
            markersize=4.2 if issued else 3.2,
            label=name,
            zorder=3 if issued else 2,
        )
    ax.axhline(profiles["base_rate"], color="#64748B", lw=0.9, ls=":", label=f"Base rate  {profiles['base_rate']:.3f}")
    style_axes(ax)
    ax.set_xlabel("Review budget (% of test ventures)")
    ax.set_ylabel("Precision among reviewed ventures")
    ax.set_xlim(8, 52)
    ax.set_ylim(0.45, 0.95)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("Precision at a fixed review budget", loc="left", fontsize=11, pad=8)
    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=400, facecolor="white")
    plt.close(fig)


def draw_rules(path: Path, firing: list[dict]) -> None:
    """Held-out firing rates. Retention was fixed before these years were scored."""
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "axes.linewidth": 0.8})
    order = ["R7", "R5", "R4", "R6", "R3", "R2", "R1"]
    by_code = {row["code"]: row for row in firing}
    rows = [by_code[code] for code in order]
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    y = np.arange(len(rows))
    height = 0.36
    closure = [row["fire_rate_failure"] for row in rows]
    exit_rate = [row["fire_rate_exit"] for row in rows]
    ax.barh(y + height / 2, closure, height=height, color="#1F4E79", label="Later closure", zorder=2)
    ax.barh(y - height / 2, exit_rate, height=height, color="#1B9E77", label="Recorded exit", zorder=2)
    ax.axhline(2.5, color="#94A3B8", lw=0.7, ls="--")
    labels = []
    for row in rows:
        status = "retained" if row["retained"] else "withdrawn"
        labels.append(f"{row['code']}  {row['name']}  ({status})")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 0.62)
    ax.set_xlabel("Share of ventures with rule strength at least 0.5")
    style_axes(ax)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.55, -0.16), ncol=2, fontsize=8)
    ax.set_title("Held-out rule firing, first rounds in 2012–2013", loc="left", fontsize=11, pad=8)
    fig.tight_layout(pad=0.45)
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(path, dpi=400, facecolor="white")
    plt.close(fig)


def json_ready(value):
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def main() -> None:
    run_checks()
    download()
    table, flow = build_table()
    parts = split_years(table)
    cuts = fit_cuts(parts["train"])
    retained, revision = revise_knowledge_base(parts["train"], cuts)
    matrices, columns = design_matrix(parts)
    fitted = fit_models(parts, matrices, cuts, retained, columns)
    rows = evaluate(fitted)
    firing = rule_firing(fitted["y"]["test"], fitted["strength_test"], fitted["retained_codes"])
    contrasts = paired_table(fitted)
    profiles = budget_profiles(fitted)
    sensitivity = margin_sensitivity(parts, cuts)
    contrast = bootstrap_difference(
        fitted["y"]["test"],
        fitted["calibrated"]["Stacked fusion"]["test"],
        fitted["calibrated"]["Gradient boosting"]["test"],
    )
    OUT.mkdir(parents=True, exist_ok=True)
    cards = write_warning_cards(parts["test"], fitted, OUT / "warning_cards.csv")
    draw_roc(OUT / "fig1_roc.png", fitted, rows)
    draw_budget(OUT / "fig2_budget.png", profiles)
    draw_rules(OUT / "fig3_rules.png", firing)
    payload = {
        "data": {
            "source": "Crunchbase Data Export, 4 December 2015",
            "redistribution": f"notpeter/crunchbase-data@{COMMIT}",
            "license": "CC-BY-NC-4.0",
            "decision_time": "first funding round with a positive disclosed amount",
            "label": "closed=1; acquired or ipo=0; operating excluded",
            "cohort": "digital-category ventures funded within 10 years of founding",
            "years": {"train": "2006-2010", "validation": "2011", "test": "2012-2013"},
            "features": columns,
            "excluded_because_they_leak_outcome": list(BANNED_FEATURES),
            "flow": flow,
            "sample": sample_block(parts),
            "cuts": {key: {"lo": lo, "hi": hi} for key, (lo, hi) in cuts.items()},
            "knowledge_revision": revision,
        },
        "model": {
            "boosting_iterations": fitted["boosting_iterations"],
            "tree_depth": fitted["tree_depth"],
            "fusion_intercept": fitted["fusion_intercept"],
            "fusion_coef_logit_hgb": fitted["fusion_coef_logit_hgb"],
            "fusion_coef_cf": fitted["fusion_coef_cf"],
            "kbews_minus_boosting_auc": contrast,
            "logistic_coefficients": fitted["logistic_coefficients"],
            "retention_margin_sensitivity": sensitivity,
        },
        "rules": firing,
        "warning_cards": cards,
        "performance": rows,
        "paired_tests": contrasts,
        "budget_profiles": profiles,
    }
    (OUT / "metrics.json").write_text(json.dumps(json_ready(payload), indent=2))
    print(json.dumps(json_ready({"performance": rows, "paired_tests": contrasts, "sensitivity": sensitivity}), indent=2))


if __name__ == "__main__":
    main()
