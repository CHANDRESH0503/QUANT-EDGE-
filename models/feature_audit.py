# models/feature_audit.py
# Audits per-feature predictive power against the 3-class label.
#
# For each feature, computes:
#   - Information Value (IV) against the LONG/FLAT/SHORT label
#   - Mean, std, non-zero fraction
#   - Correlation with forward returns
#
# IV interpretation:
#   < 0.02 = not predictive — consider dropping
#   < 0.10 = weak
#   < 0.30 = medium
#   > 0.30 = strong (verify it isn't leakage!)
#
# Run: python -m models.feature_audit --model=swing --window=recent_with_features

import argparse
import logging
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Features that Phase 1-4 unlocks (14 priority-1 features)
PRIORITY_1_FEATURES = [
    # Options
    "pcr_norm", "pcr_roc_norm", "max_pain_dist_norm",
    "iv_percentile_norm", "implied_move_norm",
    "call_wall_dist_norm", "put_wall_dist_norm",
    # FII / DII flow
    "fii_5d_norm", "fii_signal", "dii_3d_norm",
    "flow_confluence", "institutional_bias",
    # FinBERT
    "finbert_score_24h", "finbert_momentum", "finbert_news_spike",
]


def _information_value(
    feature: pd.Series,
    label:   pd.Series,
    n_bins:  int = 10,
) -> float:
    """
    IV for a 3-class label: treat LONG (2) as 'good', SHORT (0) as 'bad',
    ignore FLAT (1). This is the standard binary-IV adaptation that works
    well for direction-classification with a neutral class.
    """
    mask = label != 1
    if mask.sum() < 50:
        return 0.0

    x = feature[mask].astype(float).fillna(0.0)
    y = (label[mask] == 2).astype(int)   # 1=LONG, 0=SHORT

    if x.std() == 0:
        return 0.0

    try:
        bins = pd.qcut(x, q=n_bins, duplicates="drop")
    except ValueError:
        return 0.0

    total_good = max(y.sum(), 1)
    total_bad  = max((1 - y).sum(), 1)

    iv = 0.0
    for _, idx in y.groupby(bins, observed=False).groups.items():
        if len(idx) == 0:
            continue
        good = y.loc[idx].sum()
        bad  = (1 - y.loc[idx]).sum()
        pg = (good + 0.5) / (total_good + 0.5)   # Laplace smoothing
        pb = (bad  + 0.5) / (total_bad  + 0.5)
        iv += (pg - pb) * np.log(pg / pb)
    return float(iv)


def _grade_iv(iv: float) -> str:
    if iv < 0.02: return "DROP"
    if iv < 0.10: return "WEAK"
    if iv < 0.30: return "MEDIUM"
    if iv < 0.60: return "STRONG"
    return "SUSPICIOUS"   # likely leakage


def audit(
    model_type:  str = "swing",
    data_window: str = "recent_with_features",
    db_path:     str = "database/trading.db",
    ticker:      str = "HDFCBANK.NS",
) -> Dict:
    from data.price_fetcher       import PriceFetcher
    from features.feature_builder import FeatureBuilder

    pf = PriceFetcher(ticker)
    fb = FeatureBuilder(ticker=ticker, db_path=db_path)

    if model_type == "intraday":
        df = pf.get_intraday(interval="5m")
        fwd_days, threshold = 6, 0.004
    elif model_type == "positional":
        df = pf.get_weekly(start="2000-01-01")
        fwd_days, threshold = 3, 0.05
    else:
        df = pf.get_daily(start="2000-01-01")
        fwd_days, threshold = 5, 0.025

    if df.empty:
        logger.error("No price data — aborting audit")
        return {}

    logger.info(f"Building {model_type} training set (window={data_window})...")
    X, y = fb.build_training_dataset(
        df, model_type=model_type,
        forward_days=fwd_days, threshold=threshold,
        data_window=data_window,
    )
    if X.empty:
        logger.error("Empty feature matrix — check data backfill")
        return {}

    forward = df["Close"].pct_change(fwd_days).shift(-fwd_days)
    # Align forward returns with the X rows. build_training_dataset starts at
    # i = min_hist (60) and ends at len(df) - forward_days.
    min_hist = 60
    fwd_aligned = forward.iloc[min_hist : min_hist + len(X)].reset_index(drop=True)

    report_rows: List[Dict] = []
    for feat in X.columns:
        col = X[feat].astype(float)
        iv = _information_value(col, y)
        nz_pct = float((col != 0).mean())
        corr   = float(col.corr(fwd_aligned)) if nz_pct > 0.01 else 0.0
        row = {
            "feature":   feat,
            "iv":        round(iv, 4),
            "grade":     _grade_iv(iv),
            "mean":      round(float(col.mean()), 4),
            "std":       round(float(col.std()),  4),
            "non_zero":  round(nz_pct, 3),
            "fwd_corr":  round(corr if not np.isnan(corr) else 0.0, 4),
            "priority1": feat in PRIORITY_1_FEATURES,
        }
        report_rows.append(row)

    report_df = pd.DataFrame(report_rows).sort_values("iv", ascending=False)

    print("\n" + "=" * 80)
    print(f"FEATURE AUDIT — {model_type} | window={data_window} | n_samples={len(X)}")
    print("=" * 80)
    print(report_df.to_string(index=False))

    p1 = report_df[report_df["priority1"]]
    print("\n" + "-" * 80)
    print(f"PRIORITY-1 features ({len(p1)} present, target >= 14):")
    print("-" * 80)
    print(p1.to_string(index=False))

    not_present = set(PRIORITY_1_FEATURES) - set(report_df["feature"])
    if not_present:
        print(f"\n⚠ Missing P1 features in training matrix: {sorted(not_present)}")

    weak = p1[p1["iv"] < 0.02]
    if len(weak):
        print(f"\n⚠ P1 features with IV < 0.02 (low signal — verify data backfill):")
        print(weak[["feature", "iv", "non_zero", "std"]].to_string(index=False))

    return {
        "model_type":  model_type,
        "data_window": data_window,
        "n_samples":   len(X),
        "n_features":  len(report_df),
        "report":      report_df.to_dict(orient="records"),
    }


def _cli() -> None:
    p = argparse.ArgumentParser(description="Audit per-feature IV")
    p.add_argument("--model",  default="swing",
                   choices=["swing", "intraday", "positional"])
    p.add_argument("--window", default="recent_with_features",
                   choices=["full_history", "recent_with_features"])
    p.add_argument("--db",     default="database/trading.db")
    p.add_argument("--ticker", default="HDFCBANK.NS")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    audit(args.model, args.window, args.db, args.ticker)


if __name__ == "__main__":
    _cli()
