# notebooks/02_feature_analysis.py
# Deep dive into the 90 features — distributions, correlations, predictive power
# Run this BEFORE training to validate your features are working correctly

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from data.price_fetcher          import PriceFetcher
from processing.technical        import TechnicalProcessor
from features.technical_features import TechnicalFeatures

print("=" * 60)
print("QUANT EDGE — Feature Analysis")
print("Validating 90 features for ML model quality")
print("=" * 60)

# ── 1. Build Features ─────────────────────────────────────────
print("\n[1] Building features on 5 years of data...")
fetcher  = PriceFetcher("HDFCBANK.NS")
df       = fetcher.get_daily(start="2020-01-01")
proc     = TechnicalProcessor()
feat_obj = TechnicalFeatures()
tech_df  = proc.build_features(df)
print(f"  ✅ Built {len(tech_df.columns)} technical features on {len(tech_df)} bars")

# ── 2. Feature Completeness ────────────────────────────────────
print("\n[2] FEATURE COMPLETENESS CHECK")
nan_pcts = tech_df.isnull().mean() * 100
high_nan = nan_pcts[nan_pcts > 5]
if high_nan.empty:
    print("  ✅ All features have < 5% NaN values")
else:
    print(f"  ⚠️  Features with high NaN%:")
    for feat, pct in high_nan.items():
        print(f"    {feat}: {pct:.1f}%")

inf_count = np.isinf(tech_df.select_dtypes(include=np.number)).sum().sum()
print(f"  {'✅' if inf_count == 0 else '⚠️'} Infinite values: {inf_count}")

# ── 3. Feature Distributions ──────────────────────────────────
print("\n[3] KEY FEATURE DISTRIBUTIONS")
key_features = ["rsi_14", "adx", "bb_pct_b", "volume_ratio",
                "returns_1d", "atr_ratio", "ema_spread"]

for feat in key_features:
    if feat not in tech_df.columns:
        continue
    s = tech_df[feat].dropna()
    print(
        f"  {feat:20}: "
        f"mean={s.mean():+.4f}  "
        f"std={s.std():.4f}  "
        f"[{s.min():.3f}, {s.max():.3f}]"
    )

# ── 4. Predictive Power (Information Coefficient) ─────────────
print("\n[4] PREDICTIVE POWER — IC vs 5-day forward return")
print("  (IC > 0.05 = useful, > 0.10 = strong signal)")
forward_ret = df["Close"].pct_change(5).shift(-5)
tech_df_aligned = tech_df.copy()
tech_df_aligned["fwd_ret"] = forward_ret

ics = {}
for feat in TechnicalFeatures.FEATURE_NAMES:
    if feat not in tech_df_aligned.columns:
        continue
    try:
        pair   = tech_df_aligned[[feat, "fwd_ret"]].dropna()
        if len(pair) < 50:
            continue
        ic     = pair[feat].corr(pair["fwd_ret"])
        ics[feat] = round(float(ic), 4)
    except Exception:
        pass

sorted_ics = sorted(ics.items(), key=lambda x: abs(x[1]), reverse=True)
print("  Top 10 by |IC|:")
for feat, ic in sorted_ics[:10]:
    bar  = "█" * int(abs(ic) * 100)
    sign = "+" if ic > 0 else ""
    print(f"    {feat:25}: IC={sign}{ic:.4f}  {bar}")

print("\n  Bottom 5 (potentially useless):")
for feat, ic in sorted_ics[-5:]:
    print(f"    {feat:25}: IC={ic:+.4f}")

# ── 5. Feature Correlation Analysis ───────────────────────────
print("\n[5] HIGHLY CORRELATED FEATURE PAIRS (|r| > 0.85)")
print("  These carry redundant information:")
feat_cols  = [f for f in TechnicalFeatures.FEATURE_NAMES if f in tech_df.columns]
corr_matrix= tech_df[feat_cols].corr().abs()

high_corr  = []
for i in range(len(feat_cols)):
    for j in range(i + 1, len(feat_cols)):
        r = float(corr_matrix.iloc[i, j])
        if r > 0.85:
            high_corr.append((feat_cols[i], feat_cols[j], r))

if high_corr:
    for f1, f2, r in sorted(high_corr, key=lambda x: x[2], reverse=True)[:10]:
        print(f"  {f1:20} ↔ {f2:20}: r={r:.3f}")
else:
    print("  ✅ No highly correlated pairs found")

# ── 6. RSI Divergence Frequency ───────────────────────────────
print("\n[6] RSI DIVERGENCE ANALYSIS")
if "rsi_divergence" in tech_df.columns:
    div = tech_df["rsi_divergence"]
    bull_divs = (div > 0).sum()
    bear_divs = (div < 0).sum()
    total     = len(div)
    print(f"  Bullish divergences:  {bull_divs} ({bull_divs/total*100:.1f}% of days)")
    print(f"  Bearish divergences:  {bear_divs} ({bear_divs/total*100:.1f}% of days)")
    print(f"  No divergence:        {total-bull_divs-bear_divs} ({(total-bull_divs-bear_divs)/total*100:.1f}%)")

    # Test divergence predictive power
    bull_fwd = forward_ret[div > 0].mean() * 100
    bear_fwd = forward_ret[div < 0].mean() * 100
    no_div   = forward_ret[div == 0].mean() * 100
    print(f"\n  5-day forward return after divergence:")
    print(f"  After bullish div: {bull_fwd:+.3f}%")
    print(f"  After bearish div: {bear_fwd:+.3f}%")
    print(f"  No divergence:     {no_div:+.3f}%")

# ── 7. BB Squeeze Backtest ─────────────────────────────────────
print("\n[7] BOLLINGER BAND SQUEEZE ANALYSIS")
if "bb_squeeze" in tech_df.columns:
    squeeze    = tech_df["bb_squeeze"].astype(bool)
    next_5d_hi = (df["High"].rolling(5).max().shift(-5) - df["Close"]) / df["Close"] * 100
    squeeze_moves   = next_5d_hi[squeeze].dropna()
    nosqueeze_moves = next_5d_hi[~squeeze].dropna()
    print(f"  During squeeze ({squeeze.sum()} days):")
    print(f"    Avg 5d high above close: {squeeze_moves.mean():.2f}%")
    print(f"    % of days with 2%+ move: {(squeeze_moves > 2).mean()*100:.1f}%")
    print(f"  No squeeze ({(~squeeze).sum()} days):")
    print(f"    Avg 5d high above close: {nosqueeze_moves.mean():.2f}%")
    print(f"    % of days with 2%+ move: {(nosqueeze_moves > 2).mean()*100:.1f}%")

# ── 8. Volume Analysis ─────────────────────────────────────────
print("\n[8] VOLUME CONFIRMATION ANALYSIS")
if "volume_ratio" in tech_df.columns:
    high_vol  = tech_df["volume_ratio"] > 2.0
    price_up  = df["Close"].pct_change() > 0
    high_vol_up   = (high_vol & price_up).sum()
    high_vol_down = (high_vol & ~price_up).sum()
    print(f"  High volume days (2x+): {high_vol.sum()}")
    print(f"    → Up days:   {high_vol_up} ({high_vol_up/high_vol.sum()*100:.0f}%)")
    print(f"    → Down days: {high_vol_down} ({high_vol_down/high_vol.sum()*100:.0f}%)")
    fwd_high_vol = forward_ret[high_vol].mean() * 100
    fwd_low_vol  = forward_ret[~high_vol].mean() * 100
    print(f"  5d fwd return after high volume: {fwd_high_vol:+.3f}%")
    print(f"  5d fwd return after low volume:  {fwd_low_vol:+.3f}%")

# ── 9. Feature Count Verification ─────────────────────────────
print(f"\n[9] FEATURE VECTOR VERIFICATION")
from features.technical_features import TechnicalFeatures
feat_result = feat_obj.extract(tech_df)
print(f"  Feature names defined:  {len(TechnicalFeatures.FEATURE_NAMES)}")
print(f"  Features extracted:     {len(feat_result)}")
print(f"  Features with NaN:      {sum(1 for v in feat_result.values() if not np.isfinite(float(v)))}")
print(f"  Features exactly 0:     {sum(1 for v in feat_result.values() if float(v) == 0.0)}")

all_bounded = all(
    -1.0 <= float(v) <= 100.1  # RSI can be up to 100
    for v in feat_result.values()
    if isinstance(v, (int, float))
)
print(f"  All features finite:    {'✅ YES' if all_bounded else '⚠️ NO'}")

print("\n" + "=" * 60)
print("Feature analysis complete.")
print("Review IC values — features with |IC| < 0.01 may not help the model.")
print("=" * 60)