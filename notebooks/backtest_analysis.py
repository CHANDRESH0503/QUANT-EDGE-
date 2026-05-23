# notebooks/04_backtest_analysis.py
# Full backtest + walk-forward analysis with comprehensive reporting
# Run this after training to decide whether system is ready for live trading

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("QUANT EDGE — Backtest Analysis")
print("Full historical performance + walk-forward validation")
print("=" * 60)

STARTING_CAPITAL = 100_000.0

try:
    from xgboost import XGBClassifier
    from sklearn.model_selection import TimeSeriesSplit
    DEPS_OK = True
except ImportError:
    print("pip install xgboost scikit-learn")
    DEPS_OK = False

if not DEPS_OK:
    exit()

from data.price_fetcher       import PriceFetcher
from models.train_swing       import SwingModelTrainer
from backtest.engine          import BacktestEngine
from backtest.metrics         import BacktestMetrics
from backtest.walk_forward    import WalkForwardValidator
from features.feature_builder import FeatureBuilder

# ── 1. Fetch data ──────────────────────────────────────────────
print("\n[1] Fetching HDFC Bank data...")
fetcher = PriceFetcher("HDFCBANK.NS")
df      = fetcher.get_daily(start="2000-01-01")
print(f"  {len(df)} daily bars ({str(df.index[0])[:10]} to {str(df.index[-1])[:10]})")

# ── 2. Train the swing model ───────────────────────────────────
print("\n[2] Training swing model...")
fb      = FeatureBuilder()
X, y    = fb.build_training_dataset(
    df, model_type="swing",
    forward_days=5, threshold=0.025,
)
print(f"  Training set: {len(X)} samples | {len(X.columns)} features")

trainer = SwingModelTrainer()
metrics = trainer.train(X, y, list(X.columns), save=False)
print(f"  CV accuracy: {metrics.get('cv_accuracy_mean', 0):.3f} ± {metrics.get('cv_accuracy_std', 0):.3f}")

if trainer.model is None:
    print("  ⚠️  Model training failed — check data")
    exit()

# ── 3. Full backtest ───────────────────────────────────────────
print("\n[3] Running full backtest (2000–present)...")
engine  = BacktestEngine(
    starting_capital=STARTING_CAPITAL,
    capital_mode="FULL",
)
result  = engine.run(
    df=df,
    model=trainer.model,
    feature_names=trainer.features,
    model_type="swing",
    forward_days=5,
    threshold=0.025,
    verbose=False,
)
m       = result.get("metrics", {})

print(f"\n{'─'*50}")
print(f"FULL BACKTEST RESULTS (2000–present)")
print(f"{'─'*50}")
print(f"  Period:          {result.get('backtest_period', 'N/A')}")
print(f"  Total trades:    {m.get('total_trades', 0)}")
print(f"  Win rate:        {m.get('win_rate', 0):.1%}")
print(f"  Profit factor:   {m.get('profit_factor', 0):.2f}")
print(f"  Sharpe ratio:    {m.get('sharpe_ratio', 0):.2f}")
print(f"  Sortino ratio:   {m.get('sortino_ratio', 0):.2f}")
print(f"  Max drawdown:    {m.get('max_drawdown_pct', 0):.1%}")
print(f"  Avg win:         {m.get('avg_win_pct', 0):.2%}")
print(f"  Avg loss:        {m.get('avg_loss_pct', 0):.2%}")
print(f"  Expectancy:      {m.get('expectancy_pct', 0):.3%}")
print(f"  Total return:    {m.get('total_return_pct', 0):.1%}")
print(f"  Ending capital:  ₹{m.get('ending_capital', STARTING_CAPITAL):,.2f}")
print(f"  Max consec wins: {m.get('max_consec_wins', 0)}")
print(f"  Max consec loss: {m.get('max_consec_losses', 0)}")
print(f"  Avg hold (days): {m.get('avg_hold_days', 0):.1f}")
print(f"{'─'*50}")
print(f"  Grade:           {m.get('grade', 'D')}")
print(f"  Deployable:      {'✅ YES' if m.get('is_deployable') else '❌ NO'}")
if m.get("issues"):
    for issue in m["issues"]:
        print(f"  ⚠️  {issue}")

# ── 4. The 70% rule check ──────────────────────────────────────
win_rate = float(m.get("win_rate", 0))
print(f"\n[4] THE 70% RULE CHECK")
if win_rate > 0.70:
    print(f"  🚨 WIN RATE {win_rate:.0%} > 70% — LIKELY OVERFITTING")
    print(f"     Check for lookahead bias in feature building")
    print(f"     Verify TimeSeriesSplit was used (not random split)")
elif win_rate > 0.60:
    print(f"  ✅ Win rate {win_rate:.0%} — Realistic and healthy")
elif win_rate > 0.52:
    print(f"  🟡 Win rate {win_rate:.0%} — Acceptable but room to improve")
else:
    print(f"  ❌ Win rate {win_rate:.0%} — Below minimum threshold")

# ── 5. Direction breakdown ─────────────────────────────────────
print(f"\n[5] DIRECTION BREAKDOWN")
print(f"  LONG  trades: {m.get('long_trades', 0)}  WR={m.get('long_win_rate', 0):.0%}")
print(f"  SHORT trades: {m.get('short_trades', 0)}  WR={m.get('short_win_rate', 0):.0%}")
if m.get("short_trades", 0) < 20:
    print(f"  ⚠️  Few SHORT trades — model may be biased toward LONG")

# ── 6. Walk-forward validation ─────────────────────────────────
print(f"\n[6] WALK-FORWARD VALIDATION (2yr train / 6mo test windows)")
print(f"    This is the honest test — uses only data available at the time")
wf_result = None
try:
    validator = WalkForwardValidator(
        train_years=2, test_months=6,
        starting_capital=STARTING_CAPITAL,
    )
    wf_result = validator.validate(
        df, model_type="swing",
        forward_days=5, threshold=0.025,
    )
except Exception as e:
    print(f"  ⚠️  Walk-forward failed: {e}")

if wf_result:
    wf_agg  = wf_result.get("aggregate_metrics", {})
    wf_stab = wf_result.get("stability", {})
    assess  = wf_result.get("assessment", {})

    print(f"\n  Walk-forward aggregate results ({wf_result.get('n_folds')} folds):")
    print(f"  Win rate:          {wf_agg.get('win_rate', 0):.1%}")
    print(f"  Profit factor:     {wf_agg.get('profit_factor', 0):.2f}")
    print(f"  Sharpe:            {wf_agg.get('sharpe_ratio', 0):.2f}")
    print(f"  Max drawdown:      {wf_agg.get('max_drawdown_pct', 0):.1%}")

    print(f"\n  Stability across folds:")
    print(f"  WR mean:           {wf_stab.get('win_rate_mean', 0):.1%}")
    print(f"  WR std:            {wf_stab.get('win_rate_std', 0):.1%}")
    print(f"  WR min (worst):    {wf_stab.get('win_rate_min', 0):.1%}")
    print(f"  Profitable folds:  {wf_stab.get('profitable_folds', 0)}/{wf_stab.get('total_folds', 0)}")
    print(f"  Consistency:       {wf_stab.get('consistency', 0):.0%}")

    # ── Walk-forward vs in-sample comparison ──────────────────
    print(f"\n  In-sample vs Walk-forward comparison:")
    is_wr   = float(m.get("win_rate", 0))
    wf_wr   = float(wf_agg.get("win_rate", 0))
    degr    = is_wr - wf_wr
    print(f"  In-sample WR:      {is_wr:.1%}")
    print(f"  Walk-forward WR:   {wf_wr:.1%}")
    print(f"  Degradation:       {degr:.1%} ({'✅ acceptable' if degr < 0.08 else '⚠️ too much — possible overfit'})")

    print(f"\n  DEPLOYMENT ASSESSMENT:")
    print(f"  {assess.get('recommendation', 'No recommendation')}")

# ── 7. Equity curve summary ────────────────────────────────────
print(f"\n[7] EQUITY CURVE MILESTONES")
curve = result.get("equity_log", [])
if curve and len(curve) > 4:
    n    = len(curve)
    pts  = [0, n//4, n//2, 3*n//4, n-1]
    for i in pts:
        entry = curve[min(i, len(curve)-1)]
        date  = entry.get("date", "")[:7]
        eq    = float(entry.get("equity", STARTING_CAPITAL))
        ret   = (eq - STARTING_CAPITAL) / STARTING_CAPITAL * 100
        print(f"  {date}: ₹{eq:>12,.2f}  ({ret:+.1f}%)")

# ── 8. Monthly P&L distribution ───────────────────────────────
print(f"\n[8] MONTHLY RETURN DISTRIBUTION")
trades = result.get("trades", [])
if trades:
    df_trades = pd.DataFrame(trades)
    if "exit_date" in df_trades.columns:
        df_trades["month"] = pd.to_datetime(
            df_trades["exit_date"], errors="coerce"
        ).dt.to_period("M")
        monthly = df_trades.groupby("month")["pnl_amount"].sum()
        pos_months = (monthly > 0).sum()
        neg_months = (monthly <= 0).sum()
        best_month = monthly.max()
        worst_month= monthly.min()
        print(f"  Positive months:   {pos_months} ({pos_months/(pos_months+neg_months)*100:.0f}%)")
        print(f"  Negative months:   {neg_months}")
        print(f"  Best month:        ₹{best_month:+,.2f}")
        print(f"  Worst month:       ₹{worst_month:+,.2f}")
        print(f"  Avg monthly P&L:   ₹{monthly.mean():+,.2f}")

# ── 9. Final verdict ───────────────────────────────────────────
print(f"\n{'='*60}")
print(f"FINAL DEPLOYMENT VERDICT")
print(f"{'='*60}")

conditions = [
    (win_rate >= 0.52,              f"Win rate {win_rate:.0%} >= 52%"),
    (float(m.get("profit_factor", 0)) >= 1.5,  f"Profit factor >= 1.5"),
    (float(m.get("sharpe_ratio", 0)) >= 1.0,   f"Sharpe ratio >= 1.0"),
    (abs(float(m.get("max_drawdown_pct", -1))) <= 0.20, f"Max DD <= 20%"),
    (int(m.get("total_trades", 0)) >= 50,       f"50+ trades"),
]

all_pass = all(c[0] for c in conditions)
for passed, desc in conditions:
    print(f"  {'✅' if passed else '❌'} {desc}")

if wf_result:
    wf_pass = wf_result.get("is_deployable", False)
    print(f"  {'✅' if wf_pass else '❌'} Walk-forward validation passed")
    all_pass = all_pass and wf_pass

print(f"\n  Final verdict: {'✅ DEPLOY' if all_pass else '❌ DO NOT DEPLOY — fix issues first'}")
print("=" * 60)