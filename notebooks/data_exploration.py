# notebooks/01_data_exploration.py
# Run as: jupyter nbconvert --to notebook --execute this file
# OR simply run as a Python script: python 01_data_exploration.py
# Explores HDFC Bank price data quality and statistical properties

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from data.price_fetcher import PriceFetcher

print("=" * 60)
print("QUANT EDGE — Data Exploration")
print("HDFC Bank (HDFCBANK.NS) — 25 Year Analysis")
print("=" * 60)

# ── 1. Fetch Data ──────────────────────────────────────────────
fetcher   = PriceFetcher("HDFCBANK.NS")
df_daily  = fetcher.get_daily(start="2000-01-01")
df_weekly = fetcher.get_weekly(start="2000-01-01")

print(f"\n[1] DATA SUMMARY")
print(f"Daily bars:  {len(df_daily):,} ({str(df_daily.index[0])[:10]} to {str(df_daily.index[-1])[:10]})")
print(f"Weekly bars: {len(df_weekly):,}")
print(f"Current price: ₹{df_daily['Close'].iloc[-1]:,.2f}")
print(f"All-time high: ₹{df_daily['High'].max():,.2f}")
print(f"All-time low:  ₹{df_daily['Low'].min():,.2f}")

# ── 2. Return Distribution ─────────────────────────────────────
print(f"\n[2] RETURN DISTRIBUTION (Daily)")
daily_ret  = df_daily["Close"].pct_change().dropna()
print(f"Mean daily return:  {daily_ret.mean()*100:+.4f}%")
print(f"Std daily return:   {daily_ret.std()*100:.4f}%")
print(f"Annualised return:  {daily_ret.mean()*252*100:.2f}%")
print(f"Annualised vol:     {daily_ret.std()*np.sqrt(252)*100:.2f}%")
print(f"Skewness:           {daily_ret.skew():.4f}")
print(f"Kurtosis:           {daily_ret.kurtosis():.4f}")
print(f"Max single day:     {daily_ret.max()*100:+.2f}%")
print(f"Min single day:     {daily_ret.min()*100:+.2f}%")

# ── 3. Extreme Move Analysis ────────────────────────────────────
print(f"\n[3] EXTREME MOVES (|return| > 3%)")
extreme    = daily_ret[daily_ret.abs() > 0.03]
print(f"Count: {len(extreme)} ({len(extreme)/len(daily_ret)*100:.1f}% of days)")
print(f"Most extreme moves:")
for date, ret in daily_ret.abs().nlargest(5).items():
    print(f"  {str(date)[:10]}: {daily_ret[date]*100:+.2f}%")

# ── 4. Volatility Regimes ──────────────────────────────────────
print(f"\n[4] VOLATILITY REGIMES (Rolling 30d HV)")
hv30       = daily_ret.rolling(30).std() * np.sqrt(252) * 100
print(f"Current 30d HV:   {hv30.iloc[-1]:.1f}%")
print(f"25th pct HV:      {hv30.quantile(0.25):.1f}%")
print(f"Median HV:        {hv30.quantile(0.50):.1f}%")
print(f"75th pct HV:      {hv30.quantile(0.75):.1f}%")
print(f"95th pct HV:      {hv30.quantile(0.95):.1f}%")

# ── 5. Monthly Seasonality ─────────────────────────────────────
print(f"\n[5] MONTHLY SEASONALITY (Avg return by month)")
df_daily["month"] = df_daily.index.month
df_daily["ret"]   = df_daily["Close"].pct_change()
monthly_avg = df_daily.groupby("month")["ret"].mean() * 100
months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
for m, avg in zip(months, monthly_avg):
    bar = "█" * int(abs(avg) * 20)
    sign = "+" if avg > 0 else ""
    print(f"  {m:3}: {sign}{avg:.3f}% {bar}")

# ── 6. Day-of-Week Effect ──────────────────────────────────────
print(f"\n[6] DAY-OF-WEEK EFFECT")
df_daily["dow"] = df_daily.index.dayofweek
dow_avg = df_daily.groupby("dow")["ret"].agg(["mean","std"])
days    = ["Monday","Tuesday","Wednesday","Thursday","Friday"]
for i, day in enumerate(days):
    if i in dow_avg.index:
        mu  = float(dow_avg.loc[i, "mean"]) * 100
        std = float(dow_avg.loc[i, "std"])  * 100
        print(f"  {day:10}: mean={mu:+.3f}%  std={std:.3f}%")

# ── 7. Volume Analysis ─────────────────────────────────────────
print(f"\n[7] VOLUME ANALYSIS")
vol = df_daily["Volume"]
print(f"Avg daily volume:   {vol.mean():,.0f}")
print(f"Median volume:      {vol.median():,.0f}")
print(f"Max volume:         {vol.max():,.0f}")
print(f"Volume trend 1yr:   {(vol.iloc[-252:].mean()/vol.iloc[-504:-252].mean()-1)*100:+.1f}%")

# ── 8. Price Levels ────────────────────────────────────────────
print(f"\n[8] SUPPORT / RESISTANCE LEVELS (Volume nodes)")
from processing.support_resistance import SupportResistanceEngine
sr  = SupportResistanceEngine()
sr_result = sr.get_sr_features(df_daily)
print(f"Current price:       ₹{df_daily['Close'].iloc[-1]:,.2f}")
print(f"Nearest support:     ₹{sr_result['nearest_support']:,.2f}  ({sr_result['support_distance_pct']:.2f}% below)")
print(f"Nearest resistance:  ₹{sr_result['nearest_resistance']:,.2f}  ({sr_result['resistance_distance_pct']:.2f}% above)")
print(f"Entry quality:       {sr_result['entry_quality']}")
print(f"Reward/Risk:         {sr_result['reward_risk_sr']:.2f}")

# ── 9. Data Quality Check ──────────────────────────────────────
print(f"\n[9] DATA QUALITY CHECK")
issues = 0
nan_count = df_daily.isnull().sum().sum()
if nan_count > 0:
    print(f"  ⚠️  NaN values: {nan_count}")
    issues += 1
else:
    print(f"  ✅ No NaN values")

gap_days = pd.date_range(df_daily.index[0], df_daily.index[-1], freq="B")
missing  = len(gap_days) - len(df_daily)
print(f"  {'⚠️' if missing > 10 else '✅'} Missing trading days: {missing}")

price_anomalies = (daily_ret.abs() > 0.25).sum()
print(f"  {'⚠️' if price_anomalies > 0 else '✅'} Price anomalies (>25%/day): {price_anomalies}")

print(f"\n{'='*60}")
print(f"Data quality: {'GOOD' if issues == 0 else 'NEEDS REVIEW'}")
print(f"Ready for model training: {'YES' if len(df_daily) >= 1000 else 'NO — need more data'}")
print("=" * 60)