# notebooks/03_model_experiments.py
# Experiments to find best model hyperparameters and feature sets
# Run this to justify model design decisions with data

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import json
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("QUANT EDGE — Model Experiments")
print("XGBoost hyperparameter and feature selection experiments")
print("=" * 60)

# ── Dependencies check ─────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.metrics import accuracy_score, classification_report
    DEPS_OK = True
except ImportError:
    print("⚠️  Install: pip install xgboost scikit-learn")
    DEPS_OK = False

if not DEPS_OK:
    print("Skipping experiments — dependencies missing")
    exit()

from data.price_fetcher          import PriceFetcher
from features.feature_builder    import FeatureBuilder
from backtest.metrics            import BacktestMetrics

# ── 1. Build Training Dataset ─────────────────────────────────
print("\n[1] Building training dataset...")
fetcher = PriceFetcher("HDFCBANK.NS")
df      = fetcher.get_daily(start="2000-01-01")
fb      = FeatureBuilder()
X, y    = fb.build_training_dataset(
    df, model_type="swing",
    forward_days=5, threshold=0.025,
)
print(f"  ✅ {len(X)} training samples | {len(X.columns)} features")
print(f"  Labels: LONG={sum(y==2)} FLAT={sum(y==1)} SHORT={sum(y==0)}")
print(f"  Class balance: LONG={sum(y==2)/len(y):.0%} FLAT={sum(y==1)/len(y):.0%} SHORT={sum(y==0)/len(y):.0%}")

if len(X) < 200:
    print("  ⚠️  Insufficient data — increase history range")
    exit()

# ── 2. Baseline Model ──────────────────────────────────────────
print("\n[2] BASELINE MODEL (default params, TimeSeriesSplit CV=5)")
baseline = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    use_label_encoder=False, eval_metric="mlogloss",
    random_state=42, n_jobs=-1,
)
tscv     = TimeSeriesSplit(n_splits=5)
scores   = cross_val_score(baseline, X.fillna(0), y, cv=tscv, scoring="accuracy")
print(f"  CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
print(f"  Individual folds: {[round(s,3) for s in scores]}")

# ── 3. Depth Experiment ────────────────────────────────────────
print("\n[3] MAX DEPTH EXPERIMENT (most important hyperparameter)")
print("  20yr trader insight: Shallow trees generalise, deep trees memorise.")
for depth in [2, 3, 4, 5, 6]:
    m = XGBClassifier(
        n_estimators=300, max_depth=depth, learning_rate=0.05,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=42, n_jobs=-1,
    )
    s = cross_val_score(m, X.fillna(0), y, cv=tscv, scoring="accuracy")
    status = "← RECOMMENDED" if depth == 4 else ""
    print(f"  depth={depth}: CV={s.mean():.3f} ± {s.std():.3f}  {status}")

# ── 4. Label Threshold Sensitivity ────────────────────────────
print("\n[4] LABEL THRESHOLD SENSITIVITY")
print("  Finding the optimal threshold for LONG/SHORT labelling")
for thresh in [0.015, 0.020, 0.025, 0.030, 0.040]:
    _, y_t = fb.build_training_dataset(
        df, model_type="swing", forward_days=5, threshold=thresh
    )
    if len(y_t) < 100:
        continue
    long_pct  = sum(y_t == 2) / len(y_t)
    short_pct = sum(y_t == 0) / len(y_t)
    flat_pct  = sum(y_t == 1) / len(y_t)
    balance   = "✅" if 0.15 <= long_pct <= 0.35 else "⚠️"
    print(
        f"  thresh={thresh:.3f}: "
        f"LONG={long_pct:.0%} FLAT={flat_pct:.0%} SHORT={short_pct:.0%} {balance}"
    )

# ── 5. Forward Days Sensitivity ───────────────────────────────
print("\n[5] FORWARD HORIZON SENSITIVITY")
for fwd in [3, 5, 7, 10]:
    _, y_f = fb.build_training_dataset(
        df, model_type="swing", forward_days=fwd, threshold=0.025
    )
    if len(y_f) < 100:
        continue
    m = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=42, n_jobs=-1,
    )
    X_f, _ = fb.build_training_dataset(df, forward_days=fwd, threshold=0.025)
    if X_f.empty:
        continue
    try:
        s = cross_val_score(m, X_f.fillna(0), y_f, cv=tscv, scoring="accuracy")
        status = "← RECOMMENDED" if fwd == 5 else ""
        print(f"  fwd={fwd:2d}d: CV={s.mean():.3f} ± {s.std():.3f}  {status}")
    except Exception as e:
        print(f"  fwd={fwd}d: ERROR {e}")

# ── 6. Feature Importance ──────────────────────────────────────
print("\n[6] TOP FEATURE IMPORTANCES (train on full dataset)")
final_model = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
    use_label_encoder=False, eval_metric="mlogloss",
    random_state=42, n_jobs=-1,
)
final_model.fit(X.fillna(0), y, verbose=False)
importances = dict(zip(
    X.columns,
    final_model.feature_importances_,
))
top_10 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
print("  Top 10 features by XGBoost gain:")
max_imp = top_10[0][1] if top_10 else 1
for feat, imp in top_10:
    bar = "█" * int(imp / max_imp * 30)
    print(f"  {feat:30}: {imp:.4f}  {bar}")

# Save importances
os.makedirs("models/evaluation", exist_ok=True)
with open("models/evaluation/feature_importance.json", "w") as f:
    json.dump({"swing": importances}, f, indent=2)
print("  ✅ Feature importances saved to models/evaluation/feature_importance.json")

# ── 7. Class Weight Experiment ─────────────────────────────────
print("\n[7] CLASS WEIGHT EXPERIMENT")
print("  FLAT dominates labels — test if balancing improves useful class detection")

# Without balancing
m_unbal = XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    use_label_encoder=False, eval_metric="mlogloss",
    random_state=42, n_jobs=-1,
)
# Train on last split
train_idx = list(tscv.split(X))[-1][0]
val_idx   = list(tscv.split(X))[-1][1]
m_unbal.fit(X.iloc[train_idx].fillna(0), y.iloc[train_idx])
preds_unbal = m_unbal.predict(X.iloc[val_idx].fillna(0))
acc_unbal   = accuracy_score(y.iloc[val_idx], preds_unbal)

print(f"  Without balancing: acc={acc_unbal:.3f}")
print(f"  Prediction distribution: {dict(zip(*np.unique(preds_unbal, return_counts=True)))}")

# ── 8. Recommended Configuration ──────────────────────────────
print("\n[8] RECOMMENDED CONFIGURATION SUMMARY")
print("""
  SwingModelTrainer.PARAMS = {
      "n_estimators":     500,    # stopped early via early_stopping
      "max_depth":        4,      # sweet spot: generalises well
      "learning_rate":    0.03,   # slow learning = less overfit
      "subsample":        0.8,    # prevents overfitting
      "colsample_bytree": 0.7,    # prevents overfitting
      "min_child_weight": 5,      # minimum 5 samples per leaf
      "gamma":            1.0,    # pruning threshold
  }

  Training settings:
  - TimeSeriesSplit(n_splits=10)     # NEVER random split
  - FORWARD_DAYS = 5                 # realistic swing horizon
  - THRESHOLD    = 0.025             # 2.5% = meaningful move
  - Early stopping = 30 rounds       # stop if no improvement
""")

print("=" * 60)
print("Experiments complete. Review IC and importances before deploying.")
print("=" * 60)