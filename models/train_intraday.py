# models/train_intraday.py
# Intraday model — predicts next 30-90 min direction using 5-min candles
# VWAP, order flow, ORB are the primary features here
# Shorter horizon = noisier labels = simpler model needed

import numpy as np
import pandas as pd
import logging
import json
import os
import joblib
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, brier_score_loss
    from sklearn.utils.class_weight import compute_sample_weight
    DEPS_OK = True
except ImportError:
    DEPS_OK = False
    logger.error("pip install xgboost scikit-learn")

METRICS_PATH  = "models/evaluation/model_metrics.json"


def _paths_for(variant: str, ticker_safe: str = "HDFCBANK") -> Dict[str, str]:
    """Resolve model file paths for a given variant and bank ticker."""
    base = f"models/saved/{ticker_safe}_intraday_{variant}"
    return {
        "model":      f"{base}.pkl",
        "calibrated": f"{base}_calibrated.pkl",
        "features":   f"{base}_features.pkl",
    }

FORWARD_BARS = 6      # 6 × 5min = 30 min horizon
THRESHOLD    = 0.004  # 0.4% — tight for intraday HDFC Bank


class IntradayModelTrainer:
    """
    Intraday XGBoost model — 5-min candles, 30-min horizon.

    20yr trader insight on intraday modelling:
    Intraday data is far noisier than daily. A model that
    works on daily data at 62% win rate will likely drop
    to 55% intraday. Compensate with:
    1. Simpler model (max_depth=3, not 4)
    2. Stronger regularisation
    3. Focus on structural features (VWAP, ORB, volume)
       rather than momentum-only features
    4. Only trade the first 2 hours and last 30 minutes —
       intraday signals during lunch hour are unreliable

    60 days × 75 candles/day = ~4,500 samples.
    That is enough for a shallow model — not for a deep one.
    """

    PARAMS = {
        "n_estimators":     300,
        "max_depth":        3,        # shallower than swing — less data
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.7,
        "min_child_weight": 10,       # more conservative — noisy data
        "gamma":            2.0,
        "reg_alpha":        0.5,
        "reg_lambda":       1.0,
        "eval_metric":      "mlogloss",
        "objective":        "multi:softprob",
        "num_class":        3,
        "tree_method":      "hist",
        "random_state":     42,
        "n_jobs":           -1,
    }

    N_SPLITS       = 5    # less splits — less data available
    EARLY_STOPPING = 20

    # Class-balanced sample weights. The 0.4%/30-min threshold makes FLAT the
    # vast majority of labels, so the unweighted model collapsed to FLAT≈1.0
    # (it almost never fired → 0 trades in the holdout despite the strongest
    # AUC). 'balanced' lets it express LONG/SHORT on genuine setups; the
    # calibrated probability + Gate 6's 65% threshold filter the weak ones.
    CLASS_BALANCE = "balanced"

    def __init__(self, variant: str = "technical", ticker: str = "HDFCBANK.NS"):
        self.variant     = variant
        self.ticker      = ticker
        self.ticker_safe = ticker.replace(".NS", "")
        self._paths      = _paths_for(variant, self.ticker_safe)
        self.model       = None
        self.features    = None
        self.metrics     = {}

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: List[str],
        save: bool = True,
    ) -> Dict:
        """Train intraday model with walk-forward CV."""
        if not DEPS_OK:
            return {}
        if len(X) < 100:
            logger.error(f"Need 100+ intraday samples, got {len(X)}")
            return {}

        X_clean       = self._clean(X, feature_names)
        self.features = [f for f in feature_names if f in X_clean.columns]

        logger.info(
            f"Intraday training: {len(X_clean)} bars | "
            f"{len(self.features)} features"
        )

        tscv        = TimeSeriesSplit(n_splits=self.N_SPLITS)
        fold_scores = []

        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_clean)):
            X_tr, X_val = X_clean.iloc[tr_idx], X_clean.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx],       y.iloc[val_idx]

            m = XGBClassifier(**self.PARAMS, early_stopping_rounds=self.EARLY_STOPPING)
            sw_tr = (compute_sample_weight(self.CLASS_BALANCE, y_tr)
                     if self.CLASS_BALANCE else None)
            m.fit(
                X_tr, y_tr,
                sample_weight=sw_tr,
                eval_set=[(X_val, y_val)],   # unweighted → early-stop on real perf
                verbose=False,
            )
            acc = accuracy_score(y_val, m.predict(X_val))
            fold_scores.append(acc)
            logger.info(f"  Intraday fold {fold+1}/{self.N_SPLITS}: {acc:.3f}")

        cv_mean = float(np.mean(fold_scores))

        self.model = XGBClassifier(**self.PARAMS)
        sw = (compute_sample_weight(self.CLASS_BALANCE, y)
              if self.CLASS_BALANCE else None)
        self.model.fit(X_clean[self.features], y, sample_weight=sw, verbose=False)

        # Calibration on most-recent 15% of data (time-ordered)
        cal_brier = None
        try:
            from models.calibration import make_calibrator
            cal_n = max(30, int(len(X_clean) * 0.15))
            X_cal, y_cal = X_clean[self.features].iloc[-cal_n:], y.iloc[-cal_n:]
            # Fit unweighted so calibrated probs reflect the true (FLAT-heavy)
            # base rate even though training used balanced weights.
            self.calibrated_model = make_calibrator(self.model, len(y_cal))
            self.calibrated_model.fit(X_cal, y_cal)
            cal_brier = round(float(brier_score_loss(
                (y_cal == 2).astype(int),
                self.calibrated_model.predict_proba(X_cal)[:, 2]
            )), 4)
        except Exception as e:
            logger.warning(f"Intraday calibration failed: {e}")
            self.calibrated_model = None

        self.metrics = {
            "model_type":       "intraday",
            "trained_at":       str(datetime.now()),
            "n_samples":        len(X_clean),
            "n_features":       len(self.features),
            "forward_bars":     FORWARD_BARS,
            "threshold":        THRESHOLD,
            "cv_accuracy_mean": round(cv_mean, 4),
            "calibrated":       self.calibrated_model is not None,
            "brier_score_long": cal_brier,
            "label_dist":       y.value_counts().to_dict(),
        }

        if save:
            self._save()
            self._save_metrics()

        logger.info(f"Intraday model trained | CV={cv_mean:.3f}")
        return self.metrics

    def load(self) -> bool:
        try:
            self.model    = joblib.load(self._paths["model"])
            self.features = joblib.load(self._paths["features"])
            self.calibrated_model = (
                joblib.load(self._paths["calibrated"])
                if os.path.exists(self._paths["calibrated"]) else None
            )
            logger.info(
                f"Intraday model loaded ({self.variant}, "
                f"calibrated={self.calibrated_model is not None})"
            )
            return True
        except Exception as e:
            logger.warning(f"Intraday model load failed ({self.variant}): {e}")
            return False

    def predict(self, X: pd.DataFrame) -> Dict:
        if self.model is None:
            if not self.load():
                return self._empty()
        try:
            X_clean = self._clean(X, self.features)
            for f in self.features:
                if f not in X_clean.columns:
                    X_clean[f] = 0.0
            row       = X_clean[self.features].fillna(0).iloc[-1:]
            predictor = self.calibrated_model if self.calibrated_model else self.model
            probs     = predictor.predict_proba(row)[0]
            pred      = int(np.argmax(probs))
            lmap      = {0: "SHORT", 1: "FLAT", 2: "LONG"}
            return {
                "signal":     lmap[pred],
                "confidence": round(float(probs[pred]), 4),
                "prob_long":  round(float(probs[2]), 4),
                "prob_flat":  round(float(probs[1]), 4),
                "prob_short": round(float(probs[0]), 4),
                "model_type": "intraday",
                "calibrated": self.calibrated_model is not None,
            }
        except Exception as e:
            logger.error(f"Intraday prediction error: {e}")
            return self._empty()

    def _clean(self, X: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        cols = [f for f in features if f in X.columns]
        return X[cols].fillna(0).replace([np.inf, -np.inf], 0).astype(np.float32)

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._paths["model"]), exist_ok=True)
        joblib.dump(self.model, self._paths["model"])
        joblib.dump(self.features, self._paths["features"])
        if self.calibrated_model is not None:
            joblib.dump(self.calibrated_model, self._paths["calibrated"])
        logger.info(f"Intraday model ({self.variant}) saved")

    def _save_metrics(self) -> None:
        os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
        existing = {}
        if os.path.exists(METRICS_PATH):
            try:
                with open(METRICS_PATH) as f:
                    existing = json.load(f)
            except Exception:
                pass
        key = "intraday" if self.variant == "technical" else f"intraday_{self.variant}"
        existing[key] = self.metrics
        with open(METRICS_PATH, "w") as f:
            json.dump(existing, f, indent=2, default=str)

    def _empty(self) -> Dict:
        return {
            "signal": "FLAT", "confidence": 0.0,
            "prob_long": 0.33, "prob_flat": 0.34, "prob_short": 0.33,
            "model_type": "intraday",
        }