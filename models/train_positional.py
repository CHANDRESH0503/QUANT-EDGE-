# models/train_positional.py
# Positional model — 2-4 week predictions using weekly candles
# Heavy on fundamentals, macro, alternative data
# Only 1,300 weekly samples — keep model simple

import numpy as np
import pandas as pd
import logging
import json
import os
import joblib
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, brier_score_loss
    from sklearn.utils.class_weight import compute_sample_weight
    DEPS_OK = True
except ImportError:
    DEPS_OK = False

METRICS_PATH  = "models/evaluation/model_metrics.json"


def _paths_for(variant: str, ticker_safe: str = "HDFCBANK") -> Dict[str, str]:
    """Resolve model file paths for a given variant and bank ticker."""
    base = f"models/saved/{ticker_safe}_positional_{variant}"
    return {
        "model":      f"{base}.pkl",
        "calibrated": f"{base}_calibrated.pkl",
        "features":   f"{base}_features.pkl",
    }

FORWARD_WEEKS = 3
THRESHOLD     = 0.05   # 5% — meaningful positional move


class PositionalModelTrainer:
    """
    Positional XGBoost model — weekly candles, 3-week horizon.

    20yr trader insight:
    Positional trades live or die by fundamentals and macro.
    When NIM is expanding, NPA is falling, FII flows are
    positive and the macro cycle favours banks — you hold
    for weeks, not days. The model must reflect this.

    With only 1,300 weekly samples, keep it extremely simple:
    - max_depth = 2 (very shallow)
    - high regularisation
    - fewer features than swing model
    - validate carefully before trusting any positional signal

    The positional model is the least reliable of the three
    due to limited training data. Weight its signal lower
    than swing when combining in the alignment check.
    """

    PARAMS = {
        "n_estimators":     200,
        "max_depth":        2,        # very shallow — 1300 samples only
        "learning_rate":    0.05,
        "subsample":        0.7,
        "colsample_bytree": 0.6,
        "min_child_weight": 15,       # high — prevents overfitting
        "gamma":            3.0,
        "reg_alpha":        1.0,
        "reg_lambda":       2.0,
        "eval_metric":      "mlogloss",
        "objective":        "multi:softprob",
        "num_class":        3,
        "tree_method":      "hist",
        "random_state":     42,
        "n_jobs":           -1,
    }

    N_SPLITS       = 5
    EARLY_STOPPING = 20

    # Class-balanced sample weights — the critical fix for the positional
    # LONG-bias (CR-1A). With only ~1,300 weekly samples, heavy regularisation
    # and a LONG-skewed prior (25yr bank uptrend), the model defaulted to LONG
    # on sparse/zero input → in BEAR it always voted LONG → hard-blocked →
    # positional was effectively dead market-wide. 'balanced' equalises the
    # per-class influence so it can vote SHORT when the setup warrants it.
    # Calibration is fit UNweighted to keep probabilities true-frequency.
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
        X:             pd.DataFrame,
        y:             pd.Series,
        feature_names: List[str],
        save:          bool = True,
    ) -> Dict:
        """Train positional model with strict walk-forward CV."""
        if not DEPS_OK:
            return {}
        if len(X) < 80:
            logger.error(f"Need 80+ weekly samples, got {len(X)}")
            return {}

        X_clean       = self._clean(X, feature_names)
        self.features = [f for f in feature_names if f in X_clean.columns]

        logger.info(
            f"Positional training: {len(X_clean)} weeks | "
            f"{len(self.features)} features | "
            f"dist={y.value_counts().to_dict()}"
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
            logger.info(f"  Positional fold {fold+1}/{self.N_SPLITS}: {acc:.3f}")

        cv_mean = float(np.mean(fold_scores))

        # Final model on full data
        self.model = XGBClassifier(**self.PARAMS)
        sw = (compute_sample_weight(self.CLASS_BALANCE, y)
              if self.CLASS_BALANCE else None)
        self.model.fit(X_clean[self.features], y, sample_weight=sw, verbose=False)
        train_acc = accuracy_score(y, self.model.predict(X_clean[self.features]))

        # Calibration (positional has fewer samples — use last 20%)
        cal_brier = None
        try:
            from models.calibration import make_calibrator
            cal_n = max(20, int(len(X_clean) * 0.20))
            X_cal, y_cal = X_clean[self.features].iloc[-cal_n:], y.iloc[-cal_n:]
            # Small positional tail → make_calibrator picks Platt (stable),
            # avoiding the small-n isotonic collapse. Fit unweighted.
            self.calibrated_model = make_calibrator(self.model, len(y_cal))
            self.calibrated_model.fit(X_cal, y_cal)
            cal_brier = round(float(brier_score_loss(
                (y_cal == 2).astype(int),
                self.calibrated_model.predict_proba(X_cal)[:, 2]
            )), 4)
        except Exception as e:
            logger.warning(f"Positional calibration failed: {e}")
            self.calibrated_model = None

        self.metrics = {
            "model_type":       "positional",
            "trained_at":       str(datetime.now()),
            "n_samples":        len(X_clean),
            "n_features":       len(self.features),
            "forward_weeks":    FORWARD_WEEKS,
            "threshold":        THRESHOLD,
            "cv_accuracy_mean": round(cv_mean, 4),
            "train_accuracy":   round(train_acc, 4),
            "calibrated":       self.calibrated_model is not None,
            "brier_score_long": cal_brier,
            "label_dist":       y.value_counts().to_dict(),
        }

        if save:
            self._save()
            self._save_metrics()

        logger.info(f"Positional model trained | CV={cv_mean:.3f} | Train={train_acc:.3f}")
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
                f"Positional model loaded ({self.variant}, "
                f"calibrated={self.calibrated_model is not None})"
            )
            return True
        except Exception as e:
            logger.warning(f"Positional model load failed ({self.variant}): {e}")
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
                "model_type": "positional",
                "calibrated": self.calibrated_model is not None,
            }
        except Exception as e:
            logger.error(f"Positional prediction error: {e}")
            return self._empty()

    def _clean(self, X: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        cols = [f for f in features if f in X.columns]
        return X[cols].fillna(0).replace([np.inf, -np.inf], 0).astype(np.float32)

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._paths["model"]), exist_ok=True)
        joblib.dump(self.model,    self._paths["model"])
        joblib.dump(self.features, self._paths["features"])
        if self.calibrated_model is not None:
            joblib.dump(self.calibrated_model, self._paths["calibrated"])
        logger.info(f"Positional model ({self.variant}) saved")

    def _save_metrics(self) -> None:
        os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
        existing = {}
        if os.path.exists(METRICS_PATH):
            try:
                with open(METRICS_PATH) as f:
                    existing = json.load(f)
            except Exception:
                pass
        key = "positional" if self.variant == "technical" else f"positional_{self.variant}"
        existing[key] = self.metrics
        with open(METRICS_PATH, "w") as f:
            json.dump(existing, f, indent=2, default=str)

    def _empty(self) -> Dict:
        return {
            "signal": "FLAT", "confidence": 0.0,
            "prob_long": 0.33, "prob_flat": 0.34, "prob_short": 0.33,
            "model_type": "positional",
        }