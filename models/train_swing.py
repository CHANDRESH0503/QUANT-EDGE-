# models/train_swing.py
# Trains the PRIMARY swing trading model — 3-7 day predictions
# 25 years of HDFC Bank daily data = ~6300 samples
# XGBoost 3-class classifier: LONG=2, FLAT=1, SHORT=0

import numpy as np
import pandas as pd
import logging
import json
import os
import joblib
from datetime import datetime
from typing import Dict, Tuple, Optional, List

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.error("xgboost not installed. pip install xgboost")

try:
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import (
        accuracy_score, classification_report,
        confusion_matrix, brier_score_loss,
    )
    from sklearn.utils.class_weight import compute_sample_weight
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

METRICS_PATH     = "models/evaluation/model_metrics.json"


def _paths_for(variant: str, ticker_safe: str = "HDFCBANK") -> Dict[str, str]:
    """Resolve model file paths for a given variant and bank ticker."""
    base = f"models/saved/{ticker_safe}_{variant}"
    return {
        "model":      f"{base}.pkl",
        "calibrated": f"{base}_calibrated.pkl",
        "features":   f"{base}_features.pkl",
    }

# Label thresholds — calibrated for HDFC Bank volatility
FORWARD_DAYS = 5
THRESHOLD    = 0.025   # 2.5% move = LONG or SHORT label


class SwingModelTrainer:
    """
    Trains the swing trading XGBoost model.

    20yr trader design decisions:
    - 5-day forward horizon: realistic for swing trades on HDFC Bank
    - 2.5% threshold: calibrated to ATR, avoids noise labels
    - TimeSeriesSplit: no random split ever — prevents lookahead
    - Early stopping: prevents overfitting on training set
    - Shallow tree depth (4): forces generalisation
    - Class weight balancing: FLAT dominates naturally — correct for it

    Label distribution target:
    LONG:  20–25%
    FLAT:  50–60%
    SHORT: 20–25%
    If FLAT > 70%, reduce threshold.
    If LONG/SHORT > 40% each, increase threshold.
    """

    PARAMS = {
        "n_estimators":       500,
        "max_depth":          4,       # shallow = generalises better
        "learning_rate":      0.03,    # slow learning = less overfit
        "subsample":          0.8,
        "colsample_bytree":   0.7,
        "min_child_weight":   5,       # requires 5+ samples per leaf
        "gamma":              1.0,     # pruning regulariser
        "reg_alpha":          0.1,     # L1 regularisation
        "reg_lambda":         1.0,     # L2 regularisation
        "eval_metric":        "mlogloss",
        "objective":          "multi:softprob",
        "num_class":          3,
        "tree_method":        "hist",  # fast on CPU
        "random_state":       42,
        "n_jobs":             -1,
    }

    N_SPLITS       = 10    # walk-forward CV folds
    EARLY_STOPPING = 30    # stop if no improvement for 30 rounds

    # Class-balanced sample weights. 25yr of HDFC is a secular uptrend, so
    # raw priors lean LONG and FLAT dominates — the model defaults to the
    # majority direction on a sparse/zero input (the CR-1A LONG-bias). Passing
    # 'balanced' sample_weight equalises the per-class influence so the model
    # expresses an honest direction; the downstream gates (Gate 5 R:R, Gate 6
    # threshold) still do the filtering. Calibration is fit UNweighted so the
    # output probabilities stay true-frequency. Set None to disable.
    CLASS_BALANCE = "balanced"

    def __init__(self, variant: str = "technical", ticker: str = "HDFCBANK.NS"):
        """
        Args:
            variant: "technical" | "full"
            ticker:  bank ticker — used to namespace model files
        """
        self.variant     = variant
        self.ticker      = ticker
        self.ticker_safe = ticker.replace(".NS", "")
        self._paths      = _paths_for(variant, self.ticker_safe)
        self.model       = None
        self.features    = None
        self.metrics     = {}

    # ─────────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────────

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: List[str],
        save: bool = True,
    ) -> Dict:
        """
        Full training pipeline with walk-forward cross-validation.

        Args:
            X:             Feature matrix (from FeatureBuilder.build_training_dataset)
            y:             Labels (0=SHORT, 1=FLAT, 2=LONG)
            feature_names: Ordered list of feature names
            save:          Whether to persist model to disk

        Returns:
            metrics dict with accuracy, classification report, etc.
        """
        if not XGB_AVAILABLE or not SKLEARN_AVAILABLE:
            logger.error("XGBoost or sklearn not available")
            return {}

        if X.empty or len(X) < 200:
            logger.error(f"Insufficient training data: {len(X)} rows (need 200+)")
            return {}

        X_clean = self._clean(X, feature_names)
        self.features = [f for f in feature_names if f in X_clean.columns]

        logger.info(
            f"Swing model training: {len(X_clean)} rows | "
            f"{len(self.features)} features | "
            f"label dist: {y.value_counts().to_dict()}"
        )

        # ── Walk-forward cross-validation ─────────────────────────
        tscv       = TimeSeriesSplit(n_splits=self.N_SPLITS)
        fold_scores= []
        fold_reports = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_clean)):
            X_tr, X_val = X_clean.iloc[train_idx], X_clean.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx],       y.iloc[val_idx]

            model = XGBClassifier(**self.PARAMS, early_stopping_rounds=self.EARLY_STOPPING)
            sw_tr = (compute_sample_weight(self.CLASS_BALANCE, y_tr)
                     if self.CLASS_BALANCE else None)
            model.fit(
                X_tr, y_tr,
                sample_weight=sw_tr,
                eval_set=[(X_val, y_val)],   # unweighted → early-stop on real perf
                verbose=False,
            )

            preds  = model.predict(X_val)
            acc    = accuracy_score(y_val, preds)
            fold_scores.append(acc)

            report = classification_report(
                y_val, preds,
                target_names=["SHORT", "FLAT", "LONG"],
                output_dict=True,
                zero_division=0,
            )
            fold_reports.append(report)
            logger.info(f"  Fold {fold+1:2d}/{self.N_SPLITS}: acc={acc:.3f}")

        cv_mean = float(np.mean(fold_scores))
        cv_std  = float(np.std(fold_scores))
        logger.info(f"CV accuracy: {cv_mean:.3f} ± {cv_std:.3f}")

        # ── Final model on full dataset ───────────────────────────
        self.model = XGBClassifier(**self.PARAMS)
        sw = (compute_sample_weight(self.CLASS_BALANCE, y)
              if self.CLASS_BALANCE else None)
        self.model.fit(
            X_clean[self.features], y,
            sample_weight=sw,
            verbose=False,
        )

        # ── Isotonic calibration on most-recent 15% of data ──────
        # Time-ordered split: train on first 85%, calibrate on last 15%.
        # This keeps calibration forward-looking (no lookahead).
        cal_brier = None
        try:
            from models.calibration import make_calibrator
            cal_n = max(50, int(len(X_clean) * 0.15))
            X_cal = X_clean[self.features].iloc[-cal_n:]
            y_cal = y.iloc[-cal_n:]
            # Fit on the natural (unweighted) tail so probabilities stay
            # true-frequency despite the balanced training weights.
            self.calibrated_model = make_calibrator(self.model, len(y_cal))
            self.calibrated_model.fit(X_cal, y_cal)
            cal_probs = self.calibrated_model.predict_proba(X_cal)[:, 2]  # LONG
            y_bin     = (y_cal == 2).astype(int)
            cal_brier = round(float(brier_score_loss(y_bin, cal_probs)), 4)
            logger.info(f"Swing calibration done | Brier (LONG)={cal_brier:.4f}")
        except Exception as e:
            logger.warning(f"Calibration failed — using uncalibrated model: {e}")
            self.calibrated_model = None

        # ── Metrics ───────────────────────────────────────────────
        final_preds = self.model.predict(X_clean[self.features])
        final_acc   = accuracy_score(y, final_preds)

        self.metrics = {
            "model_type":        "swing",
            "trained_at":        str(datetime.now()),
            "n_samples":         len(X_clean),
            "n_features":        len(self.features),
            "forward_days":      FORWARD_DAYS,
            "threshold":         THRESHOLD,
            "cv_accuracy_mean":  round(cv_mean, 4),
            "cv_accuracy_std":   round(cv_std, 4),
            "train_accuracy":    round(final_acc, 4),
            "calibrated":        self.calibrated_model is not None,
            "brier_score_long":  cal_brier,
            "label_distribution": y.value_counts().to_dict(),
            "feature_names":     self.features,
        }

        if save:
            self._save()
            self._save_metrics()

        logger.info(
            f"Swing model trained | "
            f"CV={cv_mean:.3f} | "
            f"Train={final_acc:.3f} | "
            f"Features={len(self.features)}"
        )
        return self.metrics

    def load(self) -> bool:
        """Load pre-trained model from disk. Returns True if successful."""
        try:
            self.model    = joblib.load(self._paths["model"])
            self.features = joblib.load(self._paths["features"])
            # Load calibrated wrapper if available
            if os.path.exists(self._paths["calibrated"]):
                self.calibrated_model = joblib.load(self._paths["calibrated"])
                logger.info(
                    f"Swing model loaded ({self.variant}, calibrated): "
                    f"{len(self.features)} features"
                )
            else:
                self.calibrated_model = None
                logger.info(
                    f"Swing model loaded ({self.variant}, uncalibrated): "
                    f"{len(self.features)} features"
                )
            return True
        except Exception as e:
            logger.warning(f"Swing model load failed ({self.variant}): {e}")
            return False

    def predict(self, X: pd.DataFrame) -> Dict:
        """
        Predict on a single row (live signal generation).
        Returns probabilities for all three classes.
        """
        if self.model is None:
            if not self.load():
                return self._empty_prediction()

        try:
            X_clean = self._clean(X, self.features)
            for f in self.features:
                if f not in X_clean.columns:
                    X_clean[f] = 0.0
            X_aligned = X_clean[self.features].fillna(0).iloc[-1:]

            # Prefer calibrated model; fall back to raw model
            predictor = self.calibrated_model if self.calibrated_model else self.model
            probs = predictor.predict_proba(X_aligned)[0]
            pred  = int(np.argmax(probs))
            label_map = {0: "SHORT", 1: "FLAT", 2: "LONG"}

            return {
                "signal":      label_map[pred],
                "confidence":  round(float(probs[pred]), 4),
                "prob_long":   round(float(probs[2]), 4),
                "prob_flat":   round(float(probs[1]), 4),
                "prob_short":  round(float(probs[0]), 4),
                "model_type":  "swing",
                "calibrated":  self.calibrated_model is not None,
            }
        except Exception as e:
            logger.error(f"Swing prediction error: {e}")
            return self._empty_prediction()

    # ─────────────────────────────────────────────────────────────
    # PRIVATE
    # ─────────────────────────────────────────────────────────────

    def _clean(self, X: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Align, clean, and validate feature matrix."""
        cols = [f for f in features if f in X.columns]
        return (
            X[cols]
            .fillna(0)
            .replace([np.inf, -np.inf], 0)
            .astype(np.float32)
        )

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._paths["model"]), exist_ok=True)
        joblib.dump(self.model,    self._paths["model"])
        joblib.dump(self.features, self._paths["features"])
        if self.calibrated_model is not None:
            joblib.dump(self.calibrated_model, self._paths["calibrated"])
            logger.info(
                f"Swing model ({self.variant}, calibrated) saved to "
                f"{self._paths['calibrated']}"
            )
        logger.info(f"Swing model ({self.variant}) saved to {self._paths['model']}")

    def _save_metrics(self) -> None:
        os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
        existing = {}
        if os.path.exists(METRICS_PATH):
            try:
                with open(METRICS_PATH) as f:
                    existing = json.load(f)
            except Exception:
                pass
        key = "swing" if self.variant == "technical" else f"swing_{self.variant}"
        existing[key] = self.metrics
        with open(METRICS_PATH, "w") as f:
            json.dump(existing, f, indent=2, default=str)

    def _empty_prediction(self) -> Dict:
        return {
            "signal": "FLAT", "confidence": 0.0,
            "prob_long": 0.33, "prob_flat": 0.34, "prob_short": 0.33,
            "model_type": "swing",
        }