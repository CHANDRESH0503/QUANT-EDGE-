# models/train_all.py
# Master trainer — orchestrates all 4 models in correct sequence
# Called every Sunday midnight by scheduler
# Also called manually when system is first deployed

import numpy as np
import pandas as pd
import logging
import json
import os
import joblib
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)

from models.train_swing      import SwingModelTrainer
from models.train_intraday   import IntradayModelTrainer
from models.train_positional import PositionalModelTrainer
from models.train_regime     import RegimeModelTrainer
from models.feature_selector import FeatureSelector

from data.price_fetcher      import PriceFetcher
from features.feature_builder import FeatureBuilder

_METRICS_PATH_TEMPLATE = "models/evaluation/{ticker_safe}_model_metrics.json"


class ModelTrainer:
    """
    Orchestrates training of all 4 models in the correct sequence.

    Training order:
    1. Regime model (HMM)    — needs raw OHLCV only
    2. Positional model      — weekly candles + fundamentals
    3. Swing model           — daily candles + all features (PRIMARY)
    4. Intraday model        — 5-min candles + order flow

    20yr trader rule on retraining:
    Retrain every Sunday midnight with the latest week's data added.
    Never retrain mid-week — you need a stable model during live trading.
    Replace the live model ONLY if new model accuracy >= old - 1%.
    If new model is significantly worse — keep the old one.
    The market may have changed temporarily (regime shift, news event).
    One bad week does not mean the model is broken.

    Weekly retrain takes ~15-30 minutes on CPU.
    Do it at midnight when no trades are active.
    """

    def __init__(
        self,
        ticker:  str   = "HDFCBANK.NS",
        capital: float = 100_000.0,
        db_path: str   = "database/trading.db",
    ):
        self.ticker      = ticker
        self.ticker_safe = ticker.replace(".NS", "")
        self.capital     = capital
        self.db_path     = db_path
        self.metrics_path = _METRICS_PATH_TEMPLATE.format(ticker_safe=self.ticker_safe)

        # Trainers — technical variant trains on 25yr history (baseline / fallback)
        self.swing_trainer      = SwingModelTrainer(variant="technical", ticker=ticker)
        self.intraday_trainer   = IntradayModelTrainer(variant="technical", ticker=ticker)
        self.positional_trainer = PositionalModelTrainer(variant="technical", ticker=ticker)
        # Trainers — full variant trains on 2020+ window with options/FII/news features
        self.swing_trainer_full      = SwingModelTrainer(variant="full", ticker=ticker)
        self.intraday_trainer_full   = IntradayModelTrainer(variant="full", ticker=ticker)
        self.positional_trainer_full = PositionalModelTrainer(variant="full", ticker=ticker)
        # Trainers — prod variant: 25yr history + verified-non-zero production features only
        # Higher priority than full: more samples, no zero-inflation overfit
        self.swing_trainer_prod      = SwingModelTrainer(variant="prod", ticker=ticker)
        self.intraday_trainer_prod   = IntradayModelTrainer(variant="prod", ticker=ticker)
        self.positional_trainer_prod = PositionalModelTrainer(variant="prod", ticker=ticker)
        self.regime_trainer     = RegimeModelTrainer(ticker=ticker)
        self.feature_selector   = FeatureSelector()

        # Data
        self.price_fetcher  = PriceFetcher(ticker)
        self.feature_builder = FeatureBuilder(ticker, capital, db_path)

    # ─────────────────────────────────────────────────────────────
    # PUBLIC — MAIN ENTRY POINTS
    # ─────────────────────────────────────────────────────────────

    def train_all(self, force: bool = False) -> Dict:
        """
        Full training pipeline — all 4 models.
        Called every Sunday midnight by scheduler.

        Args:
            force: If True, replace model even if accuracy drops.
                   Use only for initial deployment.

        Returns:
            Summary dict with metrics for all 4 models.
        """
        t0      = datetime.now()
        summary = {"started_at": str(t0), "models": {}}

        logger.info("=" * 60)
        logger.info("WEEKLY RETRAIN — All models")
        logger.info("=" * 60)

        # ── 1. Regime model ───────────────────────────────────────
        logger.info("\n[1/4] Training REGIME model (HMM)...")
        regime_metrics = self._train_regime()
        summary["models"]["regime"] = regime_metrics

        # ── 2. Swing model ────────────────────────────────────────
        logger.info("\n[2/4] Training SWING model (primary)...")
        swing_metrics = self._train_swing(force=force)
        summary["models"]["swing"] = swing_metrics

        # ── 3. Positional model ───────────────────────────────────
        logger.info("\n[3/4] Training POSITIONAL model...")
        pos_metrics = self._train_positional(force=force)
        summary["models"]["positional"] = pos_metrics

        # ── 4. Intraday model ─────────────────────────────────────
        logger.info("\n[4/4] Training INTRADAY model...")
        intra_metrics = self._train_intraday(force=force)
        summary["models"]["intraday"] = intra_metrics

        # ── Summary ───────────────────────────────────────────────
        elapsed = (datetime.now() - t0).total_seconds()
        summary["completed_at"]   = str(datetime.now())
        summary["elapsed_seconds"]= round(elapsed, 1)
        summary["success"]        = all(
            bool(m) for m in summary["models"].values()
        )

        self._save_summary(summary)

        logger.info("\n" + "=" * 60)
        logger.info(f"RETRAIN COMPLETE — {elapsed:.0f}s")
        logger.info(
            f"Swing CV:      {swing_metrics.get('cv_accuracy_mean', 0):.3f}"
        )
        logger.info(
            f"Intraday CV:   {intra_metrics.get('cv_accuracy_mean', 0):.3f}"
        )
        logger.info(
            f"Positional CV: {pos_metrics.get('cv_accuracy_mean', 0):.3f}"
        )
        logger.info("=" * 60)

        return summary

    def train_swing_only(self, force: bool = False) -> Dict:
        """Retrain only the swing model. Useful for quick updates."""
        return self._train_swing(force=force)

    def train_regime_only(self) -> Dict:
        """Retrain only the regime model."""
        return self._train_regime()

    def load_all(self) -> Dict:
        """
        Load all pre-trained models from disk (both variants where available).
        Called at system startup to restore state.
        Returns dict of load success per model.
        """
        # Load all variants; Gate 4 picks the best available: prod → full → technical
        self.swing_trainer_prod.load()
        self.swing_trainer_full.load()
        self.intraday_trainer_prod.load()
        self.intraday_trainer_full.load()
        self.positional_trainer_prod.load()
        self.positional_trainer_full.load()
        return {
            "swing":      self.swing_trainer.load(),
            "intraday":   self.intraday_trainer.load(),
            "positional": self.positional_trainer.load(),
            "regime":     self.regime_trainer.load(),
        }

    def predict_all(
        self,
        df_daily:    pd.DataFrame,
        df_intraday: pd.DataFrame,
        df_weekly:   pd.DataFrame,
        feature_vector: Dict,
    ) -> Dict:
        """
        Run all 3 trading models and return combined prediction.
        Regime is used separately in gate 1.

        Args:
            df_daily:       Daily OHLCV
            df_intraday:    5-min OHLCV
            df_weekly:      Weekly OHLCV
            feature_vector: Raw feature dict from FeatureBuilder

        Returns:
            Combined prediction dict for signal_engine.py
        """
        # Convert feature vector to DataFrames for each model
        raw = feature_vector.get("raw_features", {})

        swing_row  = pd.DataFrame([{
            f: raw.get(f, 0.0)
            for f in self.swing_trainer.features or []
        }])
        intra_row  = pd.DataFrame([{
            f: raw.get(f, 0.0)
            for f in self.intraday_trainer.features or []
        }])
        pos_row    = pd.DataFrame([{
            f: raw.get(f, 0.0)
            for f in self.positional_trainer.features or []
        }])

        swing_pred      = self.swing_trainer.predict(swing_row)
        intraday_pred   = self.intraday_trainer.predict(intra_row)
        positional_pred = self.positional_trainer.predict(pos_row)
        regime_result   = self.regime_trainer.detect(df_daily)

        # Timeframe alignment
        signals = [
            positional_pred["signal"],
            swing_pred["signal"],
            intraday_pred["signal"],
        ]
        non_flat = [s for s in signals if s != "FLAT"]
        if len(set(non_flat)) == 1 and len(non_flat) == 3:
            alignment = "A+"
            conf_boost = 0.15
        elif len(set(non_flat)) == 1 and len(non_flat) == 2:
            alignment = "A"
            conf_boost = 0.10
        elif len(non_flat) == 1:
            alignment = "B"
            conf_boost = 0.05
        elif len(set(non_flat)) > 1:
            alignment = "F"
            conf_boost = -0.20
        else:
            alignment = "C"
            conf_boost = 0.0

        # Boosted swing confidence is the primary confidence
        primary_conf = min(
            0.95,
            swing_pred["confidence"] + conf_boost
        )

        return {
            "positional":       positional_pred,
            "swing":            swing_pred,
            "intraday":         intraday_pred,
            "regime":           regime_result,
            "alignment":        alignment,
            "conf_boost":       conf_boost,
            "primary_signal":   swing_pred["signal"],
            "primary_conf":     round(primary_conf, 4),
            "predicted_at":     str(datetime.now()),
        }

    def get_model_metrics(self) -> Dict:
        """Load and return all saved model metrics."""
        try:
            if os.path.exists(self.metrics_path):
                with open(self.metrics_path) as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load metrics: {e}")
        return {}

    # ─────────────────────────────────────────────────────────────
    # PRIVATE TRAINING METHODS
    # ─────────────────────────────────────────────────────────────

    def _train_regime(self) -> Dict:
        """Train HMM regime model on 25yr daily data."""
        try:
            df = self.price_fetcher.get_daily(start="2000-01-01")
            if df.empty:
                logger.error("No daily data for regime training")
                return {}
            return self.regime_trainer.train(df, save=True)
        except Exception as e:
            logger.error(f"Regime training failed: {e}")
            return {}

    def _train_swing(self, force: bool = False) -> Dict:
        """
        Train BOTH swing model variants:
          - technical: 25yr daily data, technical features only (fallback)
          - full:      2020+ daily data with options/FII/FinBERT features

        Returns metrics from the full variant if successful, else technical.
        """
        try:
            df = self.price_fetcher.get_daily(start="2000-01-01")
            if df.empty:
                return {}

            # ── Technical variant (25yr baseline) ─────────────────
            logger.info("Building swing training dataset (technical, 25yr)...")
            X_t, y_t = self.feature_builder.build_training_dataset(
                df, model_type="swing",
                forward_days=5, threshold=0.025,
                data_window="full_history",
            )
            tech_metrics = self._train_one_variant(
                trainer="swing", variant_name="technical",
                X=X_t, y=y_t, force=force,
            )

            # ── Full variant (2020+ with external features) ──────
            logger.info("Building swing training dataset (full, 2020+)...")
            X_f, y_f = self.feature_builder.build_training_dataset(
                df, model_type="swing",
                forward_days=5, threshold=0.025,
                data_window="recent_with_features",
            )
            full_metrics = self._train_one_variant(
                trainer="swing", variant_name="full",
                X=X_f, y=y_f, force=force,
            )

            # ── Prod variant (25yr + verified non-zero features) ──
            # More samples than full (5,800+ vs 1,500), no zero-inflation.
            # This is the preferred variant for Gate 4's primary signal.
            logger.info("Building swing training dataset (prod, 25yr)...")
            X_p, y_p = self.feature_builder.build_training_dataset(
                df, model_type="swing_prod",
                forward_days=5, threshold=0.025,
                data_window="full_history",
            )
            prod_metrics = self._train_one_variant(
                trainer="swing", variant_name="prod",
                X=X_p, y=y_p, force=force,
            )

            return prod_metrics or full_metrics or tech_metrics

        except Exception as e:
            logger.error(f"Swing training failed: {e}")
            return {}

    def _train_one_variant(
        self,
        trainer:      str,
        variant_name: str,
        X:            pd.DataFrame,
        y:            pd.Series,
        force:        bool,
    ) -> Dict:
        """
        Train a single trainer variant, gate save on CV vs existing.

        Args:
            trainer:      "swing" | "intraday" | "positional"
            variant_name: "technical" | "full"
        """
        if X.empty:
            return {}

        if variant_name == "technical":
            trainer_attr = f"{trainer}_trainer"
        elif variant_name == "prod":
            trainer_attr = f"{trainer}_trainer_prod"
        else:
            trainer_attr = f"{trainer}_trainer_full"
        t = getattr(self, trainer_attr)
        metrics_key = trainer if variant_name == "technical" else f"{trainer}_{variant_name}"

        old_acc = self._get_existing_accuracy(metrics_key)
        metrics = t.train(X, y, feature_names=list(X.columns), save=False)
        new_acc = metrics.get("cv_accuracy_mean", 0)

        if force or old_acc is None or new_acc >= old_acc - 0.01:
            t._save()
            t._save_metrics()
            tag = f"{trainer}({variant_name})"
            logger.info(
                f"{tag} REPLACED: {old_acc:.3f} → {new_acc:.3f}"
                if old_acc else f"{tag} SAVED: CV={new_acc:.3f}"
            )
        else:
            logger.warning(
                f"{trainer}({variant_name}) NOT replaced: "
                f"new={new_acc:.3f} < old={old_acc:.3f}"
            )
        return metrics

    def _train_positional(self, force: bool = False) -> Dict:
        """Train BOTH positional variants (technical 25yr + full 2020+)."""
        try:
            df_weekly = self.price_fetcher.get_weekly(start="2000-01-01")
            if df_weekly.empty:
                return {}

            logger.info("Building positional training dataset (technical, 25yr)...")
            X_t, y_t = self.feature_builder.build_training_dataset(
                df_weekly, model_type="positional",
                forward_days=3, threshold=0.05,
                data_window="full_history",
            )
            tech_metrics = self._train_one_variant(
                trainer="positional", variant_name="technical",
                X=X_t, y=y_t, force=force,
            )

            logger.info("Building positional training dataset (full, 2020+)...")
            X_f, y_f = self.feature_builder.build_training_dataset(
                df_weekly, model_type="positional",
                forward_days=3, threshold=0.05,
                data_window="recent_with_features",
            )
            full_metrics = self._train_one_variant(
                trainer="positional", variant_name="full",
                X=X_f, y=y_f, force=force,
            )

            logger.info("Building positional training dataset (prod, 25yr)...")
            X_p, y_p = self.feature_builder.build_training_dataset(
                df_weekly, model_type="positional_prod",
                forward_days=3, threshold=0.05,
                data_window="full_history",
            )
            prod_metrics = self._train_one_variant(
                trainer="positional", variant_name="prod",
                X=X_p, y=y_p, force=force,
            )

            return prod_metrics or full_metrics or tech_metrics

        except Exception as e:
            logger.error(f"Positional training failed: {e}")
            return {}

    def _train_intraday(self, force: bool = False) -> Dict:
        """
        Train BOTH intraday variants. Intraday only has 60 days of 5-min data,
        so the 'full' variant typically still trains on the same window —
        but with external features joined when available.
        """
        try:
            df_5m = self.price_fetcher.get_intraday(interval="5m")
            if df_5m.empty:
                logger.warning("No 5-min data — intraday model not retrained")
                return {}

            logger.info("Building intraday training dataset (technical)...")
            X_t, y_t = self.feature_builder.build_training_dataset(
                df_5m, model_type="intraday",
                forward_days=6, threshold=0.004,
                data_window="full_history",
            )
            tech_metrics = self._train_one_variant(
                trainer="intraday", variant_name="technical",
                X=X_t, y=y_t, force=force,
            )

            logger.info("Building intraday training dataset (full)...")
            X_f, y_f = self.feature_builder.build_training_dataset(
                df_5m, model_type="intraday",
                forward_days=6, threshold=0.004,
                data_window="recent_with_features",
            )
            full_metrics = self._train_one_variant(
                trainer="intraday", variant_name="full",
                X=X_f, y=y_f, force=force,
            )

            logger.info("Building intraday training dataset (prod)...")
            X_p, y_p = self.feature_builder.build_training_dataset(
                df_5m, model_type="intraday_prod",
                forward_days=6, threshold=0.004,
                data_window="full_history",
            )
            prod_metrics = self._train_one_variant(
                trainer="intraday", variant_name="prod",
                X=X_p, y=y_p, force=force,
            )

            return prod_metrics or full_metrics or tech_metrics

        except Exception as e:
            logger.error(f"Intraday training failed: {e}")
            return {}

    # ─────────────────────────────────────────────────────────────
    # ACCURACY MANAGEMENT
    # ─────────────────────────────────────────────────────────────

    def _get_existing_accuracy(self, model_type: str) -> Optional[float]:
        """
        Load CV accuracy of the currently deployed model.
        Used to decide whether to replace with newly trained model.
        Returns None if no existing model metrics found.
        """
        try:
            metrics = self.get_model_metrics()
            return float(metrics.get(model_type, {}).get("cv_accuracy_mean", 0)) or None
        except Exception:
            return None

    # ─────────────────────────────────────────────────────────────
    # PERSISTENCE
    # ─────────────────────────────────────────────────────────────

    def _save_summary(self, summary: Dict) -> None:
        """Save training summary for audit trail."""
        path = "models/evaluation/retrain_history.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        history = []
        if os.path.exists(path):
            try:
                with open(path) as f:
                    history = json.load(f)
            except Exception:
                pass
        history.append(summary)
        # Keep last 52 weeks
        history = history[-52:]
        with open(path, "w") as f:
            json.dump(history, f, indent=2, default=str)
        logger.info(f"Retrain summary saved to {path}")