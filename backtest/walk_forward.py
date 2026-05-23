# backtest/walk_forward.py
# Walk-forward validation — the gold standard for live trading readiness
# Train on N years, test on next M months, roll forward, repeat
# Connected to: backtest/engine.py, models/train_all.py

import numpy as np
import pandas as pd
import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

from backtest.engine       import BacktestEngine
from backtest.metrics      import BacktestMetrics
from models.train_swing    import SwingModelTrainer
from models.feature_selector import FeatureSelector

RESULTS_PATH = "models/evaluation/backtest_results.json"


class WalkForwardValidator:
    """
    Walk-forward validation — simulates how system would perform in real life.

    20yr trader perspective:
    Standard backtests lie. You optimise parameters on all historical data,
    then test on the same historical data. Of course it looks great.

    Walk-forward is honest:
    1. Train on 2 years of data
    2. Test on NEXT 6 months (data the model has never seen)
    3. Roll forward 6 months
    4. Retrain on 2 years ending at new date
    5. Test on next 6 months
    6. Repeat until no more data

    If walk-forward performance is close to in-sample — system is robust.
    If walk-forward is significantly worse — system is overfit. Do NOT deploy.

    Acceptable degradation: walk-forward win rate within 8% of in-sample.
    E.g. in-sample 62% → walk-forward minimum 54%.
    """

    def __init__(
        self,
        train_years:   int   = 2,
        test_months:   int   = 6,
        starting_capital: float = 100_000.0,
    ):
        self.train_days    = train_years * 252
        self.test_days     = test_months * 21   # ~21 trading days/month
        self.starting_capital = starting_capital
        self.engine        = BacktestEngine(starting_capital)
        self.metrics_calc  = BacktestMetrics()
        self.feature_sel   = FeatureSelector()

    # ─────────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────────

    def validate(
        self,
        df:           pd.DataFrame,
        model_type:   str   = "swing",
        forward_days: int   = 5,
        threshold:    float = 0.025,
    ) -> Dict:
        """
        Run full walk-forward validation.

        Args:
            df:           Full historical OHLCV DataFrame
            model_type:   "swing" | "intraday" | "positional"
            forward_days: Prediction horizon
            threshold:    Signal threshold

        Returns:
            Aggregated walk-forward results with stability assessment
        """
        min_required = self.train_days + self.test_days + forward_days
        if len(df) < min_required:
            logger.error(
                f"Need {min_required}+ bars for walk-forward, "
                f"got {len(df)}"
            )
            return {}

        logger.info(
            f"Walk-forward validation: {model_type} | "
            f"train={self.train_days}d | test={self.test_days}d | "
            f"{len(df)} total bars"
        )

        windows      = self._generate_windows(len(df))
        fold_results = []
        all_trades   = []

        for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(windows):
            logger.info(
                f"  Fold {fold_idx+1}/{len(windows)}: "
                f"train [{str(df.index[train_start])[:10]} → "
                f"{str(df.index[train_end-1])[:10]}] | "
                f"test  [{str(df.index[test_start])[:10]} → "
                f"{str(df.index[test_end-1])[:10]}]"
            )

            df_train = df.iloc[train_start:train_end].copy()
            df_test  = df.iloc[test_start:test_end].copy()

            # Train model on this fold's training data
            trainer, feature_names = self._train_fold(
                df_train, model_type, forward_days, threshold
            )
            if trainer is None:
                logger.warning(f"  Fold {fold_idx+1}: Training failed — skipping")
                continue

            # Test on out-of-sample period
            fold_result = self.engine.run(
                df_test,
                model     = trainer.model,
                feature_names = feature_names,
                model_type    = model_type,
                forward_days  = forward_days,
                threshold     = threshold,
            )
            if not fold_result or not fold_result.get("trades"):
                logger.warning(f"  Fold {fold_idx+1}: No trades generated")
                continue

            fold_metrics = fold_result["metrics"]
            fold_results.append({
                "fold":          fold_idx + 1,
                "train_start":   str(df.index[train_start])[:10],
                "train_end":     str(df.index[train_end-1])[:10],
                "test_start":    str(df.index[test_start])[:10],
                "test_end":      str(df.index[test_end-1])[:10],
                "n_trades":      len(fold_result["trades"]),
                "win_rate":      fold_metrics.get("win_rate", 0),
                "profit_factor": fold_metrics.get("profit_factor", 0),
                "sharpe":        fold_metrics.get("sharpe_ratio", 0),
                "max_dd":        fold_metrics.get("max_drawdown_pct", 0),
                "total_return":  fold_metrics.get("total_return_pct", 0),
            })
            all_trades.extend(fold_result["trades"])

            logger.info(
                f"    → WR={fold_metrics.get('win_rate',0):.0%} | "
                f"PF={fold_metrics.get('profit_factor',0):.2f} | "
                f"Trades={len(fold_result['trades'])}"
            )

        if not fold_results:
            logger.error("Walk-forward produced no fold results")
            return {}

        # ── Aggregate across all folds ────────────────────────────
        agg_metrics    = self.metrics_calc.calculate(all_trades, self.starting_capital)
        stability      = self._stability_analysis(fold_results)
        assessment     = self._assess_deployability(agg_metrics, stability)

        result = {
            "model_type":         model_type,
            "n_folds":            len(fold_results),
            "total_trades":       len(all_trades),
            "fold_results":       fold_results,
            "aggregate_metrics":  agg_metrics,
            "stability":          stability,
            "assessment":         assessment,
            "is_deployable":      assessment["deployable"],
            "completed_at":       str(datetime.now()),
        }

        self._save_wf_results(result, model_type)

        logger.info(
            f"Walk-forward complete: {model_type} | "
            f"Folds={len(fold_results)} | "
            f"Trades={len(all_trades)} | "
            f"WR={agg_metrics.get('win_rate',0):.0%} | "
            f"Deployable={assessment['deployable']}"
        )
        return result

    # ─────────────────────────────────────────────────────────────
    # PRIVATE
    # ─────────────────────────────────────────────────────────────

    def _generate_windows(
        self, total_bars: int
    ) -> List[Tuple[int, int, int, int]]:
        """Generate (train_start, train_end, test_start, test_end) index tuples."""
        windows   = []
        start     = 0
        while start + self.train_days + self.test_days <= total_bars:
            train_start = start
            train_end   = start + self.train_days
            test_start  = train_end
            test_end    = min(test_start + self.test_days, total_bars)
            if test_end - test_start >= 20:  # need at least 20 test bars
                windows.append((train_start, train_end, test_start, test_end))
            start      += self.test_days
        return windows

    def _train_fold(
        self,
        df_train:     pd.DataFrame,
        model_type:   str,
        forward_days: int,
        threshold:    float,
    ):
        """Train a fresh model on one fold's training data."""
        try:
            from features.feature_builder import FeatureBuilder
            fb = FeatureBuilder()
            X, y = fb.build_training_dataset(
                df_train, model_type=model_type,
                forward_days=forward_days, threshold=threshold,
            )
            if X.empty or len(X) < 50:
                return None, []

            trainer = SwingModelTrainer()
            trainer.train(X, y, list(X.columns), save=False)
            return trainer, list(X.columns)
        except Exception as e:
            logger.warning(f"Fold training error: {e}")
            return None, []

    def _stability_analysis(self, fold_results: List[Dict]) -> Dict:
        """Analyse consistency across folds."""
        win_rates   = [f["win_rate"]      for f in fold_results]
        prof_factors= [f["profit_factor"] for f in fold_results]
        sharpes     = [f["sharpe"]        for f in fold_results]

        return {
            "win_rate_mean":   round(float(np.mean(win_rates)),    4),
            "win_rate_std":    round(float(np.std(win_rates)),     4),
            "win_rate_min":    round(float(np.min(win_rates)),     4),
            "pf_mean":         round(float(np.mean(prof_factors)), 3),
            "pf_min":          round(float(np.min(prof_factors)),  3),
            "sharpe_mean":     round(float(np.mean(sharpes)),      3),
            "profitable_folds":sum(1 for wr in win_rates if wr > 0.50),
            "total_folds":     len(fold_results),
            "consistency":     round(
                sum(1 for wr in win_rates if wr > 0.50) / len(win_rates), 3
            ),
        }

    def _assess_deployability(
        self, metrics: Dict, stability: Dict
    ) -> Dict:
        """
        Final go/no-go assessment for live deployment.
        Must pass ALL criteria.
        """
        checks   = {}
        reasons  = []

        checks["win_rate_ok"] = metrics.get("win_rate", 0) >= 0.52
        if not checks["win_rate_ok"]:
            reasons.append(f"WR {metrics.get('win_rate',0):.0%} < 52%")

        checks["profit_factor_ok"] = metrics.get("profit_factor", 0) >= 1.4
        if not checks["profit_factor_ok"]:
            reasons.append(f"PF {metrics.get('profit_factor',0):.2f} < 1.4")

        checks["max_dd_ok"] = abs(metrics.get("max_drawdown_pct", -1)) <= 0.22
        if not checks["max_dd_ok"]:
            reasons.append(f"MaxDD {metrics.get('max_drawdown_pct',0):.1%} > 22%")

        checks["consistency_ok"] = stability.get("consistency", 0) >= 0.60
        if not checks["consistency_ok"]:
            reasons.append(
                f"Only {stability['profitable_folds']}/{stability['total_folds']} "
                f"profitable folds"
            )

        checks["enough_trades"] = metrics.get("total_trades", 0) >= 30
        if not checks["enough_trades"]:
            reasons.append(f"Only {metrics.get('total_trades',0)} trades (need 30+)")

        deployable = all(checks.values())

        return {
            "deployable":     deployable,
            "checks":         checks,
            "fail_reasons":   reasons,
            "recommendation": (
                "✅ DEPLOY — system meets all walk-forward criteria"
                if deployable else
                f"❌ DO NOT DEPLOY — {len(reasons)} criteria failed: "
                + "; ".join(reasons)
            ),
        }

    def _save_wf_results(self, result: Dict, model_type: str) -> None:
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        existing = {}
        if os.path.exists(RESULTS_PATH):
            try:
                with open(RESULTS_PATH) as f:
                    existing = json.load(f)
            except Exception:
                pass
        existing[f"{model_type}_walk_forward"] = {
            "aggregate_metrics": result["aggregate_metrics"],
            "stability":         result["stability"],
            "assessment":        result["assessment"],
            "n_folds":           result["n_folds"],
            "total_trades":      result["total_trades"],
            "completed_at":      result["completed_at"],
        }
        with open(RESULTS_PATH, "w") as f:
            json.dump(existing, f, indent=2, default=str)
        logger.info(f"Walk-forward results saved: {RESULTS_PATH}")