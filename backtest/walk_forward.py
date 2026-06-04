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
from backtest.pipeline_backtest import PipelineBacktest
from backtest.metrics      import BacktestMetrics
from models.train_swing    import SwingModelTrainer
from models.feature_selector import FeatureSelector

RESULTS_PATH = "models/evaluation/backtest_results.json"
PIPELINE_RESULTS_PATH = "models/evaluation/pipeline_validation.json"


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
        use_pipeline:  bool  = False,
    ):
        self.train_days    = train_years * 252
        self.test_days     = test_months * 21   # ~21 trading days/month
        self.starting_capital = starting_capital
        # use_pipeline=True (P0-3) replays the FULL 6-gate decision logic with
        # costs (PipelineBacktest) instead of the bare model + 0.55 threshold
        # (BacktestEngine). The bare engine remains the default for the legacy
        # single-model accuracy backtest.
        self.use_pipeline  = use_pipeline
        self.engine        = BacktestEngine(starting_capital)
        self.pipeline      = PipelineBacktest(starting_capital)
        self.metrics_calc  = BacktestMetrics()
        self.feature_sel   = FeatureSelector()

    def _run_oos(self, df_test, model, feature_names, model_type, forward_days, threshold):
        """Run one out-of-sample test window through the chosen backtester."""
        if self.use_pipeline:
            return self.pipeline.run(
                df_test, model=model, feature_names=feature_names,
                category=model_type, forward_days=forward_days,
            )
        return self.engine.run(
            df_test, model=model, feature_names=feature_names,
            model_type=model_type, forward_days=forward_days, threshold=threshold,
        )

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

            # Test on out-of-sample period (bare model OR full gate pipeline)
            fold_result = self._run_oos(
                df_test, trainer.model, feature_names,
                model_type, forward_days, threshold,
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
        # Per-regime breakdown (pipeline mode tags every trade with the regime
        # in force at entry — tells you e.g. "counter-regime intraday PF 0.6").
        per_regime     = self._per_regime(all_trades) if self.use_pipeline else {}

        result = {
            "model_type":         model_type,
            "mode":               "pipeline" if self.use_pipeline else "bare_model",
            "n_folds":            len(fold_results),
            "total_trades":       len(all_trades),
            "fold_results":       fold_results,
            "aggregate_metrics":  agg_metrics,
            "per_regime_metrics": per_regime,
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

    def _per_regime(self, trades: List[Dict]) -> Dict:
        """Slice aggregate metrics by regime_at_entry (pipeline trades only)."""
        out: Dict[str, Dict] = {}
        regimes = sorted({t.get("regime_at_entry", "UNKNOWN") for t in trades})
        for r in regimes:
            subset = [t for t in trades if t.get("regime_at_entry") == r]
            if len(subset) >= 5:
                m = self.metrics_calc.calculate(subset, self.starting_capital)
                out[r] = {
                    "total_trades":  m.get("total_trades", 0),
                    "win_rate":      m.get("win_rate", 0),
                    "profit_factor": m.get("profit_factor", 0),
                    "expectancy_pct":m.get("expectancy_pct", 0),
                    "max_drawdown_pct": m.get("max_drawdown_pct", 0),
                }
            else:
                out[r] = {"total_trades": len(subset), "note": "too few for metrics"}
        return out

    def validate_holdout(
        self,
        df:             pd.DataFrame,
        model_type:     str   = "swing",
        holdout_months: int   = 12,
        forward_days:   int   = 5,
        threshold:      float = 0.025,
    ) -> Dict:
        """
        Single train-on-past / test-on-held-out-tail split (P0-3 core deliverable).

        Reserves the LAST `holdout_months` of data as a strict out-of-sample
        window the model never trains on, trains on everything before it, then
        replays the full gate pipeline (with costs) over the holdout. This is the
        honest "did the STRATEGY work on data it has never seen" test that gates
        capital scaling — distinct from CV accuracy (which is not strategy edge).
        """
        holdout_days = int(holdout_months * 21)
        if len(df) < self.train_days // 2 + holdout_days + forward_days:
            logger.error(
                f"Need more bars for a {holdout_months}-month holdout: have {len(df)}"
            )
            return {}

        split   = len(df) - holdout_days
        df_train = df.iloc[:split].copy()
        df_test  = df.iloc[split:].copy()
        logger.info(
            f"Holdout validation: {model_type} | "
            f"train [{str(df_train.index[0])[:10]} → {str(df_train.index[-1])[:10]}] | "
            f"holdout [{str(df_test.index[0])[:10]} → {str(df_test.index[-1])[:10]}] "
            f"({len(df_test)} bars)"
        )

        trainer, feature_names = self._train_fold(
            df_train, model_type, forward_days, threshold
        )
        if trainer is None:
            logger.error("Holdout training failed")
            return {}

        # Force the full gate pipeline for the holdout test regardless of the
        # instance default — the holdout exists to validate the STRATEGY.
        result = self.pipeline.run(
            df_test, model=trainer.model, feature_names=feature_names,
            category=model_type, forward_days=forward_days,
        )
        if not result:
            return {}

        metrics    = result["metrics"]
        stability  = {"consistency": 1.0, "profitable_folds": 1, "total_folds": 1}
        assessment = self._assess_deployability(metrics, stability)
        return {
            "model_type":         model_type,
            "mode":               "pipeline_holdout",
            "holdout_period":     result["backtest_period"],
            "holdout_months":     holdout_months,
            "total_trades":       metrics.get("total_trades", 0),
            "aggregate_metrics":  metrics,
            "per_regime_metrics": result.get("per_regime_metrics", {}),
            "funnel":             result.get("funnel", {}),
            "diagnostics":        result.get("diagnostics", {}),
            "assessment":         assessment,
            "is_deployable":      assessment["deployable"],
            "completed_at":       str(datetime.now()),
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


# ─────────────────────────────────────────────────────────────────
# CLI — P0-3 holdout validation of the FULL gate pipeline
# ─────────────────────────────────────────────────────────────────

# Scale-up bar (CLAUDE.md / PLAN.md): WR>52, PF>1.5, Sharpe>1, MaxDD<20, 50+ trades.
SCALE_UP_BAR = {
    "win_rate": 0.52, "profit_factor": 1.5, "sharpe_ratio": 1.0,
    "max_drawdown_pct": 0.20, "total_trades": 50,
}


def _scale_gate(metrics: Dict) -> Dict:
    """Go/no-go against the documented capital-scaling bar."""
    checks = {
        "win_rate":     metrics.get("win_rate", 0)      >= SCALE_UP_BAR["win_rate"],
        "profit_factor":metrics.get("profit_factor", 0) >= SCALE_UP_BAR["profit_factor"],
        "sharpe":       metrics.get("sharpe_ratio", 0)  >= SCALE_UP_BAR["sharpe_ratio"],
        "max_dd":       abs(metrics.get("max_drawdown_pct", -1)) <= SCALE_UP_BAR["max_drawdown_pct"],
        "n_trades":     metrics.get("total_trades", 0)  >= SCALE_UP_BAR["total_trades"],
    }
    return {"pass": all(checks.values()), "checks": checks}


def run_pipeline_holdout(
    ticker:         str   = "HDFCBANK.NS",
    holdout_months: int   = 12,
    categories:     tuple = ("swing", "intraday", "positional"),
    starting_capital: float = 500_000.0,
) -> Dict:
    """
    P0-3 deliverable: train on history, replay the full 6-gate pipeline (with
    costs) over a held-out tail, and report WR/PF/Sharpe/MaxDD per category AND
    per regime, plus the capital-scaling go/no-go. Writes a metrics JSON.
    """
    from data.price_fetcher import PriceFetcher

    forward_by_cat = {"swing": 5, "intraday": 1, "positional": 21}
    fetcher = PriceFetcher(ticker)
    df = fetcher.get_daily()
    if df is None or df.empty:
        logger.error(f"No price history for {ticker}")
        return {}

    validator = WalkForwardValidator(
        starting_capital=starting_capital, use_pipeline=True
    )

    report: Dict = {"ticker": ticker, "holdout_months": holdout_months, "categories": {}}
    for cat in categories:
        res = validator.validate_holdout(
            df, model_type=cat, holdout_months=holdout_months,
            forward_days=forward_by_cat.get(cat, 5),
        )
        if not res:
            report["categories"][cat] = {"error": "validation produced no result"}
            continue
        res["scale_gate"] = _scale_gate(res["aggregate_metrics"])
        report["categories"][cat] = res
        m = res["aggregate_metrics"]
        logger.info(
            f"[{cat}] holdout: trades={m.get('total_trades',0)} "
            f"WR={m.get('win_rate',0):.0%} PF={m.get('profit_factor',0):.2f} "
            f"Sharpe={m.get('sharpe_ratio',0):.2f} "
            f"MaxDD={m.get('max_drawdown_pct',0):.1%} | "
            f"scale={'PASS' if res['scale_gate']['pass'] else 'FAIL'}"
        )
        # Per-category completion marker (progress while sweeping banks/cats).
        print(
            f"✓ {ticker:12s} {cat:11s} DONE — "
            f"trades={m.get('total_trades',0):3d} "
            f"WR={m.get('win_rate',0):4.0%} PF={m.get('profit_factor',0):5.2f} "
            f"Sharpe={m.get('sharpe_ratio',0):6.2f} "
            f"MaxDD={m.get('max_drawdown_pct',0):6.1%} | "
            f"scale={'PASS ✅' if res['scale_gate']['pass'] else 'FAIL ❌'}",
            flush=True,
        )

    report["completed_at"] = str(datetime.now())
    report["scale_ready"]  = all(
        c.get("scale_gate", {}).get("pass", False)
        for c in report["categories"].values()
        if "scale_gate" in c
    ) and bool(report["categories"])

    os.makedirs(os.path.dirname(PIPELINE_RESULTS_PATH), exist_ok=True)
    with open(PIPELINE_RESULTS_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Pipeline holdout report saved: {PIPELINE_RESULTS_PATH}")
    return report


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="P0-3 full-pipeline holdout validation")
    ap.add_argument("--ticker", default="HDFCBANK.NS")
    ap.add_argument("--holdout-months", type=int, default=12)
    ap.add_argument("--capital", type=float, default=500_000.0)
    ap.add_argument("--category", default="all",
                    help="swing|intraday|positional|all")
    args = ap.parse_args()

    cats = (("swing", "intraday", "positional")
            if args.category == "all" else (args.category,))
    rep = run_pipeline_holdout(
        ticker=args.ticker, holdout_months=args.holdout_months,
        categories=cats, starting_capital=args.capital,
    )
    print("\n" + "=" * 64)
    print(f"PIPELINE HOLDOUT — {args.ticker} (last {args.holdout_months} months OOS)")
    print("=" * 64)
    for cat, res in rep.get("categories", {}).items():
        if "aggregate_metrics" not in res:
            print(f"  {cat:11s}: {res.get('error','—')}")
            continue
        m = res["aggregate_metrics"]
        gate = "PASS ✅" if res.get("scale_gate", {}).get("pass") else "FAIL ❌"
        print(
            f"  {cat:11s}: trades={m.get('total_trades',0):3d} "
            f"WR={m.get('win_rate',0):5.0%} PF={m.get('profit_factor',0):5.2f} "
            f"Sharpe={m.get('sharpe_ratio',0):5.2f} "
            f"MaxDD={m.get('max_drawdown_pct',0):6.1%} | scale={gate}"
        )
        for reg, rm in res.get("per_regime_metrics", {}).items():
            if rm.get("total_trades", 0) >= 5:
                print(
                    f"      └ {reg:16s} n={rm['total_trades']:3d} "
                    f"WR={rm.get('win_rate',0):4.0%} PF={rm.get('profit_factor',0):4.2f}"
                )
        # ── Funnel: where every candidate bar died ──
        funnel = res.get("funnel", {})
        if funnel:
            total = sum(funnel.values())
            order = ["gate4_FLAT", "gate1_CHOPPY", "positional_counter_regime",
                     "positional_HIGH_VOL", "gate5_RR_hardfloor", "gate5_other",
                     "counter_regime_gradeD", "gate6_conf", "open_failed_sizing",
                     "gate4_predict_error", "OPENED"]
            print(f"      funnel ({total} candidate bars):")
            for k in order:
                if funnel.get(k):
                    pct = 100 * funnel[k] / total
                    print(f"          {k:26s} {funnel[k]:4d}  ({pct:4.1f}%)")
            d = res.get("diagnostics", {})
            print(f"      diag: dir={d.get('model_direction_dist')} "
                  f"g5_grades={d.get('gate5_grade_dist')} "
                  f"g5_rr_med={d.get('gate5_rr_median')} "
                  f"g6_gap_min={d.get('gate6_gap_min')}")
    print("=" * 64)
    print(f"SCALE-READY (all categories pass bar): "
          f"{'YES ✅' if rep.get('scale_ready') else 'NO ❌'}")
    print("=" * 64)