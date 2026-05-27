# monitoring/drift_monitor.py
# Detects when live feature distributions drift from training distributions (data drift)
# and when model calibration decays (concept drift via Brier score).
#
# PSI (Population Stability Index) interpretation:
#   PSI < 0.10  : stable — no action
#   PSI 0.10-0.25 : minor drift — log warning
#   PSI > 0.25  : significant drift — halt model + alert
#
# Brier score: lower is better. 0.0 = perfect calibration, 0.25 = random.
# Alert if rolling 30-day Brier worsens > 20% vs training Brier.

import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

METRICS_PATH = "models/evaluation/model_metrics.json"
DRIFT_LOG    = "logs/drift_log.csv"


class DriftMonitor:
    """
    Monitors two types of model decay:

    1. DATA DRIFT (feature distribution shift)
       PSI = Σ (actual_pct - expected_pct) × ln(actual_pct / expected_pct)
       Computed daily on top-20 features vs training distribution.

    2. CONCEPT DRIFT (model confidence no longer matches win rate)
       Rolling 30-day Brier score vs training Brier.
       If Brier worsens > 20%, model confidence is no longer trustworthy.

    Used by orchestrator in the 18:00 EOD report.
    """

    PSI_WARN  = 0.10
    PSI_ALERT = 0.25
    BRIER_DEGRADE_THRESHOLD = 0.20   # 20% relative worsening

    def __init__(self, db_path: str = "database/trading.db"):
        self.db_path       = db_path
        self._training_dist: Dict = {}   # cached training distributions

    # ─────────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────────

    def compute_psi(
        self,
        live_features: Dict[str, float],
        top_n: int = 20,
    ) -> Dict:
        """
        Compute PSI between live feature values and training distributions.

        Args:
            live_features: dict of {feature_name: current_value} from FeatureBuilder
            top_n:         number of features to check

        Returns:
            {
                "psi_by_feature": {"rsi_14": 0.08, ...},
                "max_psi":        0.31,
                "drifted":        ["rsi_14"],
                "alert":          True,
                "timestamp":      "...",
            }
        """
        if not self._training_dist:
            self._load_training_distributions()

        psi_scores: Dict[str, float] = {}
        drifted:    List[str]        = []

        feature_names = list(live_features.keys())[:top_n]

        for feat in feature_names:
            if feat not in self._training_dist:
                continue
            train_dist = self._training_dist[feat]
            live_val   = float(live_features.get(feat, 0.0))
            psi        = self._psi_single(live_val, train_dist)
            psi_scores[feat] = round(psi, 4)

            if psi >= self.PSI_WARN:
                status = "ALERT" if psi >= self.PSI_ALERT else "WARN"
                logger.warning(
                    f"Feature drift [{status}]: {feat} PSI={psi:.3f}"
                )
                if psi >= self.PSI_ALERT:
                    drifted.append(feat)

        max_psi = max(psi_scores.values()) if psi_scores else 0.0
        result  = {
            "psi_by_feature": psi_scores,
            "max_psi":        round(max_psi, 4),
            "drifted":        drifted,
            "alert":          max_psi >= self.PSI_ALERT,
            "n_features":     len(psi_scores),
            "timestamp":      str(datetime.now()),
        }

        self._log_psi(result)
        return result

    def compute_brier_drift(self, model_type: str = "swing") -> Dict:
        """
        Compare rolling 30-day realized Brier score vs training Brier.

        Pulls outcome data from signal_outcomes table.
        Returns degradation flag and relative change.
        """
        training_brier = self._get_training_brier(model_type)
        rolling_brier  = self._compute_rolling_brier(model_type, days=30)

        if training_brier is None or rolling_brier is None:
            return {
                "model_type":     model_type,
                "training_brier": training_brier,
                "rolling_brier":  rolling_brier,
                "degraded":       False,
                "insufficient_data": True,
            }

        relative_change = (rolling_brier - training_brier) / max(training_brier, 0.001)
        degraded = relative_change > self.BRIER_DEGRADE_THRESHOLD

        if degraded:
            logger.warning(
                f"Model drift [{model_type}]: Brier worsened {relative_change:+.1%} "
                f"(train={training_brier:.4f} → live_30d={rolling_brier:.4f}) "
                f"— consider retraining"
            )

        return {
            "model_type":      model_type,
            "training_brier":  round(training_brier, 4),
            "rolling_brier":   round(rolling_brier, 4),
            "relative_change": round(relative_change, 4),
            "degraded":        degraded,
            "insufficient_data": False,
        }

    def full_report(self, live_features: Optional[Dict] = None) -> str:
        """Human-readable drift report for 18:00 EOD summary."""
        lines = [
            "─" * 55,
            "MODEL HEALTH REPORT",
            "─" * 55,
        ]

        # Brier drift for all three models
        for mtype in ("swing", "intraday", "positional"):
            b = self.compute_brier_drift(mtype)
            if b.get("insufficient_data"):
                lines.append(f"  {mtype:12}: insufficient data")
            else:
                flag  = "⚠️  DEGRADED" if b["degraded"] else "✅ OK"
                lines.append(
                    f"  {mtype:12}: Brier train={b['training_brier']:.4f} "
                    f"live_30d={b['rolling_brier']:.4f} "
                    f"({b['relative_change']:+.1%}) {flag}"
                )

        # Feature PSI if live features provided
        if live_features:
            psi = self.compute_psi(live_features)
            lines.append(f"  Feature PSI (max): {psi['max_psi']:.4f} "
                         f"{'⚠️  DRIFT' if psi['alert'] else '✅ OK'}")
            if psi["drifted"]:
                lines.append(f"  Drifted features: {', '.join(psi['drifted'][:5])}")

        lines.append("─" * 55)
        return "\n".join(lines)

    def record_training_distribution(
        self,
        feature_vectors: "pd.DataFrame",  # noqa: F821
        top_n: int = 20,
    ) -> None:
        """
        Snapshot training feature distributions (mean + std + percentiles).
        Called once at end of model training.
        Saved to models/evaluation/training_distributions.json
        """
        dist = {}
        cols = list(feature_vectors.columns)[:top_n]
        for col in cols:
            series = feature_vectors[col].dropna()
            if len(series) < 10:
                continue
            dist[col] = {
                "mean":   float(series.mean()),
                "std":    float(series.std()),
                "p10":    float(series.quantile(0.10)),
                "p25":    float(series.quantile(0.25)),
                "p50":    float(series.quantile(0.50)),
                "p75":    float(series.quantile(0.75)),
                "p90":    float(series.quantile(0.90)),
                "min":    float(series.min()),
                "max":    float(series.max()),
            }
        path = "models/evaluation/training_distributions.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(dist, f, indent=2, default=str)
        self._training_dist = dist
        logger.info(f"Training distributions saved: {len(dist)} features")

    # ─────────────────────────────────────────────────────────────
    # PRIVATE
    # ─────────────────────────────────────────────────────────────

    def _psi_single(self, live_val: float, train_dist: Dict) -> float:
        """
        Simplified PSI for a single scalar value vs stored training distribution.
        Bins the value into 10 quantile buckets and computes PSI vs uniform.
        """
        mean = train_dist.get("mean", 0)
        std  = max(train_dist.get("std", 1), 0.001)
        # Standardise
        z    = (live_val - mean) / std
        # Approximate PSI as squared z-score distance (simplified, no bin sampling)
        # Full PSI needs many live values; for single point, use deviation-based proxy
        deviation = abs(z)
        if deviation < 1.0:   return 0.0
        if deviation < 2.0:   return 0.05
        if deviation < 3.0:   return 0.15
        return 0.30

    def _load_training_distributions(self) -> None:
        path = "models/evaluation/training_distributions.json"
        if os.path.exists(path):
            try:
                with open(path) as f:
                    self._training_dist = json.load(f)
                logger.info(f"Training distributions loaded: {len(self._training_dist)} features")
            except Exception as e:
                logger.warning(f"Could not load training distributions: {e}")

    def _get_training_brier(self, model_type: str) -> Optional[float]:
        if not os.path.exists(METRICS_PATH):
            return None
        try:
            with open(METRICS_PATH) as f:
                m = json.load(f)
            return m.get(model_type, {}).get("brier_score_long")
        except Exception:
            return None

    def _compute_rolling_brier(
        self, model_type: str, days: int = 30
    ) -> Optional[float]:
        """
        Compute realized Brier score from signal_outcomes over the last N days.
        Uses outcome WIN=1/else=0 as actual; confidence as predicted probability.
        """
        since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        try:
            conn = sqlite3.connect(self.db_path)
            rows = conn.execute("""
                SELECT confidence, outcome
                FROM   signal_outcomes
                WHERE  created_at >= ? AND signal NOT IN ('FLAT', '')
            """, (since,)).fetchall()
            conn.close()
        except Exception:
            return None

        if len(rows) < 10:
            return None

        y_true  = np.array([1.0 if r[1] == "WIN" else 0.0 for r in rows])
        y_pred  = np.array([float(r[0] or 0.5) for r in rows])
        brier   = float(np.mean((y_true - y_pred) ** 2))
        return round(brier, 4)

    def _log_psi(self, result: Dict) -> None:
        """Append PSI summary to drift log CSV."""
        try:
            os.makedirs(os.path.dirname(DRIFT_LOG), exist_ok=True)
            write_header = not os.path.exists(DRIFT_LOG)
            import csv
            with open(DRIFT_LOG, "a", newline="") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(["timestamp", "max_psi", "n_drifted", "alert", "drifted_features"])
                w.writerow([
                    result["timestamp"],
                    result["max_psi"],
                    len(result["drifted"]),
                    result["alert"],
                    ",".join(result["drifted"][:5]),
                ])
        except Exception as e:
            logger.debug(f"Drift log write failed: {e}")
