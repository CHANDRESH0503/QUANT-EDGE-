# processing/anomaly_detector.py
# Detects unusual market behavior before it shows in price
# Uses Isolation Forest — unsupervised ML, no labels needed
# Connected to: technical.py (features), signal_engine.py (gate 2)

import numpy as np
import pandas as pd
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not installed. pip install scikit-learn")


class AnomalyDetector:
    """
    Detects abnormal market behavior using Isolation Forest.

    20yr trader rule: When you see something you can not explain,
    step aside. Anomalies precede large unexpected moves — either
    a massive opportunity or a trap. Either way, size down.

    Three anomaly types detected:
    1. Volume anomaly    — 3x+ normal volume without price justification
    2. Price anomaly     — gap or spike outside 3-sigma band
    3. Combined anomaly  — multiple signals firing simultaneously

    Connects to:
    - technical.py: uses ATR, volume_ratio, returns features
    - signal_engine.py gate 2: anomaly = reduce position size 50%
    """

    CONTAMINATION = 0.05  # expect 5% of days to be anomalous

    def __init__(self, db_path: str = "database/trading.db"):
        self.db_path = db_path
        self._model  = None
        self._scaler = None

    # ─────────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> None:
        """
        Train Isolation Forest on historical data.
        Called during weekly retrain cycle.
        Uses last 252 days of OHLCV data.
        """
        if not SKLEARN_AVAILABLE or len(df) < 50:
            return

        X = self._build_anomaly_features(df).dropna()
        if X.empty:
            return

        self._scaler = StandardScaler()
        X_scaled     = self._scaler.fit_transform(X)

        self._model  = IsolationForest(
            n_estimators=200,
            contamination=self.CONTAMINATION,
            random_state=42,
            n_jobs=-1,
        )
        self._model.fit(X_scaled)
        logger.info(f"Anomaly detector trained on {len(X)} samples")

    def detect(self, df: pd.DataFrame) -> Dict:
        """
        Detect anomalies in the most recent bar.
        Returns anomaly flag, type, severity, and recommended action.
        Called every 15 minutes by signal engine.
        """
        # Rule-based detection always runs (no model needed)
        rule_result = self._rule_based_detection(df)

        # ML-based detection if model trained
        ml_result = self._ml_detection(df) if (
            self._model is not None and SKLEARN_AVAILABLE
        ) else {"ml_anomaly": False, "ml_score": 0.0}

        # Combine
        is_anomaly  = rule_result["rule_anomaly"] or ml_result["ml_anomaly"]
        severity    = self._calculate_severity(rule_result, ml_result, df)
        action      = self._recommended_action(is_anomaly, severity)

        return {
            "is_anomaly":         is_anomaly,
            "anomaly_type":       rule_result.get("anomaly_type", "NONE"),
            "severity":           severity,           # LOW / MEDIUM / HIGH
            "rule_anomaly":       rule_result["rule_anomaly"],
            "ml_anomaly":         ml_result["ml_anomaly"],
            "volume_spike_ratio": rule_result.get("volume_ratio", 1.0),
            "price_sigma":        rule_result.get("price_sigma", 0.0),
            "recommended_action": action,
            "position_size_mult": 0.5 if is_anomaly and severity == "HIGH" else
                                  0.75 if is_anomaly else 1.0,
        }

    # ─────────────────────────────────────────────────────────────
    # DETECTION METHODS
    # ─────────────────────────────────────────────────────────────

    def _rule_based_detection(self, df: pd.DataFrame) -> Dict:
        """
        Fast rule-based checks that run every 15 minutes.
        These are the patterns a senior trader notices immediately.
        """
        result = {"rule_anomaly": False, "anomaly_type": "NONE"}
        if len(df) < 20:
            return result

        close  = df["Close"]
        volume = df["Volume"]
        high   = df["High"]
        low    = df["Low"]

        # ── Volume anomaly ──
        avg_vol    = float(volume.rolling(20).mean().iloc[-1])
        today_vol  = float(volume.iloc[-1])
        vol_ratio  = today_vol / max(avg_vol, 1)
        result["volume_ratio"] = round(vol_ratio, 2)

        if vol_ratio > 5.0:
            result["rule_anomaly"] = True
            result["anomaly_type"] = "EXTREME_VOLUME"

        # ── Price sigma anomaly ──
        daily_ret  = close.pct_change().dropna()
        mu         = float(daily_ret.rolling(20).mean().iloc[-1])
        sigma      = float(daily_ret.rolling(20).std().iloc[-1])
        latest_ret = float(daily_ret.iloc[-1])
        price_sigma = (latest_ret - mu) / max(sigma, 0.0001)
        result["price_sigma"] = round(price_sigma, 2)

        if abs(price_sigma) > 3.5:
            result["rule_anomaly"] = True
            result["anomaly_type"] = "PRICE_SHOCK"

        # ── Absorption: high volume + tiny price move ──
        atr_approx = float((high - low).rolling(14).mean().iloc[-1])
        today_range = float(high.iloc[-1] - low.iloc[-1])
        if vol_ratio > 2.5 and today_range < atr_approx * 0.3:
            result["rule_anomaly"] = True
            result["anomaly_type"] = "ABSORPTION"

        # ── Gap anomaly ──
        gap_pct = abs(float(
            (close.iloc[-1] - close.iloc[-2]) / max(close.iloc[-2], 1)
        ))
        if gap_pct > 0.04 and vol_ratio < 1.0:
            result["rule_anomaly"] = True
            result["anomaly_type"] = "LOW_VOL_GAP"

        return result

    def _ml_detection(self, df: pd.DataFrame) -> Dict:
        """Isolation Forest prediction on latest bar."""
        try:
            X = self._build_anomaly_features(df).dropna()
            if X.empty:
                return {"ml_anomaly": False, "ml_score": 0.0}

            latest   = X.iloc[-1:].values
            scaled   = self._scaler.transform(latest)
            pred     = self._model.predict(scaled)[0]          # -1=anomaly, 1=normal
            score    = float(self._model.score_samples(scaled)[0])  # more negative = more anomalous

            return {
                "ml_anomaly": pred == -1,
                "ml_score":   round(score, 4),
            }
        except Exception as e:
            logger.warning(f"ML anomaly detection error: {e}")
            return {"ml_anomaly": False, "ml_score": 0.0}

    def _build_anomaly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features used to train and predict anomalies."""
        f = pd.DataFrame(index=df.index)
        close  = df["Close"]
        volume = df["Volume"]
        high   = df["High"]
        low    = df["Low"]

        f["ret_1d"]      = close.pct_change(1)
        f["ret_abs"]     = f["ret_1d"].abs()
        f["vol_ratio"]   = volume / volume.rolling(20).mean()
        f["range_ratio"] = (high - low) / (high - low).rolling(14).mean()
        f["gap"]         = (close - close.shift(1)).abs() / close.shift(1)
        f["vol_x_ret"]   = f["vol_ratio"] * f["ret_abs"]
        return f.replace([np.inf, -np.inf], np.nan)

    def _calculate_severity(self, rule: Dict, ml: Dict, df: pd.DataFrame) -> str:
        vol   = rule.get("volume_ratio", 1.0)
        sigma = abs(rule.get("price_sigma", 0.0))
        flags = sum([rule["rule_anomaly"], ml["ml_anomaly"]])

        if flags >= 2 or vol > 8 or sigma > 4:
            return "HIGH"
        elif flags == 1 or vol > 3 or sigma > 2.5:
            return "MEDIUM"
        return "LOW"

    def _recommended_action(self, is_anomaly: bool, severity: str) -> str:
        if not is_anomaly:
            return "NORMAL_TRADING"
        if severity == "HIGH":
            return "HALT_NEW_TRADES — review anomaly before next entry"
        if severity == "MEDIUM":
            return "REDUCE_SIZE_50% — elevated risk detected"
        return "CAUTION — monitor closely"

    def _connect(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c