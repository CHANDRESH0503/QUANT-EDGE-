# models/train_regime.py
# Trains Hidden Markov Model for market regime detection
# 4 regimes: BULL_TRENDING, BEAR_TRENDING, HIGH_VOLATILITY, CHOPPY_SIDEWAYS
# This is Gate 1 — the master gate that overrides everything else

import numpy as np
import pandas as pd
import logging
import json
import os
import joblib
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.error("pip install hmmlearn")

try:
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

MODEL_PATH = "models/saved/regime_model.pkl"


class RegimeModelTrainer:
    """
    Trains a Gaussian HMM to detect market regimes from price data.

    20yr trader insight:
    The regime determines the strategy — not the signal.
    A 70% confident LONG signal in a CHOPPY_SIDEWAYS regime
    has a real win rate closer to 45%. That same signal in a
    BULL_TRENDING regime has a real win rate closer to 70%.

    HMM is ideal for regime detection because:
    1. Markets transition between hidden states — exactly what HMM models
    2. States overlap and transition gradually — not hard rule boundaries
    3. Unsupervised — learns from data without manual labelling
    4. Probabilistic — gives confidence per regime, not just a label

    Training data: 25 years daily OHLCV = 6,300 bars
    Features: returns, 10-day volatility, 5/20-day volume ratio
    States: 4 (mapped to regimes after training by return profile)
    """

    REGIME_NAMES   = ["BULL_TRENDING", "BEAR_TRENDING",
                      "HIGH_VOLATILITY", "CHOPPY_SIDEWAYS"]
    N_COMPONENTS   = 4
    N_ITER         = 1000
    COV_TYPE       = "full"

    def __init__(self):
        self._model         = None
        self._scaler        = None
        self._state_map     = {}    # HMM state int → regime name
        self._metrics       = {}

    # ─────────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame, save: bool = True) -> Dict:
        """
        Train HMM on full historical OHLCV data.

        Args:
            df:   Daily OHLCV DataFrame from PriceFetcher.get_daily()
            save: Persist model to disk after training
        """
        if not HMM_AVAILABLE or not SKLEARN_AVAILABLE:
            logger.error("hmmlearn or sklearn not available")
            return {}

        if len(df) < 200:
            logger.error(f"Need 200+ daily bars for regime training, got {len(df)}")
            return {}

        X = self._build_features(df).dropna()
        if X.empty:
            logger.error("Feature build returned empty DataFrame")
            return {}

        logger.info(f"Regime HMM training: {len(X)} bars | {X.shape[1]} features")

        # Standardise — HMM is sensitive to scale
        self._scaler = StandardScaler()
        X_scaled     = self._scaler.fit_transform(X)

        # Train HMM
        self._model = hmm.GaussianHMM(
            n_components   = self.N_COMPONENTS,
            covariance_type= self.COV_TYPE,
            n_iter         = self.N_ITER,
            random_state   = 42,
        )
        self._model.fit(X_scaled)

        # Decode all states
        states = self._model.predict(X_scaled)

        # Map integer states to meaningful regime names
        self._map_states(X, states)

        # Calculate regime distribution
        regime_dist = {}
        for state_int, regime_name in self._state_map.items():
            pct = float(np.sum(states == state_int) / len(states) * 100)
            regime_dist[regime_name] = round(pct, 1)

        # Current regime
        current_state  = int(states[-1])
        current_regime = self._state_map.get(current_state, "CHOPPY_SIDEWAYS")

        # Stability: fraction of last 10 bars in same regime
        recent         = states[-10:]
        stability      = float(np.sum(recent == current_state) / len(recent))

        self._metrics = {
            "model_type":      "regime",
            "trained_at":      str(datetime.now()),
            "n_samples":       len(X),
            "n_components":    self.N_COMPONENTS,
            "regime_dist_pct": regime_dist,
            "current_regime":  current_regime,
            "state_map":       self._state_map,
            "stability":       round(stability, 3),
        }

        if save:
            self._save()
            self._save_metrics()

        logger.info(
            f"Regime model trained | "
            f"Current: {current_regime} ({stability:.0%} stable) | "
            f"Dist: {regime_dist}"
        )
        return self._metrics

    def load(self) -> bool:
        """Load pre-trained regime model from disk."""
        try:
            data = joblib.load(MODEL_PATH)
            self._model     = data["model"]
            self._scaler    = data["scaler"]
            self._state_map = data["state_map"]
            logger.info(f"Regime model loaded | state_map={self._state_map}")
            return True
        except Exception as e:
            logger.warning(f"Regime model load failed: {e}")
            return False

    def detect(self, df: pd.DataFrame) -> Dict:
        """
        Detect current regime from recent price data.
        Falls back to rule-based if HMM not trained.
        """
        if self._model is None:
            if not self.load():
                return self._rule_based(df)

        try:
            X = self._build_features(df).dropna()
            if X.empty:
                return self._rule_based(df)

            X_scaled = self._scaler.transform(X)
            states   = self._model.predict(X_scaled)
            probs    = self._model.predict_proba(X_scaled)

            current_state = int(states[-1])
            current_probs = probs[-1]
            regime        = self._state_map.get(current_state, "CHOPPY_SIDEWAYS")

            # Stability
            recent    = states[-min(10, len(states)):]
            stability = float(np.sum(recent == current_state) / len(recent))

            # Map state probs → regime probs
            regime_probs = {r: 0.0 for r in self.REGIME_NAMES}
            for s, r in self._state_map.items():
                if s < len(current_probs):
                    regime_probs[r] += float(current_probs[s])

            return self._build_result(regime, stability, regime_probs)

        except Exception as e:
            logger.warning(f"HMM detection failed, using rule-based: {e}")
            return self._rule_based(df)

    # ─────────────────────────────────────────────────────────────
    # FEATURE ENGINEERING
    # ─────────────────────────────────────────────────────────────

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        3 features fed to HMM — simple and stable.
        Returns: rolling returns, volatility, volume trend
        """
        f    = pd.DataFrame(index=df.index)
        ret  = df["Close"].pct_change().fillna(0)

        f["returns"]    = ret
        f["volatility"] = ret.rolling(10, min_periods=5).std()
        f["vol_trend"]  = (
            df["Volume"].rolling(5, min_periods=3).mean() /
            (df["Volume"].rolling(20, min_periods=10).mean() + 1e-8)
        )
        return f.replace([np.inf, -np.inf], np.nan)

    # ─────────────────────────────────────────────────────────────
    # STATE MAPPING
    # ─────────────────────────────────────────────────────────────

    def _map_states(self, X: pd.DataFrame, states: np.ndarray) -> None:
        """
        Map HMM integer states to meaningful regime names.

        Mapping logic (deterministic — same data = same mapping):
        - Highest avg return     → BULL_TRENDING
        - Most negative return   → BEAR_TRENDING
        - Highest volatility     → HIGH_VOLATILITY
        - Remaining              → CHOPPY_SIDEWAYS
        """
        state_stats = {}
        for s in range(self.N_COMPONENTS):
            mask = states == s
            if mask.sum() < 5:
                continue
            state_stats[s] = {
                "avg_ret": float(X["returns"][mask].mean()),
                "avg_vol": float(X["volatility"][mask].mean()),
                "count":   int(mask.sum()),
            }

        mapping     = {}
        assigned    = set()
        by_ret      = sorted(state_stats, key=lambda s: state_stats[s]["avg_ret"])
        by_vol      = sorted(state_stats, key=lambda s: state_stats[s]["avg_vol"])

        # Highest avg return → BULL
        if by_ret:
            mapping[by_ret[-1]] = "BULL_TRENDING"
            assigned.add(by_ret[-1])

        # Most negative return → BEAR
        if by_ret and by_ret[0] not in assigned:
            mapping[by_ret[0]] = "BEAR_TRENDING"
            assigned.add(by_ret[0])

        # Highest volatility (unassigned) → HIGH_VOL
        for s in reversed(by_vol):
            if s not in assigned:
                mapping[s] = "HIGH_VOLATILITY"
                assigned.add(s)
                break

        # Remaining → CHOPPY
        for s in state_stats:
            if s not in assigned:
                mapping[s] = "CHOPPY_SIDEWAYS"

        self._state_map = mapping
        logger.info(f"State mapping: {mapping}")

    # ─────────────────────────────────────────────────────────────
    # RULE-BASED FALLBACK
    # ─────────────────────────────────────────────────────────────

    def _rule_based(self, df: pd.DataFrame) -> Dict:
        """
        Deterministic regime detection when HMM is unavailable.
        Uses EMA trend + ADX-proxy + volatility.
        Less accurate but always works.
        """
        if len(df) < 30:
            return self._build_result("CHOPPY_SIDEWAYS", 0.5,
                                      {r: 0.25 for r in self.REGIME_NAMES})

        close = df["Close"]
        ret   = close.pct_change().dropna()
        ema20 = float(close.ewm(span=20).mean().iloc[-1])
        ema50 = float(close.ewm(span=50).mean().iloc[-1])
        price = float(close.iloc[-1])

        # Trend
        trend_up   = price > ema20 > ema50
        trend_down = price < ema20 < ema50

        # Consistency (ADX proxy)
        direction   = np.sign(ret)
        consistency = float(abs(direction.rolling(14).mean().iloc[-1]))

        # Volatility
        hv10     = float(ret.rolling(10).std() * np.sqrt(252) * 100)
        high_vol = hv10 > 28

        if high_vol and hv10 > 32:
            regime  = "HIGH_VOLATILITY"
            probs   = {"BULL_TRENDING": 0.10, "BEAR_TRENDING": 0.10,
                       "HIGH_VOLATILITY": 0.70, "CHOPPY_SIDEWAYS": 0.10}
        elif trend_up and consistency > 0.35:
            regime  = "BULL_TRENDING"
            probs   = {"BULL_TRENDING": 0.70, "BEAR_TRENDING": 0.05,
                       "HIGH_VOLATILITY": 0.10, "CHOPPY_SIDEWAYS": 0.15}
        elif trend_down and consistency > 0.35:
            regime  = "BEAR_TRENDING"
            probs   = {"BULL_TRENDING": 0.05, "BEAR_TRENDING": 0.70,
                       "HIGH_VOLATILITY": 0.10, "CHOPPY_SIDEWAYS": 0.15}
        else:
            regime  = "CHOPPY_SIDEWAYS"
            probs   = {"BULL_TRENDING": 0.20, "BEAR_TRENDING": 0.20,
                       "HIGH_VOLATILITY": 0.15, "CHOPPY_SIDEWAYS": 0.45}

        return self._build_result(regime, consistency, probs)

    # ─────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────

    def _build_result(
        self, regime: str, stability: float, probs: Dict
    ) -> Dict:
        rules = {
            "BULL_TRENDING":   {"trade_long": True,  "trade_short": False,
                                "position_mult": 1.2, "max_hold": 8},
            "BEAR_TRENDING":   {"trade_long": False, "trade_short": True,
                                "position_mult": 0.8, "max_hold": 3},
            "HIGH_VOLATILITY": {"trade_long": True,  "trade_short": True,
                                "position_mult": 0.5, "max_hold": 2},
            "CHOPPY_SIDEWAYS": {"trade_long": False, "trade_short": False,
                                "position_mult": 0.0, "max_hold": 0},
        }.get(regime, {"trade_long": False, "trade_short": False,
                       "position_mult": 0.0, "max_hold": 0})

        return {
            "regime":         regime,
            "stability":      round(stability, 3),
            "probs":          probs,
            "trade_long":     rules["trade_long"],
            "trade_short":    rules["trade_short"],
            "position_mult":  rules["position_mult"],
            "max_hold_days":  rules["max_hold"],
            "description":    self._describe(regime),
            "detected_at":    str(datetime.now()),
        }

    def _describe(self, regime: str) -> str:
        return {
            "BULL_TRENDING":   "Strong uptrend — buy dips, hold longer, 1.2× size",
            "BEAR_TRENDING":   "Downtrend — short only, quick exits, 0.8× size",
            "HIGH_VOLATILITY": "High volatility — half size, wide stops, 2-day max",
            "CHOPPY_SIDEWAYS": "No direction — stay in cash, all signals blocked",
        }.get(regime, "Unknown regime")

    def _save(self) -> None:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump({
            "model":     self._model,
            "scaler":    self._scaler,
            "state_map": self._state_map,
        }, MODEL_PATH)
        logger.info(f"Regime model saved to {MODEL_PATH}")

    def _save_metrics(self) -> None:
        path = "models/evaluation/model_metrics.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        existing = {}
        if os.path.exists(path):
            try:
                with open(path) as f:
                    existing = json.load(f)
            except Exception:
                pass
        existing["regime"] = self._metrics
        with open(path, "w") as f:
            json.dump(existing, f, indent=2, default=str)