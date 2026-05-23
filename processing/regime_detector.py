# processing/regime_detector.py
# Detects current market regime using Hidden Markov Model (HMM)
# 4 regimes: Bull Trending, Bear Trending, High Volatility, Choppy Sideways
# Each regime requires a different trading strategy — this is the master gate

import numpy as np
import pandas as pd
import sqlite3
import logging
import joblib
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.warning("hmmlearn not installed. pip install hmmlearn. Using rule-based fallback.")


class RegimeDetector:
    """
    Identifies current market regime using Hidden Markov Model.

    20yr trader truth:
    The single most important trading decision is not WHAT to buy —
    it is WHEN the conditions are right to trade at all.
    In choppy markets, even perfect signals fail 60% of the time.
    In strong trending markets, mediocre signals work 65% of the time.
    The regime IS the edge.

    4 Regimes:
    ┌──────────────────┬────────────────────────────────────────────┐
    │ BULL_TRENDING    │ Strong uptrend, ADX>25, low VIX            │
    │                  │ → Bigger positions, hold longer, buy dips  │
    ├──────────────────┼────────────────────────────────────────────┤
    │ BEAR_TRENDING    │ Strong downtrend, ADX>25, elevated VIX     │
    │                  │ → Short only, small positions, quick exits  │
    ├──────────────────┼────────────────────────────────────────────┤
    │ HIGH_VOLATILITY  │ Large moves both ways, VIX spike            │
    │                  │ → Reduce size 50%, wider stops, be careful  │
    ├──────────────────┼────────────────────────────────────────────┤
    │ CHOPPY_SIDEWAYS  │ No clear direction, low ADX                 │
    │                  │ → DO NOTHING. Cash is a position.           │
    └──────────────────┴────────────────────────────────────────────┘

    Method: 3-feature Gaussian HMM on daily returns + volatility + volume
    Fallback: Rule-based regime detection when HMM not trained

    Connected to:
    - price_fetcher.py: uses daily OHLCV
    - signal_engine.py gate 1: regime check — first gate before ML
    - risk/capital_mode.py: regime affects position sizing multiplier
    """

    REGIME_NAMES = {
        0: "BULL_TRENDING",
        1: "BEAR_TRENDING",
        2: "HIGH_VOLATILITY",
        3: "CHOPPY_SIDEWAYS",
    }

    REGIME_RULES = {
        "BULL_TRENDING": {
            "trade_long":       True,
            "trade_short":      False,
            "position_mult":    1.2,
            "max_hold_days":    8,
            "stop_mult":        1.5,
            "description":      "Strong uptrend — buy dips, hold longer",
        },
        "BEAR_TRENDING": {
            "trade_long":       False,
            "trade_short":      True,
            "position_mult":    0.8,
            "max_hold_days":    3,
            "stop_mult":        1.2,
            "description":      "Downtrend — short only, quick exits",
        },
        "HIGH_VOLATILITY": {
            "trade_long":       True,
            "trade_short":      True,
            "position_mult":    0.5,
            "max_hold_days":    2,
            "stop_mult":        2.0,
            "description":      "High volatility — half size, wide stops",
        },
        "CHOPPY_SIDEWAYS": {
            "trade_long":       False,
            "trade_short":      False,
            "position_mult":    0.0,
            "max_hold_days":    0,
            "stop_mult":        1.0,
            "description":      "No trend — stay in cash",
        },
    }

    MODEL_PATH = "models/saved/regime_model.pkl"

    def __init__(self, db_path: str = "database/trading.db",
                 n_components: int = 4):
        self.db_path     = db_path
        self.n_components= n_components
        self._model      = None
        self._scaler     = None
        self._setup_db()
        self._load_model()

    # ─────────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> bool:
        """
        Train HMM on historical OHLCV data.
        Called during weekly retrain cycle (Sunday midnight).
        Uses last 5 years (1260 days) for stable regime learning.

        Returns True if training succeeded.
        """
        if not HMM_AVAILABLE:
            logger.warning("HMM not available — using rule-based fallback")
            return False

        if len(df) < 100:
            logger.error("Insufficient data for HMM training (need >100 days)")
            return False

        try:
            X = self._build_hmm_features(df)
            X = X.dropna()

            # Standardise features
            from sklearn.preprocessing import StandardScaler
            self._scaler = StandardScaler()
            X_scaled     = self._scaler.fit_transform(X)

            # Gaussian HMM — each state has its own mean and covariance
            self._model  = hmm.GaussianHMM(
                n_components  = self.n_components,
                covariance_type="full",
                n_iter        = 1000,
                random_state  = 42,
            )
            self._model.fit(X_scaled)

            # Decode states and map to meaningful regime names
            states = self._model.predict(X_scaled)
            self._map_states_to_regimes(X, states)
            self._save_model()

            logger.info(
                f"Regime HMM trained on {len(X)} days | "
                f"Regime distribution: {self._get_state_distribution(states)}"
            )
            return True

        except Exception as e:
            logger.error(f"HMM training failed: {e}")
            return False

    def detect(self, df: pd.DataFrame) -> Dict:
        """
        Detect current market regime from recent OHLCV data.
        Called at the start of every signal generation cycle.

        Returns full regime context including:
        - regime name and rules
        - stability score (how long in current regime)
        - probability of each regime
        - trading rules for current regime
        """
        if self._model is not None and HMM_AVAILABLE:
            result = self._hmm_detect(df)
        else:
            result = self._rule_based_detect(df)

        # EMA override for borderline CHOPPY detections.
        # The HMM can stay locked in CHOPPY even when price is clearly trending.
        # When CHOPPY + EMA spread > ±2%, override to BEAR/BULL with reduced stability
        # so the downstream gates (2/4/5/6) can make the final call.
        if result["regime"] == "CHOPPY_SIDEWAYS" and len(df) >= 50:
            close   = df["Close"]
            ema20   = float(close.ewm(span=20).mean().iloc[-1])
            ema50   = float(close.ewm(span=50).mean().iloc[-1])
            spread  = (ema20 - ema50) / ema50  # positive = bullish
            if spread < -0.02:          # clearly below 50-day EMA → mild BEAR
                result = self._build_result(
                    "BEAR_TRENDING",
                    min(result["stability"], 0.65),
                    {**result.get("probs", self._equal_probs()),
                     "BEAR_TRENDING": 0.55, "CHOPPY_SIDEWAYS": 0.30},
                )
                logger.info(
                    f"EMA override: CHOPPY→BEAR_TRENDING "
                    f"(spread={spread:.2%}, stability capped at 65%)"
                )
            elif spread > 0.02:         # clearly above 50-day EMA → mild BULL
                result = self._build_result(
                    "BULL_TRENDING",
                    min(result["stability"], 0.65),
                    {**result.get("probs", self._equal_probs()),
                     "BULL_TRENDING": 0.55, "CHOPPY_SIDEWAYS": 0.30},
                )
                logger.info(
                    f"EMA override: CHOPPY→BULL_TRENDING "
                    f"(spread={spread:.2%}, stability capped at 65%)"
                )

        # Save to DB for trend tracking
        self._save_regime_snapshot(result)

        logger.info(
            f"Regime: {result['regime']} | "
            f"Stability: {result['stability']:.0%} | "
            f"trade_long={result['rules']['trade_long']}"
        )
        return result

    def get_regime_features_for_ml(self) -> Dict:
        """
        Returns regime as ML features — one-hot encoded + probability scores.
        Called by feature_builder.py before model prediction.
        """
        conn = self._connect()
        row  = conn.execute("""
            SELECT regime, bull_prob, bear_prob, high_vol_prob,
                   choppy_prob, stability, detected_at
            FROM   regime_snapshots
            ORDER  BY detected_at DESC LIMIT 1
        """).fetchone()
        conn.close()

        if not row:
            return self._empty_regime_features()

        regime = row[0]
        return {
            # One-hot encoding
            "regime_bull":     int(regime == "BULL_TRENDING"),
            "regime_bear":     int(regime == "BEAR_TRENDING"),
            "regime_high_vol": int(regime == "HIGH_VOLATILITY"),
            "regime_choppy":   int(regime == "CHOPPY_SIDEWAYS"),

            # Probabilities (continuous, better for ML)
            "regime_bull_prob":     float(row[1] or 0),
            "regime_bear_prob":     float(row[2] or 0),
            "regime_high_vol_prob": float(row[3] or 0),
            "regime_choppy_prob":   float(row[4] or 0),

            # Stability
            "regime_stability": float(row[5] or 0),

            # For direct signal gating
            "regime_allows_long":  int(regime in ["BULL_TRENDING", "HIGH_VOLATILITY"]),
            "regime_allows_short": int(regime in ["BEAR_TRENDING", "HIGH_VOLATILITY"]),
            "regime_is_tradeable": int(regime != "CHOPPY_SIDEWAYS"),

            # Position multiplier from regime rules
            "regime_position_mult": self.REGIME_RULES.get(
                regime, {"position_mult": 1.0}
            )["position_mult"],
        }

    def get_recent_regime_trend(self, days: int = 10) -> Dict:
        """
        Has the regime been stable or shifting recently?
        Regime change within last 3 days = extra caution.
        """
        since = datetime.now() - timedelta(days=days)
        conn  = self._connect()
        rows  = conn.execute("""
            SELECT regime, detected_at FROM regime_snapshots
            WHERE  detected_at > ?
            ORDER  BY detected_at DESC
        """, (str(since),)).fetchall()
        conn.close()

        if not rows:
            return {"stable": True, "changes": 0, "dominant": "UNKNOWN"}

        regimes  = [r[0] for r in rows]
        changes  = sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i-1])
        dominant = max(set(regimes), key=regimes.count)
        stable   = changes <= 1

        return {
            "stable":   stable,
            "changes":  changes,
            "dominant": dominant,
            "current":  regimes[0] if regimes else "UNKNOWN",
        }

    # ─────────────────────────────────────────────────────────────
    # HMM DETECTION
    # ─────────────────────────────────────────────────────────────

    def _hmm_detect(self, df: pd.DataFrame) -> Dict:
        """Run trained HMM on recent data to classify current regime."""
        try:
            X = self._build_hmm_features(df).dropna()
            if X.empty or self._scaler is None:
                return self._rule_based_detect(df)

            X_scaled  = self._scaler.transform(X)
            states    = self._model.predict(X_scaled)
            probs     = self._model.predict_proba(X_scaled)

            # Use majority vote of last 3 complete bars instead of just [-1].
            # The latest bar is a partial candle during market hours and flips
            # the regime on every intraday fetch, causing CHOPPY↔BEAR oscillation.
            consensus_window = states[-min(3, len(states)):]
            current_state    = int(np.bincount(consensus_window).argmax())
            current_probs    = probs[-min(3, len(probs)):].mean(axis=0)

            # Get regime name from state mapping
            regime = self._state_to_regime.get(current_state, "CHOPPY_SIDEWAYS")

            # Stability: what % of last 10 bars were in this regime?
            recent_states = states[-min(10, len(states)):]
            stability     = float(np.sum(recent_states == current_state) / len(recent_states))

            # Probs mapped to regime names
            regime_probs  = self._map_probs_to_regimes(current_probs)

            return self._build_result(regime, stability, regime_probs)

        except Exception as e:
            logger.warning(f"HMM detection failed, using rule-based: {e}")
            return self._rule_based_detect(df)

    def _build_hmm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """3 features for HMM — returns, volatility, volume trend."""
        f = pd.DataFrame(index=df.index)
        ret = df["Close"].pct_change()

        f["returns"]    = ret
        f["volatility"] = ret.rolling(10).std()
        f["vol_trend"]  = (
            df["Volume"].rolling(5).mean() /
            (df["Volume"].rolling(20).mean() + 1e-8)
        )
        return f.replace([np.inf, -np.inf], np.nan)

    # ─────────────────────────────────────────────────────────────
    # RULE-BASED FALLBACK
    # ─────────────────────────────────────────────────────────────

    def _rule_based_detect(self, df: pd.DataFrame) -> Dict:
        """
        Rule-based regime detection — runs when HMM not trained.
        Uses ADX, volatility, trend direction.
        Less accurate than HMM but always available.
        """
        if len(df) < 30:
            return self._build_result("CHOPPY_SIDEWAYS", 0.5,
                                       self._equal_probs())

        close = df["Close"]
        high  = df["High"]
        low   = df["Low"]

        # Trend direction
        ema20  = float(close.ewm(span=20).mean().iloc[-1])
        ema50  = float(close.ewm(span=50).mean().iloc[-1])
        price  = float(close.iloc[-1])
        trend_up   = price > ema20 > ema50
        trend_down = price < ema20 < ema50

        # Trend strength (simplified ADX)
        returns    = close.pct_change().dropna()
        direction  = np.sign(returns)
        consistency= float(abs(direction.rolling(14).mean().iloc[-1]))
        strong     = consistency > 0.35

        # Volatility
        hv10  = float(returns.rolling(10).std().iloc[-1] * np.sqrt(252) * 100)
        hv30  = float(returns.rolling(30).std().iloc[-1] * np.sqrt(252) * 100)
        high_vol = hv10 > 28 or (hv10 / max(hv30, 1) > 1.6)

        # Classify
        if high_vol and hv10 > 30:
            regime     = "HIGH_VOLATILITY"
            bull_p, bear_p, hv_p, ch_p = 0.15, 0.15, 0.60, 0.10
        elif trend_up and strong:
            regime     = "BULL_TRENDING"
            bull_p, bear_p, hv_p, ch_p = 0.70, 0.05, 0.10, 0.15
        elif trend_down and strong:
            regime     = "BEAR_TRENDING"
            bull_p, bear_p, hv_p, ch_p = 0.05, 0.70, 0.10, 0.15
        else:
            regime     = "CHOPPY_SIDEWAYS"
            bull_p, bear_p, hv_p, ch_p = 0.20, 0.20, 0.20, 0.40

        probs = {
            "BULL_TRENDING":    bull_p,
            "BEAR_TRENDING":    bear_p,
            "HIGH_VOLATILITY":  hv_p,
            "CHOPPY_SIDEWAYS":  ch_p,
        }
        return self._build_result(regime, consistency, probs)

    # ─────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────

    def _build_result(self, regime: str, stability: float,
                       probs: Dict) -> Dict:
        rules = self.REGIME_RULES.get(regime, self.REGIME_RULES["CHOPPY_SIDEWAYS"])
        return {
            "regime":       regime,
            "stability":    round(stability, 3),
            "rules":        rules,
            "description":  rules["description"],
            "probs":        probs,
            "trade_long":   rules["trade_long"],
            "trade_short":  rules["trade_short"],
            "position_mult":rules["position_mult"],
            "detected_at":  str(datetime.now()),
        }

    def _map_states_to_regimes(self, X: pd.DataFrame, states: np.ndarray) -> None:
        """
        Map HMM integer states to meaningful regime names.
        State with highest avg return → BULL
        State with most negative return → BEAR
        State with highest volatility → HIGH_VOL
        Remaining state → CHOPPY
        """
        state_stats = {}
        for state in range(self.n_components):
            mask = states == state
            if mask.sum() == 0:
                continue
            state_stats[state] = {
                "avg_return": float(X["returns"][mask].mean()),
                "avg_vol":    float(X["volatility"][mask].mean()),
                "count":      int(mask.sum()),
            }

        sorted_by_ret = sorted(state_stats, key=lambda s: state_stats[s]["avg_return"])
        sorted_by_vol = sorted(state_stats, key=lambda s: state_stats[s]["avg_vol"])

        mapping = {}
        # Highest avg return = bull
        mapping[sorted_by_ret[-1]] = "BULL_TRENDING"
        # Lowest avg return = bear
        mapping[sorted_by_ret[0]]  = "BEAR_TRENDING"
        # Highest volatility (excluding already mapped) = high vol
        for s in reversed(sorted_by_vol):
            if s not in mapping:
                mapping[s] = "HIGH_VOLATILITY"
                break
        # Remaining = choppy
        for s in state_stats:
            if s not in mapping:
                mapping[s] = "CHOPPY_SIDEWAYS"

        self._state_to_regime = mapping
        logger.info(f"Regime state mapping: {mapping}")

    def _map_probs_to_regimes(self, probs: np.ndarray) -> Dict:
        """Map HMM state probabilities to regime name probabilities."""
        result = {r: 0.0 for r in self.REGIME_NAMES.values()}
        for state, regime in self._state_to_regime.items():
            result[regime] += float(probs[state])
        return result

    def _get_state_distribution(self, states: np.ndarray) -> Dict:
        """Distribution of regimes over training period."""
        dist = {}
        for s, name in self.REGIME_NAMES.items():
            pct = float(np.sum(states == s) / len(states) * 100)
            dist[self._state_to_regime.get(s, name)] = f"{pct:.1f}%"
        return dist

    def _equal_probs(self) -> Dict:
        return {r: 0.25 for r in self.REGIME_NAMES.values()}

    def _save_model(self) -> None:
        """Persist trained HMM and scaler to disk."""
        try:
            os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)
            joblib.dump(
                {"model": self._model, "scaler": self._scaler,
                 "state_map": getattr(self, "_state_to_regime", {})},
                self.MODEL_PATH,
            )
            logger.info(f"Regime model saved to {self.MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to save regime model: {e}")

    def _load_model(self) -> None:
        """Load pre-trained HMM from disk on startup."""
        if not os.path.exists(self.MODEL_PATH):
            return
        try:
            data = joblib.load(self.MODEL_PATH)
            self._model            = data["model"]
            self._scaler           = data["scaler"]
            # "state_map" is the key used by RegimeModelTrainer; "mapping" was the old key
            self._state_to_regime  = data.get("state_map", data.get("mapping", {}))
            logger.info("Regime model loaded from disk")
        except Exception as e:
            logger.warning(f"Failed to load regime model: {e}")

    def _save_regime_snapshot(self, result: Dict) -> None:
        probs = result.get("probs", self._equal_probs())
        conn  = self._connect()
        conn.execute("""
            INSERT INTO regime_snapshots
            (regime, stability, bull_prob, bear_prob,
             high_vol_prob, choppy_prob, detected_at)
            VALUES (?,?,?,?,?,?,?)
        """, (
            result["regime"],
            result["stability"],
            probs.get("BULL_TRENDING",   0),
            probs.get("BEAR_TRENDING",   0),
            probs.get("HIGH_VOLATILITY", 0),
            probs.get("CHOPPY_SIDEWAYS", 0),
            result["detected_at"],
        ))
        conn.commit()
        conn.close()

    def _empty_regime_features(self) -> Dict:
        return {
            "regime_bull": 0, "regime_bear": 0,
            "regime_high_vol": 0, "regime_choppy": 1,
            "regime_bull_prob": 0.25, "regime_bear_prob": 0.25,
            "regime_high_vol_prob": 0.25, "regime_choppy_prob": 0.25,
            "regime_stability": 0.5,
            "regime_allows_long": 0, "regime_allows_short": 0,
            "regime_is_tradeable": 0, "regime_position_mult": 0.0,
        }

    def _connect(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    def _setup_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS regime_snapshots (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                regime        TEXT,
                stability     REAL,
                bull_prob     REAL,
                bear_prob     REAL,
                high_vol_prob REAL,
                choppy_prob   REAL,
                detected_at   TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_regime_detected
            ON regime_snapshots (detected_at DESC)
        """)
        conn.commit()
        conn.close()