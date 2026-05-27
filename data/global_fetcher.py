# data/global_fetcher.py
# Fetches global market signals that affect Indian large-cap stocks
# Sources: yfinance (free) — S&P500, Nasdaq, SGX Nifty, Crude, Gold, DXY, VIX
# Run every morning at 6 AM before Indian market opens

import sqlite3
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List

from data.price_fetcher import yf_safe_download

logger = logging.getLogger(__name__)


class GlobalFetcher:
    """
    Fetches overnight global market signals before NSE opens.

    Why this matters:
    - S&P500 overnight return predicts Indian market open direction
    - SGX/Gift Nifty is the best same-day Indian market predictor
    - Dollar strength drives FII inflows/outflows
    - Crude oil spike = India inflation risk = bank headwind
    - US yields rising = capital flows from EM to US = HDFC headwind
    - VIX spike = global risk-off = sell everything

    All free via yfinance. Run at 6 AM IST daily.
    """

    TICKERS = {
        # US Markets
        "sp500":    "^GSPC",
        "nasdaq":   "^IXIC",
        "dow":      "^DJI",
        # Fear / Volatility
        "us_vix":   "^VIX",
        "india_vix": "^INDIAVIX",
        # Commodities
        "crude":    "CL=F",
        "gold":     "GC=F",
        "silver":   "SI=F",
        "copper":   "HG=F",   # global growth proxy
        # Currency
        "dxy":      "DX-Y.NYB",
        "usdinr":   "USDINR=X",
        "eurusd":   "EURUSD=X",
        # Bonds
        "us_10y":   "^TNX",
        "us_2y":    "^IRX",
        # India indices
        "nifty":    "^NSEI",
        "banknifty": "^NSEBANK",
        # Global banks (leading indicator for HDFC)
        "jpmorgan": "JPM",
        "hsbc":     "HSBA.L",
    }

    def __init__(self, db_path: str = "database/trading.db"):
        self.db_path = db_path
        self._setup_db()

    # ─────────────────────────────────────────────────────────────
    # PUBLIC METHODS
    # ─────────────────────────────────────────────────────────────

    def fetch_overnight_signals(self) -> Dict:
        """
        Master method — fetch all global signals, save, return features.
        Called at 6:00 AM IST daily before market opens.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            row  = conn.execute("SELECT MAX(fetched_at) FROM global_snapshots").fetchone()
            conn.close()
            if row and row[0]:
                last = datetime.fromisoformat(str(row[0])[:19])
                if (datetime.now() - last).total_seconds() < 1800:
                    logger.debug("Global data fresh (<30 min) — skipping fetch")
                    return self._get_latest_from_db()
        except Exception:
            pass
        prices = self._fetch_all_prices()
        if not prices:
            logger.error("GlobalFetcher: _fetch_all_prices returned empty — check yfinance connectivity")
            return self._get_latest_from_db()

        features = self._calculate_features(prices)
        features["verdict"] = self._generate_verdict(features)
        features["fetched_at"] = str(datetime.now())

        try:
            self._save_snapshot(features)
        except Exception as e:
            logger.error(f"GlobalFetcher: _save_snapshot failed: {e}", exc_info=True)

        logger.info(
            f"Global: S&P={features.get('sp500_1d_pct',0):+.1f}% | "
            f"DXY={features.get('dxy_5d_pct',0):+.1f}% | "
            f"Crude={features.get('crude_5d_pct',0):+.1f}% | "
            f"Verdict={features['verdict']}"
        )
        return features

    def get_latest(self) -> Dict:
        """Get most recent global snapshot from DB (no API call)."""
        return self._get_latest_from_db()

    def get_row_count(self) -> int:
        """How many global snapshots are in the DB? Used by bootstrap check."""
        try:
            conn = sqlite3.connect(self.db_path)
            n    = conn.execute("SELECT COUNT(*) FROM global_snapshots").fetchone()[0]
            conn.close()
            return int(n or 0)
        except Exception:
            return 0

    def is_stale(self, max_age_hours: float = 30) -> bool:
        """True if the most recent global snapshot is older than `max_age_hours`."""
        try:
            conn = sqlite3.connect(self.db_path)
            row  = conn.execute("SELECT MAX(fetched_at) FROM global_snapshots").fetchone()
            conn.close()
            if not row or not row[0]:
                return True
            last = datetime.fromisoformat(str(row[0])[:19])
            return (datetime.now() - last).total_seconds() > max_age_hours * 3600
        except Exception:
            return True

    def bootstrap_history(self, days: int = 30) -> int:
        """
        Synthesise `days` of historical snapshots from the existing 30-day
        price series we already fetch. Lookahead-safe: each synthetic row
        slices the series at index `[:-offset]` so day T's snapshot only
        sees prices ≤ day T.

        Called on orchestrator boot when row count < 7. Returns rows added.
        """
        prices = self._fetch_all_prices()
        if not prices:
            logger.warning("bootstrap_history: no price data fetched")
            return 0

        # Pick the shortest series as our anchor — every other series gets
        # truncated to match so all rows compare apples-to-apples.
        min_len = min(len(s) for s in prices.values())
        if min_len < 7:
            logger.warning(
                f"bootstrap_history: only {min_len} days of price history — "
                f"need ≥7. Skipping bootstrap."
            )
            return 0

        # Read current set of fetched_at dates so we don't dupe.
        existing_dates = set()
        try:
            conn = sqlite3.connect(self.db_path)
            for (d,) in conn.execute(
                "SELECT substr(fetched_at, 1, 10) FROM global_snapshots"
            ).fetchall():
                existing_dates.add(d)
            conn.close()
        except Exception:
            pass

        added = 0
        # Walk backward — most recent first, stopping when we have enough.
        for offset in range(0, min(days, min_len - 6)):
            sliced = {k: s.iloc[: len(s) - offset] for k, s in prices.items()}
            try:
                anchor_date = sliced[next(iter(sliced))].index[-1]
            except Exception:
                continue
            date_str = str(anchor_date)[:10]
            if date_str in existing_dates:
                continue

            features = self._calculate_features(sliced)
            features["verdict"]    = self._generate_verdict(features)
            features["fetched_at"] = f"{date_str} 18:00:00"  # synthetic EOD
            try:
                self._save_snapshot(features)
                added += 1
                existing_dates.add(date_str)
            except Exception as e:
                logger.warning(f"bootstrap_history insert for {date_str}: {e}")

        logger.info(f"GlobalFetcher.bootstrap_history: added {added} rows")
        return added

    def get_intraday_bias(self) -> str:
        """
        Simple directional bias for intraday model.
        Based on overnight S&P500 and SGX Nifty direction.
        """
        data = self.get_latest()
        sp500 = data.get("sp500_1d_pct", 0)
        india_vix = data.get("india_vix_level", 15)

        if india_vix > 25:
            return "AVOID"  # too volatile
        elif sp500 > 0.8:
            return "BULLISH_OPEN"
        elif sp500 < -0.8:
            return "BEARISH_OPEN"
        return "NEUTRAL_OPEN"

    def get_macro_score(self) -> float:
        """
        Single -1 to +1 macro score for feature building.
        Combines all global signals into one number.
        """
        data = self.get_latest()
        verdict = data.get("verdict", "NEUTRAL")
        score_map = {
            "STRONG_BULL": 1.0,
            "BULL":        0.6,
            "NEUTRAL":     0.0,
            "BEAR":       -0.6,
            "STRONG_BEAR": -1.0,
        }
        return score_map.get(verdict, 0.0)

    # ─────────────────────────────────────────────────────────────
    # CALCULATION
    # ─────────────────────────────────────────────────────────────

    def _fetch_all_prices(self) -> Dict:
        """Download latest prices for all tickers."""
        prices = {}
        for name, ticker in self.TICKERS.items():
            try:
                df = yf_safe_download(ticker, period="30d")
                if df.empty:
                    continue
                prices[name] = df["Close"].dropna()
            except Exception as e:
                logger.warning(f"Failed to fetch {name} ({ticker}): {e}")
        return prices

    def _calculate_features(self, prices: Dict) -> Dict:
        """Calculate all percentage changes and levels from price series."""
        features = {}

        def pct(name, periods):
            s = prices.get(name)
            if s is None or len(s) < periods + 1:
                return 0.0
            return round(float((s.iloc[-1] - s.iloc[-periods]) / s.iloc[-periods] * 100), 2)

        def level(name):
            s = prices.get(name)
            if s is None or s.empty:
                return 0.0
            return round(float(s.iloc[-1]), 2)

        def above_ma(name, window):
            s = prices.get(name)
            if s is None or len(s) < window:
                return 0
            return int(s.iloc[-1] > s.rolling(window).mean().iloc[-1])

        # ── US Markets ──
        features["sp500_1d_pct"]    = pct("sp500", 1)
        features["sp500_5d_pct"]    = pct("sp500", 5)
        features["nasdaq_1d_pct"]   = pct("nasdaq", 1)
        features["dow_1d_pct"]      = pct("dow", 1)

        # ── Volatility ──
        features["us_vix_level"]    = level("us_vix")
        features["us_vix_1d_pct"]   = pct("us_vix", 1)
        features["india_vix_level"] = level("india_vix")
        features["india_vix_1d_pct"]= pct("india_vix", 1)
        features["india_vix_pct20"] = self._percentile_rank("india_vix", prices, 20)

        # ── Commodities ──
        features["crude_1d_pct"]    = pct("crude", 1)
        features["crude_5d_pct"]    = pct("crude", 5)
        features["crude_level"]     = level("crude")
        features["gold_1d_pct"]     = pct("gold", 1)
        features["copper_5d_pct"]   = pct("copper", 5)   # growth proxy

        # ── Currency ──
        features["dxy_level"]       = level("dxy")
        features["dxy_5d_pct"]      = pct("dxy", 5)
        features["dxy_above_20ma"]  = above_ma("dxy", 20)
        features["usdinr_level"]    = level("usdinr")
        features["usdinr_5d_pct"]   = pct("usdinr", 5)
        features["usdinr_1d_pct"]   = pct("usdinr", 1)

        # ── Bonds ──
        features["us_10y_level"]    = level("us_10y")
        features["us_10y_5d_chg"]   = round(
            level("us_10y") - (float(prices["us_10y"].iloc[-5])
            if "us_10y" in prices and len(prices["us_10y"]) >= 5 else level("us_10y")), 3
        )
        features["yield_spread"]    = round(level("us_10y") - level("us_2y"), 3)  # curve

        # ── India indices ──
        features["nifty_1d_pct"]    = pct("nifty", 1)
        features["nifty_5d_pct"]    = pct("nifty", 5)
        features["banknifty_1d_pct"]= pct("banknifty", 1)
        features["banknifty_5d_pct"]= pct("banknifty", 5)

        # ── Global banks ──
        features["jpmorgan_5d_pct"] = pct("jpmorgan", 5)
        features["hsbc_5d_pct"]     = pct("hsbc", 5)

        return features

    def _generate_verdict(self, f: Dict) -> str:
        """
        Score all signals into a single market verdict.
        Used for dashboard display and Telegram alerts.
        """
        bull = 0
        bear = 0

        # S&P 500
        if f.get("sp500_1d_pct", 0) > 0.8:   bull += 2
        elif f.get("sp500_1d_pct", 0) < -0.8: bear += 2

        # DXY — strong dollar is bad for India
        if f.get("dxy_5d_pct", 0) > 0.8:      bear += 2
        elif f.get("dxy_5d_pct", 0) < -0.8:   bull += 2

        # USD/INR — weak rupee is bad
        if f.get("usdinr_5d_pct", 0) > 1.0:   bear += 2
        elif f.get("usdinr_5d_pct", 0) < -0.5: bull += 1

        # Crude
        if f.get("crude_5d_pct", 0) > 4.0:    bear += 1
        elif f.get("crude_5d_pct", 0) < -4.0:  bull += 1

        # India VIX
        if f.get("india_vix_level", 15) > 22:  bear += 2
        elif f.get("india_vix_level", 15) < 13: bull += 1

        # US 10Y — rising yields bad for EM
        if f.get("us_10y_5d_chg", 0) > 0.15:  bear += 1
        elif f.get("us_10y_5d_chg", 0) < -0.15: bull += 1

        # JPMorgan as global bank proxy
        if f.get("jpmorgan_5d_pct", 0) > 1.5:  bull += 1
        elif f.get("jpmorgan_5d_pct", 0) < -1.5: bear += 1

        # Bank Nifty
        if f.get("banknifty_1d_pct", 0) > 0.5:  bull += 1
        elif f.get("banknifty_1d_pct", 0) < -0.5: bear += 1

        net = bull - bear
        if net >= 4:   return "STRONG_BULL"
        if net >= 2:   return "BULL"
        if net <= -4:  return "STRONG_BEAR"
        if net <= -2:  return "BEAR"
        return "NEUTRAL"

    def _percentile_rank(self, name: str, prices: Dict, window: int) -> float:
        """Return percentile rank of latest value in its rolling window."""
        s = prices.get(name)
        if s is None or len(s) < window:
            return 50.0
        recent = s.iloc[-window:]
        current = float(s.iloc[-1])
        rank = float(np.sum(recent < current)) / len(recent) * 100
        return round(rank, 1)

    # ─────────────────────────────────────────────────────────────
    # STORAGE
    # ─────────────────────────────────────────────────────────────

    def _save_snapshot(self, data: Dict) -> None:
        conn = self._connect()
        conn.execute("""
            INSERT INTO global_snapshots
            (sp500_1d_pct, sp500_5d_pct, nasdaq_1d_pct,
             india_vix_level, india_vix_1d_pct,
             crude_1d_pct, crude_5d_pct, crude_level,
             dxy_level, dxy_5d_pct, dxy_above_20ma,
             usdinr_level, usdinr_5d_pct,
             us_10y_level, us_10y_5d_chg,
             banknifty_1d_pct, banknifty_5d_pct,
             jpmorgan_5d_pct, verdict, fetched_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            data.get("sp500_1d_pct", 0),
            data.get("sp500_5d_pct", 0),
            data.get("nasdaq_1d_pct", 0),
            data.get("india_vix_level", 15),
            data.get("india_vix_1d_pct", 0),
            data.get("crude_1d_pct", 0),
            data.get("crude_5d_pct", 0),
            data.get("crude_level", 80),
            data.get("dxy_level", 103),
            data.get("dxy_5d_pct", 0),
            data.get("dxy_above_20ma", 0),
            data.get("usdinr_level", 83),
            data.get("usdinr_5d_pct", 0),
            data.get("us_10y_level", 4.3),
            data.get("us_10y_5d_chg", 0),
            data.get("banknifty_1d_pct", 0),
            data.get("banknifty_5d_pct", 0),
            data.get("jpmorgan_5d_pct", 0),
            data.get("verdict", "NEUTRAL"),
            data.get("fetched_at", str(datetime.now())),
        ))
        conn.commit()
        conn.close()

    def _get_latest_from_db(self) -> Dict:
        conn = self._connect()
        row = conn.execute("""
            SELECT sp500_1d_pct, sp500_5d_pct, nasdaq_1d_pct,
                   india_vix_level, india_vix_1d_pct,
                   crude_1d_pct, crude_5d_pct, crude_level,
                   dxy_level, dxy_5d_pct, dxy_above_20ma,
                   usdinr_level, usdinr_5d_pct,
                   us_10y_level, us_10y_5d_chg,
                   banknifty_1d_pct, banknifty_5d_pct,
                   jpmorgan_5d_pct, verdict, fetched_at
            FROM global_snapshots
            ORDER BY fetched_at DESC LIMIT 1
        """).fetchone()
        conn.close()

        if not row:
            return {"verdict": "NEUTRAL", "india_vix_level": 15,
                    "sp500_1d_pct": 0, "dxy_5d_pct": 0}

        keys = [
            "sp500_1d_pct","sp500_5d_pct","nasdaq_1d_pct",
            "india_vix_level","india_vix_1d_pct",
            "crude_1d_pct","crude_5d_pct","crude_level",
            "dxy_level","dxy_5d_pct","dxy_above_20ma",
            "usdinr_level","usdinr_5d_pct",
            "us_10y_level","us_10y_5d_chg",
            "banknifty_1d_pct","banknifty_5d_pct",
            "jpmorgan_5d_pct","verdict","fetched_at",
        ]
        return dict(zip(keys, row))

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _setup_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS global_snapshots (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                sp500_1d_pct      REAL, sp500_5d_pct    REAL,
                nasdaq_1d_pct     REAL, india_vix_level  REAL,
                india_vix_1d_pct  REAL, crude_1d_pct     REAL,
                crude_5d_pct      REAL, crude_level      REAL,
                dxy_level         REAL, dxy_5d_pct       REAL,
                dxy_above_20ma    INTEGER,
                usdinr_level      REAL, usdinr_5d_pct    REAL,
                us_10y_level      REAL, us_10y_5d_chg    REAL,
                banknifty_1d_pct  REAL, banknifty_5d_pct REAL,
                jpmorgan_5d_pct   REAL, verdict          TEXT,
                fetched_at        TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_global_fetched
            ON global_snapshots (fetched_at DESC)
        """)
        conn.commit()
        conn.close()