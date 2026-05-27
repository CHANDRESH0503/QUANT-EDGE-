# processing/intermarket.py
# Cross-asset analysis — how global markets affect HDFC Bank
# Bond yields, dollar, crude, global banks, sector rotation
# Connected to: global_fetcher.py, fii_fetcher.py, signal_engine.py

import numpy as np
import pandas as pd
import sqlite3
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

from data.price_fetcher import yf_safe_download

logger = logging.getLogger(__name__)

# Cross-cycle, cross-bank cache for the universe-wide intermarket fetches.
# Previously each per-bank FeatureBuilder triggered fresh yfinance calls for
# peer data (5 tickers) + sector rotation (^NSEI + ^NSEBANK) — 7 calls × 5 banks
# = 35 redundant yfinance calls per cycle. We now share one fetch across the
# universe within a TTL window (5 minutes, matches the intraday signal cadence).
_INTERMARKET_CACHE_TTL_SEC = 300
_intermarket_cache: Dict[str, "object"] = {
    "peers":  None,   # peer_data dict
    "sector": None,   # sector_rotation dict
    "ts":     0.0,
}


class IntermarketAnalyzer:
    """
    Analyses cross-asset relationships that drive HDFC Bank price.

    20yr trader truth:
    No stock trades in isolation. HDFC Bank is driven by:
    1. Bank Nifty sector strength (primary — 30% of index)
    2. US bond yields (FII flow driver)
    3. Dollar index (rupee pressure = FII exit)
    4. Crude oil (inflation → RBI → rate cycle for banks)
    5. Global bank stocks JPMorgan / HSBC (lead indicator)
    6. HDFC relative strength vs peers (money rotation signal)

    Features generated:
    - bn_relative_strength : HDFC vs Bank Nifty outperformance
    - us_yield_signal      : rising yields = headwind for EM
    - dxy_signal           : strong dollar = FII outflow
    - crude_signal         : high crude = inflation = RBI risk
    - global_bank_signal   : JPMorgan leading HDFC direction
    - sector_rotation      : which bank is getting money flows
    - intermarket_score    : composite -1 to +1

    Connected to:
    - global_fetcher.py: already fetches DXY, crude, S&P500
    - fii_fetcher.py: FII flow context
    - signal_engine.py: intermarket score feeds gate 2 rule filter
    """

    # Peer banks for relative strength comparison
    PEER_TICKERS = {
        "ICICIBANK.NS":  "ICICI Bank",
        "KOTAKBANK.NS":  "Kotak Bank",
        "AXISBANK.NS":   "Axis Bank",
        "SBIN.NS":       "SBI",
        "INDUSINDBK.NS": "IndusInd",
    }

    GLOBAL_TICKERS = {
        "^NSEBANK":  "banknifty",
        "^TNX":      "us_10y",
        "DX-Y.NYB":  "dxy",
        "CL=F":      "crude",
        "JPM":       "jpmorgan",
        "USDINR=X":  "usdinr",
    }

    def __init__(self, db_path: str = "database/trading.db"):
        self.db_path = db_path

    # ─────────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────────

    def get_intermarket_features(self, df_hdfc: pd.DataFrame) -> Dict:
        """
        Master method — all cross-asset features in one call.
        Uses cached global data from global_fetcher where possible.
        Makes fresh yfinance calls only for peer comparison.

        Args:
            df_hdfc: HDFC Bank daily OHLCV from price_fetcher
        """
        global_data  = self._fetch_global_data()
        peer_data    = self._fetch_peer_data()
        rs_features  = self._relative_strength(df_hdfc, peer_data)
        bond_features= self._bond_yield_features(global_data)
        dollar_feat  = self._dollar_features(global_data)
        crude_feat   = self._crude_features(global_data)
        bank_feat    = self._global_bank_features(global_data)
        sector_feat  = self._sector_rotation(peer_data, df_hdfc)
        score        = self._composite_score(
            rs_features, bond_features, dollar_feat,
            crude_feat, bank_feat, sector_feat
        )

        return {
            **rs_features,
            **bond_features,
            **dollar_feat,
            **crude_feat,
            **bank_feat,
            **sector_feat,
            "intermarket_score":  score,
            "intermarket_signal": self._score_to_signal(score),
        }

    # ─────────────────────────────────────────────────────────────
    # DATA FETCHING
    # ─────────────────────────────────────────────────────────────

    def _fetch_global_data(self) -> Dict:
        """
        Fetch global tickers — uses DB cache from global_fetcher first.
        Falls back to fresh yfinance call if cache is stale.
        """
        # Try DB cache first (written by global_fetcher every morning)
        cached = self._get_global_cache()
        if cached:
            return cached

        # Fresh fetch if no cache
        result = {}
        for ticker, name in self.GLOBAL_TICKERS.items():
            try:
                df = yf_safe_download(ticker, period="30d")
                if df.empty:
                    continue
                close = df["Close"].dropna()
                result[name] = {
                    "level":  float(close.iloc[-1]),
                    "1d_pct": float(close.pct_change(1).iloc[-1] * 100),
                    "5d_pct": float(close.pct_change(5).iloc[-1] * 100),
                    "20d_pct":float(close.pct_change(20).iloc[-1] * 100),
                    "series": close,
                }
            except Exception as e:
                logger.warning(f"Global fetch failed for {name}: {e}")
        return result

    def _fetch_peer_data(self) -> Dict:
        """Fetch peer bank returns for relative strength analysis.
        Universe-wide; cached for `_INTERMARKET_CACHE_TTL_SEC` so the 5 banks
        in one orchestrator cycle share a single yfinance fetch (~5× saved).
        """
        cached = _intermarket_cache.get("peers")
        if cached and (time.time() - _intermarket_cache["ts"]) < _INTERMARKET_CACHE_TTL_SEC:
            return cached

        peers = {}
        for ticker, name in self.PEER_TICKERS.items():
            try:
                df = yf_safe_download(ticker, period="30d")
                if df.empty:
                    continue
                close = df["Close"].dropna()
                peers[name] = {
                    "1d_pct":  float(close.pct_change(1).iloc[-1] * 100),
                    "5d_pct":  float(close.pct_change(5).iloc[-1] * 100),
                    "20d_pct": float(close.pct_change(20).iloc[-1] * 100),
                }
            except Exception as e:
                logger.warning(f"Peer fetch failed for {name}: {e}")
        _intermarket_cache["peers"] = peers
        _intermarket_cache["ts"]    = time.time()
        return peers

    # ─────────────────────────────────────────────────────────────
    # FEATURE BUILDERS
    # ─────────────────────────────────────────────────────────────

    def _relative_strength(self, df_hdfc: pd.DataFrame,
                            peers: Dict) -> Dict:
        """
        HDFC Bank vs Bank Nifty and vs peers.
        When HDFC outperforms its own sector — institutional money
        is specifically choosing HDFC over the broader bank basket.
        That selective buying is the strongest stock-specific signal.
        """
        hdfc_1d  = float(df_hdfc["Close"].pct_change(1).iloc[-1] * 100)
        hdfc_5d  = float(df_hdfc["Close"].pct_change(5).iloc[-1] * 100)
        hdfc_20d = float(df_hdfc["Close"].pct_change(20).iloc[-1] * 100)

        # vs Bank Nifty
        bn = self._fetch_global_data().get("banknifty", {})
        bn_5d = bn.get("5d_pct", 0)
        rs_vs_bn = round(hdfc_5d - bn_5d, 3)

        # vs peer average
        peer_5d_avg = (
            np.mean([p["5d_pct"] for p in peers.values()]) if peers else 0
        )
        rs_vs_peers = round(hdfc_5d - peer_5d_avg, 3)

        # Rank among peers (1 = strongest)
        all_5d = {"HDFC": hdfc_5d, **{n: p["5d_pct"] for n, p in peers.items()}}
        rank   = sorted(all_5d, key=all_5d.get, reverse=True).index("HDFC") + 1

        return {
            "hdfc_1d_pct":          round(hdfc_1d, 3),
            "hdfc_5d_pct":          round(hdfc_5d, 3),
            "hdfc_20d_pct":         round(hdfc_20d, 3),
            "rs_vs_banknifty_5d":   rs_vs_bn,
            "rs_vs_peers_5d":       rs_vs_peers,
            "peer_rank":            rank,
            "outperforming_sector": int(rs_vs_bn > 0.5),
            "outperforming_peers":  int(rs_vs_peers > 0.5),
        }

    def _bond_yield_features(self, global_data: Dict) -> Dict:
        """
        US 10Y yield impact on HDFC Bank via FII flows.
        Rising yields → capital goes to US Treasuries → FII sell India.
        This is the primary driver of FII equity flow direction.
        """
        us10y = global_data.get("us_10y", {})
        level = us10y.get("level", 4.3)
        chg_5d= us10y.get("5d_pct", 0)

        # Basis points change
        if us10y.get("series") is not None:
            series = us10y["series"]
            bp_5d  = round((float(series.iloc[-1]) - float(series.iloc[-5])) * 100, 1) \
                     if len(series) >= 5 else 0
        else:
            bp_5d  = 0

        # Signal: >15bps rise in a week = meaningful FII headwind
        headwind = int(bp_5d > 15)
        tailwind = int(bp_5d < -15)

        return {
            "us_10y_level":    round(level, 3),
            "us_10y_bp_5d":    bp_5d,
            "yield_headwind":  headwind,
            "yield_tailwind":  tailwind,
        }

    def _dollar_features(self, global_data: Dict) -> Dict:
        """
        Dollar index and USD/INR impact.
        Strong dollar → rupee weakens → FIIs get fewer USD when selling →
        they accelerate selling → HDFC Bank falls.
        """
        dxy   = global_data.get("dxy",    {})
        inr   = global_data.get("usdinr", {})

        dxy_5d    = dxy.get("5d_pct", 0)
        inr_5d    = inr.get("5d_pct", 0)   # positive = rupee weakening
        dxy_level = dxy.get("level", 103)
        inr_level = inr.get("level", 83.5)

        # Dollar stress = DXY rising AND rupee weakening simultaneously
        dollar_stress = int(dxy_5d > 1.0 and inr_5d > 0.5)
        dollar_ease   = int(dxy_5d < -1.0 and inr_5d < -0.3)

        return {
            "dxy_level":      round(dxy_level, 2),
            "dxy_5d_pct":     round(dxy_5d, 3),
            "usdinr_level":   round(inr_level, 2),
            "usdinr_5d_pct":  round(inr_5d, 3),
            "dollar_stress":  dollar_stress,
            "dollar_ease":    dollar_ease,
        }

    def _crude_features(self, global_data: Dict) -> Dict:
        """
        Crude oil impact on Indian banks.
        High crude → import bill rises → CAD widens → rupee falls →
        RBI tightens → loan demand slows → bank stocks suffer.
        The chain takes 2-3 months to fully materialise.
        """
        crude     = global_data.get("crude", {})
        level     = crude.get("level", 80)
        pct_5d    = crude.get("5d_pct", 0)
        pct_20d   = crude.get("20d_pct", 0)

        # $90+ crude = inflation concern for India
        crude_risk    = int(level > 90 or pct_5d > 5)
        crude_benign  = int(level < 75 and pct_5d < 2)

        return {
            "crude_level":    round(level, 2),
            "crude_5d_pct":   round(pct_5d, 3),
            "crude_20d_pct":  round(pct_20d, 3),
            "crude_risk":     crude_risk,
            "crude_benign":   crude_benign,
        }

    def _global_bank_features(self, global_data: Dict) -> Dict:
        """
        JPMorgan as leading indicator for Indian bank stocks.
        Global bank sentiment often leads domestic banks by 1-2 days.
        When JPMorgan rallies 2%+ — watch for HDFC Bank to follow.
        """
        jpm    = global_data.get("jpmorgan", {})
        jpm_5d = jpm.get("5d_pct", 0)
        jpm_1d = jpm.get("1d_pct", 0)

        return {
            "jpmorgan_1d_pct": round(jpm_1d, 3),
            "jpmorgan_5d_pct": round(jpm_5d, 3),
            "global_banks_bullish": int(jpm_5d > 2.0),
            "global_banks_bearish": int(jpm_5d < -2.0),
        }

    def _sector_rotation(self, peers: Dict,
                          df_hdfc: pd.DataFrame) -> Dict:
        """
        Is money rotating INTO banks or OUT of banks?
        Universe-wide signal — cached so all 5 banks in a cycle share one fetch.
        """
        cached = _intermarket_cache.get("sector")
        if cached and (time.time() - _intermarket_cache["ts"]) < _INTERMARKET_CACHE_TTL_SEC:
            return cached

        try:
            nifty_df = yf_safe_download("^NSEI",    period="10d")
            bn_df    = yf_safe_download("^NSEBANK", period="10d")

            if nifty_df.empty or bn_df.empty:
                return self._empty_sector()

            nifty_5d = float(nifty_df["Close"].pct_change(5).iloc[-1] * 100)
            bn_5d    = float(bn_df["Close"].pct_change(5).iloc[-1] * 100)
            rs       = round(bn_5d - nifty_5d, 3)

            result = {
                "nifty_5d_pct":         round(nifty_5d, 3),
                "banknifty_5d_pct":     round(bn_5d, 3),
                "bank_vs_nifty_rs":     rs,
                "banks_outperforming":  int(rs > 0.5),
                "banks_underperforming":int(rs < -0.5),
            }
            _intermarket_cache["sector"] = result
            _intermarket_cache["ts"]     = time.time()
            return result
        except Exception as e:
            logger.warning(f"Sector rotation failed: {e}")
            return self._empty_sector()

    # ─────────────────────────────────────────────────────────────
    # COMPOSITE SCORE
    # ─────────────────────────────────────────────────────────────

    def _composite_score(self, rs: Dict, bonds: Dict,
                          dollar: Dict, crude: Dict,
                          banks: Dict, sector: Dict) -> float:
        """
        All intermarket signals → single -1 to +1 score.
        This feeds into signal_engine gate 2 rule filter.
        """
        score = 0.0

        # Relative strength (most important for stock selection)
        if rs.get("outperforming_sector"):  score += 0.25
        if rs.get("outperforming_peers"):   score += 0.15
        if rs.get("peer_rank", 5) == 1:     score += 0.10

        # Bond yields
        if bonds.get("yield_headwind"):     score -= 0.15
        if bonds.get("yield_tailwind"):     score += 0.10

        # Dollar
        if dollar.get("dollar_stress"):     score -= 0.20
        if dollar.get("dollar_ease"):       score += 0.15

        # Crude
        if crude.get("crude_risk"):         score -= 0.10
        if crude.get("crude_benign"):       score += 0.05

        # Global banks
        if banks.get("global_banks_bullish"): score += 0.10
        if banks.get("global_banks_bearish"): score -= 0.10

        # Sector rotation
        if sector.get("banks_outperforming"):  score += 0.10
        if sector.get("banks_underperforming"):score -= 0.10

        return round(max(-1.0, min(1.0, score)), 3)

    def _score_to_signal(self, score: float) -> str:
        if score >  0.4: return "STRONG_INTERMARKET_BULL"
        if score >  0.15: return "MILD_INTERMARKET_BULL"
        if score < -0.4: return "STRONG_INTERMARKET_BEAR"
        if score < -0.15: return "MILD_INTERMARKET_BEAR"
        return "INTERMARKET_NEUTRAL"

    # ─────────────────────────────────────────────────────────────
    # DB CACHE
    # ─────────────────────────────────────────────────────────────

    def _get_global_cache(self) -> Dict:
        """Read latest global snapshot from DB (written by global_fetcher)."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            row = conn.execute("""
                SELECT sp500_1d_pct, dxy_level, dxy_5d_pct,
                       usdinr_level, usdinr_5d_pct,
                       crude_level, crude_5d_pct,
                       us_10y_level, us_10y_5d_chg,
                       banknifty_5d_pct, jpmorgan_5d_pct,
                       fetched_at
                FROM   global_snapshots
                ORDER  BY fetched_at DESC LIMIT 1
            """).fetchone()
            conn.close()

            if not row:
                return {}

            # Only use cache if fetched today
            fetched = str(row["fetched_at"])[:10]
            today   = datetime.now().strftime("%Y-%m-%d")
            if fetched < today:
                return {}

            return {
                "dxy":      {"level": row["dxy_level"],     "5d_pct": row["dxy_5d_pct"]},
                "usdinr":   {"level": row["usdinr_level"],  "5d_pct": row["usdinr_5d_pct"]},
                "crude":    {"level": row["crude_level"],   "5d_pct": row["crude_5d_pct"]},
                "us_10y":   {"level": row["us_10y_level"],  "5d_pct": row["us_10y_5d_chg"]},
                "banknifty":{"5d_pct": row["banknifty_5d_pct"]},
                "jpmorgan": {"5d_pct": row["jpmorgan_5d_pct"]},
            }
        except Exception:
            return {}

    def _empty_sector(self) -> Dict:
        return {
            "nifty_5d_pct": 0, "banknifty_5d_pct": 0,
            "bank_vs_nifty_rs": 0,
            "banks_outperforming": 0, "banks_underperforming": 0,
        }