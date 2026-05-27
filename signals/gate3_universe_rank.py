# signals/gate3_universe_rank.py
# Gate 3 — Stock universe ranking. Trade the strongest bank stock.
# Switches focus from HDFC to sector leader if HDFC is not top 2.
# Connected to: price_fetcher.py (peer data), intermarket.py

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, List

from config import TradingConfig
from data.price_fetcher import yf_safe_download

logger = logging.getLogger(__name__)


class Gate3UniverseRank:
    """
    Gate 3: Is HDFC Bank the strongest in the banking sector?

    20yr trader principle:
    Never trade a stock that is underperforming its peers.
    If ICICI Bank is surging and HDFC is flat — money is
    flowing into ICICI specifically. Trade ICICI, not HDFC.
    Relative strength within a sector is free alpha.

    Ranking factors:
    1. 5-day momentum        (40%) — most important
    2. Relative strength vs Bank Nifty (30%)
    3. RSI zone (40-65 = sweet spot) (20%)
    4. Volume trend           (10%)

    Action:
    - HDFC rank 1 or 2 → continue with HDFC
    - HDFC rank 3+ → suggest switching to rank 1 stock
    - Gate still PASSES but alerts trader to better opportunity
    """

    # Universe from config — 5 private banks only (SBI excluded: PSU dynamics differ)
    UNIVERSE = {t: t.replace(".NS", "").replace("BANK", " Bank").replace("INDUSINDBK", "IndusInd")
                for t in TradingConfig.UNIVERSE}
    UNIVERSE_NAMES = {
        "HDFCBANK.NS":  "HDFC Bank",
        "ICICIBANK.NS": "ICICI Bank",
        "KOTAKBANK.NS": "Kotak Bank",
        "AXISBANK.NS":  "Axis Bank",
        "INDUSINDBK.NS":"IndusInd Bank",
    }

    PRIMARY_TICKER  = TradingConfig.TICKER

    # Minimum composite score to qualify for signal evaluation.
    # Below -5 = stock losing badly vs peers — not worth trading even with a good ML signal.
    # 20yr rule: never trade a falling knife relative to sector.
    MIN_SCORE_THRESHOLD = -5.0

    # Position size multiplier by rank. Strongest bank gets full size;
    # weaker banks get progressively smaller positions.
    # Rationale: relative strength IS the edge — reward it with size.
    RANK_SIZE_MULT = {1: 1.00, 2: 0.85, 3: 0.70, 4: 0.55, 5: 0.40}

    def check(
        self,
        hdfc_features: Dict,
        ticker: str = None,
    ) -> Tuple[bool, Dict]:
        """
        Rank all 5 banks and return size_mult for the requested ticker.

        Gate always passes — it never blocks a signal — but:
        - Assigns a rank-based size_mult so position sizing reflects RS.
        - Flags banks below MIN_SCORE_THRESHOLD (size_mult = 0 means skip).
        - Recommends switching to rank-1 when the requested ticker is weak.
        """
        ticker = ticker or self.PRIMARY_TICKER
        rankings = self._rank_stocks()
        if not rankings:
            return True, {
                "gate":           3,
                "passed":         True,
                "primary_ticker": ticker,
                "hdfc_rank":      1,
                "ticker_rank":    1,
                "size_mult":      1.0,
                "reason":         "Universe ranking unavailable — defaulting",
                "rankings":       [],
            }

        # Find ranks
        hdfc_rank   = next((i+1 for i,r in enumerate(rankings)
                            if r["ticker"] == self.PRIMARY_TICKER), len(rankings))
        ticker_rank = next((i+1 for i,r in enumerate(rankings)
                            if r["ticker"] == ticker), len(rankings))
        ticker_score= next((r["score"] for r in rankings if r["ticker"] == ticker), 0.0)

        best_ticker = rankings[0]["ticker"]
        best_name   = rankings[0]["name"]
        best_score  = rankings[0]["score"]

        # Disqualify if far below peers (falling knife filter)
        if ticker_score < self.MIN_SCORE_THRESHOLD:
            return True, {
                "gate":           3,
                "passed":         True,
                "primary_ticker": best_ticker,
                "hdfc_rank":      hdfc_rank,
                "ticker_rank":    ticker_rank,
                "size_mult":      0.0,
                "disqualified":   True,
                "reason":         f"{ticker} score {ticker_score:.1f} < threshold {self.MIN_SCORE_THRESHOLD} — skip",
                "rankings":       rankings,
            }

        size_mult    = self.RANK_SIZE_MULT.get(ticker_rank, 0.40)
        should_switch= hdfc_rank > 2

        return True, {
            "gate":           3,
            "passed":         True,
            "hdfc_rank":      hdfc_rank,
            "ticker_rank":    ticker_rank,
            "ticker_score":   round(ticker_score, 2),
            "best_ticker":    best_ticker,
            "best_name":      best_name,
            "best_score":     round(best_score, 2),
            "should_switch":  should_switch,
            "size_mult":      size_mult,
            "primary_ticker": best_ticker if should_switch else self.PRIMARY_TICKER,
            "switch_reason":  (
                f"HDFC Bank ranked #{hdfc_rank} — {best_name} is stronger today"
            ) if should_switch else None,
            "rankings":       rankings,
        }

    def get_top_tickers(self, n: int = 5) -> List[str]:
        """
        Return all banks that pass the minimum score filter, ranked by RS.
        n caps the list but all 5 are evaluated by default.

        20yr principle: scan ALL qualified names — missing a great setup in
        a rank-3 bank because of an arbitrary cutoff is opportunity cost.
        The rank-based size_mult in check() handles risk, not this list.
        """
        rankings = self._rank_stocks()
        if not rankings:
            return [self.PRIMARY_TICKER]
        qualified = [r["ticker"] for r in rankings
                     if r["score"] >= self.MIN_SCORE_THRESHOLD]
        return qualified[:n] if qualified else [self.PRIMARY_TICKER]

    # Process-wide cache: yfinance hits + ranking math are expensive and the
    # universe ranking is identical for every bank in the same cycle. One
    # cache for all Gate3UniverseRank instances; TTL matched to the orchestrator's
    # full-pipeline interval (5 banks should hit yfinance once per cycle, not 5×).
    _RANK_CACHE: tuple = (0.0, [])   # (epoch, rankings)
    _RANK_TTL_SEC: float = 120.0     # 2 min — well under FULL_INTERVAL but > one cycle

    def _rank_stocks(self) -> List[Dict]:
        """
        Fetch peer prices and compute composite ranking score.
        Memoised across all Gate3UniverseRank instances for `_RANK_TTL_SEC`
        seconds so a 5-bank cycle does 6 yfinance calls (Bank Nifty + 5 banks)
        ONCE per cycle instead of 30×. Cache invalidates after TTL.
        """
        import time as _time
        ts, cached = Gate3UniverseRank._RANK_CACHE
        if cached and (_time.time() - ts) < Gate3UniverseRank._RANK_TTL_SEC:
            return cached

        rankings = []
        try:
            # Fetch Bank Nifty for relative strength calculation
            bn_df = yf_safe_download("^NSEBANK", period="30d")
            bn_5d = float(bn_df["Close"].pct_change(5).iloc[-1] * 100) \
                    if not bn_df.empty else 0.0

            for ticker, name in self.UNIVERSE_NAMES.items():
                try:
                    df = yf_safe_download(ticker, period="30d")
                    if df.empty or len(df) < 10:
                        continue

                    close   = df["Close"]
                    volume  = df["Volume"]

                    mom_5d  = float(close.pct_change(5).iloc[-1] * 100)
                    mom_20d = float(close.pct_change(20).iloc[-1] * 100)
                    rs_bn   = mom_5d - bn_5d
                    rsi     = self._rsi(close)
                    rsi_zone= 1.0 if 40 <= rsi <= 65 else (
                              0.5 if 35 <= rsi <= 70 else 0.0
                    )
                    vol_tr  = float(
                        volume.rolling(5).mean().iloc[-1] /
                        (volume.rolling(20).mean().iloc[-1] + 1e-8)
                    )

                    score = (
                        mom_5d  * 0.40 +
                        rs_bn   * 0.30 +
                        rsi_zone* 20 * 0.20 +   # scale to similar range
                        (vol_tr - 1) * 10 * 0.10
                    )

                    rankings.append({
                        "ticker":    ticker,
                        "name":      name,
                        "score":     score,
                        "mom_5d":    round(mom_5d, 2),
                        "rs_bn":     round(rs_bn, 2),
                        "rsi":       round(rsi, 1),
                        "vol_trend": round(vol_tr, 2),
                    })

                except Exception as e:
                    logger.debug(f"Ranking failed for {ticker}: {e}")

            rankings.sort(key=lambda x: x["score"], reverse=True)

        except Exception as e:
            logger.warning(f"Universe ranking error: {e}")

        # Stamp the cache so subsequent banks in the same cycle reuse this work.
        if rankings:
            Gate3UniverseRank._RANK_CACHE = (_time.time(), rankings)
        return rankings

    def _rsi(self, close: pd.Series, window: int = 14) -> float:
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(window).mean()
        loss  = (-delta.clip(upper=0)).rolling(window).mean()
        rs    = gain / (loss + 1e-10)
        rsi   = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if not rsi.empty else 50.0