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
    Gate 3: Best bank stock within the universe for the current regime.

    20yr trader principle:
    LONG: Never trade a stock that is underperforming its peers.
          Trade the STRONGEST bank — money is flowing there.
    SHORT: Never short the strongest stock in a falling sector.
           Short the WEAKEST bank — that's where the breakdown is.
    Relative strength within a sector is free alpha in BOTH directions.

    Ranking factors (regime-aware):
    ─── BULL / LONG ─────────────────────────────────────
    1. 5-day momentum         (40%) — strongest upward mover
    2. Relative strength vs BankNifty (30%)
    3. RSI zone 40–65 = sweet spot  (20%) — momentum not overbought
    4. Volume trend            (10%)
    ─── BEAR / SHORT ────────────────────────────────────
    1. Most NEGATIVE 5d momentum  (40%) — weakest stock = best short
    2. Most NEGATIVE RS vs BankNifty (30%) — sector laggard
    3. RSI zone: ≥65 OR ≤35 = sweet spot (20%)
       - ≥65 (extended = bear-flag setup), ≤35 (confirmed breakdown)
    4. Volume trend            (10%) — same; high vol on down = conviction

    Action (BULL):
    - Ticker rank 1 or 2 → trade this stock (strongest)
    - Ticker rank 3+ → suggest switching to rank-1

    Action (BEAR):
    - Ticker rank 1 or 2 → trade this stock (weakest = best short)
    - Ticker rank 3+ → suggest switching to rank-1 (weakest)
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
        ticker: str   = None,
        regime: str   = "BULL_TRENDING",
    ) -> Tuple[bool, Dict]:
        """
        Rank all 5 banks and return size_mult for the requested ticker.

        Gate always passes — it never blocks a signal — but:
        - Assigns a rank-based size_mult so position sizing reflects RS.
        - Flags banks below MIN_SCORE_THRESHOLD (size_mult = 0 means skip).
        - Recommends switching to rank-1 when the requested ticker is sub-optimal.

        Regime-aware:
        - BULL: rank-1 = strongest (best LONG candidate) → most size.
        - BEAR: score is inverted so rank-1 = weakest (best SHORT candidate) → most size.
        - HIGH_VOL: uses BULL-style scoring (both directions valid; rank by absolute momentum).
        """
        ticker    = ticker or self.PRIMARY_TICKER
        is_bear   = regime == "BEAR_TRENDING"
        rankings  = self._rank_stocks(regime)
        if not rankings:
            return True, {
                "gate":           3,
                "passed":         True,
                "primary_ticker": ticker,
                "hdfc_rank":      1,
                "ticker_rank":    1,
                "size_mult":      1.0,
                "regime":         regime,
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

        # Disqualify if score is far from tradeable range.
        # In BULL: score < -5 = stock crashing vs peers (falling knife).
        # In BEAR: score < -5 = stock is STRONGLY RISING vs peers = worst short
        #          candidate (potential short-squeeze risk — avoid).
        if ticker_score < self.MIN_SCORE_THRESHOLD:
            return True, {
                "gate":           3,
                "passed":         True,
                "primary_ticker": best_ticker,
                "hdfc_rank":      hdfc_rank,
                "ticker_rank":    ticker_rank,
                "size_mult":      0.0,
                "disqualified":   True,
                "regime":         regime,
                "reason":         (
                    f"{ticker} score {ticker_score:.1f} < {self.MIN_SCORE_THRESHOLD} — "
                    f"{'short-squeeze risk (rising in BEAR)' if is_bear else 'falling knife vs peers'}"
                ),
                "rankings":       rankings,
            }

        size_mult    = self.RANK_SIZE_MULT.get(ticker_rank, 0.40)
        # should_switch = True when this bank is not a top-2 candidate.
        # In BULL: top-2 = strongest (best LONGs). In BEAR: top-2 = weakest (best SHORTs).
        # Use ticker_rank (not hdfc_rank) — each bank's engine should flag itself.
        should_switch = ticker_rank > 2

        regime_label = "weakest (best short)" if is_bear else "strongest (best long)"
        ticker_short  = ticker.replace(".NS", "")
        switch_label = (
            f"{ticker_short} ranked #{ticker_rank} — {best_name} is the {regime_label} today"
        ) if should_switch else None

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
            "regime":         regime,
            "primary_ticker": best_ticker if should_switch else self.PRIMARY_TICKER,
            "switch_reason":  switch_label,
            "rankings":       rankings,
        }

    def get_top_tickers(self, n: int = 5, regime: str = "BULL_TRENDING") -> List[str]:
        """
        Return all banks that pass the minimum score filter, ranked by RS.
        n caps the list but all 5 are evaluated by default.

        In BULL: returns strongest-first (best LONGs at the top).
        In BEAR: returns weakest-first after score inversion (best SHORTs at the top).
        The rank-based size_mult in check() handles risk, not this list.

        20yr principle: scan ALL qualified names — missing a great setup in
        a rank-3 bank because of an arbitrary cutoff is opportunity cost.
        """
        rankings = self._rank_stocks(regime)
        if not rankings:
            return [self.PRIMARY_TICKER]
        qualified = [r["ticker"] for r in rankings
                     if r["score"] >= self.MIN_SCORE_THRESHOLD]
        return qualified[:n] if qualified else [self.PRIMARY_TICKER]

    # Process-wide cache keyed by regime — yfinance hits + ranking math are
    # expensive. All 5 SignalEngines in a cycle share the same regime in practice.
    # TTL matched to the orchestrator's full-pipeline interval so one cycle does
    # 6 yfinance calls (BankNifty + 5 banks) per regime, not 30×.
    _RANK_CACHE: Dict  = {}           # {regime: (epoch, rankings)}
    _RANK_TTL_SEC: float = 120.0      # 2 min

    def _rank_stocks(self, regime: str = "BULL_TRENDING") -> List[Dict]:
        """
        Fetch peer prices and compute composite ranking score.

        Regime-aware scoring (memoised per-regime):
        ─── BULL / HIGH_VOL ──────────────────────────────────
          score = (+mom_5d × 0.40) + (+rs_bn × 0.30)
                + rsi_long_zone × 20 × 0.20 + vol_trend × 10 × 0.10
          rank-1 = strongest (best LONG candidate)

        ─── BEAR ─────────────────────────────────────────────
          score = (−mom_5d × 0.40) + (−rs_bn × 0.30)
                + rsi_short_zone × 20 × 0.20 + vol_trend × 10 × 0.10
          SIGNS INVERTED so the most-negative-momentum stock scores highest.
          rank-1 = weakest (best SHORT candidate)

          RSI short zone (BEAR): ≥65 (extended = bear-flag) OR ≤35 (confirmed
          breakdown). Both are valid short entries. Neutral RSI 45-65 scores 0.

          Disqualification in BEAR: score < -5 means the stock is RISING
          strongly vs peers — short-squeeze risk, avoid.
        """
        import time as _time
        is_bear = regime == "BEAR_TRENDING"

        cached_ts, cached_list = Gate3UniverseRank._RANK_CACHE.get(regime, (0.0, []))
        if cached_list and (_time.time() - cached_ts) < Gate3UniverseRank._RANK_TTL_SEC:
            return cached_list

        rankings = []
        try:
            # Fetch Bank Nifty for relative-strength baseline
            bn_df = yf_safe_download("^NSEBANK", period="30d")
            bn_5d = float(bn_df["Close"].pct_change(5).iloc[-1] * 100) \
                    if not bn_df.empty else 0.0

            for ticker, name in self.UNIVERSE_NAMES.items():
                try:
                    df = yf_safe_download(ticker, period="30d")
                    if df.empty or len(df) < 10:
                        continue

                    close  = df["Close"]
                    volume = df["Volume"]

                    mom_5d = float(close.pct_change(5).iloc[-1] * 100)
                    rs_bn  = mom_5d - bn_5d
                    rsi    = self._rsi(close)
                    vol_tr = float(
                        volume.rolling(5).mean().iloc[-1] /
                        (volume.rolling(20).mean().iloc[-1] + 1e-8)
                    )

                    if is_bear:
                        # BEAR: invert momentum/RS signs so the WEAKEST stock ranks #1.
                        # RSI sweet spot: ≥65 (overbought/extended = bear-flag setup)
                        # OR ≤35 (confirmed breakdown = momentum short). Neutral = 0.
                        rsi_zone = (
                            1.0 if (rsi >= 65 or rsi <= 35) else
                            (0.5 if (rsi >= 55 or rsi <= 45) else 0.0)
                        )
                        score = (
                            (-mom_5d) * 0.40 +     # falling stock scores higher
                            (-rs_bn)  * 0.30 +     # sector laggard scores higher
                            rsi_zone  * 20 * 0.20 +
                            (vol_tr - 1) * 10 * 0.10
                        )
                    else:
                        # BULL / HIGH_VOL: original scoring — strongest stock ranks #1.
                        # RSI sweet spot: 40-65 (momentum zone, not overbought).
                        rsi_zone = (
                            1.0 if 40 <= rsi <= 65 else
                            (0.5 if 35 <= rsi <= 70 else 0.0)
                        )
                        score = (
                            mom_5d  * 0.40 +
                            rs_bn   * 0.30 +
                            rsi_zone * 20 * 0.20 +
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
                        "regime":    regime,
                    })

                except Exception as e:
                    logger.debug(f"Ranking failed for {ticker}: {e}")

            rankings.sort(key=lambda x: x["score"], reverse=True)

        except Exception as e:
            logger.warning(f"Universe ranking error: {e}")

        # Stamp the per-regime cache
        if rankings:
            Gate3UniverseRank._RANK_CACHE[regime] = (_time.time(), rankings)
        return rankings

    def _rsi(self, close: pd.Series, window: int = 14) -> float:
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(window).mean()
        loss  = (-delta.clip(upper=0)).rolling(window).mean()
        rs    = gain / (loss + 1e-10)
        rsi   = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1]) if not rsi.empty else 50.0