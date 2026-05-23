# data/options_fetcher.py
# Fetches NSE options chain data for HDFC Bank
# Calculates: PCR, Max Pain, IV Percentile, Max OI Strikes
# Source: NSE India public API (free, no key needed)

import requests
import sqlite3
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

NSE_SYMBOL = "HDFCBANK"


class OptionsFetcher:
    """
    Fetches NSE options chain and derives key features:
    - PCR (Put-Call Ratio) — primary smart money signal
    - PCR Rate of Change   — leading indicator
    - Max Pain             — expiry week price magnet
    - IV Percentile        — options cheap or expensive
    - Max OI Strikes       — dynamic support/resistance levels
    - Implied Move         — expected price move from straddle price

    Runs every 30 minutes during market hours.
    """

    NSE_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
        "Referer": "https://www.nseindia.com/",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }

    def __init__(self, db_path: str = "database/trading.db"):
        self.db_path = db_path
        self._session = self._create_session()
        self._setup_db()

    # ─────────────────────────────────────────────────────────────
    # PUBLIC METHODS
    # ─────────────────────────────────────────────────────────────

    def fetch_and_calculate(self) -> Dict:
        """
        Master method — fetch chain, calculate all features, save to DB.
        Returns dict of all options features for use in signal engine.
        Called every 30 minutes during market hours.
        """
        chain = self._fetch_options_chain()
        if not chain:
            logger.warning("Options chain fetch returned empty data")
            return self._empty_result()

        try:
            result = {
                "pcr":                  self._calculate_pcr(chain),
                "pcr_roc":              self._calculate_pcr_roc(),
                "max_pain":             self._calculate_max_pain(chain),
                "iv_percentile":        self._calculate_iv_percentile(chain),
                "max_call_oi_strike":   self._get_max_call_oi_strike(chain),
                "max_put_oi_strike":    self._get_max_put_oi_strike(chain),
                "implied_move_pct":     self._calculate_implied_move(chain),
                "total_call_oi":        self._sum_oi(chain, "CE"),
                "total_put_oi":         self._sum_oi(chain, "PE"),
                "atm_strike":           self._get_atm_strike(chain),
                "spot_price":           chain.get("records", {}).get("underlyingValue", 0),
                "fetched_at":           str(datetime.now()),
            }

            # Save snapshot to DB for PCR ROC calculation
            self._save_snapshot(result)

            logger.info(
                f"Options: PCR={result['pcr']:.2f} | "
                f"MaxPain=₹{result['max_pain']} | "
                f"IV%ile={result['iv_percentile']:.0f}%"
            )
            return result

        except Exception as e:
            logger.error(f"Options calculation failed: {e}")
            return self._empty_result()

    def get_latest(self) -> Dict:
        """
        Get most recent options snapshot from DB (no API call).
        Used when market is closed or for quick reads.
        """
        conn = self._connect()
        row = conn.execute("""
            SELECT pcr, pcr_roc, max_pain, iv_percentile,
                   max_call_oi_strike, max_put_oi_strike,
                   implied_move_pct, spot_price, fetched_at
            FROM options_snapshots
            ORDER BY fetched_at DESC
            LIMIT 1
        """).fetchone()
        conn.close()

        if not row:
            return self._empty_result()

        return {
            "pcr":                row[0],
            "pcr_roc":            row[1],
            "max_pain":           row[2],
            "iv_percentile":      row[3],
            "max_call_oi_strike": row[4],
            "max_put_oi_strike":  row[5],
            "implied_move_pct":   row[6],
            "spot_price":         row[7],
            "fetched_at":         row[8],
        }

    def get_snapshot_at(self, as_of_date: str) -> Dict:
        """
        Point-in-time snapshot lookup for training/backtest.
        Returns the most recent options_snapshots row with fetched_at <= as_of_date.
        Same shape as get_latest(). Empty result if no row exists.

        Lookahead-safe: bhavcopy and live snapshots are end-of-day,
        so a row dated T reflects T's close — usable for T+1 decisions.
        """
        conn = self._connect()
        row = conn.execute("""
            SELECT pcr, pcr_roc, max_pain, iv_percentile,
                   max_call_oi_strike, max_put_oi_strike,
                   implied_move_pct, spot_price, fetched_at
            FROM   options_snapshots
            WHERE  fetched_at <= ?
            ORDER  BY fetched_at DESC
            LIMIT  1
        """, (as_of_date,)).fetchone()
        conn.close()

        if not row:
            return self._empty_result()

        return {
            "pcr":                row[0],
            "pcr_roc":            row[1],
            "max_pain":           row[2],
            "iv_percentile":      row[3],
            "max_call_oi_strike": row[4],
            "max_put_oi_strike":  row[5],
            "implied_move_pct":   row[6],
            "spot_price":         row[7],
            "fetched_at":         row[8],
        }

    def get_expiry_features_at(self, as_of_date: str) -> Dict:
        """
        Expiry features as of a historical date.
        Computes days-to-monthly-expiry from `as_of_date` (last Thursday rule)
        without using `datetime.now()`, so it's safe inside training loops.
        """
        try:
            today = datetime.strptime(as_of_date[:10], "%Y-%m-%d")
        except ValueError:
            today = datetime.now()

        import calendar
        year, month = today.year, today.month
        last_day = calendar.monthrange(year, month)[1]
        expiry = None
        for day in range(last_day, 0, -1):
            candidate = datetime(year, month, day)
            if candidate.weekday() == 3:  # Thursday
                if candidate >= today:
                    expiry = candidate
                break

        if expiry is None:
            if month == 12:
                year, month = year + 1, 1
            else:
                month += 1
            last_day = calendar.monthrange(year, month)[1]
            for day in range(last_day, 0, -1):
                if datetime(year, month, day).weekday() == 3:
                    expiry = datetime(year, month, day)
                    break

        days_to_expiry = (expiry - today).days if expiry else 15

        return {
            "days_to_monthly_expiry": max(0, days_to_expiry),
            "is_expiry_week":         int(0 <= days_to_expiry <= 5),
            "is_expiry_day":          int(days_to_expiry == 0),
            "max_pain_distance":      0.0,    # not derivable historically
            "max_pain_direction":     "NONE", # without spot vs max_pain at that date
        }

    def get_pcr_signal(self) -> str:
        """
        Convert PCR value to human-readable signal.
        PCR > 1.3 = BULLISH (too many puts = contrarian buy)
        PCR < 0.7 = BEARISH (too many calls = complacency)
        """
        data = self.get_latest()
        pcr = data.get("pcr", 1.0)

        if pcr > 1.3:
            return "BULLISH"
        elif pcr < 0.7:
            return "BEARISH"
        else:
            return "NEUTRAL"

    def get_expiry_features(self) -> Dict:
        """
        Calculate expiry-related features for ML model.
        NSE monthly expiry = last Thursday of the month.
        """
        today = datetime.now()
        expiry = self._get_next_expiry()
        days_to_expiry = (expiry - today).days

        return {
            "days_to_monthly_expiry": max(0, days_to_expiry),
            "is_expiry_week":         int(days_to_expiry <= 5),
            "is_expiry_day":          int(days_to_expiry == 0),
            "max_pain_distance":      self._get_max_pain_distance(),
            "max_pain_direction":     self._get_max_pain_direction(),
        }

    # ─────────────────────────────────────────────────────────────
    # CALCULATION METHODS
    # ─────────────────────────────────────────────────────────────

    def _calculate_pcr(self, chain: Dict) -> float:
        """
        Put-Call Ratio = total put OI / total call OI.
        Higher PCR = more puts = smart money hedging = contrarian bullish.
        """
        total_call_oi = self._sum_oi(chain, "CE")
        total_put_oi  = self._sum_oi(chain, "PE")

        if total_call_oi == 0:
            return 1.0

        return round(total_put_oi / total_call_oi, 3)

    def _calculate_pcr_roc(self) -> float:
        """
        PCR Rate of Change vs 5 days ago.
        Increasing PCR = institutions buying more protection = warning.
        """
        conn = self._connect()
        rows = conn.execute("""
            SELECT pcr FROM options_snapshots
            ORDER BY fetched_at DESC
            LIMIT 10
        """).fetchall()
        conn.close()

        if len(rows) < 2:
            return 0.0

        current_pcr = rows[0][0]

        # Use 5-snapshot average for smoothing
        older_pcr = np.mean([r[0] for r in rows[1:6]])
        if older_pcr == 0:
            return 0.0

        return round(current_pcr - older_pcr, 3)

    def _calculate_max_pain(self, chain: Dict) -> float:
        """
        Max pain = price where total option seller loss is minimized.
        At expiry, price gravitates toward max pain.
        """
        records = chain.get("records", {}).get("data", [])
        if not records:
            return 0.0

        strikes = {}
        for record in records:
            strike = record.get("strikePrice", 0)
            call_oi = record.get("CE", {}).get("openInterest", 0)
            put_oi  = record.get("PE", {}).get("openInterest", 0)
            strikes[strike] = {"call_oi": call_oi, "put_oi": put_oi}

        all_strikes = sorted(strikes.keys())
        if not all_strikes:
            return 0.0

        pain_at_strike = {}
        for S in all_strikes:
            total_pain = 0
            for K, oi in strikes.items():
                # Call pain: when S > K, call sellers lose (S-K)*call_oi
                if S > K:
                    total_pain += (S - K) * oi["call_oi"]
                # Put pain: when S < K, put sellers lose (K-S)*put_oi
                if S < K:
                    total_pain += (K - S) * oi["put_oi"]
            pain_at_strike[S] = total_pain

        # Max pain = strike with minimum total pain (sellers want price here)
        max_pain_strike = min(pain_at_strike, key=pain_at_strike.get)
        return float(max_pain_strike)

    def _calculate_iv_percentile(self, chain: Dict) -> float:
        """
        IV Percentile = where current ATM IV sits vs last 252 readings.
        Low IV pct = options cheap, buy them.
        High IV pct = options expensive, sell them.
        """
        current_iv = self._get_atm_iv(chain)
        if current_iv == 0:
            return 50.0

        # Get historical IV readings
        conn = self._connect()
        rows = conn.execute("""
            SELECT atm_iv FROM options_snapshots
            WHERE atm_iv > 0
            ORDER BY fetched_at DESC
            LIMIT 252
        """).fetchall()
        conn.close()

        if len(rows) < 10:
            return 50.0  # not enough history — return neutral

        historical_ivs = [r[0] for r in rows]
        below_current = sum(1 for iv in historical_ivs if iv < current_iv)
        percentile = (below_current / len(historical_ivs)) * 100

        return round(percentile, 1)

    def _calculate_implied_move(self, chain: Dict) -> float:
        """
        Implied move = (ATM call price + ATM put price) / spot price.
        This is the market's expected ±move before expiry.
        """
        atm_strike = self._get_atm_strike(chain)
        spot = chain.get("records", {}).get("underlyingValue", 0)

        if not spot or not atm_strike:
            return 0.0

        records = chain.get("records", {}).get("data", [])
        for record in records:
            if record.get("strikePrice") == atm_strike:
                call_price = record.get("CE", {}).get("lastPrice", 0)
                put_price  = record.get("PE", {}).get("lastPrice", 0)
                if spot > 0:
                    return round((call_price + put_price) / spot * 100, 2)

        return 0.0

    def _get_max_call_oi_strike(self, chain: Dict) -> float:
        """Strike with highest call OI = key resistance level."""
        records = chain.get("records", {}).get("data", [])
        max_oi = 0
        max_strike = 0
        for record in records:
            oi = record.get("CE", {}).get("openInterest", 0)
            if oi > max_oi:
                max_oi = oi
                max_strike = record.get("strikePrice", 0)
        return float(max_strike)

    def _get_max_put_oi_strike(self, chain: Dict) -> float:
        """Strike with highest put OI = key support level."""
        records = chain.get("records", {}).get("data", [])
        max_oi = 0
        max_strike = 0
        for record in records:
            oi = record.get("PE", {}).get("openInterest", 0)
            if oi > max_oi:
                max_oi = oi
                max_strike = record.get("strikePrice", 0)
        return float(max_strike)

    def _get_atm_strike(self, chain: Dict) -> float:
        """Find at-the-money strike (nearest to spot price)."""
        spot = chain.get("records", {}).get("underlyingValue", 0)
        if not spot:
            return 0.0
        records = chain.get("records", {}).get("data", [])
        strikes = [r.get("strikePrice", 0) for r in records]
        if not strikes:
            return 0.0
        return min(strikes, key=lambda x: abs(x - spot))

    def _get_atm_iv(self, chain: Dict) -> float:
        """Get implied volatility of ATM call option."""
        atm_strike = self._get_atm_strike(chain)
        records = chain.get("records", {}).get("data", [])
        for record in records:
            if record.get("strikePrice") == atm_strike:
                return float(record.get("CE", {}).get("impliedVolatility", 0))
        return 0.0

    def _sum_oi(self, chain: Dict, option_type: str) -> int:
        """Sum total open interest for calls (CE) or puts (PE)."""
        records = chain.get("records", {}).get("data", [])
        return sum(
            r.get(option_type, {}).get("openInterest", 0)
            for r in records
        )

    def _get_max_pain_distance(self) -> float:
        """Distance of current price from max pain as percentage."""
        data = self.get_latest()
        spot = data.get("spot_price", 0)
        max_pain = data.get("max_pain", 0)
        if spot == 0 or max_pain == 0:
            return 0.0
        return round((spot - max_pain) / spot * 100, 2)

    def _get_max_pain_direction(self) -> str:
        """Is price above or below max pain? Price moves toward max pain."""
        distance = self._get_max_pain_distance()
        if distance > 0.5:
            return "ABOVE_MAX_PAIN"   # price likely to fall toward max pain
        elif distance < -0.5:
            return "BELOW_MAX_PAIN"   # price likely to rise toward max pain
        return "AT_MAX_PAIN"

    # ─────────────────────────────────────────────────────────────
    # NETWORK & STORAGE
    # ─────────────────────────────────────────────────────────────

    def _fetch_options_chain(self) -> Dict:
        """Fetch live options chain from NSE."""
        url = (
            f"https://www.nseindia.com/api/option-chain-equities"
            f"?symbol={NSE_SYMBOL}"
        )
        try:
            response = self._session.get(
                url, headers=self.NSE_HEADERS, timeout=15
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"NSE options chain returned {response.status_code}")
                return {}
        except Exception as e:
            logger.error(f"Options chain fetch failed: {e}")
            return {}

    def _save_snapshot(self, data: Dict) -> None:
        """Save options snapshot to DB for PCR ROC and IV percentile."""
        conn = self._connect()
        conn.execute("""
            INSERT INTO options_snapshots
            (pcr, pcr_roc, max_pain, iv_percentile, atm_iv,
             max_call_oi_strike, max_put_oi_strike,
             implied_move_pct, total_call_oi, total_put_oi,
             spot_price, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get("pcr", 1.0),
            data.get("pcr_roc", 0.0),
            data.get("max_pain", 0),
            data.get("iv_percentile", 50),
            0.0,  # atm_iv stored separately in full flow
            data.get("max_call_oi_strike", 0),
            data.get("max_put_oi_strike", 0),
            data.get("implied_move_pct", 0),
            data.get("total_call_oi", 0),
            data.get("total_put_oi", 0),
            data.get("spot_price", 0),
            data["fetched_at"],
        ))
        conn.commit()
        conn.close()

    def _get_next_expiry(self) -> datetime:
        """
        NSE monthly expiry = last Thursday of the current month.
        If last Thursday has passed, use next month's last Thursday.
        """
        today = datetime.now()
        year, month = today.year, today.month

        # Find last Thursday of current month
        import calendar
        last_day = calendar.monthrange(year, month)[1]
        for day in range(last_day, 0, -1):
            if datetime(year, month, day).weekday() == 3:  # Thursday = 3
                expiry = datetime(year, month, day)
                if expiry > today:
                    return expiry
                break

        # Move to next month
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1

        last_day = __import__("calendar").monthrange(year, month)[1]
        for day in range(last_day, 0, -1):
            if datetime(year, month, day).weekday() == 3:
                return datetime(year, month, day)

        return today + timedelta(days=30)  # fallback

    def _create_session(self) -> requests.Session:
        """Create session with NSE cookie pre-loaded."""
        session = requests.Session()
        try:
            session.get(
                "https://www.nseindia.com",
                headers=self.NSE_HEADERS,
                timeout=10,
            )
        except Exception:
            pass
        return session

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _empty_result(self) -> Dict:
        return {
            "pcr": 1.0, "pcr_roc": 0.0, "max_pain": 0,
            "iv_percentile": 50.0, "max_call_oi_strike": 0,
            "max_put_oi_strike": 0, "implied_move_pct": 0,
            "total_call_oi": 0, "total_put_oi": 0,
            "spot_price": 0, "fetched_at": str(datetime.now()),
        }

    def _setup_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS options_snapshots (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                pcr                  REAL,
                pcr_roc              REAL,
                max_pain             REAL,
                iv_percentile        REAL,
                atm_iv               REAL,
                max_call_oi_strike   REAL,
                max_put_oi_strike    REAL,
                implied_move_pct     REAL,
                total_call_oi        INTEGER,
                total_put_oi         INTEGER,
                spot_price           REAL,
                fetched_at           TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_options_fetched
            ON options_snapshots (fetched_at DESC)
        """)
        conn.commit()
        conn.close()