# processing/fundamental.py
# Scrapes HDFC Bank fundamental data from Screener.in
# Banking-specific metrics: NIM, NPA, CASA, loan growth, P/B
# Quarterly update — provides context for positional model

import requests
import sqlite3
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

from config import get_bank_config


class FundamentalProcessor:
    """
    Extracts HDFC Bank fundamental metrics for positional model.

    Banking fundamentals that matter most (ranked by importance):
    1. NIM trend         — core profitability driver
    2. NPA trend         — asset quality / risk
    3. Loan growth       — revenue growth proxy
    4. CASA ratio        — cost of funds advantage
    5. P/B vs history    — valuation context
    6. Earnings beat streak — management credibility

    Updates quarterly. Between quarters, cached values used.
    Screener.in is free and comprehensive for Indian stocks.
    """

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }

    def __init__(self, db_path: str = "database/trading.db", ticker: str = "HDFCBANK.NS"):
        self.db_path     = db_path
        self.ticker      = ticker
        _cfg             = get_bank_config(ticker)
        self.screener_url = (
            f"https://www.screener.in/company/{_cfg['screener_id']}/consolidated/"
        )
        self._setup_db()

    # ─────────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────────

    def fetch_and_update(self) -> Dict:
        """
        Scrape Screener.in and update fundamental database.
        Called weekly by scheduler (Sundays).
        Returns latest fundamental features.
        """
        try:
            data = self._scrape_screener()
            if data:
                self._save_fundamentals(data)
                logger.info(
                    f"Fundamentals updated: P/B={data.get('pb_ratio',0):.2f} | "
                    f"ROE={data.get('roe',0):.1f}%"
                )
        except Exception as e:
            logger.error(f"Fundamental fetch failed: {e}")

        return self.get_fundamental_features()

    def get_fundamental_features(self) -> Dict:
        """
        Return all fundamental features for ML model.
        Uses latest values from DB — no scraping.
        """
        latest  = self._get_latest()
        history = self._get_history(quarters=8)

        if not latest:
            return self._empty_features()

        # Calculate trends from history
        nim_trend    = self._calc_trend([h.get("nim", 0) for h in history])
        npa_trend    = self._calc_trend([h.get("npa_gross", 0) for h in history], inverted=True)
        loan_trend   = self._calc_trend([h.get("loan_growth", 0) for h in history])
        casa_trend   = self._calc_trend([h.get("casa_ratio", 0) for h in history])

        # P/B vs 5-year average
        pb_current  = latest.get("pb_ratio", 0)
        pb_5yr_avg  = self._get_pb_avg(years=5)
        pb_ratio    = round(pb_current / max(pb_5yr_avg, 0.1), 3)

        # Beat streak
        beat_streak = self._count_beat_streak()

        # Fundamental grade (senior trader quick assessment)
        grade, grade_score = self._grade_fundamentals(latest, nim_trend, npa_trend)

        return {
            # Core banking metrics
            "nim":                latest.get("nim", 0),
            "nim_trend":          nim_trend,
            "npa_gross":          latest.get("npa_gross", 0),
            "npa_net":            latest.get("npa_net", 0),
            "npa_trend":          npa_trend,
            "casa_ratio":         latest.get("casa_ratio", 0),
            "casa_trend":         casa_trend,
            "loan_growth":        latest.get("loan_growth", 0),
            "loan_trend":         loan_trend,
            "deposit_growth":     latest.get("deposit_growth", 0),
            "car":                latest.get("car", 0),          # capital adequacy
            "pcr":                latest.get("pcr", 0),          # provision coverage

            # Profitability
            "roe":                latest.get("roe", 0),
            "roa":                latest.get("roa", 0),
            "cost_to_income":     latest.get("cost_to_income", 0),

            # Valuation
            "pb_ratio":           pb_current,
            "pe_ratio":           latest.get("pe_ratio", 0),
            "pb_vs_5yr_avg":      pb_ratio,       # >1.2 = expensive, <0.8 = cheap

            # Earnings quality
            "eps_surprise":       latest.get("eps_surprise", 0),
            "beat_streak":        beat_streak,

            # Composite
            "fundamental_grade":  grade,
            "fundamental_score":  grade_score,    # -1 to +1 for ML
            "fundamentals_missing": 0,
        }

    # ─────────────────────────────────────────────────────────────
    # SCRAPING
    # ─────────────────────────────────────────────────────────────

    def _scrape_screener(self) -> Optional[Dict]:
        """Scrape HDFC Bank data from Screener.in."""
        try:
            resp = requests.get(self.screener_url, headers=self.HEADERS, timeout=15)
            if resp.status_code != 200:
                logger.warning(f"Screener returned {resp.status_code}")
                return None

            soup = BeautifulSoup(resp.text, "html.parser")
            data = {}

            # Key ratios section
            ratios = soup.find("section", id="ratios")
            if ratios:
                for li in ratios.find_all("li"):
                    name  = li.find("span", class_="name")
                    value = li.find("span", class_="value") or li.find("span", class_="number")
                    if name and value:
                        key = name.get_text(strip=True).lower().replace(" ", "_")
                        val = self._parse_number(value.get_text(strip=True))
                        data[key] = val

            # Map common screener field names
            field_map = {
                "p/b": "pb_ratio", "p/e": "pe_ratio",
                "return_on_equity": "roe", "return_on_assets": "roa",
                "dividend_yield": "div_yield",
                "book_value": "book_value",
            }
            for raw, clean in field_map.items():
                if raw in data:
                    data[clean] = data.pop(raw)

            # Defaults for banking-specific metrics not on screener main page
            # These typically come from quarterly results
            data.setdefault("nim", self._get_latest_metric("nim") or 3.8)
            data.setdefault("npa_gross", self._get_latest_metric("npa_gross") or 1.3)
            data.setdefault("npa_net", self._get_latest_metric("npa_net") or 0.4)
            data.setdefault("casa_ratio", self._get_latest_metric("casa_ratio") or 45.0)
            data.setdefault("loan_growth", self._get_latest_metric("loan_growth") or 16.0)
            data.setdefault("car", self._get_latest_metric("car") or 18.0)
            data.setdefault("pcr", self._get_latest_metric("pcr") or 72.0)
            data.setdefault("cost_to_income", self._get_latest_metric("cost_to_income") or 42.0)
            data.setdefault("eps_surprise", 0.0)
            data.setdefault("deposit_growth", 15.0)
            data.setdefault("roa", data.get("roe", 16) / 10)  # estimate if missing

            data["fetched_at"] = str(datetime.now())
            return data

        except Exception as e:
            logger.error(f"Screener scraping failed: {e}")
            return None

    # ─────────────────────────────────────────────────────────────
    # ANALYSIS
    # ─────────────────────────────────────────────────────────────

    def _calc_trend(self, values: List[float],
                    inverted: bool = False) -> float:
        """
        Calculate trend slope from quarterly values (most recent first).
        Returns -1 to +1 where +1 = strongly improving.
        inverted=True for NPA where declining = positive.
        """
        vals = [v for v in values if v and v != 0]
        if len(vals) < 2:
            return 0.0

        # Simple linear regression slope
        x   = list(range(len(vals)))
        n   = len(vals)
        sx  = sum(x)
        sy  = sum(vals)
        sxy = sum(x[i] * vals[i] for i in range(n))
        sx2 = sum(xi ** 2 for xi in x)
        denom = n * sx2 - sx * sx
        if denom == 0:
            return 0.0

        slope = (n * sxy - sx * sy) / denom
        if inverted:
            slope = -slope

        # Normalise slope to -1 to +1
        return round(max(-1.0, min(1.0, slope / (abs(sum(vals) / n) + 1e-6))), 3)

    def _grade_fundamentals(
        self, latest: Dict, nim_trend: float, npa_trend: float
    ) -> tuple:
        """
        Quick fundamental grade as a senior trader would assess.
        Returns (grade_str, score_float)
        """
        score = 0.0

        # NIM: above 4% is excellent for Indian banks
        nim = latest.get("nim", 0)
        if nim > 4.0:   score += 0.2
        elif nim > 3.5: score += 0.1
        elif nim < 3.0: score -= 0.2

        # NPA: below 1.5% gross NPA is clean
        # Default to 1.3 (HDFC typical) rather than 999 to avoid POOR grade on missing data
        npa = float(latest.get("npa_gross") or 1.3)
        if npa < 1.0:   score += 0.2
        elif npa < 2.0: score += 0.1
        elif npa > 3.0: score -= 0.3

        # NIM trend
        score += nim_trend * 0.15

        # NPA trend (improving = positive)
        score += npa_trend * 0.15

        # ROE: above 16% is excellent
        roe = latest.get("roe", 0)
        if roe > 18:    score += 0.1
        elif roe > 15:  score += 0.05
        elif roe < 12:  score -= 0.1

        # CASA: above 45% is competitive moat
        casa = latest.get("casa_ratio", 0)
        if casa > 48:   score += 0.1
        elif casa < 40: score -= 0.1

        score = round(max(-1.0, min(1.0, score)), 3)

        if score >  0.4: grade = "EXCELLENT"
        elif score > 0.1: grade = "GOOD"
        elif score > -0.1: grade = "AVERAGE"
        elif score > -0.4: grade = "WEAK"
        else: grade = "POOR"

        return grade, score

    def _count_beat_streak(self) -> int:
        """Count consecutive quarters of positive EPS surprise."""
        conn = self._connect()
        rows = conn.execute("""
            SELECT eps_surprise FROM fundamentals
            WHERE  ticker = ?
            ORDER  BY fetched_at DESC LIMIT 8
        """, (self.ticker,)).fetchall()
        conn.close()

        streak = 0
        for row in rows:
            if (row[0] or 0) > 0:
                streak += 1
            else:
                break
        return streak

    def _get_pb_avg(self, years: int = 5) -> float:
        """Average P/B ratio over last N years."""
        since = datetime.now() - timedelta(days=years * 365)
        conn  = self._connect()
        row   = conn.execute("""
            SELECT AVG(pb_ratio) FROM fundamentals
            WHERE  fetched_at > ? AND pb_ratio > 0 AND ticker = ?
        """, (str(since), self.ticker)).fetchone()
        conn.close()
        return float(row[0]) if row and row[0] else 3.0  # HDFC historical avg

    def _get_latest_metric(self, metric: str) -> Optional[float]:
        """Get latest value of a specific metric from DB."""
        conn = self._connect()
        row  = conn.execute(f"""
            SELECT {metric} FROM fundamentals
            WHERE  {metric} IS NOT NULL AND {metric} > 0 AND ticker = ?
            ORDER  BY fetched_at DESC LIMIT 1
        """, (self.ticker,)).fetchone()
        conn.close()
        return float(row[0]) if row else None

    # ─────────────────────────────────────────────────────────────
    # STORAGE
    # ─────────────────────────────────────────────────────────────

    def _get_latest(self) -> Optional[Dict]:
        conn = self._connect()
        row  = conn.execute("""
            SELECT * FROM fundamentals WHERE ticker = ? ORDER BY fetched_at DESC LIMIT 1
        """, (self.ticker,)).fetchone()
        conn.close()
        return dict(row) if row else None

    def _get_history(self, quarters: int = 8) -> List[Dict]:
        conn = self._connect()
        rows = conn.execute("""
            SELECT nim, npa_gross, npa_net, casa_ratio,
                   loan_growth, deposit_growth, roe, roa
            FROM   fundamentals
            WHERE  ticker = ?
            ORDER  BY fetched_at DESC LIMIT ?
        """, (self.ticker, quarters)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def _save_fundamentals(self, data: Dict) -> None:
        conn = self._connect()
        conn.execute("""
            INSERT OR REPLACE INTO fundamentals
            (ticker, nim, npa_gross, npa_net, casa_ratio, loan_growth,
             deposit_growth, car, pcr, roe, roa, cost_to_income,
             pb_ratio, pe_ratio, eps_surprise, fetched_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            self.ticker,
            data.get("nim", 0),           data.get("npa_gross", 0),
            data.get("npa_net", 0),       data.get("casa_ratio", 0),
            data.get("loan_growth", 0),   data.get("deposit_growth", 0),
            data.get("car", 0),           data.get("pcr", 0),
            data.get("roe", 0),           data.get("roa", 0),
            data.get("cost_to_income", 0),data.get("pb_ratio", 0),
            data.get("pe_ratio", 0),      data.get("eps_surprise", 0),
            data.get("fetched_at", str(datetime.now())),
        ))
        conn.commit()
        conn.close()

    def _parse_number(self, text: str) -> float:
        text = re.sub(r"[^0-9.\-]", "", str(text))
        try:
            return float(text)
        except ValueError:
            return 0.0

    def _empty_features(self) -> Dict:
        # No DB rows — we used to return sector defaults here labelled GOOD,
        # which let Gate 2 pass even when fundamentals were *entirely* missing.
        # Now we return the same sector defaults BUT label the grade as
        # "DEFAULTED" so:
        #   - Gate 2 Check 5 (`fund_ok = grade not in POOR/UNKNOWN`) still passes
        #     for swing/intraday (don't paralyse trading on a fresh DB)
        #   - But Gate 2 also adds a +1pp conf bump requirement specifically
        #     for `positional` (see gate2_rule_filter.py) since long-hold trades
        #     are most exposed to fundamental surprises
        defaults = {"nim": 3.8, "npa_gross": 1.3, "roe": 16.0, "casa_ratio": 45.0}
        _, score = self._grade_fundamentals(defaults, 0.0, 0.0)
        return {
            "nim": 0, "nim_trend": 0, "npa_gross": 0, "npa_net": 0,
            "npa_trend": 0, "casa_ratio": 0, "casa_trend": 0,
            "loan_growth": 0, "loan_trend": 0, "deposit_growth": 0,
            "car": 0, "pcr": 0, "roe": 0, "roa": 0, "cost_to_income": 0,
            "pb_ratio": 0, "pe_ratio": 0, "pb_vs_5yr_avg": 1.0,
            "eps_surprise": 0, "beat_streak": 0,
            "fundamental_grade": "DEFAULTED",
            "fundamental_score": score,
            "fundamentals_missing": 1,
        }

    def _connect(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    def _setup_db(self) -> None:
        # Safety-net only — `database.db_setup.DatabaseSetup` is the authoritative
        # schema owner. Shape must match (ticker NOT NULL, UNIQUE(ticker, fetched_at)).
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fundamentals (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker          TEXT    NOT NULL DEFAULT 'HDFCBANK.NS',
                nim             REAL, npa_gross      REAL,
                npa_net         REAL, casa_ratio      REAL,
                loan_growth     REAL, deposit_growth  REAL,
                car             REAL, pcr             REAL,
                roe             REAL, roa             REAL,
                cost_to_income  REAL, pb_ratio        REAL,
                pe_ratio        REAL, eps_surprise     REAL,
                fetched_at      TEXT,
                UNIQUE(ticker, fetched_at)
            )
        """)
        conn.commit()
        conn.close()