# data/alternative_fetcher.py
# Google Trends search demand as leading fundamental indicator
# Rising searches for HDFC products = business growing 2-3 months ahead
# Source: pytrends (Google Trends unofficial API — free)

import sqlite3
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List

logger = logging.getLogger(__name__)

try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False
    logger.warning("pytrends not installed. pip install pytrends")


class AlternativeFetcher:
    """
    Fetches Google Trends data as leading fundamental indicators.

    Why this works:
    People search for HDFC home loan → they apply → loan disbursed in 30-60 days
    → loan book grows → NIM improves → quarterly results beat estimates
    → stock re-rates upward

    Search interest is a genuine 2-3 month leading indicator of loan growth.
    Available free via Google Trends. No API key needed.

    Keywords tracked:
    - HDFC home loan          → home loan demand (largest segment)
    - HDFC personal loan      → retail credit demand
    - HDFC credit card apply  → card acquisition momentum
    - HDFC bank account open  → CASA growth proxy
    - HDFC bank careers       → hiring = expansion proxy

    Runs weekly (Google Trends is weekly granularity for 3-month data).
    """

    PRODUCT_KEYWORDS = [
        "HDFC home loan",
        "HDFC personal loan",
        "HDFC credit card apply",
        "HDFC bank account open",
    ]

    HIRING_KEYWORDS = [
        "HDFC bank careers",
        "HDFC bank jobs",
    ]

    # Separate from product — fear/concern searches (bearish signals)
    CONCERN_KEYWORDS = [
        "HDFC bank complaint",
        "HDFC bank fraud",
    ]

    GEO = "IN"  # India

    def __init__(self, db_path: str = "database/trading.db"):
        self.db_path = db_path
        self._setup_db()

    # ─────────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────────

    def fetch_all(self) -> Dict:
        """
        Fetch Google Trends for all keyword groups.
        Called weekly by scheduler (Sunday midnight).
        Returns alternative data features.
        """
        if not PYTRENDS_AVAILABLE:
            logger.warning("pytrends unavailable — returning cached data")
            return self.get_alternative_features()

        self._fetch_keyword_group("product",  self.PRODUCT_KEYWORDS)
        time.sleep(3)  # avoid rate limiting
        self._fetch_keyword_group("hiring",   self.HIRING_KEYWORDS)
        time.sleep(3)
        self._fetch_keyword_group("concern",  self.CONCERN_KEYWORDS)

        features = self.get_alternative_features()
        logger.info(
            f"Alt data: demand={features.get('demand_trend_score',0):.2f} | "
            f"hiring={features.get('hiring_trend','')} | "
            f"score={features.get('alt_data_score',0):.2f}"
        )
        return features

    def get_alternative_features(self) -> Dict:
        """
        Return all alternative data features for ML model.
        Uses latest data from DB — no API call.
        """
        product_trend  = self._get_trend_score("product")
        hiring_trend   = self._get_trend_score("hiring")
        concern_trend  = self._get_trend_score("concern")

        # Demand score: rising product searches = positive
        demand_score = product_trend.get("avg_slope", 0)

        # Hiring score: rising job searches = expansion
        hiring_score = hiring_trend.get("avg_slope", 0)

        # Concern score: rising complaints = negative (inverted)
        concern_score = -concern_trend.get("avg_slope", 0)

        # Combined alt data score (-1 to +1)
        alt_score = round(
            demand_score  * 0.5 +
            hiring_score  * 0.3 +
            concern_score * 0.2,
            3
        )
        alt_score = max(-1.0, min(1.0, alt_score))

        return {
            # Product demand (leading loan growth indicator)
            "demand_trend_score":     round(demand_score, 3),
            "home_loan_trend":        product_trend.get("home_loan_slope", 0),
            "personal_loan_trend":    product_trend.get("personal_loan_slope", 0),
            "credit_card_trend":      product_trend.get("credit_card_slope", 0),

            # Hiring (business expansion proxy)
            "hiring_trend_score":     round(hiring_score, 3),
            "hiring_trend":           self._slope_to_label(hiring_score),

            # Concern (risk indicator, inverted)
            "concern_trend_score":    round(concern_score, 3),

            # Composite
            "alt_data_score":         alt_score,
            "alt_data_signal":        self._score_to_signal(alt_score),

            # Metadata
            "last_updated":           self._get_last_update(),
        }

    # ─────────────────────────────────────────────────────────────
    # TRENDS FETCHING
    # ─────────────────────────────────────────────────────────────

    def _fetch_keyword_group(self, group_name: str, keywords: List[str]) -> None:
        """
        Fetch Google Trends for a group of keywords.
        pytrends allows max 5 keywords per request.
        Weekly data over 3 months gives ~13 data points per keyword.
        """
        try:
            pytrends = TrendReq(hl="en-IN", tz=330, timeout=(10, 25))

            # pytrends supports max 5 keywords at once
            batch = keywords[:5]
            pytrends.build_payload(
                batch,
                cat=0,
                timeframe="today 3-m",
                geo=self.GEO,
            )

            interest_df = pytrends.interest_over_time()
            if interest_df.empty:
                logger.warning(f"No trends data for group: {group_name}")
                return

            conn = self._connect()
            for keyword in batch:
                if keyword not in interest_df.columns:
                    continue

                series = interest_df[keyword].dropna()
                if series.empty:
                    continue

                # Calculate 4-week slope (recent trend direction)
                recent  = float(series.iloc[-4:].mean())
                older   = float(series.iloc[-8:-4].mean()) if len(series) >= 8 else recent
                slope   = round((recent - older) / max(older, 1), 4)

                conn.execute("""
                    INSERT INTO alt_trends
                    (keyword, group_name, recent_avg, older_avg,
                     slope, latest_value, fetched_at)
                    VALUES (?,?,?,?,?,?,?)
                """, (
                    keyword, group_name,
                    round(recent, 2), round(older, 2),
                    slope, float(series.iloc[-1]),
                    str(datetime.now()),
                ))

            conn.commit()
            conn.close()
            logger.info(f"Trends fetched for group: {group_name} ({len(batch)} keywords)")

        except Exception as e:
            logger.error(f"Trends fetch failed for {group_name}: {e}")

    # ─────────────────────────────────────────────────────────────
    # FEATURE BUILDERS
    # ─────────────────────────────────────────────────────────────

    def _get_trend_score(self, group_name: str) -> Dict:
        """Get latest trend data for a keyword group from DB."""
        week_ago = datetime.now() - timedelta(days=8)
        conn = self._connect()

        rows = conn.execute("""
            SELECT keyword, slope, recent_avg, latest_value
            FROM   alt_trends
            WHERE  group_name = ?
            AND    fetched_at > ?
            ORDER  BY fetched_at DESC
        """, (group_name, str(week_ago))).fetchall()
        conn.close()

        if not rows:
            return {"avg_slope": 0.0}

        slopes = [r[1] for r in rows]
        avg_slope = float(sum(slopes) / len(slopes)) if slopes else 0.0

        result = {"avg_slope": round(avg_slope, 4)}

        # Individual keyword slopes for detailed features
        for row in rows:
            keyword = row[0].lower().replace(" ", "_").replace("hdfc_", "")
            result[f"{keyword}_slope"] = row[1]

        return result

    def _get_last_update(self) -> str:
        conn = self._connect()
        row = conn.execute("""
            SELECT fetched_at FROM alt_trends ORDER BY fetched_at DESC LIMIT 1
        """).fetchone()
        conn.close()
        return row[0] if row else "Never"

    def _slope_to_label(self, slope: float) -> str:
        if slope >  0.15: return "STRONG_EXPANSION"
        if slope >  0.05: return "EXPANSION"
        if slope < -0.15: return "STRONG_CONTRACTION"
        if slope < -0.05: return "CONTRACTION"
        return "STABLE"

    def _score_to_signal(self, score: float) -> str:
        if score >  0.3:  return "BULLISH_FUNDAMENTAL"
        if score >  0.1:  return "MILD_POSITIVE"
        if score < -0.3:  return "BEARISH_FUNDAMENTAL"
        if score < -0.1:  return "MILD_NEGATIVE"
        return "NEUTRAL"

    # ─────────────────────────────────────────────────────────────
    # STORAGE
    # ─────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    def _setup_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alt_trends (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                keyword       TEXT,
                group_name    TEXT,
                recent_avg    REAL,
                older_avg     REAL,
                slope         REAL,
                latest_value  REAL,
                fetched_at    TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_alt_fetched
            ON alt_trends (fetched_at DESC)
        """)
        conn.commit()
        conn.close()