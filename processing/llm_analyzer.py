# processing/llm_analyzer.py
# Claude API used ONLY for text events — earnings and major announcements
# NOT for signal validation (ML does that)
# LLM generates numerical SCORES that become ML input features
# Cost: ~5 API calls/month = ~₹80

import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

logger = logging.getLogger(__name__)

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("anthropic not installed. pip install anthropic")


class LLMAnalyzer:
    """
    Uses Claude to extract structured scores from unstructured financial text.

    Design principle (20yr trader perspective):
    LLM's job is ONLY to read text and return numbers.
    ML model's job is to make the actual trading decision.
    Never let LLM decide the trade — it has no statistical edge there.

    Called for:
    1. Quarterly earnings full-text analysis (4× per year)
    2. High-priority BSE announcements (≈5/month)

    NOT called for:
    - Routine news articles (FinBERT handles those)
    - Signal validation (ML confidence threshold handles that)
    - Every 15-minute check (too expensive, no edge)

    Output scores feed into ML model as features:
    - llm_earnings_score     : -1 to +1 (quarterly, cached 90 days)
    - llm_announcement_score : -1 to +1 (per announcement, cached 7 days)
    - llm_combined_score     : weighted combination
    """

    MODEL = "claude-sonnet-4-6"

    def __init__(
        self,
        db_path: str = "database/trading.db",
        api_key: Optional[str] = None,
    ):
        self.db_path = db_path
        self._client = None
        self._api_key = api_key
        self._setup_db()

        # In-memory cache to avoid repeated API calls
        self._earnings_cache: Optional[Dict] = None
        self._earnings_cache_date: Optional[datetime] = None

    # ─────────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────────

    def analyze_earnings(self, earnings_text: str) -> Dict:
        """
        Full quarterly earnings analysis.
        Called once when results are published — 4× per year.
        Result cached for 90 days until next quarter.

        Returns structured dict with numerical scores for ML.
        """
        client = self._get_client()
        if client is None:
            return self._empty_earnings()

        prompt = f"""You are a senior Indian banking sector analyst.
Analyze this HDFC Bank quarterly result and extract key metrics.
Return ONLY valid JSON — no markdown, no explanation.

{earnings_text[:4000]}

Return this exact JSON structure:
{{
    "overall_score": <-100 to 100, 100=massive beat>,
    "revenue_surprise": <-100 to 100>,
    "profit_surprise": <-100 to 100>,
    "nim_score": <-100 to 100, 100=strong expansion>,
    "npa_score": <-100 to 100, 100=strong improvement>,
    "loan_growth_score": <-100 to 100>,
    "management_tone": <"optimistic"|"neutral"|"cautious">,
    "guidance_score": <-100 to 100>,
    "beat_estimates": <true|false>,
    "key_positive": "<one line>",
    "key_risk": "<one line>",
    "reasoning": "<two sentences max>"
}}"""

        try:
            response = client.messages.create(
                model=self.MODEL,
                max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
            )
            raw  = response.content[0].text.strip()
            data = json.loads(raw)

            # Normalise all scores to -1 to +1
            normalised = {
                k: round(v / 100.0, 4) if isinstance(v, (int, float)) else v
                for k, v in data.items()
            }

            # Save to DB for persistence across restarts
            self._save_earnings_analysis(normalised)

            # Update in-memory cache
            self._earnings_cache      = normalised
            self._earnings_cache_date = datetime.now()

            logger.info(
                f"Earnings analysis: score={normalised.get('overall_score',0):.2f} | "
                f"beat={normalised.get('beat_estimates',False)}"
            )
            return normalised

        except json.JSONDecodeError as e:
            logger.error(f"LLM earnings JSON parse error: {e}")
            return self._empty_earnings()
        except Exception as e:
            logger.error(f"LLM earnings analysis failed: {e}")
            return self._empty_earnings()

    def analyze_bse_announcement(self, title: str, description: str) -> Dict:
        """
        Analyse a high-priority BSE announcement.
        Called for board meetings, dividends, acquisitions, regulatory actions.
        Returns impact score and expected timeframe.
        """
        client = self._get_client()
        if client is None:
            return self._empty_announcement()

        prompt = f"""You are an expert Indian stock market analyst.
HDFC Bank BSE Announcement:
Title: {title}
Details: {description[:1000]}

Return ONLY valid JSON:
{{
    "impact_score": <-100 to 100, 100=very positive for stock>,
    "confidence": <0 to 100>,
    "timeframe": <"immediate"|"this_week"|"this_month">,
    "magnitude": <"large"|"medium"|"small">,
    "reason": "<one sentence>"
}}"""

        try:
            response = client.messages.create(
                model=self.MODEL,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            data  = json.loads(response.content[0].text.strip())
            score = round(data.get("impact_score", 0) / 100.0, 4)

            result = {
                "impact_score":      score,
                "confidence":        data.get("confidence", 50) / 100.0,
                "timeframe":         data.get("timeframe", "this_week"),
                "magnitude":         data.get("magnitude", "small"),
                "reason":            data.get("reason", ""),
                "analyzed_at":       str(datetime.now()),
            }

            self._save_announcement_score(title, result)
            logger.info(
                f"Announcement: '{title[:60]}' → score={score:.2f}"
            )
            return result

        except Exception as e:
            logger.error(f"LLM announcement analysis failed: {e}")
            return self._empty_announcement()

    def get_llm_features_for_ml(self) -> Dict:
        """
        Returns the LLM-generated features ready for ML model input.
        Uses cached results — no API call made here.
        """
        earnings     = self._get_cached_earnings()
        announcement = self._get_latest_announcement_score()

        # Combined score weighted: earnings 70%, latest announcement 30%
        earnings_score     = earnings.get("overall_score", 0.0)
        announcement_score = announcement.get("impact_score", 0.0)
        combined           = round(earnings_score * 0.7 + announcement_score * 0.3, 4)

        # How fresh is the earnings data?
        days_since = self._days_since_earnings()

        return {
            "llm_earnings_score":      earnings_score,
            "llm_earnings_nim":        earnings.get("nim_score", 0.0),
            "llm_earnings_npa":        earnings.get("npa_score", 0.0),
            "llm_earnings_guidance":   earnings.get("guidance_score", 0.0),
            "llm_beat_estimates":      float(earnings.get("beat_estimates", False)),
            "llm_announcement_score":  announcement_score,
            "llm_combined_score":      combined,
            "llm_earnings_days_ago":   days_since,
            # Decay factor: earnings signal weakens over 90 days
            "llm_earnings_relevance":  max(0.0, 1.0 - days_since / 90.0),
        }

    # ─────────────────────────────────────────────────────────────
    # PRIVATE
    # ─────────────────────────────────────────────────────────────

    def _get_cached_earnings(self) -> Dict:
        """Return in-memory cache or load from DB."""
        # In-memory cache valid for current process
        if self._earnings_cache is not None:
            return self._earnings_cache

        # Load from DB
        conn = self._connect()
        row  = conn.execute("""
            SELECT earnings_data FROM llm_earnings_cache
            ORDER  BY analyzed_at DESC LIMIT 1
        """).fetchone()
        conn.close()

        if row:
            try:
                self._earnings_cache = json.loads(row[0])
                return self._earnings_cache
            except Exception:
                pass

        return self._empty_earnings()

    def _get_latest_announcement_score(self) -> Dict:
        """Get most recent announcement score from last 7 days."""
        week_ago = datetime.now() - timedelta(days=7)
        conn     = self._connect()
        row      = conn.execute("""
            SELECT impact_score, confidence, reason
            FROM   llm_announcement_scores
            WHERE  analyzed_at > ?
            ORDER  BY analyzed_at DESC LIMIT 1
        """, (str(week_ago),)).fetchone()
        conn.close()

        if not row:
            return self._empty_announcement()
        return {
            "impact_score": row[0],
            "confidence":   row[1],
            "reason":       row[2],
        }

    def _days_since_earnings(self) -> float:
        """How many days since last earnings analysis."""
        conn = self._connect()
        row  = conn.execute("""
            SELECT analyzed_at FROM llm_earnings_cache
            ORDER  BY analyzed_at DESC LIMIT 1
        """).fetchone()
        conn.close()

        if not row:
            return 90.0
        try:
            analyzed = datetime.strptime(str(row[0])[:19], "%Y-%m-%d %H:%M:%S")
            return float((datetime.now() - analyzed).days)
        except Exception:
            return 90.0

    def _save_earnings_analysis(self, data: Dict) -> None:
        conn = self._connect()
        conn.execute("""
            INSERT INTO llm_earnings_cache (earnings_data, analyzed_at)
            VALUES (?, ?)
        """, (json.dumps(data), str(datetime.now())))
        conn.commit()
        conn.close()

    def _save_announcement_score(self, title: str, data: Dict) -> None:
        conn = self._connect()
        conn.execute("""
            INSERT INTO llm_announcement_scores
            (title, impact_score, confidence, timeframe, reason, analyzed_at)
            VALUES (?,?,?,?,?,?)
        """, (
            title[:300],
            data.get("impact_score", 0),
            data.get("confidence", 0),
            data.get("timeframe", ""),
            data.get("reason", ""),
            data.get("analyzed_at", str(datetime.now())),
        ))
        conn.commit()
        conn.close()

    def _get_client(self):
        if self._client:
            return self._client
        if not ANTHROPIC_AVAILABLE:
            logger.error("anthropic package not installed")
            return None
        try:
            kwargs = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = anthropic.Anthropic(**kwargs)
            return self._client
        except Exception as e:
            logger.error(f"Anthropic client init failed: {e}")
            return None

    def _empty_earnings(self) -> Dict:
        return {
            "overall_score": 0.0, "nim_score": 0.0, "npa_score": 0.0,
            "revenue_surprise": 0.0, "profit_surprise": 0.0,
            "guidance_score": 0.0, "beat_estimates": False,
            "management_tone": "neutral", "key_positive": "",
            "key_risk": "", "reasoning": "",
        }

    def _empty_announcement(self) -> Dict:
        return {
            "impact_score": 0.0, "confidence": 0.5,
            "timeframe": "this_week", "magnitude": "small", "reason": "",
        }

    def _connect(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    def _setup_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS llm_earnings_cache (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                earnings_data TEXT,
                analyzed_at  TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS llm_announcement_scores (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                title        TEXT,
                impact_score REAL,
                confidence   REAL,
                timeframe    TEXT,
                reason       TEXT,
                analyzed_at  TEXT
            )
        """)
        conn.commit()
        conn.close()