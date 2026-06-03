# data/bse_fetcher.py
# Fetches official BSE/NSE corporate filings and announcements
# This is the MOST reliable news source — official corporate disclosures
# Sources: BSE India website, NSE corporate actions

import requests
import sqlite3
import logging
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

from config import get_bank_config


class BSEFetcher:
    """
    Fetches official BSE/NSE corporate announcements for HDFC Bank.

    Why this matters:
    - Official source — not filtered through media
    - Real-time: filed same day as event
    - LLM analyzes these for major events (earnings, board meetings, etc.)
    - Captures material non-public info disclosed legally

    Runs every 30 minutes during market hours.
    Runs every 2 hours after market hours.
    """

    BSE_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.bseindia.com/",
        "Accept-Language": "en-US,en;q=0.9",
    }

    NSE_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36"
        ),
        "Accept": "application/json",
        "Referer": "https://www.nseindia.com/",
    }

    # Categories that warrant immediate LLM analysis
    HIGH_PRIORITY_CATEGORIES = [
        "Financial Results",
        "Board Meeting",
        "Dividend",
        "Acquisition",
        "Merger/Amalgamation",
        "Rights Issue",
        "Buyback",
        "Change in Management",
        "Regulatory Action",
        "RBI Action",
        "Credit Rating",
        "Analyst/Institutional Investor Meet",
        "Investor Presentation",
    ]

    def __init__(self, db_path: str = "database/trading.db", ticker: str = "HDFCBANK.NS"):
        self.db_path    = db_path
        self.ticker     = ticker
        _cfg            = get_bank_config(ticker)
        self._bse_code  = _cfg["bse_code"]
        self._nse_symbol= _cfg["nse_symbol"]
        self._earnings_dates = _cfg["earnings_dates"]
        self._setup_db()
        self._session   = self._create_session()

    # ─────────────────────────────────────────────────────────────
    # PUBLIC METHODS
    # ─────────────────────────────────────────────────────────────

    def fetch_bse_announcements(self, days_back: int = 2) -> int:
        """
        Fetch recent BSE announcements for HDFC Bank.
        Returns count of new announcements saved.
        """
        try:
            url = (
                f"https://api.bseindia.com/BseIndiaAPI/api/AnnGetData/w"
                f"?scrip_cd={self._bse_code}"
                f"&Category=Company Update"
                f"&subcategory=-1"
            )
            response = self._session.get(url, headers=self.BSE_HEADERS, timeout=10)

            if response.status_code != 200:
                logger.warning(f"BSE API returned {response.status_code}")
                return 0

            data = response.json()
            # BSE intermittently returns a bare JSON string (error body) instead
            # of the expected object — guard so `.get` never hits a str and spams
            # the log with "'str' object has no attribute 'get'" every cycle.
            if not isinstance(data, dict):
                logger.warning(
                    f"BSE returned non-dict payload ({type(data).__name__}) — skipping"
                )
                return 0
            announcements = data.get("Table", [])
            return self._save_announcements(announcements, "bse")

        except Exception as e:
            logger.error(f"BSE announcement fetch failed: {e}")
            return 0

    def fetch_nse_corporate_actions(self) -> int:
        """
        Fetch NSE corporate actions (dividends, splits, bonuses).
        """
        try:
            url = (
                f"https://www.nseindia.com/api/corporates-corporateActions"
                f"?index=equities&symbol={self._nse_symbol}"
            )
            response = self._session.get(url, headers=self.NSE_HEADERS, timeout=10)

            if response.status_code != 200:
                logger.warning(f"NSE corporate actions returned {response.status_code}")
                return 0

            data = response.json()
            return self._save_corporate_actions(data)

        except Exception as e:
            logger.error(f"NSE corporate actions fetch failed: {e}")
            return 0

    def fetch_quarterly_results_calendar(self) -> Optional[datetime]:
        """
        Find the next scheduled quarterly results date.
        Checks BSE board meeting announcements for result dates.
        Returns the next earnings date or None.
        """
        try:
            conn = self._connect()

            # Look for board meeting announcements in DB — for THIS ticker only.
            rows = conn.execute("""
                SELECT title, description, announcement_date
                FROM bse_announcements
                WHERE (ticker = ? OR ticker IS NULL)
                  AND category IN ('Board Meeting', 'Financial Results')
                  AND announcement_date > ?
                ORDER BY announcement_date ASC
                LIMIT 5
            """, (self.ticker, str(datetime.now()))).fetchall()
            conn.close()

            for row in rows:
                text = f"{row[0]} {row[1]}".lower()
                if any(kw in text for kw in ["results", "financial results", "quarterly"]):
                    try:
                        return datetime.strptime(str(row[2])[:10], "%Y-%m-%d")
                    except Exception:
                        pass

            return None

        except Exception as e:
            logger.error(f"Earnings calendar lookup failed: {e}")
            return None

    def get_days_to_earnings(self) -> int:
        """
        Returns days until next earnings — used as ML feature.
        Returns 99 if no upcoming date found (safe default = no restriction).
        """
        next_date = self.fetch_quarterly_results_calendar()
        if next_date is None:
            return 99
        delta = (next_date - datetime.now()).days
        return max(0, delta)

    def get_recent_announcements(self, hours: int = 48) -> List[Dict]:
        """
        Get recent announcements for LLM analysis and Telegram alerts — for THIS ticker.
        """
        since = datetime.now() - timedelta(hours=hours)
        conn = self._connect()
        rows = conn.execute("""
            SELECT id, title, description, category,
                   announcement_date, is_high_priority, llm_analyzed
            FROM bse_announcements
            WHERE (ticker = ? OR ticker IS NULL)
              AND created_at > ?
            ORDER BY announcement_date DESC
        """, (self.ticker, str(since))).fetchall()
        conn.close()

        return [
            {
                "id":               row[0],
                "title":            row[1],
                "description":      row[2],
                "category":         row[3],
                "announcement_date": row[4],
                "is_high_priority": row[5],
                "llm_analyzed":     row[6],
            }
            for row in rows
        ]

    def sync_earnings_calendar(self) -> int:
        """
        Upsert upcoming HDFC Bank earnings dates into the earnings_calendar table.

        Two sources (in order of priority):
        1. BSE announcements DB — scans Board Meeting / Financial Results for result dates.
        2. Hardcoded HDFC_EARNINGS_2026_2027 list — seeds if table has < 4 rows.

        Returns count of rows upserted.
        """
        try:
            conn = self._connect()
            self._ensure_earnings_calendar(conn)
            upserted = 0

            # ── Source 1: BSE announcements ───────────────────────────
            try:
                rows = conn.execute("""
                    SELECT title, description, announcement_date
                    FROM bse_announcements
                    WHERE (ticker = ? OR ticker IS NULL)
                      AND category IN ('Board Meeting', 'Financial Results')
                      AND announcement_date > ?
                    ORDER BY announcement_date ASC
                    LIMIT 20
                """, (self.ticker, str(datetime.now().date()))).fetchall()

                date_pattern = re.compile(
                    r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b'
                )

                for row in rows:
                    text = f"{row[0] or ''} {row[1] or ''}".lower()
                    if not any(kw in text for kw in ["result", "financial results", "quarterly"]):
                        continue

                    # Try to extract a date from title/description
                    matches = date_pattern.findall(f"{row[0] or ''} {row[1] or ''}")
                    parsed_date = None
                    for m in matches:
                        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y",
                                    "%d/%m/%y", "%d-%m-%y"):
                            try:
                                parsed_date = datetime.strptime(m, fmt).strftime("%Y-%m-%d")
                                break
                            except ValueError:
                                continue
                        if parsed_date:
                            break

                    # Fall back to announcement_date if no date found in text
                    if not parsed_date and row[2]:
                        try:
                            parsed_date = str(row[2])[:10]
                            datetime.strptime(parsed_date, "%Y-%m-%d")  # validate
                        except ValueError:
                            parsed_date = None

                    if parsed_date:
                        try:
                            conn.execute("""
                                INSERT OR IGNORE INTO earnings_calendar
                                (ticker, result_date, source, confirmed, created_at)
                                VALUES (?, ?, 'bse', 1, ?)
                            """, (self.ticker, parsed_date, str(datetime.now())))
                            if conn.execute("SELECT changes()").fetchone()[0] > 0:
                                upserted += 1
                                logger.info(f"earnings_calendar: BSE date added {parsed_date}")
                        except Exception as e:
                            logger.warning(f"earnings_calendar BSE insert failed: {e}")

            except Exception as e:
                logger.warning(f"BSE earnings scan failed: {e}")

            # ── Source 2: Hardcoded dates if table is sparse ──────────
            existing_count = conn.execute(
                "SELECT COUNT(*) FROM earnings_calendar WHERE ticker = ?", (self.ticker,)
            ).fetchone()[0]

            if existing_count < 4:
                for date_str in self._earnings_dates:
                    try:
                        conn.execute("""
                            INSERT OR IGNORE INTO earnings_calendar
                            (ticker, result_date, source, confirmed, created_at)
                            VALUES (?, ?, 'hardcoded', 0, ?)
                        """, (self.ticker, date_str, str(datetime.now())))
                        if conn.execute("SELECT changes()").fetchone()[0] > 0:
                            upserted += 1
                    except Exception as e:
                        logger.warning(f"earnings_calendar hardcoded insert failed for {date_str}: {e}")

            conn.commit()
            conn.close()
            logger.info(f"sync_earnings_calendar: {upserted} rows upserted")
            return upserted

        except Exception as e:
            logger.error(f"sync_earnings_calendar failed: {e}")
            return 0

    def get_next_earnings_date(self) -> Optional[str]:
        """
        Returns the next upcoming HDFCBANK.NS earnings date as ISO string (YYYY-MM-DD),
        or None if no future date is found in earnings_calendar.
        """
        try:
            conn = self._connect()
            self._ensure_earnings_calendar(conn)
            today = str(datetime.now().date())
            row = conn.execute("""
                SELECT result_date FROM earnings_calendar
                WHERE ticker = ?
                AND result_date >= ?
                ORDER BY result_date ASC
                LIMIT 1
            """, (self.ticker, today)).fetchone()
            conn.close()
            return str(row[0]) if row else None
        except Exception as e:
            logger.error(f"get_next_earnings_date failed: {e}")
            return None

    def get_high_priority_unanalyzed(self) -> List[Dict]:
        """
        Get high-priority announcements not yet analyzed by LLM — for THIS ticker.
        Legacy rows with ticker IS NULL are also returned for one-time cleanup.
        These trigger immediate LLM analysis regardless of schedule.
        """
        conn = self._connect()
        rows = conn.execute("""
            SELECT id, title, description, category
            FROM bse_announcements
            WHERE (ticker = ? OR ticker IS NULL)
              AND is_high_priority = 1
              AND llm_analyzed = 0
            ORDER BY created_at DESC
            LIMIT 5
        """, (self.ticker,)).fetchall()
        conn.close()

        return [
            {"id": r[0], "title": r[1], "description": r[2], "category": r[3]}
            for r in rows
        ]

    def mark_llm_analyzed(self, announcement_id: int, impact_score: float) -> None:
        """Mark announcement as analyzed by LLM with its impact score."""
        conn = self._connect()
        conn.execute("""
            UPDATE bse_announcements
            SET llm_analyzed = 1, llm_impact_score = ?
            WHERE id = ?
        """, (impact_score, announcement_id))
        conn.commit()
        conn.close()

    # ─────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────────────────────────

    def _save_announcements(self, announcements: list, source: str) -> int:
        """Parse and save BSE announcements to database."""
        conn = self._connect()
        new_count = 0

        for ann in announcements:
            try:
                title = ann.get("HEADLINE", "") or ann.get("NEWSSUB", "")
                description = ann.get("NEWSBODY", "") or ann.get("ATTACHMENTNAME", "")
                category = ann.get("CATEGORYNAME", "General")
                ann_date = ann.get("NEWS_DT", str(datetime.now()))

                if not title:
                    continue

                is_high_priority = any(
                    cat.lower() in title.lower() or cat.lower() in category.lower()
                    for cat in self.HIGH_PRIORITY_CATEGORIES
                )

                conn.execute("""
                    INSERT OR IGNORE INTO bse_announcements
                    (ticker, title, description, category, source,
                     announcement_date, is_high_priority,
                     llm_analyzed, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)
                """, (
                    self.ticker,
                    title.strip()[:500],
                    str(description)[:2000],
                    category,
                    source,
                    ann_date,
                    int(is_high_priority),
                    str(datetime.now()),
                ))

                if conn.execute("SELECT changes()").fetchone()[0] > 0:
                    new_count += 1
                    if is_high_priority:
                        logger.info(f"[{self.ticker}] [BSE HIGH PRIORITY] {title[:80]}")

            except Exception as e:
                logger.warning(f"Failed to save announcement: {e}")

        conn.commit()
        conn.close()
        return new_count

    def _save_corporate_actions(self, data: list) -> int:
        """Save NSE corporate actions to database."""
        conn = self._connect()
        new_count = 0

        for action in data:
            try:
                title = f"{action.get('subject', '')} — {action.get('series', '')}"
                description = str(action)
                category = action.get("subject", "Corporate Action")
                ex_date = action.get("exDate", str(datetime.now()))

                conn.execute("""
                    INSERT OR IGNORE INTO bse_announcements
                    (ticker, title, description, category, source,
                     announcement_date, is_high_priority,
                     llm_analyzed, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)
                """, (
                    self.ticker,
                    title[:500],
                    description[:2000],
                    category,
                    "nse_corporate_actions",
                    ex_date,
                    1,  # corporate actions are always high priority
                    str(datetime.now()),
                ))

                if conn.execute("SELECT changes()").fetchone()[0] > 0:
                    new_count += 1

            except Exception as e:
                logger.warning(f"Failed to save corporate action: {e}")

        conn.commit()
        conn.close()
        return new_count

    def _create_session(self) -> requests.Session:
        """
        Create a persistent session that pre-loads NSE/BSE cookies.
        NSE requires a valid session cookie from the homepage.
        """
        session = requests.Session()
        try:
            session.get(
                "https://www.nseindia.com",
                headers=self.NSE_HEADERS,
                timeout=10,
            )
        except Exception:
            pass  # Cookie prefetch may fail, that is fine
        return session

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_earnings_calendar(self, conn: sqlite3.Connection) -> None:
        """
        Create earnings_calendar table if it doesn't exist (idempotent).
        Schema matches database/db_setup.py — UNIQUE(ticker, result_date) so
        all 5 banks share the table without colliding.
        """
        conn.execute("""
            CREATE TABLE IF NOT EXISTS earnings_calendar (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker      TEXT    NOT NULL DEFAULT 'HDFCBANK.NS',
                result_date TEXT    NOT NULL,
                source      TEXT    DEFAULT 'hardcoded',
                confirmed   INTEGER DEFAULT 0,
                created_at  TEXT,
                UNIQUE(ticker, result_date)
            )
        """)
        conn.commit()

    def _setup_db(self) -> None:
        """
        Create bse_announcements table if it does not exist.
        Schema matches database/db_setup.py — composite UNIQUE(ticker, title,
        announcement_date) so two banks can file announcements with the same
        title (e.g. "Board Meeting Outcome") without one clobbering the other.
        """
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bse_announcements (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker            TEXT    NOT NULL DEFAULT 'HDFCBANK.NS',
                title             TEXT,
                description       TEXT,
                category          TEXT,
                source            TEXT,
                announcement_date TEXT,
                is_high_priority  INTEGER DEFAULT 0,
                llm_analyzed      INTEGER DEFAULT 0,
                llm_impact_score  REAL    DEFAULT 0.0,
                created_at        TEXT,
                UNIQUE(ticker, title, announcement_date)
            )
        """)
        # Best-effort migration: add ticker column if the table predates this.
        cols = {row[1] for row in conn.execute("PRAGMA table_info(bse_announcements)")}
        if "ticker" not in cols:
            conn.execute(
                "ALTER TABLE bse_announcements ADD COLUMN "
                "ticker TEXT NOT NULL DEFAULT 'HDFCBANK.NS'"
            )
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_bse_created
            ON bse_announcements (created_at DESC)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_bse_priority
            ON bse_announcements (is_high_priority, llm_analyzed)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_bse_ticker_created
            ON bse_announcements (ticker, created_at DESC)
        """)
        conn.commit()
        conn.close()