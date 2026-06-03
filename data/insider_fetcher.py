# data/insider_fetcher.py
# Tracks promoter shareholding changes and block/bulk deals for HDFC Bank
# SEBI mandates disclosure within 2 trading days — this is legal alpha
# Sources: BSE shareholding pattern, NSE bulk/block deals

import requests
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np

from config import get_bank_config

logger = logging.getLogger(__name__)

HDFC_BSE_CODE  = "500180"
HDFC_NSE_SYM   = "HDFCBANK"


class InsiderFetcher:
    """
    Tracks smart money at the company level — promoters and institutions.

    The single most reliable signal in Indian markets:
    When the people who built the company are buying more shares,
    they know something. When they are selling consistently — be careful.

    Features:
    - promoter_holding_pct       : current % stake
    - promoter_qoq_change        : quarter-over-quarter change
    - promoter_trend             : increasing / stable / decreasing
    - block_deal_net_cr          : large block deal value today
    - institutional_holding_pct  : total FII + DII holding %
    - dii_holding_change         : domestic MF accumulation trend

    Data updated quarterly (shareholding) and daily (block/bulk deals).
    """

    NSE_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Referer": "https://www.nseindia.com/",
    }
    BSE_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Referer": "https://www.bseindia.com/",
    }

    def __init__(self, db_path: str = "database/trading.db",
                 ticker: str = "HDFCBANK.NS"):
        self.db_path    = db_path
        self.ticker     = ticker
        cfg             = get_bank_config(ticker)
        self.nse_symbol = cfg.get("nse_symbol", HDFC_NSE_SYM)
        self.bse_code   = cfg.get("bse_code",   HDFC_BSE_CODE)
        self._session   = self._create_session()
        self._setup_db()

    # ─────────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────────

    def fetch_all(self) -> Dict:
        """
        Fetch shareholding pattern + block deals.
        Called weekly (shareholding) and daily (block deals).
        """
        self._fetch_shareholding()
        self._fetch_block_deals()
        return self.get_insider_features()

    def get_insider_features(self) -> Dict:
        """
        Return all insider features formatted for ML model.
        Combines promoter holding trend + block deal signal.
        """
        promoter  = self._get_promoter_trend()
        blocks    = self._get_block_deal_features()
        holdings  = self._get_holding_breakdown()

        # Combined insider conviction score (-1 to +1)
        insider_score = self._calculate_insider_score(promoter, blocks)

        return {
            # Promoter
            "promoter_holding_pct":   promoter.get("current_pct", 0),
            "promoter_qoq_change":    promoter.get("qoq_change", 0),
            "promoter_trend":         promoter.get("trend", "STABLE"),
            "promoter_pledge_pct":    promoter.get("pledge_pct", 0),

            # Institutional
            "fii_holding_pct":        holdings.get("fii_pct", 0),
            "dii_holding_pct":        holdings.get("dii_pct", 0),
            "dii_holding_qoq":        holdings.get("dii_qoq", 0),
            "institutional_total_pct": holdings.get("total_institutional", 0),

            # Block deals
            "block_deal_net_cr":      blocks.get("net_cr", 0),
            "block_deal_is_buying":   blocks.get("is_buying", False),
            "block_deal_detected":    blocks.get("detected", False),
            "block_deal_5d_net_cr":   blocks.get("net_5d_cr", 0),

            # Composite
            "insider_score":          insider_score,
            "insider_signal":         self._score_to_signal(insider_score),
        }

    # ─────────────────────────────────────────────────────────────
    # SHAREHOLDING PATTERN
    # ─────────────────────────────────────────────────────────────

    def _fetch_shareholding(self) -> None:
        """
        Fetch quarterly shareholding pattern from NSE.
        NSE publishes this within 21 days of quarter end.
        """
        try:
            from data.nse_session import nse_get_json
            # NSE requires the index param (omitting it returns 200 "missing index").
            url = (
                f"https://www.nseindia.com/api/corporate-share-holdings-master"
                f"?index=equities&symbol={self.nse_symbol}"
            )
            quote_page = f"https://www.nseindia.com/get-quotes/equity?symbol={self.nse_symbol}"
            data = nse_get_json(self._session, url, referer=quote_page, page_url=quote_page)
            # Endpoint returns a BARE LIST of all historical quarters (newest first);
            # tolerate a {"data": [...]} wrapper too.
            rows = data if isinstance(data, list) else (
                   data.get("data", []) if isinstance(data, dict) else [])
            if not rows:
                return

            # Persist the most recent quarters so the 4-quarter promoter trend is
            # available immediately (not after 4 live refreshes). Each row carries
            # promoter+group % (`pr_and_prgrp`) and public % (`public_val`); FII/DII/
            # pledge live only in the linked XBRL, not this summary — left 0.
            saved = 0
            for row in rows[:8]:
                quarter = self._iso_quarter(row.get("date", ""))
                if not quarter:
                    continue
                self._save_shareholding({
                    "quarter":      quarter,
                    "promoter_pct": self._safe_float(row.get("pr_and_prgrp", 0)),
                    "fii_pct":      0.0,
                    "dii_pct":      0.0,
                    "public_pct":   self._safe_float(row.get("public_val", 0)),
                    "pledge_pct":   0.0,
                    "total_shares": 0.0,
                })
                saved += 1
            if saved:
                logger.info(f"[{self.ticker}] Shareholding: {saved} quarters saved, "
                            f"latest promoter={rows[0].get('pr_and_prgrp')}%")

        except Exception as e:
            logger.error(f"[{self.ticker}] Shareholding fetch failed: {e}")

    def _fetch_block_deals(self) -> None:
        """
        Fetch block deals from NSE — transactions > 5 lakh shares.
        These are large institutional moves that signal conviction.
        """
        try:
            # Block deals (separate from bulk deals — even larger)
            from data.nse_session import nse_get_json
            url = "https://www.nseindia.com/api/block-deal"
            block_page = "https://www.nseindia.com/market-data/block-deal-watch"
            data = nse_get_json(self._session, url, referer=block_page, page_url=block_page)
            # NSE returns either a bare list or {"data": [...]} depending on endpoint
            if isinstance(data, dict):
                data = data.get("data", [])
            if not isinstance(data, list):
                return
            # The block-deal feed is market-wide; keep only THIS bank's deals.
            bank_deals = [
                d for d in data
                if self.nse_symbol in str(d.get("symbol", "")).upper()
            ]

            conn = self._connect()
            for deal in bank_deals:
                try:
                    qty   = self._safe_float(deal.get("quantity", 0))
                    price = self._safe_float(deal.get("price", 0))
                    value_cr = (qty * price) / 1e7

                    conn.execute("""
                        INSERT OR IGNORE INTO insider_block_deals
                        (ticker, trade_date, client_name, buy_sell,
                         quantity, price, value_cr, deal_type, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        self.ticker,
                        deal.get("date", str(datetime.now().date())),
                        deal.get("clientName", "Unknown"),
                        deal.get("buySell", "BUY"),
                        qty, price, value_cr, "BLOCK",
                        str(datetime.now()),
                    ))
                except Exception:
                    pass
            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"[{self.ticker}] Block deals fetch failed: {e}")

    # ─────────────────────────────────────────────────────────────
    # FEATURE BUILDERS
    # ─────────────────────────────────────────────────────────────

    def _get_promoter_trend(self) -> Dict:
        """Calculate promoter holding trend over last 4 quarters."""
        conn = self._connect()
        rows = conn.execute("""
            SELECT quarter, promoter_pct, pledge_pct
            FROM   shareholding_pattern
            WHERE  ticker = ?
            ORDER  BY quarter DESC
            LIMIT  4
        """, (self.ticker,)).fetchall()
        conn.close()

        if not rows:
            return {"current_pct": 0, "qoq_change": 0, "trend": "UNKNOWN",
                    "pledge_pct": 0}

        current     = rows[0][0], rows[0][1], rows[0][2]
        current_pct = current[1]
        pledge_pct  = current[2]
        qoq_change  = round(rows[0][1] - rows[1][1], 3) if len(rows) > 1 else 0

        # Trend over 4 quarters
        if len(rows) >= 3:
            pcts = [r[1] for r in rows]
            if pcts[0] > pcts[1] > pcts[2]:
                trend = "INCREASING"
            elif pcts[0] < pcts[1] < pcts[2]:
                trend = "DECREASING"
            else:
                trend = "STABLE"
        else:
            trend = "STABLE"

        return {
            "current_pct": current_pct,
            "qoq_change":  qoq_change,
            "trend":       trend,
            "pledge_pct":  pledge_pct,
        }

    def _get_block_deal_features(self) -> Dict:
        """Summarise block/bulk deals for today and last 5 days."""
        today    = datetime.now().strftime("%Y-%m-%d")
        five_ago = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")

        conn = self._connect()

        # Today
        today_rows = conn.execute("""
            SELECT buy_sell, value_cr FROM insider_block_deals
            WHERE trade_date = ? AND ticker = ?
        """, (today, self.ticker)).fetchall()

        # Last 5 days
        five_rows = conn.execute("""
            SELECT buy_sell, value_cr FROM insider_block_deals
            WHERE trade_date >= ? AND ticker = ?
        """, (five_ago, self.ticker)).fetchall()

        conn.close()

        def net(rows):
            buy  = sum(r[1] for r in rows if r[0] == "BUY")
            sell = sum(r[1] for r in rows if r[0] == "SELL")
            return round(buy - sell, 2)

        today_net = net(today_rows)
        five_net  = net(five_rows)

        return {
            "net_cr":    today_net,
            "net_5d_cr": five_net,
            "is_buying": today_net > 0,
            "detected":  len(today_rows) > 0,
        }

    def _get_holding_breakdown(self) -> Dict:
        """Latest FII + DII + DII QoQ change from shareholding table."""
        conn = self._connect()
        rows = conn.execute("""
            SELECT fii_pct, dii_pct
            FROM   shareholding_pattern
            WHERE  ticker = ?
            ORDER  BY quarter DESC
            LIMIT  2
        """, (self.ticker,)).fetchall()
        conn.close()

        if not rows:
            return {"fii_pct": 0, "dii_pct": 0,
                    "dii_qoq": 0, "total_institutional": 0}

        fii = rows[0][0]
        dii = rows[0][1]
        dii_qoq = round(rows[0][1] - rows[1][1], 3) if len(rows) > 1 else 0

        return {
            "fii_pct":              fii,
            "dii_pct":              dii,
            "dii_qoq":              dii_qoq,
            "total_institutional":  round(fii + dii, 2),
        }

    def _calculate_insider_score(self, promoter: Dict, blocks: Dict) -> float:
        """
        Composite insider conviction score from -1.0 to +1.0.
        - Promoter increasing = strong positive
        - Promoter decreasing = strong negative
        - High pledge = negative (financial stress)
        - Block buying = moderate positive
        """
        score = 0.0

        # Promoter trend
        if promoter.get("trend") == "INCREASING":
            score += 0.4
        elif promoter.get("trend") == "DECREASING":
            score -= 0.4

        # Promoter QoQ change
        qoq = promoter.get("qoq_change", 0)
        score += min(0.3, max(-0.3, qoq * 10))

        # Pledge ratio — high pledge = risk
        pledge = promoter.get("pledge_pct", 0)
        if pledge > 20:
            score -= 0.3
        elif pledge > 10:
            score -= 0.1

        # Block deals
        net_5d = blocks.get("net_5d_cr", 0)
        if net_5d > 200:
            score += 0.2
        elif net_5d < -200:
            score -= 0.2

        return round(max(-1.0, min(1.0, score)), 3)

    def _score_to_signal(self, score: float) -> str:
        if score >  0.4:  return "STRONG_BULL"
        if score >  0.1:  return "MILD_BULL"
        if score < -0.4:  return "STRONG_BEAR"
        if score < -0.1:  return "MILD_BEAR"
        return "NEUTRAL"

    # ─────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────

    def _safe_float(self, val) -> float:
        try:
            return float(str(val).replace(",", "").replace("%", ""))
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _iso_quarter(date_str: str) -> str:
        """'31-MAR-2026' → '2026-03-31' so `quarter DESC` sorts chronologically."""
        for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d-%m-%Y"):
            try:
                return datetime.strptime(str(date_str).strip(), fmt).strftime("%Y-%m-%d")
            except (ValueError, AttributeError):
                continue
        return ""

    def _save_shareholding(self, data: Dict) -> None:
        conn = self._connect()
        conn.execute("""
            INSERT OR REPLACE INTO shareholding_pattern
            (ticker, quarter, promoter_pct, fii_pct, dii_pct,
             public_pct, pledge_pct, total_shares, created_at)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            self.ticker, data["quarter"], data["promoter_pct"], data["fii_pct"],
            data["dii_pct"], data["public_pct"], data["pledge_pct"],
            data["total_shares"], str(datetime.now()),
        ))
        conn.commit()
        conn.close()

    def _create_session(self) -> requests.Session:
        s = requests.Session()
        try:
            s.get("https://www.nseindia.com", headers=self.NSE_HEADERS, timeout=10)
        except Exception:
            pass
        return s

    def _connect(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    def _setup_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        # Per-bank migration: legacy tables were HDFC-only (no `ticker`, and a
        # UNIQUE on quarter alone — only one bank's row could ever exist). These
        # hold only refetchable public cache (never trades), so when the old
        # shape is detected we drop+recreate rather than ALTER (SQLite can't
        # change a UNIQUE constraint in place). Self-healing on local and VPS.
        for tbl in ("shareholding_pattern", "insider_block_deals"):
            try:
                cols = [r[1] for r in conn.execute(f"PRAGMA table_info({tbl})").fetchall()]
                if cols and "ticker" not in cols:
                    conn.execute(f"DROP TABLE {tbl}")
                    logger.info(f"Migrated {tbl} to per-bank schema (dropped legacy HDFC-only table)")
            except Exception as e:
                logger.warning(f"{tbl} migration check failed: {e}")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS shareholding_pattern (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker        TEXT    NOT NULL,
                quarter       TEXT,
                promoter_pct  REAL,
                fii_pct       REAL,
                dii_pct       REAL,
                public_pct    REAL,
                pledge_pct    REAL,
                total_shares  REAL,
                created_at    TEXT,
                UNIQUE(ticker, quarter)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS insider_block_deals (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker      TEXT    NOT NULL,
                trade_date  TEXT,
                client_name TEXT,
                buy_sell    TEXT,
                quantity    REAL,
                price       REAL,
                value_cr    REAL,
                deal_type   TEXT,
                created_at  TEXT,
                UNIQUE(ticker, trade_date, client_name, buy_sell, quantity)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sh_quarter ON shareholding_pattern(ticker, quarter DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_bd_date ON insider_block_deals(ticker, trade_date DESC)")
        conn.commit()
        conn.close()
        conn.close()