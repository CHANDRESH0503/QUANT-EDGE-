# data/fii_fetcher.py
# Fetches FII (Foreign Institutional Investor) and DII (Domestic Institutional)
# daily equity flow data from NSE India
# FII flows are the single biggest driver of large-cap Indian stocks

import os
import requests
import sqlite3
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class FIIFetcher:
    """
    Tracks institutional money flow into Indian equities.

    Features generated:
    - fii_net_daily       : net FII buying/selling today (₹ Cr)
    - fii_5d_rolling      : 5-day cumulative FII flow
    - fii_trend           : direction of last 5 days
    - dii_net_daily       : DII net buying/selling
    - combined_flow       : FII + DII combined signal
    - bulk_deals          : large single transactions for HDFC Bank

    Why this matters:
    FIIs control enough AUM to move large-cap stocks for multiple days.
    Consistent buying for 3+ days = tailwind. Selling = headwind.
    Published daily by NSE — free, reliable, official.
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
    }

    def __init__(self, db_path: str = "database/trading.db"):
        self.db_path = db_path
        self._session = self._create_session()
        self._setup_db()

    # ─────────────────────────────────────────────────────────────
    # PUBLIC METHODS
    # ─────────────────────────────────────────────────────────────

    def fetch_fii_dii_data(self) -> Dict:
        """
        Fetch today's FII/DII equity flow data.
        Returns structured dict with net values and signals.
        Called daily after market close (3:45 PM IST).
        """
        try:
            conn = sqlite3.connect(self.db_path)
            row  = conn.execute("SELECT MAX(fetched_at) FROM fii_data").fetchone()
            conn.close()
            if row and row[0]:
                last = datetime.fromisoformat(str(row[0])[:19])
                if (datetime.now() - last).total_seconds() < 1800:
                    logger.debug("FII data fresh (<30 min) — skipping fetch")
                    return self._get_latest_from_db()
        except Exception:
            pass
        try:
            url = "https://www.nseindia.com/api/fiidiiTradeReact"
            response = self._session.get(
                url, headers=self.NSE_HEADERS, timeout=10
            )

            if response.status_code != 200:
                logger.warning(f"FII/DII API returned {response.status_code}")
                return self._get_latest_from_db()

            data = response.json()
            result = self._parse_fii_dii(data)
            self._save_fii_data(result)

            logger.info(
                f"FII Net: ₹{result['fii_net_cr']:.0f} Cr | "
                f"DII Net: ₹{result['dii_net_cr']:.0f} Cr"
            )
            return result

        except Exception as e:
            logger.error(f"FII/DII fetch failed: {e}")
            return self._get_latest_from_db()

    def fetch_bulk_deals(self) -> List[Dict]:
        """
        Fetch bulk deals from NSE — large institutional transactions.
        A bulk deal > ₹50 Cr in HDFC Bank = high-conviction signal.
        Published daily by NSE.
        """
        try:
            url = "https://www.nseindia.com/api/bulk-deals"
            response = self._session.get(
                url, headers=self.NSE_HEADERS, timeout=10
            )

            if response.status_code != 200:
                return []

            data = response.json()
            hdfc_deals = [
                d for d in data
                if "HDFCBANK" in str(d.get("symbol", "")).upper()
            ]

            self._save_bulk_deals(hdfc_deals)
            return hdfc_deals

        except Exception as e:
            logger.error(f"Bulk deals fetch failed: {e}")
            return []

    def get_flow_features(self) -> Dict:
        """
        Get all FII/DII features formatted for ML model input.
        Combines multiple days to create trend features.
        """
        conn = self._connect()

        # Last 20 days of FII data
        rows = conn.execute("""
            SELECT trade_date, fii_net_cr, dii_net_cr
            FROM fii_data
            ORDER BY trade_date DESC
            LIMIT 20
        """).fetchall()
        conn.close()

        if not rows:
            return self._empty_flow_features()

        fii_values = [r[1] for r in rows]
        dii_values = [r[2] or 0.0 for r in rows]

        # Rolling sums
        fii_1d = fii_values[0]
        fii_3d = sum(fii_values[:3])
        fii_5d = sum(fii_values[:5])
        fii_10d = sum(fii_values[:10])

        dii_1d = dii_values[0]
        dii_3d = sum(dii_values[:3])

        # Trend direction
        fii_trend = self._calculate_trend(fii_values[:5])
        dii_trend = self._calculate_trend(dii_values[:5])

        # Confluence — both buying or both selling
        confluence = "BOTH_BUYING" if fii_5d > 500 and dii_3d > 0 else \
                     "BOTH_SELLING" if fii_5d < -500 and dii_3d < 0 else \
                     "MIXED"

        # Bulk deals today
        bulk_summary = self._get_bulk_deal_summary()

        return {
            # Raw values
            "fii_net_1d_cr":    round(fii_1d, 2),
            "fii_net_3d_cr":    round(fii_3d, 2),
            "fii_net_5d_cr":    round(fii_5d, 2),
            "fii_net_10d_cr":   round(fii_10d, 2),
            "dii_net_1d_cr":    round(dii_1d, 2),
            "dii_net_3d_cr":    round(dii_3d, 2),

            # Normalised signals (-1 to +1)
            "fii_signal":       self._normalise_flow(fii_5d),
            "dii_signal":       self._normalise_flow(dii_3d),

            # Direction
            "fii_trend":        fii_trend,
            "dii_trend":        dii_trend,
            "flow_confluence":  confluence,

            # Bulk deals
            "bulk_net_cr":          bulk_summary["net_cr"],
            "bulk_is_buying":       bulk_summary["is_buying"],
            "bulk_deal_detected":   bulk_summary["detected"],
        }

    def get_flow_features_at(self, as_of_date: str) -> Dict:
        """
        Point-in-time flow features for training/backtest.
        Same shape as get_flow_features() but bounded to trade_date <= as_of_date,
        so historical rows never use future FII data.
        """
        conn = self._connect()
        rows = conn.execute("""
            SELECT trade_date, fii_net_cr, dii_net_cr
            FROM   fii_data
            WHERE  trade_date <= ?
            ORDER  BY trade_date DESC
            LIMIT  20
        """, (as_of_date[:10],)).fetchall()
        conn.close()

        if not rows:
            return self._empty_flow_features()

        fii_values = [r[1] for r in rows]
        dii_values = [r[2] or 0.0 for r in rows]

        fii_1d  = fii_values[0]
        fii_3d  = sum(fii_values[:3])
        fii_5d  = sum(fii_values[:5])
        fii_10d = sum(fii_values[:10])
        dii_1d  = dii_values[0]
        dii_3d  = sum(dii_values[:3])

        fii_trend = self._calculate_trend(fii_values[:5])
        dii_trend = self._calculate_trend(dii_values[:5])

        confluence = "BOTH_BUYING"  if fii_5d > 500  and dii_3d > 0 else \
                     "BOTH_SELLING" if fii_5d < -500 and dii_3d < 0 else \
                     "MIXED"

        bulk_summary = self._get_bulk_deal_summary_at(as_of_date[:10])

        return {
            "fii_net_1d_cr":    round(fii_1d, 2),
            "fii_net_3d_cr":    round(fii_3d, 2),
            "fii_net_5d_cr":    round(fii_5d, 2),
            "fii_net_10d_cr":   round(fii_10d, 2),
            "dii_net_1d_cr":    round(dii_1d, 2),
            "dii_net_3d_cr":    round(dii_3d, 2),
            "fii_signal":       self._normalise_flow(fii_5d),
            "dii_signal":       self._normalise_flow(dii_3d),
            "fii_trend":        fii_trend,
            "dii_trend":        dii_trend,
            "flow_confluence":  confluence,
            "bulk_net_cr":          bulk_summary["net_cr"],
            "bulk_is_buying":       bulk_summary["is_buying"],
            "bulk_deal_detected":   bulk_summary["detected"],
        }

    def get_fii_signal_text(self) -> str:
        """Simple signal for Telegram alerts and dashboard."""
        features = self.get_flow_features()
        fii_5d = features.get("fii_net_5d_cr", 0)

        if fii_5d > 2000:
            return "STRONG BUY"
        elif fii_5d > 500:
            return "BUYING"
        elif fii_5d < -2000:
            return "STRONG SELL"
        elif fii_5d < -500:
            return "SELLING"
        return "NEUTRAL"

    def load_from_csv(self, csv_dir: str = "fii_cash_daily_data") -> tuple:
        """
        Backfill fii_data from NSE FII daily CSV files.
        Only inserts dates not already in the DB (INSERT OR IGNORE).
        Returns (inserted, skipped) counts.
        """
        import glob
        import csv as csvmod

        files = sorted(glob.glob(os.path.join(csv_dir, "FII_Cash_Daily_*.csv")))
        if not files:
            logger.warning(f"No CSV files found in {csv_dir}")
            return 0, 0

        conn = self._connect()
        inserted = skipped = 0

        for path in files:
            with open(path, newline="", encoding="utf-8-sig") as f:
                for row in csvmod.DictReader(f):
                    raw_net = row.get("FII_Equity_Net_Purchase_Sales", "NA").strip()
                    if raw_net in ("NA", "", "None"):
                        skipped += 1
                        continue
                    try:
                        trade_date = datetime.strptime(
                            row["Date"].strip(), "%d %b %Y"
                        ).strftime("%Y-%m-%d")
                        fii_net  = float(raw_net.replace(",", ""))
                        raw_buy  = row.get("FII_Equity_Gross_Purchase", "0").strip()
                        raw_sell = row.get("FII_Equity_Gross_Sales",    "0").strip()
                        fii_buy  = float(raw_buy.replace(",", ""))  if raw_buy  not in ("NA", "") else 0.0
                        fii_sell = float(raw_sell.replace(",", "")) if raw_sell not in ("NA", "") else 0.0
                    except (ValueError, KeyError):
                        skipped += 1
                        continue

                    cur = conn.execute(
                        "INSERT OR IGNORE INTO fii_data "
                        "(trade_date, fii_buy_cr, fii_sell_cr, fii_net_cr, "
                        " dii_buy_cr, dii_sell_cr, dii_net_cr, fetched_at) "
                        "VALUES (?, ?, ?, ?, NULL, NULL, NULL, ?)",
                        (trade_date, fii_buy, fii_sell, fii_net, "csv_backfill"),
                    )
                    if cur.rowcount:
                        inserted += 1
                    else:
                        skipped += 1

        conn.commit()
        conn.close()
        logger.info(f"FII CSV backfill complete: {inserted} inserted, {skipped} skipped")
        return inserted, skipped

    # ─────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────────────────────────

    def _parse_fii_dii(self, data: list) -> Dict:
        """
        Parse NSE FII/DII response.
        NSE returns a list of rows; field names differ across API versions.
        Handles both 'buyValue'/'sellValue' and 'buy_value'/'sell_value' layouts.
        DII is the aggregate of Mutual Funds + Insurance + other domestic institutions.
        """
        fii_buy = fii_sell = dii_buy = dii_sell = 0.0

        for row in data:
            category = str(row.get("category", "")).upper()

            # NSE uses multiple field name conventions across API versions
            try:
                buy_raw  = (row.get("buyValue")  or row.get("buy_value")  or
                            row.get("BuyValue")  or row.get("grossPurchase") or "0")
                sell_raw = (row.get("sellValue") or row.get("sell_value") or
                            row.get("SellValue") or row.get("grossSales")    or "0")
                buy_val  = float(str(buy_raw).replace(",", ""))
                sell_val = float(str(sell_raw).replace(",", ""))
            except (ValueError, TypeError):
                continue

            # FII / FPI bucket
            if "FII" in category or "FPI" in category or "FOREIGN" in category:
                fii_buy  += buy_val
                fii_sell += sell_val
            # DII bucket — NSE sometimes splits into MF, insurance, banks
            elif ("DII"       in category or "DOMESTIC"  in category or
                  "MUTUAL"    in category or "INSURANCE" in category or
                  "MF"        in category or "AMC"       in category or
                  "BANK"      in category and "FOREIGN" not in category):
                dii_buy  += buy_val
                dii_sell += sell_val

        fii_net = fii_buy - fii_sell
        dii_net = dii_buy - dii_sell

        return {
            "trade_date":   datetime.now().strftime("%Y-%m-%d"),
            "fii_buy_cr":   round(fii_buy, 2),
            "fii_sell_cr":  round(fii_sell, 2),
            "fii_net_cr":   round(fii_net, 2),
            "dii_buy_cr":   round(dii_buy, 2),
            "dii_sell_cr":  round(dii_sell, 2),
            "dii_net_cr":   round(dii_net, 2),
            "fetched_at":   str(datetime.now()),
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """Determine trend direction from a list of values (most recent first)."""
        if len(values) < 3:
            return "NEUTRAL"
        # Are the last 3 values consistently positive or negative?
        positives = sum(1 for v in values[:3] if v > 0)
        if positives >= 3:
            return "BUYING"
        elif positives == 0:
            return "SELLING"
        return "MIXED"

    def _normalise_flow(self, value_cr: float) -> float:
        """
        Normalise ₹ Cr flow to -1 to +1 scale.
        ±₹5000 Cr = ±1.0 (maximum signal strength)
        """
        return max(-1.0, min(1.0, value_cr / 5000.0))

    def _get_bulk_deal_summary(self) -> Dict:
        """Get today's bulk deal summary for HDFC Bank."""
        today = datetime.now().strftime("%Y-%m-%d")
        conn = self._connect()
        rows = conn.execute("""
            SELECT trade_type, quantity, price
            FROM bulk_deals
            WHERE trade_date = ?
        """, (today,)).fetchall()
        conn.close()

        if not rows:
            return {"net_cr": 0.0, "is_buying": False, "detected": False}

        buy_value = sum(
            r[1] * r[2] for r in rows if r[0] == "BUY"
        ) / 1e7  # convert to Cr

        sell_value = sum(
            r[1] * r[2] for r in rows if r[0] == "SELL"
        ) / 1e7

        return {
            "net_cr":   round(buy_value - sell_value, 2),
            "is_buying": buy_value > sell_value,
            "detected": len(rows) > 0,
        }

    def _get_bulk_deal_summary_at(self, trade_date: str) -> Dict:
        """Bulk deal summary at a specific historical date (no datetime.now())."""
        conn = self._connect()
        rows = conn.execute("""
            SELECT trade_type, quantity, price
            FROM   bulk_deals
            WHERE  trade_date = ?
        """, (trade_date,)).fetchall()
        conn.close()

        if not rows:
            return {"net_cr": 0.0, "is_buying": False, "detected": False}

        buy_value  = sum(r[1] * r[2] for r in rows if r[0] == "BUY")  / 1e7
        sell_value = sum(r[1] * r[2] for r in rows if r[0] == "SELL") / 1e7

        return {
            "net_cr":    round(buy_value - sell_value, 2),
            "is_buying": buy_value > sell_value,
            "detected":  len(rows) > 0,
        }

    def _save_fii_data(self, data: Dict) -> None:
        conn = self._connect()
        conn.execute("""
            INSERT OR REPLACE INTO fii_data
            (trade_date, fii_buy_cr, fii_sell_cr, fii_net_cr,
             dii_buy_cr, dii_sell_cr, dii_net_cr, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data["trade_date"], data["fii_buy_cr"], data["fii_sell_cr"],
            data["fii_net_cr"], data["dii_buy_cr"], data["dii_sell_cr"],
            data["dii_net_cr"], data["fetched_at"],
        ))
        conn.commit()
        conn.close()

    def _save_bulk_deals(self, deals: List[Dict]) -> None:
        conn = self._connect()
        for deal in deals:
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO bulk_deals
                    (trade_date, client_name, trade_type,
                     quantity, price, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    deal.get("mTDATE", str(datetime.now().date())),
                    deal.get("clientName", "Unknown"),
                    deal.get("buySell", "BUY"),
                    float(str(deal.get("quantityTraded", "0")).replace(",", "") or 0),
                    float(str(deal.get("tradedPrice", "0")).replace(",", "") or 0),
                    str(datetime.now()),
                ))
            except Exception as e:
                logger.warning(f"Bulk deal save error: {e}")
        conn.commit()
        conn.close()

    def _get_latest_from_db(self) -> Dict:
        conn = self._connect()
        row = conn.execute("""
            SELECT trade_date, fii_net_cr, dii_net_cr
            FROM fii_data ORDER BY trade_date DESC LIMIT 1
        """).fetchone()
        conn.close()
        if not row:
            return {"fii_net_cr": 0.0, "dii_net_cr": 0.0,
                    "trade_date": str(datetime.now().date())}
        return {"fii_net_cr": row[1], "dii_net_cr": row[2],
                "trade_date": row[0]}

    def _empty_flow_features(self) -> Dict:
        return {
            "fii_net_1d_cr": 0, "fii_net_3d_cr": 0,
            "fii_net_5d_cr": 0, "fii_net_10d_cr": 0,
            "dii_net_1d_cr": 0, "dii_net_3d_cr": 0,
            "fii_signal": 0.0, "dii_signal": 0.0,
            "fii_trend": "NEUTRAL", "dii_trend": "NEUTRAL",
            "flow_confluence": "MIXED",
            "bulk_net_cr": 0, "bulk_is_buying": False,
            "bulk_deal_detected": False,
        }

    def _create_session(self) -> requests.Session:
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

    def _setup_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fii_data (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_date   TEXT    UNIQUE,
                fii_buy_cr   REAL,
                fii_sell_cr  REAL,
                fii_net_cr   REAL,
                dii_buy_cr   REAL,
                dii_sell_cr  REAL,
                dii_net_cr   REAL,
                fetched_at   TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bulk_deals (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_date   TEXT,
                client_name  TEXT,
                trade_type   TEXT,
                quantity     REAL,
                price        REAL,
                created_at   TEXT,
                UNIQUE(trade_date, client_name, trade_type)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fii_date ON fii_data (trade_date DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_bulk_date ON bulk_deals (trade_date DESC)")
        conn.commit()
        conn.close()