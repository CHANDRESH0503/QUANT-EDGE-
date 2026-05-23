# data/historical/bulk_deals_backfill.py
# Backfill bulk deal transactions for HDFCBANK from NSE archive.
# Run: python -m data.historical.bulk_deals_backfill --start=2020-01-01

import argparse
import logging
import sqlite3
import time
from datetime import datetime, timedelta

import requests

logger = logging.getLogger(__name__)

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://www.nseindia.com/",
}

BULK_URL  = "https://www.nseindia.com/api/historical/bulk-deals"
SLEEP_SEC = 1.5


def _session() -> requests.Session:
    s = requests.Session()
    try:
        s.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=10)
    except Exception:
        pass
    return s


def backfill(start: str, end: str, db_path: str = "database/trading.db") -> int:
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")

    s = _session()
    conn = sqlite3.connect(db_path)
    total = 0

    # NSE bulk-deals archive accepts month-sized windows
    cur = start_dt
    while cur <= end_dt:
        window_end = min(cur + timedelta(days=30), end_dt)
        params = {
            "from": cur.strftime("%d-%m-%Y"),
            "to":   window_end.strftime("%d-%m-%Y"),
        }
        try:
            resp = s.get(BULK_URL, headers=NSE_HEADERS, params=params, timeout=20)
            if resp.status_code == 200:
                payload = resp.json()
                rows = payload.get("data", []) if isinstance(payload, dict) else payload
                hdfc = [
                    d for d in rows
                    if "HDFCBANK" in str(d.get("symbol", "")).upper()
                ]
                for d in hdfc:
                    try:
                        conn.execute("""
                            INSERT OR IGNORE INTO bulk_deals
                            (trade_date, client_name, trade_type,
                             quantity, price, created_at)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            d.get("date") or d.get("mTDATE", ""),
                            d.get("clientName", "Unknown"),
                            d.get("buySell") or d.get("buyOrSell", "BUY"),
                            float(str(d.get("quantityTraded", "0")).replace(",", "") or 0),
                            float(str(d.get("tradedPrice", "0")).replace(",", "") or 0),
                            str(datetime.now()),
                        ))
                        total += 1
                    except Exception as e:
                        logger.debug(f"bulk row insert error: {e}")
                if hdfc:
                    conn.commit()
                    logger.info(f"bulk deals {params['from']}→{params['to']}: +{len(hdfc)} HDFC rows")
            else:
                logger.debug(f"bulk {params}: HTTP {resp.status_code}")
        except Exception as e:
            logger.debug(f"bulk fetch error {params}: {e}")

        time.sleep(SLEEP_SEC)
        cur = window_end + timedelta(days=1)

    conn.close()
    logger.info(f"Bulk deals backfill complete — {total} HDFCBANK rows")
    return total


def _cli() -> None:
    p = argparse.ArgumentParser(description="Backfill HDFCBANK bulk deals from NSE")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end",   default=datetime.now().strftime("%Y-%m-%d"))
    p.add_argument("--db",    default="database/trading.db")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    backfill(args.start, args.end, args.db)


if __name__ == "__main__":
    _cli()
