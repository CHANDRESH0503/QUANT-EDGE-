# data/historical/fii_backfill.py
# Backfill daily FII/DII flow from NSE historical archive into fii_data.
# Idempotent — skips dates already present (trade_date UNIQUE).
# Run: python -m data.historical.fii_backfill --start=2020-01-01
#
# NSE endpoint: /api/historical/fiidiiTradeReact?from=DD-Mon-YYYY&to=DD-Mon-YYYY
# Returns a list of daily rows (max ~90 days per call).

import argparse
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import requests

logger = logging.getLogger(__name__)

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
    "X-Requested-With": "XMLHttpRequest",
}

HISTORICAL_URL = "https://www.nseindia.com/api/historical/fiidiiTradeReact"
BATCH_DAYS = 85   # stay under NSE's ~90-day limit
SLEEP_SEC  = 2.0  # polite delay between batch requests


def _session() -> requests.Session:
    s = requests.Session()
    try:
        # Seed session cookie — NSE requires a prior homepage visit
        s.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=15)
        time.sleep(1)
    except Exception:
        pass
    return s


def _batches(start: datetime, end: datetime) -> List[Tuple[datetime, datetime]]:
    """Split [start, end] into ~85-day windows."""
    batches = []
    cur = start
    while cur <= end:
        nxt = min(cur + timedelta(days=BATCH_DAYS), end)
        batches.append((cur, nxt))
        cur = nxt + timedelta(days=1)
    return batches


def _fetch_batch(s: requests.Session, from_dt: datetime, to_dt: datetime) -> List[Dict]:
    """
    Fetch one 85-day batch from NSE historical endpoint.
    NSE format: from=DD-Mon-YYYY&to=DD-Mon-YYYY  (e.g. 01-Jan-2020)
    Returns list of row dicts or [] on failure.
    """
    params = {
        "from": from_dt.strftime("%d-%b-%Y"),
        "to":   to_dt.strftime("%d-%b-%Y"),
    }
    try:
        resp = s.get(HISTORICAL_URL, headers=NSE_HEADERS, params=params, timeout=20)
        if resp.status_code == 401:
            # Session expired — re-seed and retry once
            logger.debug("NSE 401 — re-seeding session")
            s.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=15)
            time.sleep(2)
            resp = s.get(HISTORICAL_URL, headers=NSE_HEADERS, params=params, timeout=20)

        if resp.status_code != 200:
            logger.warning(
                f"FII batch {params['from']}→{params['to']}: HTTP {resp.status_code}"
            )
            return []

        payload = resp.json()
        # Endpoint returns {"data": [...]} or a bare list
        if isinstance(payload, list):
            return payload
        return payload.get("data", [])
    except Exception as e:
        logger.warning(f"FII batch {params['from']}→{params['to']} error: {e}")
        return []


def _parse_row(raw: Dict) -> Dict | None:
    """
    Convert one NSE row into fii_data shape.
    NSE row keys: date, buyValue, sellValue, category
    Each date has multiple rows (FPI, DII, etc.) — caller must aggregate per date.
    """
    try:
        date_str  = raw.get("date", "").strip()          # e.g. "01-Jan-2020"
        category  = str(raw.get("category", "")).upper()
        buy_val   = float(str(raw.get("buyValue",  "0")).replace(",", "") or 0)
        sell_val  = float(str(raw.get("sellValue", "0")).replace(",", "") or 0)
        return {
            "date":     date_str,
            "category": category,
            "buy":      buy_val,
            "sell":     sell_val,
        }
    except (ValueError, TypeError):
        return None


def _aggregate_by_date(rows: List[Dict]) -> Dict[str, Dict]:
    """
    Group raw NSE rows by date; sum FPI→FII bucket and DII bucket.
    Returns {trade_date_iso: {fii_buy, fii_sell, fii_net, dii_buy, dii_sell, dii_net}}
    """
    buckets: Dict[str, Dict] = {}
    for raw in rows:
        parsed = _parse_row(raw)
        if not parsed:
            continue

        # Normalise date → YYYY-MM-DD
        try:
            dt = datetime.strptime(parsed["date"], "%d-%b-%Y")
            iso = dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

        if iso not in buckets:
            buckets[iso] = dict(fii_buy=0.0, fii_sell=0.0, dii_buy=0.0, dii_sell=0.0)

        cat = parsed["category"]
        if "FII" in cat or "FPI" in cat or "FOREIGN" in cat:
            buckets[iso]["fii_buy"]  += parsed["buy"]
            buckets[iso]["fii_sell"] += parsed["sell"]
        elif "DII" in cat or "DOMESTIC" in cat:
            buckets[iso]["dii_buy"]  += parsed["buy"]
            buckets[iso]["dii_sell"] += parsed["sell"]

    return buckets


def _save_batch(conn: sqlite3.Connection, buckets: Dict[str, Dict]) -> int:
    """Insert/replace rows into fii_data. Returns number of rows written."""
    written = 0
    for iso, b in sorted(buckets.items()):
        fii_net = round(b["fii_buy"] - b["fii_sell"], 2)
        dii_net = round(b["dii_buy"] - b["dii_sell"], 2)
        # Skip zero rows — likely holidays returned as noise
        if b["fii_buy"] == 0 and b["dii_buy"] == 0:
            continue
        conn.execute("""
            INSERT OR REPLACE INTO fii_data
            (trade_date, fii_buy_cr, fii_sell_cr, fii_net_cr,
             dii_buy_cr, dii_sell_cr, dii_net_cr, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """, (
            iso,
            round(b["fii_buy"],  2), round(b["fii_sell"],  2), fii_net,
            round(b["dii_buy"],  2), round(b["dii_sell"],  2), dii_net,
        ))
        written += 1
    return written


def backfill(start: str, end: str, db_path: str = "database/trading.db") -> int:
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")

    logger.info(f"FII backfill: {start} → {end}")
    s    = _session()
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")

    total  = 0
    batches = _batches(start_dt, end_dt)
    logger.info(f"FII backfill: {len(batches)} batch(es) of ~{BATCH_DAYS} days each")

    for i, (from_dt, to_dt) in enumerate(batches, 1):
        # Skip batch if all dates already covered
        existing = conn.execute("""
            SELECT COUNT(*) FROM fii_data
            WHERE trade_date BETWEEN ? AND ?
        """, (from_dt.strftime("%Y-%m-%d"), to_dt.strftime("%Y-%m-%d"))).fetchone()[0]

        if existing >= (to_dt - from_dt).days * 0.7:  # ~70% coverage = already done
            logger.info(
                f"Batch {i}/{len(batches)} "
                f"{from_dt.strftime('%Y-%m-%d')}→{to_dt.strftime('%Y-%m-%d')} "
                f"already covered ({existing} rows) — skipping"
            )
            continue

        logger.info(
            f"Batch {i}/{len(batches)}: "
            f"{from_dt.strftime('%d-%b-%Y')} → {to_dt.strftime('%d-%b-%Y')}"
        )
        raw_rows = _fetch_batch(s, from_dt, to_dt)
        if not raw_rows:
            logger.warning(f"  Batch {i} returned 0 rows — NSE may be throttling")
            time.sleep(SLEEP_SEC * 3)
            continue

        buckets = _aggregate_by_date(raw_rows)
        written = _save_batch(conn, buckets)
        conn.commit()
        total += written
        logger.info(f"  → {len(raw_rows)} raw rows → {written} trading days saved "
                    f"(cumulative: {total})")
        time.sleep(SLEEP_SEC)

    conn.commit()
    conn.close()
    logger.info(f"FII backfill complete — {total} days written to fii_data")
    return total


def _cli() -> None:
    p = argparse.ArgumentParser(description="Backfill NSE FII/DII historical flow")
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
