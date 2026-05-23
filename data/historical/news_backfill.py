# data/historical/news_backfill.py
# Backfill historical HDFC Bank news from GDELT into the `news` table,
# then optionally run FinBERT scoring across the backlog.
#
# GDELT (https://api.gdeltproject.org) indexes global media since 2015 and
# returns headline + timestamp + URL with no API key. Coverage is broader
# than Google News RSS (which only goes back ~30 days).
#
# Run: python -m data.historical.news_backfill --start=2020-01-01 --score

import argparse
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List

import requests

logger = logging.getLogger(__name__)

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
QUERY     = '"HDFC Bank" sourcelang:eng'   # GDELT supports quoted phrases
MAX_RECORDS_PER_CALL = 250
BATCH_DAYS = 14   # 2-week windows balance request size and gap coverage
SLEEP_SEC  = 1.0


def _gdelt_window(start: datetime, end: datetime) -> List[Dict]:
    """Pull articles for [start, end] window from GDELT."""
    params = {
        "query":       QUERY,
        "format":      "json",
        "mode":        "ArtList",
        "maxrecords":  MAX_RECORDS_PER_CALL,
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime":   end.strftime("%Y%m%d%H%M%S"),
        "sort":        "DateDesc",
    }
    try:
        resp = requests.get(GDELT_URL, params=params, timeout=30)
        if resp.status_code != 200:
            logger.debug(f"GDELT HTTP {resp.status_code} for {start.date()}→{end.date()}")
            return []
        try:
            payload = resp.json()
        except ValueError:
            return []
        return payload.get("articles", [])
    except Exception as e:
        logger.debug(f"GDELT fetch error {start.date()}: {e}")
        return []


def _save(conn: sqlite3.Connection, article: Dict) -> bool:
    """Insert one article. Returns True if new row was created."""
    try:
        title = (article.get("title") or "").strip()
        if not title:
            return False
        # GDELT 'seendate' is YYYYMMDDTHHMMSSZ
        sd = article.get("seendate") or ""
        try:
            published = datetime.strptime(sd, "%Y%m%dT%H%M%SZ")
        except ValueError:
            published = datetime.now()

        cur = conn.execute("""
            INSERT OR IGNORE INTO news
            (title, summary, published, source, link,
             sentiment_score, processed, llm_analyzed, created_at)
            VALUES (?, ?, ?, ?, ?, 0.0, 0, 0, ?)
        """, (
            title[:500],
            (article.get("title") or "")[:1000],   # GDELT has no summary
            str(published),
            article.get("domain", "gdelt"),
            article.get("url", ""),
            str(published),     # historical articles backdated to publish time
        ))
        return cur.rowcount > 0
    except Exception as e:
        logger.debug(f"news insert error: {e}")
        return False


def backfill_news(start: str, end: str, db_path: str) -> int:
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")

    conn = sqlite3.connect(db_path)
    total_new = 0

    cur = start_dt
    while cur <= end_dt:
        nxt = min(cur + timedelta(days=BATCH_DAYS), end_dt)
        articles = _gdelt_window(cur, nxt + timedelta(hours=23, minutes=59))
        new_in_window = 0
        for a in articles:
            if _save(conn, a):
                new_in_window += 1
        if new_in_window:
            conn.commit()
            total_new += new_in_window
            logger.info(
                f"news {cur.date()}→{nxt.date()}: "
                f"+{new_in_window} new ({len(articles)} fetched)"
            )
        time.sleep(SLEEP_SEC)
        cur = nxt + timedelta(days=1)

    conn.close()
    logger.info(f"News backfill complete — {total_new} new articles")
    return total_new


def score_all(db_path: str) -> int:
    """Run FinBERT across every processed=0 row in the news table."""
    from processing.finbert_sentiment import FinBERTSentiment
    fb = FinBERTSentiment(db_path=db_path)
    scored = 0
    while True:
        n = fb.process_unscored(batch_size=200)
        if not n:
            break
        scored += n
        logger.info(f"FinBERT scored {scored} articles so far...")
    logger.info(f"FinBERT scoring complete — {scored} articles")
    return scored


def _cli() -> None:
    p = argparse.ArgumentParser(description="Backfill HDFC Bank news from GDELT")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end",   default=datetime.now().strftime("%Y-%m-%d"))
    p.add_argument("--db",    default="database/trading.db")
    p.add_argument("--score", action="store_true",
                   help="After fetching, run FinBERT across all unscored news")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    backfill_news(args.start, args.end, args.db)
    if args.score:
        score_all(args.db)


if __name__ == "__main__":
    _cli()
