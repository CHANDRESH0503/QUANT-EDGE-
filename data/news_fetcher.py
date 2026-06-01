# data/news_fetcher.py
# Collects news from RSS feeds for all 5 banks every 15 minutes.
# Stores in SQLite with per-bank ticker tagging — FinBERT processes separately.
# Sources: Google News (per-bank), Economic Times (per-bank), Moneycontrol, Business Standard

import feedparser
import sqlite3
import logging
import os
import re
import gzip
import time
import json
import zlib
import urllib.request
import urllib.error
import urllib.parse

from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Set QE_IS_VPS=1 in the systemd EnvironmentFile on the production host so we
# skip Google News on DigitalOcean (datacenter IPs are in Google's blocklist —
# the Chrome UA doesn't help because IP filtering happens before UA evaluation).
IS_VPS = os.environ.get("QE_IS_VPS", "0") == "1"
_VPS_LOGGED = False

# Hard recency cutoff. Articles older than this at insert time are dropped —
# anything months/years old has no signal for swing/intraday trading decisions.
MAX_ARTICLE_AGE_DAYS = 7

# Google News query operator that restricts results to the last N days AND
# sorts them by date (newest first). Empirically returns ~100 entries vs the
# relevance-sorted default which mixes in articles from years ago.
GOOGLE_NEWS_RECENCY_WINDOW = "7d"

# ── HTTP headers used for every feed request ──────────────────────────────────
# VPS / cloud IPs are blocked by Google News and most Indian news sites if no
# User-Agent is sent (feedparser's default UA is "feedparser/<version>", which
# is immediately identified as a bot and rejected).  A realistic Chrome UA
# passes most site-side filters.
_BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
_REQUEST_HEADERS = {
    "User-Agent":      _BROWSER_UA,
    "Accept":          "application/rss+xml, application/xml, text/xml, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Referer":         "https://www.google.com/",
    "Cache-Control":   "no-cache",
}
_FETCH_TIMEOUT = 20   # seconds per feed request

# ── GDELT DOC 2.0 API — datacenter-friendly bank-specific source ──────────────
# Google News and the ET topic RSS feeds are both unusable from the VPS:
# Google News IP-blocks DigitalOcean ranges, and ET's per-bank topic RSS now
# returns an empty 776-byte shell (0 entries). GDELT's DOC API returns recent
# global news as RSS, needs no key, and serves cloud IPs without WAF blocks —
# so it is the bank news PRIMARY on the VPS.
#
# CRITICAL — GDELT aggressively rate-limits (HTTP 429): one query every few
# seconds at most, and bursts trip a multi-minute penalty box. Fetching one
# query PER BANK (5×/cycle, ×3 with retries) reliably 429s every bank after the
# first. So we fetch ONE COMBINED query for all 5 banks per cycle and cache it
# class-level (universe pattern, like intermarket — Rule #19); each bank then
# keyword-filters the shared result. That's 1 GDELT request per ~13 min.
GDELT_DOC_API        = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_TIMESPAN       = "7d"    # matches MAX_ARTICLE_AGE_DAYS
GDELT_MAXRECORDS     = 250     # combined 5-bank query — give every bank coverage
# One OR'd query covering the whole universe. Each bank's keyword filter
# (`_is_relevant`) selects its own articles from the shared result set.
GDELT_COMBINED_QUERY = (
    '("HDFC Bank" OR "ICICI Bank" OR "Kotak Mahindra Bank" '
    'OR "Axis Bank" OR "IndusInd Bank")'
)
# Cache TTL must be < the 15-min news cycle so each cycle gets one fresh fetch.
_GDELT_CACHE_TTL_SEC = 780     # 13 min


def _gdelt_url(query: str, maxrecords: int = GDELT_MAXRECORDS, fmt: str = "json") -> str:
    """Build a GDELT DOC API URL for a query (newest first).

    JSON is the default and the format we actually use — GDELT's RSS `<pubDate>`
    is malformed (feedparser misreads it as a month-old date and the recency
    filter then drops fresh articles). The JSON `seendate` field is a clean
    `YYYYMMDDTHHMMSSZ` timestamp, so we parse that instead.
    """
    q = urllib.parse.quote(query)
    return (
        f"{GDELT_DOC_API}?query={q}&mode=ArtList&format={fmt}"
        f"&timespan={GDELT_TIMESPAN}&sortby=datedesc&maxrecords={maxrecords}"
    )


def _fetch_feed_bytes(url: str, _retried_429: bool = False) -> Optional[bytes]:
    """Download a feed URL with a browser-like UA.

    Returns raw bytes on success, None on network/HTTP error.
    Logs the HTTP status and entry-level detail so VPS debugging is easy.
    """
    try:
        req = urllib.request.Request(url, headers=_REQUEST_HEADERS)
        with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT) as resp:
            status = getattr(resp, "status", "?")
            data   = resp.read()
            # We advertise `Accept-Encoding: gzip, deflate`, so most feeds
            # (ET, Moneycontrol, CNBC, LiveMint) reply compressed. urllib does
            # NOT auto-decompress, and feedparser cannot parse compressed bytes
            # — it silently returns 0 entries. Decompress here based on the
            # Content-Encoding header. This is THE reason news died on the VPS:
            # once Google News (the one feed that replied uncompressed) is
            # disabled with QE_IS_VPS=1, every remaining feed parsed to 0.
            encoding = (resp.headers.get("Content-Encoding") or "").lower()
            if encoding == "gzip":
                try:
                    data = gzip.decompress(data)
                except OSError:
                    pass  # not actually gzipped — use bytes as-is
            elif encoding == "deflate":
                try:
                    data = zlib.decompress(data)
                except zlib.error:
                    try:
                        data = zlib.decompress(data, -zlib.MAX_WBITS)  # raw deflate
                    except zlib.error:
                        pass
            logger.debug(
                f"_fetch_feed_bytes: status={status} bytes={len(data)} "
                f"enc={encoding or 'none'} url={url[:80]}"
            )
            return data
    except urllib.error.HTTPError as e:
        # GDELT throttles bursts with 429 — back off once and retry before giving up.
        if e.code == 429 and not _retried_429:
            logger.info(f"HTTP 429 (rate-limited) — backing off 10s and retrying: {url[:80]}")
            time.sleep(10)
            return _fetch_feed_bytes(url, _retried_429=True)
        logger.warning(f"HTTP {e.code} fetching feed: {url[:80]}")
        return None
    except urllib.error.URLError as e:
        logger.warning(f"URL error fetching feed ({e.reason}): {url[:80]}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error fetching feed ({e}): {url[:80]}")
        return None


# ── Per-bank news configuration ───────────────────────────────────────────────
BANK_NEWS_CONFIG: Dict[str, Dict] = {
    "HDFCBANK.NS": {
        "google_query": "HDFC+Bank+NSE+stock+India",
        "gdelt_query":  '"HDFC Bank" India',
        "et_feed": (
            "https://economictimes.indiatimes.com/topic/hdfc-bank"
            "/rssfeeds/1715249571.cms"
        ),
        "keywords": [
            "hdfc bank", "hdfcbank", "hdfc", "sashidhar jagdishan",
            "housing development finance",
        ],
    },
    "ICICIBANK.NS": {
        "google_query": "ICICI+Bank+NSE+stock+India",
        "gdelt_query":  '"ICICI Bank" India',
        "et_feed": (
            "https://economictimes.indiatimes.com/topic/icici-bank"
            "/rssfeeds/315110307.cms"
        ),
        "keywords": [
            "icici bank", "icicibank", "icici", "sandeep bakhshi",
        ],
    },
    "KOTAKBANK.NS": {
        "google_query": "Kotak+Mahindra+Bank+NSE+stock+India",
        "gdelt_query":  '"Kotak Mahindra Bank"',
        "et_feed": (
            "https://economictimes.indiatimes.com/topic/kotak-mahindra-bank"
            "/rssfeeds/1715255371.cms"
        ),
        "keywords": [
            "kotak bank", "kotak mahindra bank", "kotakbank",
            "ashok vaswani", "kotak mahindra",
        ],
    },
    "AXISBANK.NS": {
        "google_query": "Axis+Bank+NSE+stock+India",
        "gdelt_query":  '"Axis Bank" India',
        "et_feed": (
            "https://economictimes.indiatimes.com/topic/axis-bank"
            "/rssfeeds/1715249541.cms"
        ),
        "keywords": [
            "axis bank", "axisbank", "axis bank ltd",
            "amitabh chaudhry",
        ],
    },
    "INDUSINDBK.NS": {
        "google_query": "IndusInd+Bank+NSE+stock+India",
        "gdelt_query":  '"IndusInd Bank"',
        "et_feed": (
            "https://economictimes.indiatimes.com/topic/indusind-bank"
            "/rssfeeds/1715250067.cms"
        ),
        "keywords": [
            "indusind bank", "indusindbk", "sumant kathpalia",
            "indusind",
        ],
    },
}

# Generic feeds shared across all banks — articles tagged by keyword matching.
# All fetched with browser UA so VPS IPs are not blocked.
# ET topic feeds: previously returned 0 entries without UA; re-enabled with UA fix.
# Business Standard: 403 even with UA — remains removed.
_SHARED_FEEDS = {
    "moneycontrol": "https://www.moneycontrol.com/rss/marketreports.xml",
    "livemint":     "https://www.livemint.com/rss/markets",
    "cnbctv18":     "https://www.cnbctv18.com/commonfeeds/v1/cne/rss/market.xml",
}

# ET banking RSS — per-bank, fetched in addition to google_news.
# Works from VPS when browser UA is sent.  Used as primary on VPS when Google
# News is blocked (DigitalOcean/Railway IPs are in Google's datacenter blocklist).
_ET_BANKING_RSS = "https://economictimes.indiatimes.com/industry/banking/finance/banking/rssfeeds/7202578.cms"

HIGH_IMPORTANCE_KEYWORDS = [
    "quarterly results", "q1 results", "q2 results", "q3 results", "q4 results",
    "earnings", "profit", "revenue", "npa", "nim", "dividend",
    "acquisition", "merger", "board meeting", "rbi penalty",
    "bulk deal", "block deal", "promoter", "stake sale",
    "guidance", "analyst upgrade", "analyst downgrade",
]

ALL_BANK_TICKERS = list(BANK_NEWS_CONFIG.keys())


class NewsFetcher:
    """
    Fetches financial news for a specific bank (or all banks) from RSS feeds.

    Pass ticker="HDFCBANK.NS" for single-bank or use fetch_all_banks() to
    collect for every bank in one scheduler call.
    """

    def __init__(
        self,
        db_path: str = "database/trading.db",
        ticker:  str = "HDFCBANK.NS",
    ):
        self.db_path = db_path
        self.ticker  = ticker if ticker in BANK_NEWS_CONFIG else "HDFCBANK.NS"
        self._setup_db()

    # ─────────────────────────────────────────────────────────────
    # PUBLIC METHODS
    # ─────────────────────────────────────────────────────────────

    # Class-level state for the empty-cycle alarm. Counts consecutive
    # `fetch_all_banks` calls that ingested zero articles total. Resets to 0
    # whenever at least one row is inserted.
    _consecutive_empty_cycles: int = 0
    _empty_alarm_sent: bool = False

    # Class-level GDELT cache — ONE combined universe query shared across all 5
    # per-bank fetchers in a cycle, so we make a single GDELT request per ~13
    # min instead of 5×3 (which 429s). Holds parsed feedparser entries.
    _gdelt_cache_ts: float = 0.0
    _gdelt_cache_entries: list = []

    def fetch_all(self) -> Dict[str, int]:
        """
        Fetch RSS feeds for this bank and save new articles.
        Returns a dict of {source_name: inserted_count}; sum is total new.
        Called every 15 minutes by scheduler (via fetch_all_banks).
        """
        try:
            conn = sqlite3.connect(self.db_path)
            row  = conn.execute(
                "SELECT MAX(created_at) FROM news WHERE ticker = ?",
                (self.ticker,),
            ).fetchone()
            conn.close()
            if row and row[0]:
                last = datetime.fromisoformat(str(row[0])[:19])
                if (datetime.now() - last).total_seconds() < 900:
                    logger.debug(f"News fresh (<15 min) for {self.ticker} — skipping")
                    return {}
        except Exception:
            pass

        cfg   = BANK_NEWS_CONFIG[self.ticker]
        feeds = self._build_feeds(cfg)

        per_source: Dict[str, int] = {}
        for source_name, feed_url in feeds.items():
            try:
                count = self._fetch_feed(source_name, feed_url)
                per_source[source_name] = count
                if count:
                    logger.info(f"[{self.ticker}][{source_name}] {count} new articles")
            except Exception as e:
                logger.error(f"[{self.ticker}][{source_name}] Feed error: {e}")
                per_source[source_name] = 0

        total_new = sum(per_source.values())
        logger.info(f"News fetch [{self.ticker}] complete — {total_new} new articles")
        return per_source

    @classmethod
    def fetch_all_banks(cls, db_path: str = "database/trading.db", telegram=None) -> int:
        """
        Fetch news for all 5 banks in sequence.
        Logs one per-source cycle summary at the end.
        Raises a Telegram alert (if `telegram` is given) when two consecutive
        cycles ingest zero articles — that signals every feed source is down.
        """
        agg: Dict[str, int] = {}
        for ticker in ALL_BANK_TICKERS:
            try:
                fetcher = cls(db_path=db_path, ticker=ticker)
                for source, n in fetcher.fetch_all().items():
                    agg[source] = agg.get(source, 0) + n
            except Exception as e:
                logger.error(f"fetch_all_banks failed for {ticker}: {e}")

        total = sum(agg.values())
        summary = " ".join(f"{k}={v}" for k, v in sorted(agg.items()))
        logger.info(f"news cycle: {summary} [total={total}]")

        if total == 0:
            cls._consecutive_empty_cycles += 1
            if cls._consecutive_empty_cycles >= 2 and not cls._empty_alarm_sent:
                msg = (
                    f"⚠️ News pipeline empty for {cls._consecutive_empty_cycles} "
                    f"consecutive cycles — every feed source returned 0 articles."
                )
                logger.warning(msg)
                if telegram is not None:
                    try:
                        telegram.send_raw(msg)
                        cls._empty_alarm_sent = True
                    except Exception as e:
                        logger.warning(f"news empty-cycle alarm telegram failed: {e}")
        else:
            cls._consecutive_empty_cycles = 0
            cls._empty_alarm_sent = False
        return total

    def get_recent_unprocessed(self, limit: int = 50) -> List[Dict]:
        """Articles not yet scored by FinBERT for this bank."""
        conn = self._connect()
        rows = conn.execute("""
            SELECT id, title, summary, source, published
            FROM news
            WHERE processed = 0 AND ticker = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (self.ticker, limit)).fetchall()
        conn.close()
        return [
            {
                "id":        row[0],
                "title":     row[1],
                "summary":   row[2],
                "source":    row[3],
                "published": row[4],
                "text":      f"{row[1]}. {row[2] or ''}",
            }
            for row in rows
        ]

    def get_recent_scored(self, hours: int = 24) -> List[Dict]:
        """Scored articles from last N hours for this bank."""
        since = datetime.now() - timedelta(hours=hours)
        conn  = self._connect()
        rows  = conn.execute("""
            SELECT id, title, summary, source, published,
                   sentiment_score, is_important
            FROM news
            WHERE processed = 1 AND ticker = ? AND created_at > ?
            ORDER BY created_at DESC
        """, (self.ticker, str(since))).fetchall()
        conn.close()
        return [
            {
                "id":              row[0],
                "title":           row[1],
                "summary":         row[2],
                "source":          row[3],
                "published":       row[4],
                "sentiment_score": row[5],
                "is_important":    row[6],
            }
            for row in rows
        ]

    def get_daily_stats(self, hours: int = 24) -> Dict:
        """Aggregate sentiment stats for dashboard and feature building."""
        since = datetime.now() - timedelta(hours=hours)
        conn  = self._connect()
        stats = conn.execute("""
            SELECT
                COUNT(*)                                                      AS total_articles,
                AVG(sentiment_score)                                          AS avg_sentiment,
                MIN(sentiment_score)                                          AS min_sentiment,
                MAX(sentiment_score)                                          AS max_sentiment,
                STDEV(sentiment_score)                                        AS std_sentiment,
                SUM(CASE WHEN sentiment_score >  0.2 THEN 1 ELSE 0 END)      AS positive_count,
                SUM(CASE WHEN sentiment_score < -0.2 THEN 1 ELSE 0 END)      AS negative_count,
                SUM(CASE WHEN is_important = 1 THEN 1 ELSE 0 END)            AS important_count
            FROM news
            WHERE processed = 1 AND ticker = ? AND created_at > ?
        """, (self.ticker, str(since))).fetchone()
        conn.close()

        if not stats or stats[0] == 0:
            return {
                "total_articles": 0, "avg_sentiment": 0.0,
                "min_sentiment":  0.0, "max_sentiment": 0.0,
                "std_sentiment":  0.0, "positive_count": 0,
                "negative_count": 0,  "important_count": 0,
                "quality": "NO_NEWS",
            }

        return {
            "total_articles":  stats[0],
            "avg_sentiment":   round(stats[1] or 0.0, 4),
            "min_sentiment":   round(stats[2] or 0.0, 4),
            "max_sentiment":   round(stats[3] or 0.0, 4),
            "std_sentiment":   round(stats[4] or 0.0, 4),
            "positive_count":  stats[5] or 0,
            "negative_count":  stats[6] or 0,
            "important_count": stats[7] or 0,
            "quality":         "GOOD" if stats[0] >= 3 else "LOW",
        }

    def mark_processed(self, article_id: int, sentiment_score: float) -> None:
        conn = self._connect()
        conn.execute("""
            UPDATE news SET sentiment_score = ?, processed = 1 WHERE id = ?
        """, (sentiment_score, article_id))
        conn.commit()
        conn.close()

    def mark_llm_analyzed(self, article_id: int, llm_score: float) -> None:
        conn = self._connect()
        conn.execute("""
            UPDATE news SET llm_analyzed = 1, llm_score = ? WHERE id = ?
        """, (llm_score, article_id))
        conn.commit()
        conn.close()

    def get_important_unanalyzed(self) -> List[Dict]:
        conn = self._connect()
        rows = conn.execute("""
            SELECT id, title, summary, source
            FROM news
            WHERE is_important = 1 AND llm_analyzed = 0 AND processed = 1
            AND ticker = ?
            ORDER BY created_at DESC LIMIT 10
        """, (self.ticker,)).fetchall()
        conn.close()
        return [{"id": r[0], "title": r[1], "summary": r[2], "source": r[3]}
                for r in rows]

    # ─────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ─────────────────────────────────────────────────────────────

    def _build_feeds(self, cfg: Dict) -> Dict[str, str]:
        """Build the RSS feed dict for this bank.

        Priority (in fetch order):
          1. gdelt         — per-bank GDELT DOC API; the VPS bank-specific PRIMARY
                             (datacenter-friendly, replaces dead Google News + ET)
          2. google_news   — date-sorted entries; LOCAL ONLY (IP-blocked on VPS)
          3. moneycontrol  — shared generic; VPS-friendly with browser UA
          4. livemint      — shared generic; VPS-friendly with browser UA
          5. cnbctv18      — shared generic; VPS-friendly with browser UA

        The ET per-bank topic RSS (`cfg["et_feed"]`) and ET banking RSS are NO
        LONGER fetched — both now return an empty 776-byte shell (0 entries).
        GDELT replaces them as the bank-specific source. The `et_feed` field is
        retained in BANK_NEWS_CONFIG so it can be re-enabled if ET restores it.
        """
        global _VPS_LOGGED
        feeds: Dict[str, str] = {}
        # GDELT — the bank-specific primary now that Google News (IP-blocked) and
        # ET topic RSS (empty) are dead from the datacenter. The value is a
        # sentinel: `_fetch_feed` serves "gdelt" from the shared class-level
        # combined-query cache (one request/cycle for all banks), NOT this URL.
        feeds["gdelt"] = "combined"
        if not IS_VPS:
            # Google News: date-restricted via when:Nd, sorted newest first.
            # Skipped on VPS — DigitalOcean IPs are in Google's datacenter
            # blocklist and the request is rejected before UA is read.
            feeds["google_news"] = (
                f"https://news.google.com/rss/search"
                f"?q={cfg['google_query']}+when:{GOOGLE_NEWS_RECENCY_WINDOW}"
                f"&hl=en-IN&gl=IN&ceid=IN:en"
            )
        elif not _VPS_LOGGED:
            logger.info(
                "VPS mode (QE_IS_VPS=1): Google News disabled (IP-blocklisted); "
                "GDELT is the per-bank primary"
            )
            _VPS_LOGGED = True
        feeds.update(_SHARED_FEEDS)
        return feeds

    @staticmethod
    def _parse_published(raw: str) -> Optional[datetime]:
        """Parse the RSS `published` field (RFC822 or ISO 8601).

        Returns a naive UTC-ish datetime, or None if the field is unparseable.
        RFC822 ("Sat, 23 May 2026 09:30:25 +0000") covers Google News and most
        feeds; ISO 8601 covers Moneycontrol and a handful of others.
        """
        if not raw:
            return None
        try:
            dt = parsedate_to_datetime(raw)
            if dt is not None:
                if dt.tzinfo is not None:
                    dt = dt.replace(tzinfo=None)
                return dt
        except (TypeError, ValueError, IndexError):
            pass
        try:
            return datetime.fromisoformat(raw[:19])
        except ValueError:
            return None

    @classmethod
    def _get_gdelt_entries(cls) -> list:
        """Return cached combined-GDELT entries, refreshing once per TTL window.

        ONE GDELT request per cycle for the whole universe (shared across all 5
        per-bank fetchers) — avoids the per-bank 429 storm. On a fetch failure
        (429/network) the stale cache is reused rather than going dark.

        Uses GDELT JSON mode and the `seendate` timestamp (the RSS `<pubDate>`
        is malformed → misparsed as month-old → dropped by recency). Returns a
        list of plain dicts shaped like feedparser entries (`.get` compatible).
        """
        age = time.monotonic() - cls._gdelt_cache_ts
        if cls._gdelt_cache_entries and age < _GDELT_CACHE_TTL_SEC:
            return cls._gdelt_cache_entries

        url = _gdelt_url(GDELT_COMBINED_QUERY, maxrecords=GDELT_MAXRECORDS, fmt="json")
        raw = _fetch_feed_bytes(url)
        if raw is None:
            logger.warning(
                "[gdelt] combined fetch failed (429/network) — reusing "
                f"{len(cls._gdelt_cache_entries)} cached entries"
            )
            return cls._gdelt_cache_entries

        try:
            articles = json.loads(raw).get("articles", [])
        except (ValueError, AttributeError) as e:
            logger.warning(f"[gdelt] JSON parse failed ({e}) — keeping cache")
            return cls._gdelt_cache_entries

        entries = []
        for a in articles:
            title = (a.get("title") or "").strip()
            if not title:
                continue
            # seendate "YYYYMMDDTHHMMSSZ" → ISO "YYYY-MM-DD HH:MM:SS" (parseable
            # by _parse_published). Falls back to empty (→ stamped "now") if odd.
            seen = a.get("seendate", "")
            try:
                pub = datetime.strptime(seen, "%Y%m%dT%H%M%SZ").isoformat(sep=" ", timespec="seconds")
            except (ValueError, TypeError):
                pub = ""
            entries.append({
                "title":     title,
                "summary":   "",            # DOC API returns headline only
                "published": pub,
                "link":      a.get("url", ""),
            })

        if entries:
            cls._gdelt_cache_entries = entries
            cls._gdelt_cache_ts      = time.monotonic()
            logger.info(f"[gdelt] combined universe fetch: {len(entries)} entries cached")
        else:
            logger.warning("[gdelt] combined fetch returned 0 entries — keeping cache")
        return cls._gdelt_cache_entries

    def _fetch_feed(self, source_name: str, feed_url: str) -> int:
        """Fetch one source's entries and ingest the relevant+recent ones.

        GDELT uses the shared class-level combined-query cache (one request per
        cycle for all banks). All other feeds fetch their own RSS with a
        browser-like UA (so VPS IPs aren't blocked), falling back to bare
        feedparser only if the browser-UA fetch fails entirely.
        """
        if source_name == "gdelt":
            entries = self._get_gdelt_entries()
            return self._ingest_entries(source_name, entries)

        raw = _fetch_feed_bytes(feed_url)
        if raw is not None:
            feed = feedparser.parse(raw)
        else:
            logger.info(f"[{self.ticker}][{source_name}] browser fetch failed, trying feedparser direct")
            feed = feedparser.parse(feed_url)

        n_entries   = len(feed.entries)
        http_status = getattr(feed, "status", "local")
        if n_entries == 0:
            logger.warning(
                f"[{self.ticker}][{source_name}] 0 entries "
                f"(HTTP {http_status}, bozo={feed.bozo})"
                + (f" bozo_exc={feed.bozo_exception}" if feed.bozo else "")
            )
        else:
            logger.debug(
                f"[{self.ticker}][{source_name}] {n_entries} entries (HTTP {http_status})"
            )
        return self._ingest_entries(source_name, feed.entries)

    def _ingest_entries(self, source_name: str, entries: list) -> int:
        """Filter entries by this bank's keywords + recency and insert new rows.

        Shared by RSS feeds and the GDELT combined cache — `entries` is any list
        of objects supporting `.get("title"/"summary"/"published"/"link")`.
        """
        conn      = self._connect()
        keywords  = BANK_NEWS_CONFIG[self.ticker]["keywords"]
        new_count = 0
        skipped_old = 0
        now       = datetime.now()
        cutoff    = now - timedelta(days=MAX_ARTICLE_AGE_DAYS)

        for entry in entries:
            title   = entry.get("title", "").strip()
            summary = self._clean_html(entry.get("summary", ""))
            if not title:
                continue
            if not self._is_relevant(title + " " + summary, keywords):
                continue

            published_raw = entry.get("published", "")
            published_dt  = self._parse_published(published_raw)
            # No parseable date → assume it's a freshly published item from a
            # feed that omits dates (rare). Keep it, stamped as "now".
            effective_dt  = published_dt or now
            if effective_dt < cutoff:
                skipped_old += 1
                continue

            is_important   = self._is_important(title + " " + summary)
            link           = entry.get("link", "")
            published_iso  = effective_dt.isoformat(sep=" ", timespec="seconds")

            try:
                conn.execute("""
                    INSERT OR IGNORE INTO news
                    (title, summary, published, published_iso, source, link,
                     ticker, is_important, processed, llm_analyzed, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?)
                """, (
                    title,
                    summary[:1000],
                    published_raw or published_iso,
                    published_iso,
                    source_name,
                    link,
                    self.ticker,
                    int(is_important),
                    str(now),
                ))
                if conn.execute("SELECT changes()").fetchone()[0] > 0:
                    new_count += 1
            except Exception as e:
                logger.warning(f"Insert error '{title[:50]}': {e}")

        conn.commit()
        conn.close()
        if skipped_old:
            logger.debug(
                f"[{self.ticker}][{source_name}] skipped {skipped_old} "
                f"articles older than {MAX_ARTICLE_AGE_DAYS}d"
            )
        return new_count

    def _is_relevant(self, text: str, keywords: List[str]) -> bool:
        text_lower = text.lower()
        return any(kw in text_lower for kw in keywords)

    def _is_important(self, text: str) -> bool:
        text_lower = text.lower()
        return any(kw in text_lower for kw in HIGH_IMPORTANCE_KEYWORDS)

    def _clean_html(self, text: str) -> str:
        clean = re.sub(r"<[^>]+>", " ", text)
        return re.sub(r"\s+", " ", clean).strip()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.create_aggregate("STDEV", 1, _StdDev)
        return conn

    def _setup_db(self) -> None:
        """Create/migrate news table with ticker + published_iso columns."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS news (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                title           TEXT    UNIQUE,
                summary         TEXT,
                published       TEXT,
                published_iso   TEXT,
                source          TEXT,
                link            TEXT,
                ticker          TEXT    DEFAULT 'HDFCBANK.NS',
                sentiment_score REAL    DEFAULT 0.0,
                llm_score       REAL    DEFAULT 0.0,
                is_important    INTEGER DEFAULT 0,
                processed       INTEGER DEFAULT 0,
                llm_analyzed    INTEGER DEFAULT 0,
                created_at      TEXT
            )
        """)
        # Migration: add ticker column to existing tables
        try:
            conn.execute("ALTER TABLE news ADD COLUMN ticker TEXT DEFAULT 'HDFCBANK.NS'")
            conn.execute("UPDATE news SET ticker = 'HDFCBANK.NS' WHERE ticker IS NULL")
            conn.commit()
            logger.info("news table: added ticker column")
        except Exception:
            pass  # column already exists
        # Migration: add published_iso column for proper date sorting/filtering
        try:
            conn.execute("ALTER TABLE news ADD COLUMN published_iso TEXT")
            conn.commit()
            logger.info("news table: added published_iso column")
        except Exception:
            pass  # column already exists
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_news_created
            ON news (created_at DESC)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_news_processed
            ON news (processed, created_at DESC)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_news_ticker
            ON news (ticker, created_at DESC)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_news_published
            ON news (ticker, published_iso DESC)
        """)
        conn.commit()
        conn.close()


# ── SQLite STDEV aggregate (not built-in) ────────────────────────────────────
class _StdDev:
    def __init__(self):
        self.values = []

    def step(self, value):
        if value is not None:
            self.values.append(float(value))

    def finalize(self):
        if len(self.values) < 2:
            return 0.0
        mean     = sum(self.values) / len(self.values)
        variance = sum((x - mean) ** 2 for x in self.values) / len(self.values)
        return variance ** 0.5
