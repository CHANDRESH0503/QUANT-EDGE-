# data/social_fetcher.py
# Tracks retail sentiment from Twitter/X and Reddit
# Used as CONTRARIAN indicator — extreme retail bullishness = warning signal
# Free tier: Twitter API Basic, Reddit PRAW (free)

import sqlite3
import logging
import re
import time
import os
from dotenv import load_dotenv

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np

load_dotenv()
bearer_token = os.getenv("TWITTER_BEARER")

logger = logging.getLogger(__name__)

# ── Try importing optional social libraries ──────────────────
try:
    import praw
    REDDIT_AVAILABLE = True
except ImportError:
    REDDIT_AVAILABLE = False
    logger.warning("praw not installed — Reddit sentiment disabled. pip install praw")

try:
    import tweepy
    TWITTER_AVAILABLE = True
except ImportError:
    TWITTER_AVAILABLE = False
    logger.warning("tweepy not installed — Twitter sentiment disabled. pip install tweepy")


class SocialFetcher:
    """
    Collects and scores retail social sentiment for HDFC Bank.

    Key insight from 20 years of trading:
    Social sentiment is MOST useful as a contrarian indicator.
    When 85%+ of retail posts are bullish — smart money is often selling.
    When retail is extremely bearish on a quality stock — accumulation likely.

    Sources:
    - Twitter/X: search HDFCBANK mentions (Twitter API Basic tier, free)
    - Reddit: r/IndiaInvestments, r/Dalal_Street_Bets (PRAW, free)

    Runs every 2 hours. Saves raw posts + scored sentiment to SQLite.
    """

    SEARCH_QUERIES = [
        "HDFCBANK", "HDFC Bank stock", "HDFC Bank NSE",
        "#HDFCBANK", "HDFC Bank share price",
    ]

    SUBREDDITS = [
        "IndiaInvestments",
        "Dalal_Street_Bets",
        "IndianStreetBets",
        "stocksIndia",
    ]

    BULLISH_KEYWORDS = [
        "buy", "long", "bullish", "upside", "target", "accumulate",
        "strong", "outperform", "breakout", "support", "bounce",
        "rally", "good results", "beat", "positive", "upgrade",
    ]

    BEARISH_KEYWORDS = [
        "sell", "short", "bearish", "downside", "overvalued", "expensive",
        "weak", "underperform", "breakdown", "resistance", "fall",
        "dump", "miss", "negative", "downgrade", "avoid",
    ]

    def __init__(
        self,
        db_path: str = "database/trading.db",
        twitter_bearer_token: Optional[str] = bearer_token,
        reddit_client_id: Optional[str] = None,
        reddit_client_secret: Optional[str] = None,
        reddit_user_agent: str = "TradingBot/1.0",
    ):
        self.db_path = db_path
        self.twitter_bearer = twitter_bearer_token
        self.reddit_id = reddit_client_id
        self.reddit_secret = reddit_client_secret
        self.reddit_ua = reddit_user_agent
        self._setup_db()

        # Lazy-init clients
        self._twitter_client = None
        self._reddit_client = None

    # ─────────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────────

    def fetch_all(self) -> Dict:
        """
        Fetch Twitter + Reddit, score sentiment, save, return features.
        Called every 2 hours by scheduler.
        """
        twitter_count = self._fetch_twitter() if self._can_use_twitter() else 0
        reddit_count  = self._fetch_reddit()  if self._can_use_reddit()  else 0

        features = self.get_social_features()
        logger.info(
            f"Social: +{twitter_count} tweets, +{reddit_count} posts | "
            f"Bull ratio={features.get('bull_ratio', 0):.0%} | "
            f"Signal={features.get('contrarian_signal', 'NEUTRAL')}"
        )
        return features

    def get_social_features(self) -> Dict:
        """
        Aggregate last 24h posts into ML features.
        Returns bull_ratio, contrarian_signal, activity_score.
        """
        since = datetime.now() - timedelta(hours=24)
        conn  = self._connect()

        rows = conn.execute("""
            SELECT sentiment_label, score, source
            FROM   social_posts
            WHERE  created_at > ?
        """, (str(since),)).fetchall()
        conn.close()

        if not rows:
            return self._empty_social_features()

        scores  = [r[1] for r in rows]
        labels  = [r[0] for r in rows]
        total   = len(labels)
        bullish = labels.count("BULL")
        bearish = labels.count("BEAR")

        bull_ratio = bullish / total if total > 0 else 0.5
        bear_ratio = bearish / total if total > 0 else 0.5
        avg_score  = float(np.mean(scores)) if scores else 0.0

        # Historical baseline for activity spike detection
        avg_daily = self._get_avg_daily_count()
        activity_ratio = total / max(avg_daily, 1)

        # Contrarian signal: extremes are fade opportunities
        if bull_ratio >= 0.85:
            contrarian = "CONTRARIAN_BEAR"   # too much retail euphoria
        elif bear_ratio >= 0.80:
            contrarian = "CONTRARIAN_BULL"   # extreme retail fear
        elif bull_ratio >= 0.65:
            contrarian = "MILD_BULL"
        elif bear_ratio >= 0.65:
            contrarian = "MILD_BEAR"
        else:
            contrarian = "NEUTRAL"

        return {
            "total_posts_24h":     total,
            "bull_count":          bullish,
            "bear_count":          bearish,
            "bull_ratio":          round(bull_ratio, 3),
            "bear_ratio":          round(bear_ratio, 3),
            "avg_sentiment_score": round(avg_score, 3),
            "activity_ratio":      round(activity_ratio, 2),
            "contrarian_signal":   contrarian,
            # Normalised feature for ML (-1 = extreme bull = bearish for us)
            "social_ml_score":     round(self._contrarian_score(bull_ratio), 3),
        }

    # ─────────────────────────────────────────────────────────────
    # TWITTER
    # ─────────────────────────────────────────────────────────────

    def _fetch_twitter(self) -> int:
        """Fetch recent tweets mentioning HDFC Bank."""
        client = self._get_twitter_client()
        if client is None:
            return 0

        saved = 0
        for query in self.SEARCH_QUERIES[:2]:  # limit API calls
            try:
                tweets = client.search_recent_tweets(
                    query=f"{query} -is:retweet lang:en",
                    max_results=50,
                    tweet_fields=["created_at", "public_metrics"],
                )
                if tweets.data:
                    for tweet in tweets.data:
                        text  = tweet.text
                        label, score = self._score_text(text)
                        self._save_post(text, label, score, "twitter")
                        saved += 1
                time.sleep(1)  # rate limit respect
            except Exception as e:
                logger.warning(f"Twitter fetch error for '{query}': {e}")

        return saved

    # ─────────────────────────────────────────────────────────────
    # REDDIT
    # ─────────────────────────────────────────────────────────────

    def _fetch_reddit(self) -> int:
        """Fetch recent Reddit posts from Indian investing subreddits."""
        client = self._get_reddit_client()
        if client is None:
            return 0

        saved  = 0
        cutoff = datetime.now() - timedelta(hours=24)

        for sub_name in self.SUBREDDITS:
            try:
                subreddit = client.subreddit(sub_name)
                for post in subreddit.new(limit=50):
                    created = datetime.fromtimestamp(post.created_utc)
                    if created < cutoff:
                        break

                    text = f"{post.title} {post.selftext}"
                    if not self._is_hdfc_related(text):
                        continue

                    label, score = self._score_text(text)
                    self._save_post(text[:500], label, score, f"reddit_{sub_name}")
                    saved += 1

                time.sleep(0.5)
            except Exception as e:
                logger.warning(f"Reddit fetch error for r/{sub_name}: {e}")

        return saved

    # ─────────────────────────────────────────────────────────────
    # SCORING
    # ─────────────────────────────────────────────────────────────

    def _score_text(self, text: str):
        """
        Simple keyword-based sentiment scoring.
        Not as accurate as FinBERT but fast and free.
        FinBERT is used for news articles (higher quality text).
        Social posts are informal — keyword scoring is sufficient.
        Returns (label, score) where score is -1.0 to +1.0.
        """
        text_lower = text.lower()
        bull_hits = sum(1 for kw in self.BULLISH_KEYWORDS if kw in text_lower)
        bear_hits = sum(1 for kw in self.BEARISH_KEYWORDS if kw in text_lower)

        if bull_hits == 0 and bear_hits == 0:
            return "NEUTRAL", 0.0

        total = bull_hits + bear_hits
        score = (bull_hits - bear_hits) / total

        if score > 0.2:
            label = "BULL"
        elif score < -0.2:
            label = "BEAR"
        else:
            label = "NEUTRAL"

        return label, round(score, 3)

    def _is_hdfc_related(self, text: str) -> bool:
        """Check if post mentions HDFC Bank."""
        text_lower = text.lower()
        return any(kw in text_lower for kw in [
            "hdfc", "hdfcbank", "hdfc bank", "500180"
        ])

    def _contrarian_score(self, bull_ratio: float) -> float:
        """
        Convert bull ratio to contrarian ML score.
        0.85 bull_ratio → -0.85 (very bullish retail = bearish signal for us)
        0.15 bull_ratio → +0.70 (very bearish retail = bullish signal)
        0.50 bull_ratio → 0.0  (neutral)
        This inverts retail sentiment for use as a contrarian feature.
        """
        return round(-(bull_ratio - 0.5) * 2, 3)

    def _get_avg_daily_count(self) -> float:
        """Get average daily post count from last 7 days for spike detection."""
        week_ago = datetime.now() - timedelta(days=7)
        conn = self._connect()
        row = conn.execute("""
            SELECT COUNT(*) / 7.0 FROM social_posts WHERE created_at > ?
        """, (str(week_ago),)).fetchone()
        conn.close()
        return float(row[0]) if row else 10.0

    def _empty_social_features(self) -> Dict:
        return {
            "total_posts_24h": 0, "bull_count": 0, "bear_count": 0,
            "bull_ratio": 0.5, "bear_ratio": 0.5,
            "avg_sentiment_score": 0.0, "activity_ratio": 1.0,
            "contrarian_signal": "NEUTRAL", "social_ml_score": 0.0,
        }

    # ─────────────────────────────────────────────────────────────
    # CLIENT INIT
    # ─────────────────────────────────────────────────────────────

    def _get_twitter_client(self):
        if self._twitter_client:
            return self._twitter_client
        if not TWITTER_AVAILABLE or not self.twitter_bearer:
            return None
        try:
            self._twitter_client = tweepy.Client(
                bearer_token=self.twitter_bearer,
                wait_on_rate_limit=True,
            )
            return self._twitter_client
        except Exception as e:
            logger.error(f"Twitter client init failed: {e}")
            return None

    def _get_reddit_client(self):
        if self._reddit_client:
            return self._reddit_client
        if not REDDIT_AVAILABLE or not self.reddit_id:
            return None
        try:
            self._reddit_client = praw.Reddit(
                client_id=self.reddit_id,
                client_secret=self.reddit_secret,
                user_agent=self.reddit_ua,
            )
            return self._reddit_client
        except Exception as e:
            logger.error(f"Reddit client init failed: {e}")
            return None

    def _can_use_twitter(self) -> bool:
        return TWITTER_AVAILABLE and bool(self.twitter_bearer)

    def _can_use_reddit(self) -> bool:
        return REDDIT_AVAILABLE and bool(self.reddit_id)

    # ─────────────────────────────────────────────────────────────
    # STORAGE
    # ─────────────────────────────────────────────────────────────

    def _save_post(self, text: str, label: str, score: float, source: str) -> None:
        conn = self._connect()
        try:
            conn.execute("""
                INSERT OR IGNORE INTO social_posts
                (text_hash, text_snippet, sentiment_label, score, source, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                str(hash(text[:100])),
                text[:200],
                label, score, source,
                str(datetime.now()),
            ))
            conn.commit()
        except Exception:
            pass
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _setup_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS social_posts (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                text_hash        TEXT    UNIQUE,
                text_snippet     TEXT,
                sentiment_label  TEXT,
                score            REAL,
                source           TEXT,
                created_at       TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_social_created
            ON social_posts (created_at DESC)
        """)
        conn.commit()
        conn.close()