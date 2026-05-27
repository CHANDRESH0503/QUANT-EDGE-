# processing/finbert_sentiment.py
# Scores news articles using FinBERT — finance-specific BERT model
# Far superior to keyword matching for financial text
# Understands: "NPA declining" = positive, "margin pressure" = negative

import sqlite3
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

try:
    # Prevent loky/multiprocessing semaphore leak on Python 3.14 / macOS.
    # Must be set before importing tokenizers (pulled in by transformers).
    import os as _os
    _os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _os.environ.setdefault("OMP_NUM_THREADS", "1")
    _os.environ.setdefault("MKL_NUM_THREADS", "1")
    from transformers import pipeline, Pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not installed. pip install transformers torch")


class FinBERTSentiment:
    """
    Scores financial news using ProsusAI/finbert.

    Why FinBERT over basic sentiment:
    - Trained on 10,000+ financial news articles
    - Understands finance-specific language
    - "NPA declining" → positive (generic model says negative)
    - "Rate hike" → context-aware for banking stocks
    - Returns positive/negative/neutral with confidence score

    Integration with data layer:
    - Reads unprocessed articles from news table (news_fetcher.py)
    - Updates each row with sentiment_score after processing
    - Aggregates daily/hourly sentiment for feature building
    - Feeds sentiment features into all three ML models

    Score range: -1.0 (very negative) to +1.0 (very positive)
    """

    MODEL_NAME = "ProsusAI/finbert"

    # Class-level pipeline cache — one FinBERT model shared across every
    # FinBERTSentiment(ticker=...) instance in this process. Loading the
    # model is ~1.6 GB RAM + 30 s, so per-instance loading turns a 5-bank
    # orchestrator into an OOM. With this cache, RAM stays at ~1.6 GB
    # regardless of how many tickers we serve.
    _shared_pipeline: Optional["Pipeline"] = None

    def __init__(
        self,
        db_path: str = "database/trading.db",
        ticker:  str = "HDFCBANK.NS",
    ):
        self.db_path = db_path
        self.ticker  = ticker

    # ─────────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────────

    def process_unscored(self, batch_size: int = 50) -> int:
        """
        Score all unprocessed news articles in the database.
        Uses FinBERT when available, falls back to keyword scoring.
        Called every 15 minutes after news_fetcher.fetch_all().
        Returns count of articles processed.

        Round-robin by ticker (2026-05-26): the previous global
        `ORDER BY created_at DESC LIMIT 50` starved smaller banks whenever
        HDFCBANK news volume dominated. We now query per-ticker with an even
        share of `batch_size` (max(5, batch_size//N)) so every bank's pipeline
        sees fresh FinBERT scores within a single 15-min cycle.

        Also rescores articles that were previously scored 0.0 by the old
        bucketed approach (pre continuous-scoring fix), limiting to the last
        30 days so we don't reprocess the entire 6-year archive. Only runs
        the rescore pass when FinBERT is available (not keyword fallback).
        """
        model = self._get_pipeline()   # None when transformers not installed
        use_keyword_fallback = (model is None)
        if use_keyword_fallback:
            logger.info("FinBERT unavailable — using keyword scorer")

        conn  = self._connect()

        # Discover the distinct ticker set from the unscored backlog so the
        # round-robin works whether the table has 1 ticker or 50.
        ticker_rows = conn.execute("""
            SELECT ticker, COUNT(*)
            FROM   news
            WHERE  processed = 0
            GROUP  BY ticker
        """).fetchall()
        tickers = [r[0] or "HDFCBANK.NS" for r in ticker_rows] or ["HDFCBANK.NS"]
        per_ticker = max(5, batch_size // len(tickers))

        # Pass 1: unprocessed articles, balanced per ticker.
        rows = []
        for t in tickers:
            tr = conn.execute("""
                SELECT id, title, summary
                FROM   news
                WHERE  processed = 0 AND ticker = ?
                ORDER  BY created_at DESC
                LIMIT  ?
            """, (t, per_ticker)).fetchall()
            rows.extend(tr)
            if len(rows) >= batch_size:
                rows = rows[:batch_size]
                break

        # Pass 2: articles scored exactly 0.0 by old bucketed method (FinBERT only)
        # Continuous scorer produces non-zero for nearly all articles; exact 0.0
        # means either a genuinely flat article (rare) or a stale bucketed score.
        # We re-run them through FinBERT so they get proper continuous scores.
        # Round-robin here too so a small bank with a few stale rows isn't
        # crowded out by HDFC's archive.
        rescore_rows = []
        if not use_keyword_fallback:
            for t in tickers:
                tr = conn.execute("""
                    SELECT id, title, summary
                    FROM   news
                    WHERE  processed = 1 AND ticker = ?
                      AND  ABS(sentiment_score) < 0.001
                      AND  created_at > datetime('now', '-30 days')
                    ORDER  BY created_at DESC
                    LIMIT  ?
                """, (t, per_ticker)).fetchall()
                rescore_rows.extend(tr)
                if len(rescore_rows) >= batch_size:
                    rescore_rows = rescore_rows[:batch_size]
                    break

        if not rows and not rescore_rows:
            conn.close()
            return 0

        processed = 0
        for row in rows:
            article_id, title, summary = row[0], row[1], row[2] or ""
            text = f"{title}. {summary}"[:512]
            score = self._keyword_score(text) if use_keyword_fallback else self._score_text(model, text)
            conn.execute("""
                UPDATE news
                SET    sentiment_score = ?, processed = 1
                WHERE  id = ?
            """, (score, article_id))
            processed += 1

        rescored = 0
        for row in rescore_rows:
            article_id, title, summary = row[0], row[1], row[2] or ""
            text  = f"{title}. {summary}"[:512]
            score = self._score_text(model, text)
            # Only overwrite if FinBERT actually produced a non-zero value;
            # if it returns 0.0 again the article is genuinely neutral.
            if abs(score) >= 0.001:
                conn.execute("""
                    UPDATE news SET sentiment_score = ? WHERE id = ?
                """, (score, article_id))
                rescored += 1

        conn.commit()
        conn.close()
        method = 'Keyword' if use_keyword_fallback else 'FinBERT'
        logger.info(f"{method} scored {processed} new + {rescored} rescored articles")
        return processed + rescored

    def score_text(self, text: str) -> float:
        """
        Score a single piece of text. Used by LLM analyzer and BSE fetcher
        for immediate scoring of important announcements.
        """
        model = self._get_pipeline()
        if model is None:
            return 0.0
        return self._score_text(model, text[:512])

    def get_daily_sentiment(self, hours: int = 24) -> Dict:
        """
        Aggregate sentiment features for the last N hours of *published* news.

        Live signal generation filters by COALESCE(published_iso, created_at)
        so an article published 3 days ago but freshly ingested today doesn't
        pollute the 24h window. Legacy rows without published_iso fall back
        to created_at. (Backtest path get_daily_sentiment_at() still uses
        created_at <= as_of_date to avoid lookahead bias.)

        Returns:
            score        : mean sentiment score (-1 to +1)
            score_3d     : 3-day rolling mean (trend)
            momentum     : today vs yesterday improvement
            count        : number of articles
            std          : disagreement between sources
            positive_ratio: fraction of bullish articles
            quality      : GOOD / LOW / NO_NEWS
            news_spike   : is article volume 2.5x normal?
        """
        since  = datetime.now() - timedelta(hours=hours)
        conn   = self._connect()

        # ts = effective publication timestamp (publication first, ingest fallback)
        rows   = conn.execute("""
            SELECT sentiment_score,
                   COALESCE(published_iso, created_at) AS ts,
                   source
            FROM   news
            WHERE  processed = 1 AND ticker = ?
            AND    COALESCE(published_iso, created_at) > ?
            ORDER  BY ts DESC
        """, (self.ticker, str(since))).fetchall()

        three_days_ago = datetime.now() - timedelta(hours=72)
        rows_3d = conn.execute("""
            SELECT sentiment_score
            FROM   news
            WHERE  processed = 1 AND ticker = ?
            AND    COALESCE(published_iso, created_at) > ?
        """, (self.ticker, str(three_days_ago))).fetchall()

        avg_daily = conn.execute("""
            SELECT COUNT(*) / 7.0
            FROM   news
            WHERE  ticker = ?
              AND  COALESCE(published_iso, created_at) > ?
        """, (self.ticker, str(datetime.now() - timedelta(days=7)))).fetchone()[0]

        conn.close()

        if not rows:
            return self._empty_sentiment()

        scores     = [r[0] for r in rows if r[0] is not None]
        scores_3d  = [r[0] for r in rows_3d if r[0] is not None]

        if not scores:
            return self._empty_sentiment()

        count        = len(scores)
        avg_score    = float(np.mean(scores))
        avg_3d       = float(np.mean(scores_3d)) if scores_3d else avg_score
        std_score    = float(np.std(scores)) if len(scores) > 1 else 0.0
        pos_ratio    = sum(1 for s in scores if s > 0.2) / count
        neg_ratio    = sum(1 for s in scores if s < -0.2) / count

        # Momentum: is today's sentiment better than yesterday's?
        yesterday    = datetime.now() - timedelta(hours=24)
        prev_scores  = [r[0] for r in rows if str(r[1]) < str(yesterday) and r[0]]
        curr_scores  = [r[0] for r in rows if str(r[1]) >= str(yesterday) and r[0]]
        momentum     = 0.0
        if prev_scores and curr_scores:
            momentum = float(np.mean(curr_scores) - np.mean(prev_scores))

        # News spike: is today's volume unusual?
        avg_daily    = max(float(avg_daily or 1), 1.0)
        news_spike   = count / avg_daily

        return {
            "score":          round(avg_score, 4),
            "score_3d":       round(avg_3d, 4),
            "momentum":       round(momentum, 4),
            "count":          count,
            "std":            round(std_score, 4),
            "positive_ratio": round(pos_ratio, 3),
            "negative_ratio": round(neg_ratio, 3),
            "news_spike":     round(news_spike, 2),
            "quality":        "GOOD" if count >= 3 else "LOW",
            "signal":         self._score_to_signal(avg_score),
        }

    def get_daily_sentiment_at(self, as_of_date: str, hours: int = 24) -> Dict:
        """
        Point-in-time sentiment aggregation for training/backtest.
        Same shape as get_daily_sentiment() but the window is
        [as_of_date - hours, as_of_date] using DB created_at,
        so historical rows never use future news.
        """
        try:
            anchor = datetime.strptime(as_of_date[:10], "%Y-%m-%d")
        except ValueError:
            anchor = datetime.now()

        window_start = anchor - timedelta(hours=hours)
        window_3d    = anchor - timedelta(hours=72)
        avg_window   = anchor - timedelta(days=7)

        conn = self._connect()

        rows = conn.execute("""
            SELECT sentiment_score, created_at
            FROM   news
            WHERE  processed = 1 AND ticker = ?
              AND  created_at >  ?
              AND  created_at <= ?
            ORDER  BY created_at DESC
        """, (self.ticker, str(window_start), str(anchor))).fetchall()

        rows_3d = conn.execute("""
            SELECT sentiment_score
            FROM   news
            WHERE  processed = 1 AND ticker = ?
              AND  created_at >  ?
              AND  created_at <= ?
        """, (self.ticker, str(window_3d), str(anchor))).fetchall()

        avg_daily = conn.execute("""
            SELECT COUNT(*) / 7.0
            FROM   news
            WHERE  ticker = ?
              AND  created_at >  ?
              AND  created_at <= ?
        """, (self.ticker, str(avg_window), str(anchor))).fetchone()[0]

        conn.close()

        if not rows:
            return self._empty_sentiment()

        scores    = [r[0] for r in rows    if r[0] is not None]
        scores_3d = [r[0] for r in rows_3d if r[0] is not None]
        if not scores:
            return self._empty_sentiment()

        count     = len(scores)
        avg_score = float(np.mean(scores))
        avg_3d    = float(np.mean(scores_3d)) if scores_3d else avg_score
        std_score = float(np.std(scores)) if len(scores) > 1 else 0.0
        pos_ratio = sum(1 for s in scores if s > 0.2) / count
        neg_ratio = sum(1 for s in scores if s < -0.2) / count

        yesterday   = anchor - timedelta(hours=24)
        prev_scores = [r[0] for r in rows if str(r[1]) <  str(yesterday) and r[0]]
        curr_scores = [r[0] for r in rows if str(r[1]) >= str(yesterday) and r[0]]
        momentum    = 0.0
        if prev_scores and curr_scores:
            momentum = float(np.mean(curr_scores) - np.mean(prev_scores))

        avg_daily  = max(float(avg_daily or 1), 1.0)
        news_spike = count / avg_daily

        return {
            "score":          round(avg_score, 4),
            "score_3d":       round(avg_3d, 4),
            "momentum":       round(momentum, 4),
            "count":          count,
            "std":            round(std_score, 4),
            "positive_ratio": round(pos_ratio, 3),
            "negative_ratio": round(neg_ratio, 3),
            "news_spike":     round(news_spike, 2),
            "quality":        "GOOD" if count >= 3 else "LOW",
            "signal":         self._score_to_signal(avg_score),
        }

    def get_sentiment_features_for_ml_at(self, as_of_date: str) -> Dict:
        """Same shape as get_sentiment_features_for_ml() but point-in-time."""
        s24 = self.get_daily_sentiment_at(as_of_date, hours=24)
        s72 = self.get_daily_sentiment_at(as_of_date, hours=72)
        momentum_3d = round(s24["score"] - s72["score"], 4)
        return {
            "finbert_score_24h":      s24["score"],
            "finbert_score_72h":      s72["score"],
            "finbert_momentum":       s24["momentum"],
            "finbert_std":            s24["std"],
            "finbert_positive_ratio": s24["positive_ratio"],
            "finbert_news_count":     s24["count"],
            "finbert_news_spike":     s24["news_spike"],
            "finbert_score_trend":    momentum_3d,
            "finbert_momentum_3d":    momentum_3d,   # alpha feature (alias)
        }

    def get_sentiment_features_for_ml(self) -> Dict:
        """
        Returns the exact feature dict fed into XGBoost models.
        Combines 24h and 72h windows for trend detection.
        """
        s24  = self.get_daily_sentiment(hours=24)
        s72  = self.get_daily_sentiment(hours=72)
        momentum_3d = round(s24["score"] - s72["score"], 4)

        return {
            "finbert_score_24h":      s24["score"],
            "finbert_score_72h":      s72["score"],
            "finbert_momentum":       s24["momentum"],
            "finbert_std":            s24["std"],
            "finbert_positive_ratio": s24["positive_ratio"],
            "finbert_news_count":     s24["count"],
            "finbert_news_spike":     s24["news_spike"],
            "finbert_score_trend":    momentum_3d,
            # Alpha feature (2026-05-26): explicit name for `score_24h − score_72h`
            # sentiment momentum. Same value as finbert_score_trend; alias kept
            # for readability in feature lists.
            "finbert_momentum_3d":    momentum_3d,
        }

    # ─────────────────────────────────────────────────────────────
    # PRIVATE
    # ─────────────────────────────────────────────────────────────

    def _score_text(self, model: "Pipeline", text: str) -> float:
        """
        Run FinBERT on text, return continuous -1 to +1 score.

        Critical change: use `p_positive - p_negative` from the full
        probability distribution instead of bucketing on the top label.
        FinBERT often classifies finance headlines as "neutral" with
        moderate confidence — but the underlying probs still lean
        positive or negative. Throwing them away (returning 0.0) kills
        ~50% of available sentiment signal. Continuous scoring keeps
        that information at proportional magnitude.

        Audit (2026-05-22): 6136/11371 articles were scored 0.0 with the
        bucketed approach; continuous scoring brings the bulk of these
        into the ±0.1-0.4 range — small but real edge for ML features.
        """
        try:
            # top_k=None returns scores for ALL labels; older versions of
            # transformers used return_all_scores=True. Try both for
            # back-compat.
            try:
                results = model(text, top_k=None)
            except TypeError:
                results = model(text, return_all_scores=True)

            # results is List[List[Dict]] for batched or List[Dict] for single
            scores = results[0] if (results and isinstance(results[0], list)) else results
            probs  = {r["label"].lower(): float(r["score"]) for r in scores}
            p_pos  = probs.get("positive", 0.0)
            p_neg  = probs.get("negative", 0.0)
            # Continuous score: positive lean → +, negative lean → −.
            # Magnitude reflects model conviction; ambiguous headlines
            # get small values close to 0 rather than exactly 0.
            return round(p_pos - p_neg, 4)
        except Exception as e:
            logger.warning(f"FinBERT scoring failed: {e}")
            return 0.0

    def _get_pipeline(self) -> Optional["Pipeline"]:
        """Lazy-load FinBERT pipeline. Shared across all instances in the
        process via FinBERTSentiment._shared_pipeline — see class docstring."""
        if FinBERTSentiment._shared_pipeline is not None:
            return FinBERTSentiment._shared_pipeline

        if not TRANSFORMERS_AVAILABLE:
            logger.error("transformers not available")
            return None

        try:
            logger.info("Loading FinBERT model (first load may take 30s)...")
            # Pin torch to 1 intra-op thread so it never spawns loky workers
            # that leak semaphores on Python 3.14 / macOS at shutdown.
            try:
                import torch
                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
            except Exception:
                pass
            FinBERTSentiment._shared_pipeline = pipeline(
                "sentiment-analysis",
                model=self.MODEL_NAME,
                revision="main",    # pin to stable branch; prevents PR-branch checks
                device=-1,          # CPU — change to 0 for GPU
                truncation=True,
                max_length=512,
            )
            logger.info("FinBERT loaded successfully (shared across tickers)")
            return FinBERTSentiment._shared_pipeline
        except Exception as e:
            logger.error(f"FinBERT load failed: {e}")
            return None

    # ─────────────────────────────────────────────────────────────
    # KEYWORD FALLBACK (used when transformers not installed)
    # ─────────────────────────────────────────────────────────────

    _BULLISH_WORDS = [
        "profit", "growth", "surge", "beat", "record", "rally", "strong",
        "upgrade", "outperform", "buy", "positive", "gain", "rise", "boost",
        "increase", "expand", "recovery", "momentum", "bullish", "upside",
        "npa declining", "margin expansion", "loan growth", "rate cut",
        "net interest income", "roe improving", "dividend", "acquisition win",
    ]
    _BEARISH_WORDS = [
        "loss", "decline", "fall", "drop", "miss", "weak", "concern",
        "downgrade", "underperform", "sell", "negative", "risk", "sell-off",
        "decrease", "contract", "slump", "bearish", "downside", "npa",
        "margin pressure", "npa rising", "bad loan", "default", "rbi action",
        "fraud", "penalty", "investigation", "rate hike hurt", "credit cost",
    ]

    def _keyword_score(self, text: str) -> float:
        """
        Lightweight keyword-based sentiment scorer.
        Returns -1.0 to +1.0. Used when FinBERT (transformers) is not installed.
        Less accurate than FinBERT but functional — still feeds the feature pipeline.
        """
        t = text.lower()
        bull = sum(1 for w in self._BULLISH_WORDS if w in t)
        bear = sum(1 for w in self._BEARISH_WORDS if w in t)
        total = bull + bear
        if total == 0:
            return 0.0
        score = (bull - bear) / total           # -1 to +1
        # Scale down slightly — keywords are noisier than FinBERT
        return round(max(-0.8, min(0.8, score * 0.75)), 4)

    def _score_to_signal(self, score: float) -> str:
        if score >  0.5: return "VERY_POSITIVE"
        if score >  0.2: return "POSITIVE"
        if score < -0.5: return "VERY_NEGATIVE"
        if score < -0.2: return "NEGATIVE"
        return "NEUTRAL"

    def _empty_sentiment(self) -> Dict:
        return {
            "score": 0.0, "score_3d": 0.0, "momentum": 0.0,
            "count": 0, "std": 0.0, "positive_ratio": 0.5,
            "negative_ratio": 0.5, "news_spike": 1.0,
            "quality": "NO_NEWS", "signal": "NEUTRAL",
        }

    def _connect(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c
