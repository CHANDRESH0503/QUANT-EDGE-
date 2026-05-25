# features/feature_builder.py
# Master orchestrator — assembles the complete feature vector for all 3 ML models
# Connects every module: data/ → processing/ → features/ → models/ → signals/
# This is the single most important file in the entire system

import json
import numpy as np
import pandas as pd
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

# ── Processing layer ───────────────────────────────────────────────────────────
from processing.technical          import TechnicalProcessor
from processing.finbert_sentiment  import FinBERTSentiment
from processing.llm_analyzer       import LLMAnalyzer
from processing.fundamental        import FundamentalProcessor
from processing.macro_analyzer     import MacroAnalyzer
from processing.pattern_detector   import PatternDetector
from processing.anomaly_detector   import AnomalyDetector
from processing.earnings_predictor import EarningsPredictor
from processing.order_flow         import OrderFlowAnalyzer
from processing.support_resistance import SupportResistanceEngine
from processing.intermarket        import IntermarketAnalyzer
from processing.regime_detector    import RegimeDetector

# ── Data layer ─────────────────────────────────────────────────────────────────
from data.price_fetcher       import PriceFetcher
from data.options_fetcher     import OptionsFetcher
from data.fii_fetcher         import FIIFetcher
from data.global_fetcher      import GlobalFetcher
from data.social_fetcher      import SocialFetcher
from data.insider_fetcher     import InsiderFetcher
from data.alternative_fetcher import AlternativeFetcher

# ── Feature assemblers ─────────────────────────────────────────────────────────
from features.technical_features   import TechnicalFeatures
from features.sentiment_features   import SentimentFeatures
from features.fundamental_features import FundamentalFeatures
from features.macro_features       import MacroFeatures
from features.options_features     import OptionsFeatures
from features.flow_features        import FlowFeatures
from features.calendar_features    import CalendarFeatures
from features.alternative_features import AlternativeFeatures
from features.meta_features        import MetaFeatures
from features.risk_features        import RiskFeatures


class FeatureBuilder:
    """
    Master feature assembly pipeline — the brain of the trading system.

    20yr senior trader perspective:
    Every morning I used to check 15 data sources manually before placing
    a single trade. Price action, FII flow, options PCR, global cues,
    sector rotation, fundamental quality, news sentiment — all of it.
    This class automates that entire pre-trade checklist into one
    coherent feature vector built in under 60 seconds.

    The feature vector is the difference between a guess and an edge.
    Without it you are trading on gut feeling. With it, you are trading
    on everything that matters, simultaneously, every single time.

    Architecture:
    ┌─────────────────────────────────────────────────────┐
    │  data/ (9 sources)  →  processing/ (12 processors) │
    │       ↓                                             │
    │  features/ (10 assemblers)  →  90 clean features   │
    │       ↓                                             │
    │  3 model vectors (swing / intraday / positional)    │
    └─────────────────────────────────────────────────────┘

    Design rules:
    1. Every feature is bounded [-1, +1] or [0, 1]
    2. Missing data → 0.0 (neutral), never NaN
    3. No lookahead: only past data used at each point
    4. Features selected by predictive power, not availability
    """

    # ──────────────────────────────────────────────────────────────────────────
    # FEATURE REGISTRIES
    # These lists define EXACTLY what each model sees.
    # Change with care — retraining required after any modification.
    # ──────────────────────────────────────────────────────────────────────────

    SWING_FEATURES = [
        # Technical momentum (6)
        "rsi_14", "stoch_rsi", "rsi_divergence",
        "macd_hist", "macd_hist_roc", "adx",
        # Technical trend (5)
        "ema_spread", "ema_spread_roc", "supertrend",
        "support_dist", "resistance_dist",
        # Technical volatility (5)
        "bb_pct_b", "bb_squeeze", "atr_ratio",
        "hv_ratio", "range_pct_52w",
        # Technical volume (4)
        "obv_roc", "mfi", "volume_ratio", "close_pct_range",
        # Technical price action (3)
        "returns_1d", "returns_5d", "returns_20d",
        # Sentiment (9)
        "finbert_score_24h", "finbert_score_trend",
        "finbert_momentum", "finbert_news_spike",
        "llm_earnings_score", "llm_earnings_relevance",
        "llm_announcement_score", "llm_combined_score",
        "social_contrarian_score",
        # Options (8)
        "pcr_norm", "pcr_roc_norm", "max_pain_dist_norm",
        "iv_percentile_norm", "implied_move_norm",
        "expiry_proximity", "call_wall_dist_norm", "put_wall_dist_norm",
        # Flow (8)
        "fii_5d_norm", "fii_signal", "dii_3d_norm",
        "flow_confluence", "insider_score",
        "order_flow_score", "delta_divergence", "institutional_bias",
        # Macro (8)
        "india_vix_norm", "rbi_proximity_norm", "rbi_cutting_cycle",
        "dollar_stress", "rupee_stress",
        "yield_headwind", "macro_score", "intermarket_score",
        # Intermarket (5)
        "rs_vs_banknifty", "banks_outperforming_nifty",
        "peer_rank_norm", "jpmorgan_norm", "crude_risk",
        # Calendar (8)
        "earnings_proximity", "earnings_risk_flag",
        "pre_earnings_drift", "earnings_beat_prob",
        "is_expiry_week", "is_monday",
        "month_seasonality", "holiday_proximity",
        # Pattern (4)
        "pattern_score", "chart_golden_cross",
        "chart_volume_breakout", "chart_bb_breakout",
        # Regime (5)
        "regime_bull", "regime_bear", "regime_high_vol",
        "regime_choppy", "regime_stability",
        # Meta (5)
        "tf_alignment_score", "rolling_accuracy_10",
        "losing_streak_norm", "model_health_score", "avg_model_confidence",
    ]  # Total: 83 features

    # ──────────────────────────────────────────────────────────────────────────
    # PRODUCTION FEATURE SETS  (verified non-zero in 1,500+ live snapshots)
    # Drops: DII daily, FII daily, insider, LLM/social, call/put walls,
    #        chart_* patterns (TA-Lib absent), and all meta features (all 0).
    # Use these for the "prod" training variant — full 25yr history + clean
    # features only, so XGBoost doesn't overfit to structural zero columns.
    # ──────────────────────────────────────────────────────────────────────────

    SWING_FEATURES_PROD = [
        # Technical momentum (6)
        "rsi_14", "stoch_rsi", "rsi_divergence",
        "macd_hist", "macd_hist_roc", "adx",
        # Technical trend (5)
        "ema_spread", "ema_spread_roc", "supertrend",
        "support_dist", "resistance_dist",
        # Technical volatility (5)
        "bb_pct_b", "bb_squeeze", "atr_ratio", "hv_ratio", "range_pct_52w",
        # Technical volume (4)
        "obv_roc", "mfi", "volume_ratio", "close_pct_range",
        # Price action (3)
        "returns_1d", "returns_5d", "returns_20d",
        # Sentiment — FinBERT only (3)
        "finbert_score_24h", "finbert_score_trend", "finbert_news_spike",
        # Options — confirmed non-zero (5)
        "pcr_norm", "max_pain_dist_norm", "iv_percentile_norm",
        "implied_move_norm", "expiry_proximity",
        # Flow — monthly FII only (2)
        "fii_5d_norm", "fii_signal",
        # Macro — all confirmed non-zero (9)
        "india_vix_norm", "rbi_proximity_norm", "rbi_cutting_cycle",
        "dollar_stress", "rupee_stress", "yield_headwind",
        "crude_risk", "macro_score", "intermarket_score",
        # Calendar (5)
        "earnings_proximity", "earnings_risk_flag",
        "is_expiry_week", "is_monday", "holiday_proximity",
        # Regime (5)
        "regime_bull", "regime_bear", "regime_high_vol",
        "regime_choppy", "regime_stability",
        # Pattern (1)
        "pattern_score",
    ]  # Total: 48 features

    INTRADAY_FEATURES_PROD = [
        # Technical — fast momentum (10)
        "rsi_14", "stoch_rsi", "macd_hist", "macd_hist_roc",
        "bb_pct_b", "bb_squeeze", "atr_ratio",
        "volume_ratio", "returns_1d", "close_pct_range",
        # Price levels (4)
        "gap_pct", "support_dist", "resistance_dist",
        "vwap_dist",
        # Sentiment — FinBERT (2)
        "finbert_score_24h", "finbert_news_spike",
        # Options (3)
        "pcr_norm", "max_pain_dist_norm", "expiry_proximity",
        # Calendar (4)
        "is_expiry_week", "is_monday", "earnings_risk_flag", "holiday_proximity",
        # Macro (2)
        "india_vix_norm", "dollar_stress",
        # Pattern (1)
        "pattern_score",
        # Regime (3)
        "regime_bull", "regime_bear", "regime_choppy",
    ]  # Total: 29 features

    POSITIONAL_FEATURES_PROD = [
        # Technical — weekly / long-term trend (9)
        "rsi_14", "adx", "ema_spread", "ema_spread_roc",
        "supertrend", "returns_5d", "returns_20d",
        "range_pct_52w", "hv_ratio",
        # Price levels (2)
        "support_dist", "resistance_dist",
        # Sentiment — FinBERT (2)
        "finbert_score_24h", "finbert_score_trend",
        # Macro / Intermarket (9)
        "india_vix_norm", "rbi_proximity_norm", "rbi_cutting_cycle",
        "dollar_stress", "rupee_stress", "yield_headwind",
        "crude_risk", "macro_score", "intermarket_score",
        # Flow — monthly FII only (2)
        "fii_5d_norm", "fii_signal",
        # Options (3)
        "pcr_norm", "iv_percentile_norm", "implied_move_norm",
        # Calendar (4)
        "earnings_proximity", "earnings_risk_flag",
        "is_expiry_week", "holiday_proximity",
        # Regime (3)
        "regime_bull", "regime_bear", "regime_stability",
    ]  # Total: 34 features

    INTRADAY_FEATURES = [
        # Technical — fast momentum
        "rsi_14", "stoch_rsi", "macd_hist", "macd_hist_roc",
        "bb_pct_b", "bb_squeeze", "atr_ratio",
        "volume_ratio", "returns_1d", "close_pct_range",
        "gap_pct", "support_dist", "resistance_dist",
        # VWAP — most important intraday reference
        "vwap_dist", "price_above_vwap",
        # Order flow — primary intraday signal
        "order_flow_score", "cum_delta_norm", "delta_divergence",
        "absorption_flag", "institutional_bias",
        # Sentiment (same-day)
        "finbert_score_24h", "finbert_news_spike",
        "llm_announcement_score",
        # Options — expiry structure
        "pcr_norm", "max_pain_dist_norm",
        "expiry_proximity", "is_expiry_week", "is_expiry_day",
        # Calendar — intraday timing patterns
        "is_monday", "is_friday", "is_midweek",
        "earnings_risk_flag", "holiday_proximity",
        # Macro — quick directional read
        "india_vix_norm", "dollar_stress",
        # Pattern — intraday setups
        "pattern_score", "chart_inside_bar",
        # Regime
        "regime_bull", "regime_bear", "regime_choppy",
        # Meta
        "tf_alignment_score", "losing_streak_norm",
    ]  # Total: 41 features

    POSITIONAL_FEATURES = [
        # Technical — weekly / long-term trend
        "rsi_14", "adx", "ema_spread", "ema_spread_roc",
        "supertrend", "returns_5d", "returns_20d",
        "range_pct_52w", "hv_ratio",
        "support_dist", "resistance_dist",
        # Fundamental — ONLY positional model uses these
        "nim_norm", "nim_trend", "npa_norm", "npa_trend",
        "casa_norm", "loan_growth_norm", "roe_norm",
        "pb_vs_avg_norm", "beat_streak_norm", "fundamental_score",
        # Sentiment
        "finbert_score_24h", "finbert_score_trend",
        "llm_earnings_score", "llm_earnings_relevance",
        "llm_combined_score",
        # Alternative data — ONLY positional model uses these
        "demand_trend", "home_loan_trend",
        "hiring_trend", "alt_data_score",
        "insider_score", "promoter_trend_score",
        # Macro / Intermarket
        "india_vix_norm", "rbi_proximity_norm", "rbi_cutting_cycle",
        "dollar_stress", "rupee_stress",
        "yield_headwind", "crude_risk",
        "macro_score", "intermarket_score",
        "rs_vs_banknifty", "peer_rank_norm",
        # Flow
        "fii_5d_norm", "fii_signal", "flow_confluence",
        "dii_accumulation",
        # Options
        "pcr_norm", "iv_percentile_norm", "implied_move_norm",
        # Calendar
        "earnings_proximity", "earnings_beat_prob",
        "month_seasonality", "is_q4_india",
        # Regime
        "regime_bull", "regime_bear", "regime_stability",
        # Meta
        "rolling_accuracy_10", "model_health_score",
    ]  # Total: 55 features

    # ──────────────────────────────────────────────────────────────────────────
    # INITIALISATION
    # ──────────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        ticker:  str   = "HDFCBANK.NS",
        capital: float = 100_000.0,
        db_path: str   = "database/trading.db",
    ):
        self.ticker  = ticker
        self.capital = capital
        self.db_path = db_path

        # ── Processing layer ───────────────────────────────────────
        self.technical   = TechnicalProcessor()
        self.finbert     = FinBERTSentiment(db_path, ticker=ticker)
        self.llm         = LLMAnalyzer(db_path)
        self.fundamental = FundamentalProcessor(db_path, ticker=ticker)
        self.macro       = MacroAnalyzer(db_path)
        self.patterns    = PatternDetector()
        self.anomaly     = AnomalyDetector(db_path)
        self.earnings    = EarningsPredictor(db_path, ticker=ticker)
        self.order_flow  = OrderFlowAnalyzer()
        self.sr_engine   = SupportResistanceEngine()
        self.intermarket = IntermarketAnalyzer(db_path)
        self.regime      = RegimeDetector(db_path, ticker=ticker)

        # ── Data layer ─────────────────────────────────────────────
        self.price_fetcher    = PriceFetcher(ticker)
        self.options_fetcher  = OptionsFetcher(db_path)
        self.fii_fetcher      = FIIFetcher(db_path)
        self.global_fetcher   = GlobalFetcher(db_path)
        self.social_fetcher   = SocialFetcher(db_path)
        self.insider_fetcher  = InsiderFetcher(db_path)
        self.alt_fetcher      = AlternativeFetcher(db_path)

        # ── Feature assemblers ─────────────────────────────────────
        self.tech_feats  = TechnicalFeatures()
        self.sent_feats  = SentimentFeatures()
        self.fund_feats  = FundamentalFeatures()
        self.macro_feats = MacroFeatures()
        self.opts_feats  = OptionsFeatures()
        self.flow_feats  = FlowFeatures()
        self.cal_feats   = CalendarFeatures()
        self.alt_feats   = AlternativeFeatures()
        self.meta_feats  = MetaFeatures()
        self.risk_feats  = RiskFeatures()

        self._setup_db()
        logger.info(
            f"FeatureBuilder ready — ticker={ticker} "
            f"capital=₹{capital:,.0f} db={db_path}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # PRIMARY PUBLIC METHOD
    # ──────────────────────────────────────────────────────────────────────────

    def build_all(
        self,
        positional_signal: str   = "FLAT",
        swing_signal:      str   = "FLAT",
        intraday_signal:   str   = "FLAT",
        positional_conf:   float = 0.5,
        swing_conf:        float = 0.5,
        intraday_conf:     float = 0.5,
        signal_uuid:       str   = "",
    ) -> Dict:
        """
        Build the complete feature set for all three models.

        Called every 15 minutes by signal_engine.py.
        Signals from the PREVIOUS cycle are passed in to compute
        meta-features (alignment, confidence agreement).

        Returns:
            {
              raw_features:          dict of all 90+ features
              swing_vector:          np.ndarray for swing XGBoost
              intraday_vector:       np.ndarray for intraday XGBoost
              positional_vector:     np.ndarray for positional XGBoost
              swing_feature_names:   list (same order as vector)
              intraday_feature_names:list
              positional_feature_names:list
              risk_context:          dict — position sizing params
              regime:                dict — current market regime
              sr_levels:             dict — key price levels
              current_price:         float
              atr:                   float
              data_quality:          dict — data source health
              built_at:              str
            }
        """
        t0 = datetime.now()
        logger.info("━" * 55)
        logger.info("FeatureBuilder.build_all() — assembling features")

        # ── STEP 1: Price data ─────────────────────────────────────
        logger.info("[1/10] Price data")
        df_daily    = self.price_fetcher.get_latest_daily(days=300)
        df_intraday = self.price_fetcher.get_latest_intraday()

        if df_daily.empty:
            logger.error("Daily price fetch returned empty — aborting")
            return self._empty_result()

        current_price = float(df_daily["Close"].iloc[-1])

        # ── STEP 2: Technical features ─────────────────────────────
        logger.info("[2/10] Technical processing")
        tech_df      = self.technical.build_features(df_daily)
        tech_feats   = self.tech_feats.extract(tech_df)

        # atr_14 lives in tech_df (raw TechnicalProcessor output) but is not in
        # TechnicalFeatures.FEATURE_NAMES — so tech_feats.get("atr_14") was
        # always falling back to the hardcoded 20.0 default. Pull it directly
        # from tech_df so position sizing uses each bank's actual volatility.
        # Without this, stops on KOTAK (real ATR ₹9) were as wide as stops on
        # AXIS (real ATR ₹27) — a 3× error in risk geometry.
        try:
            atr_14 = float(tech_df["atr_14"].iloc[-1]) if "atr_14" in tech_df.columns else 0.0
            if not (atr_14 > 0):
                atr_14 = max(0.01 * current_price, 1.0)   # 1% of price as last-resort floor
        except Exception:
            atr_14 = max(0.01 * current_price, 1.0)

        # Intraday technical (5-min candles)
        tech_feats_5m = {}
        of_raw        = {}
        if not df_intraday.empty:
            tech_5m_df    = self.technical.build_features(df_intraday)
            tech_feats_5m = self.tech_feats.extract(tech_5m_df)
            of_raw        = self.order_flow.get_order_flow_features(
                df_intraday,
                atr=atr_14,
            )

        # ── STEP 3: Sentiment ──────────────────────────────────────
        logger.info("[3/10] Sentiment (FinBERT + LLM + Social)")
        finbert_raw = self.finbert.get_sentiment_features_for_ml()
        llm_raw     = self.llm.get_llm_features_for_ml()
        social_raw  = self.social_fetcher.get_social_features()
        sent_feats  = self.sent_feats.extract(finbert_raw, llm_raw, social_raw)

        # ── STEP 4: Fundamental ────────────────────────────────────
        logger.info("[4/10] Fundamentals (Screener.in)")
        fund_raw    = self.fundamental.get_fundamental_features()
        fund_feats  = self.fund_feats.extract(fund_raw)

        # ── STEP 5: Macro + Intermarket ───────────────────────────
        logger.info("[5/10] Macro + intermarket")
        macro_raw   = self.macro.get_macro_features()
        im_raw      = self.intermarket.get_intermarket_features(df_daily)
        macro_feats = self.macro_feats.extract(macro_raw, im_raw)

        # ── STEP 6: Options ────────────────────────────────────────
        logger.info("[6/10] Options chain (PCR, max pain, IV)")
        opts_raw    = self.options_fetcher.get_latest()
        expiry_raw  = self.options_fetcher.get_expiry_features()
        opts_feats  = self.opts_feats.extract(opts_raw, expiry_raw, current_price)

        # ── STEP 7: Flow (FII + Insider + Order flow) ─────────────
        logger.info("[7/10] Money flow")
        fii_raw     = self.fii_fetcher.get_flow_features()
        insider_raw = self.insider_fetcher.get_insider_features()
        flow_feats  = self.flow_feats.extract(fii_raw, insider_raw, of_raw)

        # ── STEP 8: Calendar ───────────────────────────────────────
        logger.info("[8/10] Calendar + earnings")
        earn_raw    = self.earnings.get_earnings_features(
            df_daily,
            implied_move_pct=float(opts_raw.get("implied_move_pct", 0)),
        )
        cal_feats   = self.cal_feats.extract(earn_raw, expiry_raw)

        # ── STEP 9: Pattern + Anomaly + S/R + Regime + Alt ────────
        logger.info("[9/10] Pattern / anomaly / S/R / regime / alt data")
        pattern_raw  = self.patterns.get_pattern_features(df_daily)
        anomaly_raw  = self.anomaly.detect(df_daily)
        sr_raw       = self.sr_engine.get_sr_features(
            df_daily,
            max_call_oi_strike=float(opts_raw.get("max_call_oi_strike", 0)),
            max_put_oi_strike=float(opts_raw.get("max_put_oi_strike",  0)),
        )
        regime_result = self.regime.detect(df_daily)
        regime_feats  = self.regime.get_regime_features_for_ml()
        regime_trend  = self.regime.get_recent_regime_trend()

        alt_raw      = self.alt_fetcher.get_alternative_features()
        alt_feats    = self.alt_feats.extract(alt_raw, insider_raw)

        # ── STEP 10: Meta + Risk ───────────────────────────────────
        logger.info("[10/10] Meta + risk features")
        meta_feats   = self.meta_feats.extract(
            positional_signal=positional_signal,
            swing_signal=swing_signal,
            intraday_signal=intraday_signal,
            positional_conf=positional_conf,
            swing_conf=swing_conf,
            intraday_conf=intraday_conf,
            regime_features=regime_feats,
            db_path=self.db_path,
        )
        risk_feats   = self.risk_feats.extract(
            capital=self.capital,
            anomaly_features=anomaly_raw,
            sr_features=sr_raw,
            meta_features=meta_feats,
            db_path=self.db_path,
        )

        # ── ASSEMBLE MASTER FLAT DICT ──────────────────────────────
        raw = {}
        raw.update(tech_feats)
        raw.update(sent_feats)
        raw.update(fund_feats)
        raw.update(macro_feats)
        raw.update(opts_feats)
        raw.update(flow_feats)
        raw.update(cal_feats)
        raw.update(alt_feats)
        raw.update(meta_feats)
        raw.update(risk_feats)
        raw.update(regime_feats)

        # ── Pattern scalars ────────────────────────────────────────
        raw["pattern_score"]         = self._clip(pattern_raw.get("pattern_score", 0) / 100)
        raw["chart_golden_cross"]    = float(pattern_raw.get("chart_golden_cross", 0))
        raw["chart_death_cross"]     = float(pattern_raw.get("chart_death_cross", 0))
        raw["chart_volume_breakout"] = float(pattern_raw.get("chart_volume_breakout", 0))
        raw["chart_bb_breakout"]     = float(pattern_raw.get("chart_bb_breakout", 0))
        raw["chart_inside_bar"]      = float(pattern_raw.get("chart_inside_bar", 0))

        # ── S/R overrides (more accurate than TechnicalProcessor) ─
        raw["support_dist"]    = self._clip(sr_raw.get("support_distance_pct", 2.0) / 5)
        raw["resistance_dist"] = self._clip(sr_raw.get("resistance_distance_pct", 2.0) / 5)

        # ── Intermarket extras ─────────────────────────────────────
        raw["jpmorgan_norm"] = self._clip(im_raw.get("jpmorgan_5d_pct", 0) / 5)
        raw["rs_vs_banknifty"] = self._clip(
            im_raw.get("rs_vs_banknifty_5d", 0) / 3
        )

        # ── Order flow extras for intraday vector ──────────────────
        raw["vwap_dist"]        = self._clip(of_raw.get("vwap_distance", 0) * 10)
        raw["price_above_vwap"] = float(of_raw.get("price_above_vwap", 0))
        raw["cum_delta_norm"]   = self._clip(of_raw.get("cum_delta_norm", 0))
        raw["absorption_flag"]  = float(of_raw.get("absorption_detected", 0))
        raw["gap_pct"]          = float(tech_feats_5m.get("gap_pct",
                                         tech_feats.get("gap_pct", 0)))

        # Intraday calendar extras
        raw["is_midweek"]   = cal_feats.get("is_midweek", 0.0)
        raw["is_expiry_day"]= float(expiry_raw.get("is_expiry_day", 0))

        # ── Gate 2 raw pass-throughs (not normalised, needed as-is) ──
        raw["days_to_rbi"]        = macro_raw.get("days_to_rbi",         30)
        raw["india_vix_level"]    = macro_raw.get("india_vix_level",     15)
        raw["usdinr_5d_pct"]      = macro_raw.get("usdinr_5d_pct",        0)
        raw["nifty_5d_pct"]       = macro_raw.get("nifty_5d_pct",          0)  # now in macro_analyzer
        raw["days_to_earnings"]   = earn_raw.get("days_to_earnings",     99)
        raw["macro_event_mult"]   = macro_raw.get("macro_event_mult",    1.0)
        raw["earnings_size_mult"] = earn_raw.get("earnings_size_mult",   1.0)
        raw["fundamental_grade"]  = fund_raw.get("fundamental_grade", "GOOD")

        # Override regime_* features in raw_features with rule-based classification
        # so the ML model sees the same definition as was used during training.
        # Gate 1 still uses the HMM-based regime_result dict (not raw_features).
        _adx    = float(raw.get("adx",        0))
        _esp    = float(raw.get("ema_spread",  0))
        _r20    = float(raw.get("returns_20d", 0))
        _r5     = float(raw.get("returns_5d",  0))
        _rule_bear = int(_adx > 22 and _esp < -0.01 and _r20 < 0)
        _rule_bull = int(_adx > 22 and _esp >  0.01 and _r20 > 0)
        _rule_hvol = int(_adx > 30 and abs(_r5) > 0.025)
        _rule_chop = int(not _rule_bear and not _rule_bull and not _rule_hvol)
        raw["regime_bear"]      = _rule_bear
        raw["regime_bull"]      = _rule_bull
        raw["regime_high_vol"]  = _rule_hvol
        raw["regime_choppy"]    = _rule_chop
        # Stability from HMM is still meaningful — keep it
        raw["regime_stability"] = float(raw.get("regime_stability", 0.5))

        # Sanitise entire dict — no NaN, no inf
        raw = {
            k: (round(float(v), 6) if np.isfinite(float(v)) else 0.0)
               if isinstance(v, (int, float, np.floating, np.integer))
               else v
            for k, v in raw.items()
        }

        # ── Build model vectors ────────────────────────────────────
        swing_vec   = self._to_vector(raw, self.SWING_FEATURES)
        intra_vec   = self._to_vector(
            {**raw, **{k: tech_feats_5m.get(k, raw.get(k, 0.0))
                       for k in tech_feats_5m}},
            self.INTRADAY_FEATURES,
        )
        pos_vec     = self._to_vector(raw, self.POSITIONAL_FEATURES)

        elapsed = (datetime.now() - t0).total_seconds()
        quality = self._data_quality(raw)
        logger.info(
            f"Feature build done in {elapsed:.1f}s | "
            f"{len(raw)} features | quality={quality['quality']}"
        )

        result = {
            # Vectors for ML models
            "raw_features":              raw,
            "swing_vector":              swing_vec,
            "intraday_vector":           intra_vec,
            "positional_vector":         pos_vec,
            "swing_feature_names":       self.SWING_FEATURES,
            "intraday_feature_names":    self.INTRADAY_FEATURES,
            "positional_feature_names":  self.POSITIONAL_FEATURES,

            # Risk gate — used by signal_engine before placing any trade
            "risk_context": {
                "trading_allowed":       int(risk_feats.get("trading_allowed", 1)),
                "final_size_mult":       float(risk_feats.get("final_size_mult", 1.0)),
                "capital_mode":          risk_feats.get("capital_mode", "FULL"),
                "entry_quality":         risk_feats.get("entry_quality", "B"),
                "monthly_dd_flag":       int(risk_feats.get("monthly_dd_flag", 0)),
                "consecutive_loss_halt": int(risk_feats.get("consecutive_loss_halt", 0)),
                "anomaly_detected":      int(anomaly_raw.get("is_anomaly", False)),
                "anomaly_type":          anomaly_raw.get("anomaly_type", "NONE"),
                "anomaly_severity":      anomaly_raw.get("severity", "LOW"),
                "anomaly_action":        anomaly_raw.get("recommended_action", "NORMAL"),
                # Gate 6 VIX threshold adjustment — MUST be raw level (not normalised)
                "india_vix":             float(macro_raw.get("india_vix_level", 15.0)),
            },

            # Regime context — gate 1 in signal_engine
            "regime": {
                "regime":       regime_result.get("regime", "UNKNOWN"),
                "stability":    float(regime_result.get("stability", 0.5)),
                "trade_long":   bool(regime_result.get("trade_long", True)),
                "trade_short":  bool(regime_result.get("trade_short", False)),
                "position_mult":float(regime_result.get("position_mult", 1.0)),
                "description":  regime_result.get("description", ""),
                "regime_stable":regime_trend.get("stable", True),
                "regime_changes":regime_trend.get("changes", 0),
            },

            # S/R levels — gate 5 in signal_engine + exit engine targets
            "sr_levels": {
                "nearest_support":    float(sr_raw.get("nearest_support", 0)),
                "nearest_resistance": float(sr_raw.get("nearest_resistance", 0)),
                "second_support":     float(sr_raw.get("second_support", 0)),
                "second_resistance":  float(sr_raw.get("second_resistance", 0)),
                "entry_quality":      sr_raw.get("entry_quality", "B"),
                "reward_risk_sr":     float(sr_raw.get("reward_risk_sr", 2.0)),
                "near_support":       int(sr_raw.get("near_support", 0)),
                "near_resistance":    int(sr_raw.get("near_resistance", 0)),
                "near_breakout":      int(sr_raw.get("near_breakout", 0)),
            },

            # Price reference
            "current_price": current_price,
            "atr":           atr_14,   # real bank-specific ATR, not the 20.0 fallback

            # Metadata
            "data_quality":  quality,
            "build_time_sec":round(elapsed, 2),
            "built_at":      str(datetime.now()),
            "signal_uuid":   signal_uuid,
        }

        self._save_snapshot(result)
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # TRAINING DATASET BUILDER
    # ──────────────────────────────────────────────────────────────────────────

    def build_training_dataset(
        self,
        df:           pd.DataFrame,
        model_type:   str   = "swing",
        forward_days: int   = 5,
        threshold:    float = 0.025,
        data_window:  str   = "full_history",
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build labelled historical feature matrix for XGBoost training.

        Senior trader rule on labelling:
        Do not be greedy with the threshold.
        2.5% in 5 days = realistic swing target for HDFC Bank.
        Too tight → too many flat labels, model learns nothing.
        Too wide  → too few labels, class imbalance.

        Labels:
          2 = LONG  (price up  > threshold in forward_days)
          0 = SHORT (price down > threshold)
          1 = FLAT  (within ±threshold)

        Args:
            df:           Full historical OHLCV (daily) from PriceFetcher
            model_type:   "swing" | "intraday" | "positional"
            forward_days: Lookahead for label creation
            threshold:    Min price move to qualify as LONG or SHORT
            data_window:  "full_history"          → 25yr, technical features only
                          "recent_with_features"  → 2020-01-01+, full features
                                                    (options, FII/DII, FinBERT joined
                                                     per-bar via point-in-time getters).

        Lookahead safety (recent_with_features mode):
        Every per-bar getter uses `<= bar_date` (not `<`) intentionally —
        bhavcopy, FII flow, and end-of-day news are published after T's close,
        so a row dated T reflects T's close which is available before T+1's open.
        """
        feature_names = {
            "swing":            self.SWING_FEATURES,
            "swing_prod":       self.SWING_FEATURES_PROD,
            "intraday":         self.INTRADAY_FEATURES,
            "intraday_prod":    self.INTRADAY_FEATURES_PROD,
            "positional":       self.POSITIONAL_FEATURES,
            "positional_prod":  self.POSITIONAL_FEATURES_PROD,
        }.get(model_type, self.SWING_FEATURES)

        join_external = (data_window == "recent_with_features")

        # Trim history for recent-window training
        if join_external:
            cutoff = pd.Timestamp("2020-01-01")
            if hasattr(df.index, "tz") and df.index.tz is not None:
                cutoff = cutoff.tz_localize(df.index.tz)
            df = df[df.index >= cutoff]

        logger.info(
            f"Building {model_type} training set: "
            f"{len(df)} bars | horizon={forward_days}d | "
            f"threshold={threshold:.1%} | window={data_window}"
        )

        min_hist = 60

        try:
            # Build features for the entire dataset in one pass (O(n) not O(n²)).
            # Technical indicators are purely backward-looking so there is no
            # lookahead bias — each row only uses data up to that timestamp.
            tech_df = self.technical.build_features(df)
            if tech_df.empty:
                logger.error("Technical feature build returned empty DataFrame")
                return pd.DataFrame(), pd.Series(dtype=int)
        except Exception as e:
            logger.error(f"Technical feature build failed: {e}")
            return pd.DataFrame(), pd.Series(dtype=int)

        # Pre-compute per-bar regime labels from technical data (backward-looking).
        # Uses ADX + EMA spread + rolling volatility — no HMM, no lookahead bias.
        # Provides regime_bull / regime_bear / regime_high_vol / regime_choppy /
        # regime_stability features across the FULL history so prod models can
        # learn regime-conditional predictions without requiring the HMM at train time.
        _regime_series = self._compute_bulk_regime_features(tech_df)

        # Align feature rows with valid label indices
        rows   = []
        labels = []
        for i in range(min_hist, len(df) - forward_days):
            try:
                row = self.tech_feats.extract(tech_df.iloc[: i + 1])

                # Attach per-bar regime features
                regime_row = _regime_series.iloc[i].to_dict()
                row.update(regime_row)

                # Attach per-bar FII features for full_history when data exists
                bar_idx  = df.index[i]
                bar_date = (bar_idx.strftime("%Y-%m-%d")
                            if hasattr(bar_idx, "strftime") else str(bar_idx)[:10])
                fii_raw  = self.fii_fetcher.get_flow_features_at(bar_date)
                if fii_raw.get("fii_net_5d_cr", 0) != 0 or fii_raw.get("fii_signal", 0) != 0:
                    row.update(self.flow_feats.extract(
                        fii_raw, insider_features={}, order_flow={},
                    ))

                if join_external:
                    bar_idx = df.index[i]
                    bar_date = (
                        bar_idx.strftime("%Y-%m-%d")
                        if hasattr(bar_idx, "strftime") else str(bar_idx)[:10]
                    )
                    bar_close = float(df["Close"].iloc[i])

                    # ── Options (per-bar point-in-time) ──────────────
                    opts_raw   = self.options_fetcher.get_snapshot_at(bar_date)
                    expiry_raw = self.options_fetcher.get_expiry_features_at(bar_date)
                    row.update(self.opts_feats.extract(
                        opts_raw, expiry_raw, current_price=bar_close,
                    ))

                    # ── FII / DII flow (per-bar point-in-time) ──────
                    fii_raw = self.fii_fetcher.get_flow_features_at(bar_date)
                    row.update(self.flow_feats.extract(
                        fii_raw, insider_features={}, order_flow={},
                    ))

                    # ── FinBERT sentiment (per-bar point-in-time) ───
                    row.update(self.finbert.get_sentiment_features_for_ml_at(bar_date))

                fwd   = float(df["Close"].iloc[i + forward_days] / df["Close"].iloc[i] - 1)
                label = 2 if fwd > threshold else (0 if fwd < -threshold else 1)
                rows.append(row)
                labels.append(label)
            except Exception as e:
                logger.debug(f"Training row {i} skipped: {e}")

        if not rows:
            logger.error("Training dataset empty — check price data")
            return pd.DataFrame(), pd.Series(dtype=int)

        X = pd.DataFrame(rows)[
            [f for f in feature_names if f in pd.DataFrame(rows).columns]
        ].fillna(0).replace([np.inf, -np.inf], 0)

        y = pd.Series(labels, name="label")

        label_dist = y.value_counts().to_dict()
        logger.info(
            f"Training set: {len(X)} rows | "
            f"LONG={label_dist.get(2,0)} "
            f"FLAT={label_dist.get(1,0)} "
            f"SHORT={label_dist.get(0,0)}"
        )
        return X, y

    # ──────────────────────────────────────────────────────────────────────────
    # CONVENIENCE WRAPPERS
    # ──────────────────────────────────────────────────────────────────────────

    def get_swing_vector(self) -> Tuple[np.ndarray, List[str]]:
        """Return (vector, feature_names) for swing model. Single call."""
        r = self.build_all()
        return r["swing_vector"], r["swing_feature_names"]

    def get_intraday_vector(self) -> Tuple[np.ndarray, List[str]]:
        r = self.build_all()
        return r["intraday_vector"], r["intraday_feature_names"]

    def get_positional_vector(self) -> Tuple[np.ndarray, List[str]]:
        r = self.build_all()
        return r["positional_vector"], r["positional_feature_names"]

    def update_capital(self, new_capital: float) -> None:
        """Update capital — called after each trade closes."""
        self.capital = new_capital
        logger.info(f"Capital updated to ₹{new_capital:,.0f}")

    # ──────────────────────────────────────────────────────────────────────────
    # INTERNAL UTILITIES
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_bulk_regime_features(self, tech_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-bar regime features from already-built technical indicators.
        Purely backward-looking — no HMM, no future data.

        Rule-based classification:
        - BEAR: ADX > 22 AND EMA spread < -0.01 AND 20d return < 0
        - BULL: ADX > 22 AND EMA spread > +0.01 AND 20d return > 0
        - HIGH_VOL: ADX > 30 AND |5d return| > 2.5%
        - CHOPPY: otherwise

        Stability = rolling 10-bar fraction in same regime (backward).
        """
        adx      = tech_df.get("adx",          pd.Series(20.0, index=tech_df.index))
        ema_sp   = tech_df.get("ema_spread",    pd.Series(0.0,  index=tech_df.index))
        ret20    = tech_df.get("returns_20d",   pd.Series(0.0,  index=tech_df.index))
        ret5     = tech_df.get("returns_5d",    pd.Series(0.0,  index=tech_df.index))

        bear = ((adx > 22) & (ema_sp < -0.01) & (ret20 < 0)).astype(int)
        bull = ((adx > 22) & (ema_sp >  0.01) & (ret20 > 0)).astype(int)
        hvol = ((adx > 30) & (ret5.abs() > 0.025)).astype(int)
        chop = (1 - bear - bull - hvol).clip(0, 1)

        # Stability: rolling 10-bar majority regime fraction (raw=True → numpy array)
        dominant_regime = bear * 1 + bull * 2 + hvol * 3  # 0=choppy
        stability = (
            dominant_regime.rolling(10, min_periods=3)
            .apply(lambda x: (x == x[-1]).sum() / len(x), raw=True)
        ).fillna(0.5)

        return pd.DataFrame({
            "regime_bear":      bear.values,
            "regime_bull":      bull.values,
            "regime_high_vol":  hvol.values,
            "regime_choppy":    chop.values,
            "regime_stability": stability.values,
        }, index=tech_df.index)

    def _to_vector(self, raw: Dict, names: List[str]) -> np.ndarray:
        """
        Extract ordered float32 array from flat feature dict.
        Missing keys → 0.0. NaN/inf → 0.0.
        """
        vec = np.zeros(len(names), dtype=np.float32)
        for idx, name in enumerate(names):
            val = raw.get(name, 0.0)
            try:
                v = float(val)
                vec[idx] = v if np.isfinite(v) else 0.0
            except (TypeError, ValueError):
                vec[idx] = 0.0
        return vec

    def _clip(self, val: float,
              lo: float = -1.0, hi: float = 1.0) -> float:
        try:
            v = float(val)
            return round(max(lo, min(hi, v)), 6) if np.isfinite(v) else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _data_quality(self, raw: Dict) -> Dict:
        """
        Assess what fraction of numeric features are non-zero.
        Detects data source failures early.
        """
        numeric = {k: v for k, v in raw.items()
                   if isinstance(v, (int, float, np.floating, np.integer))}
        total   = len(numeric)
        zeros   = sum(1 for v in numeric.values() if v == 0.0)
        miss    = zeros / max(total, 1)

        quality = (
            "EXCELLENT" if miss < 0.10 else
            "GOOD"      if miss < 0.25 else
            "DEGRADED"  if miss < 0.50 else
            "POOR"
        )
        return {
            "total_features": total,
            "zero_features":  zeros,
            "missing_pct":    round(miss, 3),
            "quality":        quality,
        }

    def _save_snapshot(self, result: Dict) -> None:
        """Audit trail — saves build metadata + full feature values to DB."""
        try:
            raw = result.get("raw_features", {})
            # Serialize only finite numerics — strip non-serialisable objects
            feature_json = json.dumps(
                {k: round(float(v), 6)
                 for k, v in raw.items()
                 if isinstance(v, (int, float)) and np.isfinite(float(v))},
                separators=(",", ":"),
            )
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT INTO feature_snapshots
                (signal_uuid, built_at, build_time_sec, data_quality,
                 n_features, current_price, regime, risk_allowed,
                 feature_values)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (
                result.get("signal_uuid", ""),
                result["built_at"],
                result["build_time_sec"],
                result["data_quality"]["quality"],
                result["data_quality"]["total_features"],
                result["current_price"],
                result["regime"]["regime"],
                result["risk_context"]["trading_allowed"],
                feature_json,
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Snapshot save skipped: {e}")

    def _setup_db(self) -> None:
        """Create feature_snapshots table if not present."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_snapshots (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_uuid    TEXT,
                    built_at       TEXT,
                    build_time_sec REAL,
                    data_quality   TEXT,
                    n_features     INTEGER,
                    current_price  REAL,
                    regime         TEXT,
                    risk_allowed   INTEGER,
                    feature_values TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signal_outcomes (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal      TEXT,
                    outcome     TEXT,
                    pnl_pct     REAL,
                    pnl_amount  REAL,
                    created_at  TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS open_trades (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal      TEXT,
                    entry_price REAL,
                    stop_price  REAL,
                    target_price REAL,
                    risk_amount REAL,
                    status      TEXT DEFAULT 'OPEN',
                    opened_at   TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS closed_trades (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal       TEXT,
                    entry_price  REAL,
                    exit_price   REAL,
                    pnl_amount   REAL,
                    pnl_pct      REAL,
                    close_date   TEXT,
                    status       TEXT DEFAULT 'CLOSED'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS macro_data (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    rbi_rate      REAL,
                    credit_growth REAL,
                    fetched_at    TEXT
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"DB setup warning: {e}")

    def _empty_result(self) -> Dict:
        """Safe empty result when price data fetch fails."""
        return {
            "raw_features":              {},
            "swing_vector":              np.zeros(len(self.SWING_FEATURES),      dtype=np.float32),
            "intraday_vector":           np.zeros(len(self.INTRADAY_FEATURES),   dtype=np.float32),
            "positional_vector":         np.zeros(len(self.POSITIONAL_FEATURES), dtype=np.float32),
            "swing_feature_names":       self.SWING_FEATURES,
            "intraday_feature_names":    self.INTRADAY_FEATURES,
            "positional_feature_names":  self.POSITIONAL_FEATURES,
            "risk_context":  {"trading_allowed": 0, "final_size_mult": 0.0,
                               "capital_mode": "SMALL"},
            "regime":        {"name": "UNKNOWN", "trade_long": False,
                               "trade_short": False, "position_mult": 0.0},
            "sr_levels":     {"nearest_support": 0, "nearest_resistance": 0,
                               "entry_quality": "D"},
            "current_price": 0.0,
            "atr":           0.0,
            "data_quality":  {"quality": "POOR", "missing_pct": 1.0,
                               "total_features": 0},
            "build_time_sec":0.0,
            "built_at":      str(datetime.now()),
        }
