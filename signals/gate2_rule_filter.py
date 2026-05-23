# signals/gate2_rule_filter.py
# Gate 2 — 11-condition rule filter. Basic sanity checks.
# Each check is a necessary (not sufficient) condition for trading.
# Connected to: all processors, feature_builder.py

import logging
from typing import Dict, Tuple, List

logger = logging.getLogger(__name__)


class Gate2RuleFilter:
    """
    Gate 2: 11-condition pre-ML sanity filter.

    20yr trader checklist before any trade:
    1.  Trend exists           → ADX > 18 or |EMA spread| > 1%
    2.  Volume confirming      → volume_ratio ≥ 0.7x average
    3.  Market direction ok    → Nifty 5d in line with regime direction
    4.  Not within 3d earnings → hard block (5d = soft warn, handled by size)
    5.  Fundamentals ok        → grade not POOR / UNKNOWN
    6.  No severe anomaly      → HARD FAIL on HIGH severity
    7.  VIX not at halt level  → HARD FAIL only at VIX ≥ 28 (HALT_AND_FLATTEN)
                                  VIX 25-28 allowed — Gate 6 raises threshold +10pp
    8.  Macro not hostile      → macro_score > -0.5
    9.  Intermarket aligned    → intermarket_score > -0.4
    10. Rupee not collapsing   → usdinr_5d_pct < +2%
    11. Not RBI decision day   → days_to_rbi > 0 (soft avoid only)

    Minimum pass threshold: 8 of 11 checks.
    HARD_FAILS fire regardless of n_passed — single hard fail = FLAT.
    Check 11 (RBI day) never triggers HARD_FAIL but does reduce n_passed by 1
    on an MPC decision day — size is also cut to 0.4× via macro_event_mult.
    """

    MIN_PASSES = 8       # out of 11 — allows one extra check to fail due to sparse data
    HARD_FAILS = {       # these alone block the signal regardless of n_passed
        "earnings_risk",
        "vix_halt",
        "anomaly_high",
    }

    def check(self, raw_features: Dict, context: Dict) -> Tuple[bool, Dict]:
        """
        Run all 11 rule checks.

        Args:
            raw_features: flat feature dict from FeatureBuilder
            context:      regime string + fundamental/anomaly context

        Returns:
            (passed: bool, details: Dict)
        """
        checks  = {}
        reasons = []

        # Regime direction from context (passed from signal_engine)
        # Used to make Check 3 direction-aware.
        regime = context.get("regime", "BULL_TRENDING")
        is_bear = regime == "BEAR_TRENDING"

        # ── Check 1: Trend exists ─────────────────────────────────
        adx        = float(raw_features.get("adx", 0))
        ema_spread = float(raw_features.get("ema_spread", 0))
        has_trend  = adx > 18 or abs(ema_spread) > 0.01
        checks["has_trend"] = has_trend
        if not has_trend:
            reasons.append(f"No clear trend: ADX={adx:.1f}, EMA spread={ema_spread:.3f}")

        # ── Check 2: Volume confirming ────────────────────────────
        vol_ratio = float(raw_features.get("volume_ratio", 1.0))
        vol_ok    = vol_ratio >= 0.7
        checks["volume_ok"] = vol_ok
        if not vol_ok:
            reasons.append(f"Low volume: {vol_ratio:.1f}x (need 0.7x+)")

        # ── Check 3: Market direction stable (regime-aware) ───────
        # BULL regime → block if Nifty crashing (long bias at risk)
        # BEAR regime → block if Nifty surging  (short bias at risk)
        # HIGH_VOL   → always pass (both directions allowed)
        nifty_5d = float(raw_features.get("nifty_5d_pct", 0))
        if regime == "HIGH_VOLATILITY":
            market_ok = True
        elif is_bear:
            market_ok = nifty_5d < 3.0    # block BEAR short if market surged 3%+
        else:
            market_ok = nifty_5d > -3.0   # block LONG if market crashed 3%+
        checks["market_ok"] = market_ok
        if not market_ok:
            if is_bear:
                reasons.append(f"Nifty surging {nifty_5d:+.1f}% — risk for shorts")
            else:
                reasons.append(f"Nifty collapsing {nifty_5d:.1f}% in 5 days")

        # ── Check 4: Earnings proximity ───────────────────────────
        # Hard fail within 3 days (too close for any model).
        # 3-5 days before earnings: soft warn only — positional model
        # specifically uses pre_earnings_drift features in this window.
        days_earn  = float(raw_features.get("days_to_earnings", 99))
        not_earning = days_earn > 3
        checks["earnings_risk"] = not_earning
        if not not_earning:
            reasons.append(f"Earnings in {int(days_earn)}d — hard block within 3 days")

        # ── Check 5: Fundamentals acceptable ─────────────────────
        fund_grade = context.get("fundamental_grade", "GOOD")
        fund_ok    = fund_grade not in ("POOR", "UNKNOWN")
        checks["fundamentals_ok"] = fund_ok
        if not fund_ok:
            reasons.append(f"Poor fundamental grade: {fund_grade}")

        # ── Check 6: No high-severity anomaly ────────────────────
        anomaly_sev = context.get("anomaly_severity", "LOW")
        anomaly_ok  = anomaly_sev != "HIGH"
        checks["anomaly_high"] = anomaly_ok
        if not anomaly_ok:
            reasons.append(f"High-severity anomaly: {context.get('anomaly_type')}")

        # ── Check 7: VIX not at HALT level ───────────────────────
        # VIX ≥ 28 → HALT_AND_FLATTEN (hard block here in Gate 2)
        # VIX 25-28 → allowed but Gate 6 raises confidence threshold +10pp
        # VIX 20-25 → allowed but Gate 6 raises confidence threshold +5pp
        vix    = float(raw_features.get("india_vix_level", 15))
        vix_ok = vix < 28
        checks["vix_halt"] = vix_ok
        if not vix_ok:
            reasons.append(f"VIX HALT level: {vix:.1f} ≥ 28 — flatten all positions")

        # ── Check 8: Macro not hostile ────────────────────────────
        macro_score = float(raw_features.get("macro_score", 0))
        macro_ok    = macro_score > -0.5
        checks["macro_ok"] = macro_ok
        if not macro_ok:
            reasons.append(f"Hostile macro: score={macro_score:.2f}")

        # ── Check 9: Intermarket aligned ─────────────────────────
        im_score = float(raw_features.get("intermarket_score", 0))
        im_ok    = im_score > -0.4
        checks["intermarket_ok"] = im_ok
        if not im_ok:
            reasons.append(f"Intermarket headwinds: score={im_score:.2f}")

        # ── Check 10: Rupee not collapsing ────────────────────────
        usdinr_5d = float(raw_features.get("usdinr_5d_pct", 0))
        rupee_ok  = usdinr_5d < 2.0
        checks["rupee_ok"] = rupee_ok
        if not rupee_ok:
            reasons.append(f"Rupee stress: USD/INR +{usdinr_5d:.1f}% in 5 days")

        # ── Check 11: Not RBI decision day (soft fail only) ───────
        # Soft avoid on the MPC decision day itself (days_to_rbi == 0).
        # Never counts toward HARD_FAILS — system can still trade when
        # 9+ other checks pass, but macro_event_mult reduces size to 0.4×.
        days_rbi    = float(raw_features.get("days_to_rbi", 30))
        not_rbi_day = days_rbi > 0
        checks["not_rbi_day"] = not_rbi_day
        if not not_rbi_day:
            reasons.append("RBI MPC decision day — size reduced to 0.4× via macro_event_mult")

        # ── Evaluate ──────────────────────────────────────────────
        n_passed    = sum(checks.values())
        hard_failed = [k for k in self.HARD_FAILS if not checks.get(k, True)]
        passed      = n_passed >= self.MIN_PASSES and len(hard_failed) == 0

        return passed, {
            "gate":         2,
            "passed":       passed,
            "n_passed":     n_passed,
            "n_total":      len(checks),
            "hard_failed":  hard_failed,
            "checks":       checks,
            "fail_reasons": reasons,
            "pass_rate":    round(n_passed / len(checks), 3),
            "regime":       regime,
            "vix_level":    vix,
        }