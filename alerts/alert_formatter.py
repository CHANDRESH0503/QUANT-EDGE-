# alerts/alert_formatter.py
# Formats all signal and system alerts into clean Telegram messages
# Every alert tells the trader exactly what to do — no ambiguity

from datetime import datetime
from typing import Dict, List
import pytz as _pytz

_IST = _pytz.timezone("Asia/Kolkata")


def _now_ist_str(fmt: str = "%H:%M IST") -> str:
    """Return the current time as a string in IST, regardless of server timezone."""
    return datetime.now(_IST).strftime(fmt)


class AlertFormatter:
    """
    Formats trading signals into actionable Telegram messages.

    20yr trader principle:
    An alert must answer 5 questions instantly:
    1. What direction?  (LONG / SHORT / FLAT)
    2. Where to enter?  (price)
    3. Where to stop?   (stop loss)
    4. Where to target? (profit target)
    5. How much to buy? (shares / position value)

    Everything else is context. Keep it clean.
    A cluttered alert causes hesitation. Hesitation costs money.
    """

    SIGNAL_EMOJI = {"LONG": "🟢", "SHORT": "🔴", "FLAT": "⚪"}
    REGIME_EMOJI = {
        "BULL_TRENDING": "🐂", "BEAR_TRENDING": "🐻",
        "HIGH_VOLATILITY": "⚡", "CHOPPY_SIDEWAYS": "🌫️",
    }
    ALIGN_EMOJI    = {"A+": "🏆", "A": "✅", "B": "👍", "C": "➖", "F": "❌"}
    CATEGORY_LABEL = {
        "swing":      "🟢 SWING",
        "positional": "🔵 POSITIONAL",
        "intraday":   "🟠 INTRADAY",
    }

    TICKER_NAMES = {
        "HDFCBANK.NS":  "HDFC Bank",
        "ICICIBANK.NS": "ICICI Bank",
        "KOTAKBANK.NS": "Kotak Bank",
        "AXISBANK.NS":  "Axis Bank",
        "INDUSINDBK.NS":"IndusInd Bank",
        "SBIN.NS":      "SBI",
    }

    @classmethod
    def ticker_display(cls, ticker: str) -> str:
        """Return a clean display name from a yfinance ticker symbol."""
        return cls.TICKER_NAMES.get(ticker, ticker.replace(".NS", ""))

    # ─────────────────────────────────────────────────────────────
    # SIGNAL ALERTS
    # ─────────────────────────────────────────────────────────────

    def format_signal(self, signal: Dict, ticker: str = "") -> str:
        """
        Full signal alert — sent when a new signal passes all 6 gates.
        Includes all trade parameters and key data context.
        """
        raw_ticker = ticker or signal.get("ticker", "HDFCBANK.NS")
        ticker     = self.ticker_display(raw_ticker)
        direction  = signal.get("signal", "FLAT")
        sig_emoji  = self.SIGNAL_EMOJI.get(direction, "⚪")
        regime     = signal.get("regime", "UNKNOWN")
        alignment = signal.get("alignment", "B")
        conf      = float(signal.get("confidence", 0))
        entry     = float(signal.get("entry_price", 0))
        stop      = float(signal.get("stop_loss", 0))
        target    = float(signal.get("target_price", 0))
        shares    = int(signal.get("shares", 0))
        risk_amt  = float(signal.get("risk_amount", 0))
        pos_val   = float(signal.get("position_value", 0))
        rr        = float(signal.get("reward_risk", 0))
        cap_mode  = signal.get("capital_mode", "FULL")
        eq        = signal.get("entry_quality", "B")
        now       = datetime.now().strftime("%d %b %Y %H:%M IST")

        # Sub-model predictions
        sw  = signal.get("swing",      {})
        itr = signal.get("intraday",   {})
        pos = signal.get("positional", {})
        sw_sig   = sw.get("signal",  "FLAT")
        itr_sig  = itr.get("signal", "FLAT")
        pos_sig  = pos.get("signal", "FLAT")
        sw_conf  = float(sw.get("confidence",  0))
        itr_conf = float(itr.get("confidence", 0))
        pos_conf = float(pos.get("confidence", 0))

        # Sentiment, flow, macro context
        raw = signal.get("raw_features", {})
        sent_score = float(raw.get("finbert_score_24h", 0))
        fii_5d     = float(raw.get("fii_net_5d_cr", 0)) if "fii_net_5d_cr" in raw else None
        pcr        = float(raw.get("pcr_norm", 0))
        vix        = float(raw.get("india_vix_level", 0)) if "india_vix_level" in raw else None
        vol_ratio  = float(raw.get("volume_ratio", 1))

        category    = (signal.get("category") or signal.get("trade_type") or "swing").lower()
        cat_label   = self.CATEGORY_LABEL.get(category, category.upper())

        lines = [
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"{sig_emoji} *{ticker} — {cat_label} {direction}*",
            f"",
            f"{self.ALIGN_EMOJI.get(alignment, '')} Alignment: *{alignment}*  |  "
            f"{self.REGIME_EMOJI.get(regime, '')} Regime: *{regime.replace('_', ' ')}*",
            f"",
            f"📊 *Model Predictions*",
            f"  Positional: {self.SIGNAL_EMOJI.get(pos_sig,'')} {pos_sig} ({pos_conf:.0%})",
            f"  Swing ★:    {self.SIGNAL_EMOJI.get(sw_sig, '')} {sw_sig} ({sw_conf:.0%})",
            f"  Intraday:   {self.SIGNAL_EMOJI.get(itr_sig,'')} {itr_sig} ({itr_conf:.0%})",
            f"",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"💰 *Trade Setup*",
            f"  Entry:   ₹{entry:,.2f}",
            f"  Stop:    ₹{stop:,.2f}  ({abs(stop-entry)/entry*100:.1f}%)",
            f"  Target:  ₹{target:,.2f}  (+{abs(target-entry)/entry*100:.1f}%)",
            f"  R:R:     {rr:.1f} : 1",
            f"  Quality: {eq}-grade entry",
            f"",
            f"📦 *Position*",
            f"  Shares:  {shares}",
            f"  Value:   ₹{pos_val:,.2f}",
            f"  Risk:    ₹{risk_amt:,.2f}",
            f"  Mode:    {cap_mode}",
            f"",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"📡 *Data Context*",
        ]

        if sent_score != 0:
            sent_label = "Positive ✓" if sent_score > 0.2 else ("Negative ⚠️" if sent_score < -0.2 else "Neutral")
            lines.append(f"  Sentiment: {sent_score:+.2f} ({sent_label})")

        if fii_5d is not None:
            fii_label = "Buying ✓" if fii_5d > 500 else ("Selling ⚠️" if fii_5d < -500 else "Neutral")
            lines.append(f"  FII 5-day: ₹{fii_5d:+,.0f} Cr ({fii_label})")

        if vix is not None:
            vix_label = "Low ✓" if vix < 15 else ("High ⚠️" if vix > 20 else "Normal")
            lines.append(f"  India VIX: {vix:.1f} ({vix_label})")

        lines.append(f"  Volume:    {vol_ratio:.1f}× avg")

        reasons = signal.get("reasons", [])
        if reasons:
            lines += ["", f"✅ *Gates Passed*"]
            for r in reasons[:5]:
                lines.append(f"  • {r}")

        lines += [
            f"",
            f"⏰ {now}",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ]

        return "\n".join(lines)

    def format_flat(self, reason: str, regime: str = "", category: str = "") -> str:
        """
        Brief FLAT alert — no trade with concise reason.
        category: optional "swing" / "positional" / "intraday" — when supplied,
        the alert is scoped to that category ("SWING no longer active") rather
        than a global "No Signal".
        """
        now = datetime.now().strftime("%H:%M IST")
        r   = self.REGIME_EMOJI.get(regime, "")
        if category:
            cat_label = self.CATEGORY_LABEL.get(category.lower(), category.upper())
            return (
                f"⚪ *{cat_label} — STOOD DOWN ({now})*\n"
                f"{r} {regime.replace('_', ' ') if regime else ''}\n"
                f"Reason: {reason}"
            )
        return (
            f"⚪ *No Signal — {now}*\n"
            f"{r} {regime.replace('_', ' ') if regime else ''}\n"
            f"Reason: {reason}"
        )

    # ─────────────────────────────────────────────────────────────
    # EXIT ALERTS
    # ─────────────────────────────────────────────────────────────

    def format_exit(self, exit_info: Dict) -> str:
        """Exit recommendation alert with P&L."""
        signal   = exit_info.get("signal", "LONG")
        price    = float(exit_info.get("exit_price", 0))
        pnl_pct  = float(exit_info.get("pnl_pct", 0))
        pnl_amt  = float(exit_info.get("pnl_amount", 0)) if "pnl_amount" in exit_info else 0
        reason   = exit_info.get("reason", "")
        message  = exit_info.get("message", "")
        ticker   = exit_info.get("ticker", "")
        pnl_emoji= "✅" if pnl_pct >= 0 else "❌"
        now      = _now_ist_str()   # always IST regardless of server timezone

        bank = ticker.replace(".NS", "") if ticker else ""
        return (
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🚪 *EXIT {signal} @ ₹{price:,.2f}*"
            + (f"  [{bank}]" if bank else "") + "\n"
            f"\n"
            f"Reason: {reason}\n"
            f"Detail: {message}\n"
            f"\n"
            f"{pnl_emoji} P&L: {pnl_pct:+.2%}"
            + (f" (₹{pnl_amt:+,.2f})" if pnl_amt else "") +
            f"\n"
            f"⏰ {now}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

    # ─────────────────────────────────────────────────────────────
    # SYSTEM ALERTS
    # ─────────────────────────────────────────────────────────────

    def format_market_open(self, signal: Dict) -> str:
        """9:15 AM market open confirmation."""
        direction = signal.get("signal", "FLAT")
        sig_emoji = self.SIGNAL_EMOJI.get(direction, "⚪")
        entry     = float(signal.get("entry_price", 0))
        return (
            f"🔔 *NSE Market Open — 9:15 AM*\n"
            f"\n"
            f"Active signal: {sig_emoji} {direction}\n"
            f"Entry level:   ₹{entry:,.2f}\n"
            f"\n"
            f"Confirm fill and set stop + target orders."
        )

    def format_intraday_close_warning(self) -> str:
        """3:15 PM warning — close intraday positions."""
        return (
            f"⚠️ *3:15 PM — Close Intraday Positions*\n"
            f"\n"
            f"NSE closes at 3:30 PM.\n"
            f"Exit all intraday trades NOW.\n"
            f"Do not carry intraday positions to next day."
        )

    def format_daily_summary(self, stats: Dict) -> str:
        """End-of-day performance summary."""
        equity   = float(stats.get("current_equity", 0))
        month_pnl= float(stats.get("monthly_pnl", 0))
        month_dd = float(stats.get("monthly_dd_pct", 0))
        win_rate = float(stats.get("win_rate", 0))
        pf       = float(stats.get("profit_factor", 0))
        total    = int(stats.get("total_trades", 0))
        c_losses = int(stats.get("consecutive_losses", 0))
        heat     = float(stats.get("portfolio_heat", 0))
        now      = datetime.now().strftime("%d %b %Y")

        dd_emoji = "✅" if month_dd > -0.02 else ("⚠️" if month_dd > -0.05 else "🚨")
        cl_emoji = "✅" if c_losses == 0 else ("⚠️" if c_losses < 3 else "🚨")

        return (
            f"📊 *Daily Summary — {now}*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"💼 Equity:      ₹{equity:,.2f}\n"
            f"📅 Month P&L:   ₹{month_pnl:+,.2f} ({month_dd:+.1%})\n"
            f"🔥 Heat:        {heat:.1%} risk deployed\n"
            f"\n"
            f"📈 Win Rate:    {win_rate:.0%}\n"
            f"⚖️  Profit Factor: {pf:.2f}\n"
            f"📋 Total Trades: {total}\n"
            f"\n"
            f"{dd_emoji} Monthly DD:  {month_dd:+.1%}\n"
            f"{cl_emoji} Consec Losses: {c_losses}\n"
            f"━━━━━━━━━━━━━━━━━━━━"
        )

    def format_retrain_report(self, summary: Dict) -> str:
        """Sunday midnight retrain completion report."""
        models = summary.get("models", {})
        elapsed= float(summary.get("elapsed_seconds", 0))
        now    = datetime.now().strftime("%d %b %Y %H:%M")

        lines = [
            f"🔄 *Weekly Retrain Complete — {now}*",
            f"Duration: {elapsed:.0f}s",
            f"",
        ]
        for name, m in models.items():
            if not m:
                lines.append(f"  {name.title()}: ❌ Failed")
                continue
            cv = float(m.get("cv_accuracy_mean", 0))
            n  = int(m.get("n_samples", 0))
            lines.append(f"  {name.title()}: ✅ {cv:.1%} CV ({n:,} samples)")

        return "\n".join(lines)

    def format_circuit_breaker(self, cb_status: Dict) -> str:
        """Circuit breaker triggered alert — urgent."""
        level   = cb_status.get("level", "WARN")
        dd      = float(cb_status.get("monthly_dd_pct", 0))
        losses  = int(cb_status.get("consecutive_losses", 0))
        warnings= cb_status.get("warnings", [])
        emoji   = {"WARN": "⚠️", "PAUSE": "🛑", "HALT": "🚨"}.get(level, "⚠️")

        lines = [
            f"{emoji} *CIRCUIT BREAKER: {level}*",
            f"",
            f"Monthly DD:   {dd:+.1%}",
            f"Consec Losses: {losses}",
            f"",
            f"Action required:",
        ]
        for w in warnings:
            lines.append(f"  • {w}")

        if level == "HALT":
            lines += ["", "⛔ ALL TRADING HALTED — manual review required"]
        elif level == "PAUSE":
            lines += ["", "🛑 No new entries until conditions improve"]

        return "\n".join(lines)

    def format_model_reversal(self, reversal: Dict) -> str:
        """
        Advisory alert sent when an open position's supporting model has reversed
        direction or the regime has turned adverse.

        This is NOT an automatic exit — it tells the trader to review the trade.
        The existing stop-loss in the exit engine is the hard safety net.
        """
        ticker    = self.ticker_display(reversal.get("ticker", ""))
        cat       = reversal.get("category", "swing")
        cat_label = self.CATEGORY_LABEL.get(cat, cat.upper())
        direction = reversal.get("open_signal",  "LONG")
        model_dir = reversal.get("model_signal", "FLAT")
        rtype     = reversal.get("type", "MODEL_REVERSAL")
        regime    = reversal.get("regime", "")
        entry     = float(reversal.get("entry_price", 0))
        stop      = float(reversal.get("stop_price",  0))
        now       = datetime.now().strftime("%H:%M IST")

        dir_emoji   = self.SIGNAL_EMOJI.get(direction, "⚪")
        model_emoji = self.SIGNAL_EMOJI.get(model_dir, "⚪")
        reg_emoji   = self.REGIME_EMOJI.get(regime, "")

        if rtype == "MODEL_REVERSAL":
            headline = f"⚠️ *MODEL REVERSAL — {cat_label}*"
            body = (
                f"Open trade:  {dir_emoji} *{direction}*\n"
                f"Model now:   {model_emoji} *{model_dir}*\n"
                f"\n"
                f"→ Model is now calling the opposite direction.\n"
                f"  Consider tightening stop or closing {direction}."
            )
        else:
            headline = f"⚠️ *REGIME ADVERSE — {cat_label}*"
            body = (
                f"Open trade:  {dir_emoji} *{direction}*\n"
                f"{reg_emoji} Regime: *{regime.replace('_', ' ')}*\n"
                f"\n"
                f"→ Market regime no longer supports {direction}.\n"
                f"  Review position — stop remains at ₹{stop:,.2f}."
            )

        return (
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"{headline}\n"
            f"*{ticker}* | {now}\n"
            f"\n"
            f"{body}\n"
            f"\n"
            f"Entry: ₹{entry:,.2f}  |  Stop: ₹{stop:,.2f}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

    def format_error(self, component: str, error: str) -> str:
        """System error alert."""
        now = datetime.now().strftime("%H:%M IST")
        return (
            f"⚠️ *System Error — {now}*\n"
            f"Component: {component}\n"
            f"Error: {error[:200]}"
        )