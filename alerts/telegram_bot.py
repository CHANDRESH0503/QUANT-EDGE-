# alerts/telegram_bot.py
# Delivers all alerts to trader's phone via Telegram
# Uses python-telegram-bot async API
# Connected to: signal_engine.py, exit_engine.py, orchestrator.py

import logging
import asyncio
import os
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)

try:
    from telegram import Bot
    from telegram.constants import ParseMode
    from telegram.error import TelegramError
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.warning("python-telegram-bot not installed. pip install python-telegram-bot")

from alerts.alert_formatter import AlertFormatter


class TelegramBot:
    """
    Delivers trading signals and system alerts via Telegram.

    20yr trader perspective:
    The alert must arrive BEFORE the market opens.
    I want to wake up at 8:30 AM, read the signal on my phone,
    make my decision, and place the order manually at 9:15 AM.
    Automated execution can come later — first prove the signals.

    Delivery schedule:
    6:00 AM  → Full pre-market signal (if any)
    6:00-9:00→ Updates if signal changes materially
    9:15 AM  → Market open confirmation
    15:15    → Intraday close warning
    15:45    → Day summary
    18:00    → Performance report
    Any time → Exit recommendations + circuit breaker alerts

    Telegram setup:
    1. Message @BotFather on Telegram → create bot → get token
    2. Message your bot once → get chat_id from getUpdates
    3. Add to config.py: TELEGRAM_TOKEN and TELEGRAM_CHAT_ID
    """

    def __init__(
        self,
        token:   Optional[str] = None,
        chat_id: Optional[str] = None,
        ticker:  str = "HDFCBANK",
    ):
        self.token   = token   or os.getenv("TELEGRAM_TOKEN",  "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.ticker  = ticker
        self.formatter = AlertFormatter()
        self._bot    = None

        if not self.token or not self.chat_id:
            logger.warning(
                "Telegram credentials not set. "
                "Set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID in .env"
            )

    # ─────────────────────────────────────────────────────────────
    # PUBLIC SEND METHODS
    # ─────────────────────────────────────────────────────────────

    def send_signal(self, signal: Dict) -> bool:
        """Send a full trading signal alert."""
        direction = signal.get("signal", "FLAT")

        if direction == "FLAT":
            reason   = signal.get("reason", "No conditions met")
            regime   = signal.get("regime", "")
            category = signal.get("category", "") or signal.get("trade_type", "")
            text     = self.formatter.format_flat(reason, regime, category)
        else:
            # Prefer the ticker on the signal itself (per-category routing can
            # carry its own ticker if the engine ever spans tickers) and fall
            # back to the bot's configured ticker for backward compat.
            tkr  = signal.get("ticker") or self.ticker
            text = self.formatter.format_signal(signal, tkr)

        return self._send(text)

    def send_exit(self, exit_info: Dict) -> bool:
        """Send exit recommendation alert."""
        text = self.formatter.format_exit(exit_info)
        return self._send(text)

    def send_market_open(self, signal: Dict) -> bool:
        """9:15 AM market open confirmation."""
        text = self.formatter.format_market_open(signal)
        return self._send(text)

    def send_intraday_close_warning(self) -> bool:
        """3:15 PM close warning."""
        text = self.formatter.format_intraday_close_warning()
        return self._send(text)

    def send_daily_summary(self, stats: Dict) -> bool:
        """End-of-day summary."""
        text = self.formatter.format_daily_summary(stats)
        return self._send(text)

    def send_retrain_report(self, summary: Dict) -> bool:
        """Sunday midnight retrain report."""
        text = self.formatter.format_retrain_report(summary)
        return self._send(text)

    def send_circuit_breaker(self, cb_status: Dict) -> bool:
        """Circuit breaker alert — high priority."""
        text = self.formatter.format_circuit_breaker(cb_status)
        return self._send(text)

    def send_error(self, component: str, error: str) -> bool:
        """System error notification."""
        text = self.formatter.format_error(component, error)
        return self._send(text)

    def send_raw(self, text: str) -> bool:
        """Send a raw text message — for custom alerts."""
        return self._send(text)

    def test_connection(self) -> bool:
        """
        Verify bot is configured and reachable.
        Call this on system startup to confirm alerts will work.
        """
        return self._send(
            f"✅ QUANT EDGE connected\n"
            f"Ticker: {self.ticker}\n"
            f"Time: {datetime.now().strftime('%d %b %Y %H:%M IST')}\n"
            f"System is live and monitoring."
        )

    # ─────────────────────────────────────────────────────────────
    # CORE SEND
    # ─────────────────────────────────────────────────────────────

    def _send(self, text: str) -> bool:
        """
        Send a message. Handles both sync and async contexts.
        Uses asyncio.run() when called from synchronous scheduler.
        """
        if not TELEGRAM_AVAILABLE:
            logger.warning(f"Telegram unavailable. Message:\n{text[:100]}")
            return False

        if not self.token or not self.chat_id:
            logger.warning("Telegram credentials missing — message not sent")
            return False

        try:
            return asyncio.run(self._async_send(text))
        except RuntimeError:
            # Already inside a running event loop (FastAPI / Jupyter)
            try:
                import nest_asyncio
                nest_asyncio.apply()
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(self._async_send(text))
            except ImportError:
                # Fallback: spawn a new thread with its own event loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    future = ex.submit(asyncio.run, self._async_send(text))
                    return future.result(timeout=15)
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    async def _async_send(self, text: str) -> bool:
        """Async message delivery. Uses context manager required by PTB v20+."""
        chunks = self._split_message(text)

        # python-telegram-bot v20+ requires async context manager for connection pooling
        async with Bot(token=self.token) as bot:
            for chunk in chunks:
                try:
                    await bot.send_message(
                        chat_id    = self.chat_id,
                        text       = chunk,
                        parse_mode = ParseMode.MARKDOWN,
                    )
                except TelegramError as e:
                    # Retry once without markdown if Telegram rejects formatting
                    if "can't parse" in str(e).lower() or "parse" in str(e).lower():
                        try:
                            await bot.send_message(
                                chat_id = self.chat_id,
                                text    = self._strip_markdown(chunk),
                            )
                        except TelegramError as e2:
                            logger.error(f"Telegram retry failed: {e2}")
                            return False
                    else:
                        logger.error(f"Telegram error: {e}")
                        return False

        logger.debug(f"Telegram sent: {text[:60]}...")
        return True

    # ─────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────

    def _split_message(self, text: str, limit: int = 4096) -> list:
        """Split long messages into Telegram-compatible chunks."""
        if len(text) <= limit:
            return [text]
        chunks = []
        while text:
            if len(text) <= limit:
                chunks.append(text)
                break
            # Split at last newline before limit
            split_at = text.rfind("\n", 0, limit)
            if split_at == -1:
                split_at = limit
            chunks.append(text[:split_at])
            text = text[split_at:].lstrip("\n")
        return chunks

    def _strip_markdown(self, text: str) -> str:
        """Remove Telegram markdown for plain-text fallback."""
        for char in ["*", "_", "`", "["]:
            text = text.replace(char, "")
        return text