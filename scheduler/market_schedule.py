# scheduler/market_schedule.py
# NSE market calendar — knows exactly when to run what
# 20yr trader rule: never run a job outside its intended window

import pytz
import logging
from datetime import datetime, date, time, timedelta
from typing import List, Optional

logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")

# NSE 2026 holidays
NSE_HOLIDAYS_2026 = {
    date(2026, 1, 26), date(2026, 3, 25), date(2026, 4, 2),
    date(2026, 4, 14), date(2026, 4, 17), date(2026, 5, 1),
    date(2026, 8, 15), date(2026, 10, 2), date(2026, 10, 20),
    date(2026, 11, 4), date(2026, 12, 25),
}


class MarketSchedule:
    """
    NSE market calendar — determines what runs when.

    Market hours: 9:15 AM – 3:30 PM IST, Mon–Fri
    Pre-market:   9:00 AM – 9:15 AM IST
    Post-close:   3:30 PM – 6:00 PM IST

    Schedule:
    06:00  → Pre-market analysis (full feature build + signal)
    09:00  → Pre-open confirmation
    09:15  → Market open alert
    09:15–15:30 → Every 15 min: signal refresh + exit check
    15:15  → Intraday close warning
    15:45  → Day summary
    18:00  → Performance report
    Sunday 00:00 → Weekly model retrain
    Sunday 06:00 → Fundamental + alt data refresh
    """

    MARKET_OPEN  = time(9, 15)
    MARKET_CLOSE = time(15, 30)
    PRE_OPEN     = time(9,  0)
    INTRADAY_CUTOFF = time(15, 15)

    def now_ist(self) -> datetime:
        return datetime.now(IST)

    def today(self) -> date:
        return self.now_ist().date()

    def is_trading_day(self, dt: Optional[date] = None) -> bool:
        """Is today (or given date) a valid NSE trading day?"""
        d = dt or self.today()
        if d.weekday() >= 5:          return False   # Sat/Sun
        if d in NSE_HOLIDAYS_2026:    return False
        return True

    def is_market_open(self) -> bool:
        """Is NSE currently open?"""
        if not self.is_trading_day():
            return False
        t = self.now_ist().time()
        return self.MARKET_OPEN <= t <= self.MARKET_CLOSE

    def is_pre_market(self) -> bool:
        """9:00–9:15 AM — pre-open session."""
        if not self.is_trading_day():
            return False
        t = self.now_ist().time()
        return self.PRE_OPEN <= t < self.MARKET_OPEN

    def is_intraday_close_window(self) -> bool:
        """Within last 15 min of session — close intraday trades."""
        t = self.now_ist().time()
        return self.INTRADAY_CUTOFF <= t <= self.MARKET_CLOSE

    def is_pre_market_analysis_time(self) -> bool:
        """6:00–9:00 AM on a trading day — run full pre-market analysis."""
        if not self.is_trading_day():
            return False
        t = self.now_ist().time()
        return time(6, 0) <= t < self.PRE_OPEN

    def minutes_to_open(self) -> int:
        """Minutes until NSE opens (0 if already open)."""
        now = self.now_ist()
        if self.is_market_open():
            return 0
        open_dt = now.replace(
            hour=self.MARKET_OPEN.hour,
            minute=self.MARKET_OPEN.minute,
            second=0, microsecond=0,
        )
        if now > open_dt:
            # Already past today's open — find next trading day
            next_day = now.date() + timedelta(days=1)
            while not self.is_trading_day(next_day):
                next_day += timedelta(days=1)
            open_dt = IST.localize(datetime(
                next_day.year, next_day.month, next_day.day,
                self.MARKET_OPEN.hour, self.MARKET_OPEN.minute,
            ))
        delta = open_dt - now
        return max(0, int(delta.total_seconds() / 60))

    def next_trading_day(self) -> date:
        """Find the next NSE trading day from today."""
        d = self.today() + timedelta(days=1)
        while not self.is_trading_day(d):
            d += timedelta(days=1)
        return d

    def is_sunday_midnight(self) -> bool:
        """Sunday midnight window for weekly retrain (00:00–00:30)."""
        now = self.now_ist()
        return now.weekday() == 6 and now.hour == 0 and now.minute < 30

    def is_sunday_morning(self) -> bool:
        """Sunday 6 AM — weekly data refresh."""
        now = self.now_ist()
        return now.weekday() == 6 and now.hour == 6 and now.minute < 30

    def is_end_of_day(self) -> bool:
        """3:45–4:00 PM — post-market summary window."""
        t = self.now_ist().time()
        return time(15, 45) <= t <= time(16, 0) and self.is_trading_day()

    def is_evening_report_time(self) -> bool:
        """6:00–6:15 PM — evening performance report."""
        t = self.now_ist().time()
        return time(18, 0) <= t <= time(18, 15) and self.is_trading_day()

    def should_run_signal_pipeline(self) -> bool:
        """
        Should the 15-minute signal pipeline run right now?
        Runs during pre-market analysis (6 AM–9:15 AM) and
        during market hours (9:15 AM–3:30 PM).
        """
        return self.is_pre_market_analysis_time() or self.is_market_open()

    def get_next_run_seconds(self) -> int:
        """Seconds until the next 15-minute pipeline run."""
        now_min  = self.now_ist().minute
        now_sec  = self.now_ist().second
        # Next 15-min boundary
        next_min = ((now_min // 15) + 1) * 15
        if next_min >= 60:
            next_min -= 60
            remaining = (60 - now_min) * 60 - now_sec
        else:
            remaining = (next_min - now_min) * 60 - now_sec
        return max(10, remaining)

    def log_schedule(self) -> None:
        """Log today's schedule summary."""
        now  = self.now_ist()
        is_td= self.is_trading_day()
        logger.info(
            f"Schedule: {now.strftime('%A %d %b %Y %H:%M IST')} | "
            f"Trading day: {is_td} | "
            f"Market open: {self.is_market_open()} | "
            f"Min to open: {self.minutes_to_open()}"
        )