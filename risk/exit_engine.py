# risk/exit_engine.py
# Smart exit system — monitors open positions every 15 minutes
# Multiple exit conditions run simultaneously
# Connected to: database/trading.db, alerts/telegram_bot.py

import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ExitEngine:
    """
    Monitors all open positions for exit conditions.

    20yr trader truth:
    Entry is 30% of trading. Exit is 70%.
    Most traders have decent entries but terrible exits.
    They exit winners too early (fear) and hold losers too long (hope).
    This engine removes both emotions.

    Exit triggers (in priority order):
    1. Hard stop loss        — price hits stop, exit immediately
    2. Trailing stop         — locks in profits as price rises
    3. Signal reversal       — model flips to opposite direction
    4. Time exit             — max hold days exceeded
    5. Resistance hit        — price at major resistance, take profit
    6. Regime change         — regime shifts against the trade
    7. Sentiment reversal    — very negative news during LONG trade
    """

    MAX_HOLD = {"swing": 7, "intraday": 1, "positional": 28}
    TRAIL_TRIGGER    = 0.02    # start trailing after 2% profit
    TRAIL_DISTANCE   = 0.015   # trail stop 1.5% below highest price
    RESISTANCE_ZONE  = 0.005   # within 0.5% = near resistance

    def __init__(self, db_path: str = "database/trading.db"):
        self.db_path = db_path
        self._setup_db()

    # ─────────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────────

    def check_all_positions(
        self,
        current_price: float,
        open_price: float = 0.0,
        ticker: str = None,
    ) -> List[Dict]:
        """
        Check open positions for exit conditions.
        Returns list of exit recommendations (does NOT close in DB — caller
        must call close_position() / close_position_partial() for each hit).

        Args:
            current_price: Latest market price for this ticker.
            open_price:    Today's open price (for gap-fill logic).
            ticker:        When provided, only check positions for this ticker.
                           CRITICAL: always pass this so HDFCBANK price is never
                           used to evaluate ICICIBANK positions.
        """
        positions = self._get_open_positions(ticker=ticker)
        results   = []

        for pos in positions:
            result = self._check_gap_fill(pos, open_price) if open_price else None
            if result and result["should_exit"]:
                results.append(result)
                logger.warning(
                    f"GAP FILL EXIT: {pos['signal']} | "
                    f"open=₹{open_price:.2f} past stop=₹{pos['stop_price']:.2f}"
                )
            else:
                result = self._check_position(pos, current_price)
                if result["should_exit"]:
                    results.append(result)
                    logger.info(
                        f"EXIT SIGNAL: {pos['signal']} | "
                        f"reason={result['reason']} | "
                        f"pnl={result['pnl_pct']:+.2%}"
                    )

        return results

    def _check_gap_fill(self, pos: Dict, open_price: float) -> Optional[Dict]:
        """
        Gap-fill logic: if today's open has already blown through the stop,
        the actual fill is at open_price (not the theoretical stop level).
        This is the most realistic modelling of overnight gap risk.
        """
        if open_price <= 0:
            return None
        signal = pos["signal"]
        stop   = float(pos.get("stop_price", 0))
        pos_id = pos["id"]
        entry  = float(pos["entry_price"])

        gapped = (
            (signal == "LONG"  and open_price < stop) or
            (signal == "SHORT" and open_price > stop)
        )
        if not gapped:
            return None

        if signal == "LONG":
            pnl_pct = (open_price - entry) / entry
        else:
            pnl_pct = (entry - open_price) / entry

        return {
            "should_exit":  True,
            "position_id":  pos_id,
            "signal":       signal,
            "exit_price":   open_price,
            "pnl_pct":      round(pnl_pct, 6),
            "reason":       "GAP_FILL",
            "message":      (
                f"Gap filled past stop: open=₹{open_price:.2f}, "
                f"stop=₹{stop:.2f} — exit at open price"
            ),
        }

    def open_position(
        self,
        signal:      str,
        entry_price: float,
        stop_price:  float,
        target_price:float,
        shares:      int,
        risk_amount: float,
        trade_type:  str = "swing",
        signal_uuid: str = "",
        alignment:   str = "",
        regime:      str = "",
        confidence:  float = 0.0,
        entry_quality: str = "",
        ticker:      str = "HDFCBANK.NS",
        regime_match: Optional[bool] = None,
        size_mult:   float = 1.0,
        atr_at_entry: float = 0.0,
        reward_risk: float = 0.0,
    ) -> int:
        """Record a new open position. Returns position ID.

        The entry-context fields (confidence, alignment, regime_at_entry,
        entry_quality, regime_match, size_mult, atr_at_entry, reward_risk) are
        PERSISTED here so they can be carried into closed_trades on exit and
        used for attributable outcome analysis. Previously these were accepted
        as parameters but silently dropped — the learning loop was blind.
        """
        rm = None if regime_match is None else int(bool(regime_match))
        conn = self._connect()
        cur  = conn.execute("""
            INSERT INTO open_trades
            (signal_uuid, ticker, signal, entry_price, stop_price, target_price,
             shares, risk_amount, trade_type,
             highest_price, lowest_price, status, opened_at,
             confidence, alignment, regime_at_entry, entry_quality,
             regime_match, size_mult, atr_at_entry, reward_risk)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            signal_uuid or "",
            ticker,
            signal, entry_price, stop_price, target_price,
            shares, risk_amount, trade_type,
            entry_price,  # highest = entry at open
            entry_price,  # lowest  = entry at open
            "OPEN",
            str(datetime.now()),
            float(confidence), alignment, regime, entry_quality,
            rm, float(size_mult), float(atr_at_entry), float(reward_risk),
        ))
        pos_id = cur.lastrowid
        conn.commit()
        conn.close()
        logger.info(
            f"Position opened: id={pos_id} {signal} {shares} @ ₹{entry_price:.2f} "
            f"| {trade_type} conf={confidence:.0%} align={alignment or '?'} "
            f"SR={entry_quality or '?'} {'aligned' if rm else 'counter' if rm==0 else '?'}"
        )
        return pos_id

    def close_position(self, position_id: int, exit_price: float, reason: str) -> Dict:
        """
        Close a position: INSERT into closed_trades, DELETE from open_trades.

        Both operations are in ONE atomic transaction — either both succeed
        or neither does.  If the INSERT fails (schema mismatch, locked DB, etc.)
        the DELETE is also rolled back so the position is never silently lost.

        Returns the closed trade dict, or {} if the position doesn't exist.
        Raises on unrecoverable DB errors so the caller can log / retry.
        """
        conn = self._connect()
        try:
            pos = conn.execute(
                "SELECT * FROM open_trades WHERE id=?", (position_id,)
            ).fetchone()

            if not pos:
                logger.warning(f"close_position: id={position_id} not found in open_trades")
                return {}

            pos         = dict(pos)
            entry       = float(pos["entry_price"])
            shares      = int(pos["shares"])
            signal      = pos["signal"]
            signal_uuid = pos.get("signal_uuid") or ""
            ticker      = pos.get("ticker")     or "HDFCBANK.NS"
            trade_type  = pos.get("trade_type") or "swing"
            opened_at   = pos.get("opened_at",  str(datetime.now()))
            risk_amount = float(pos.get("risk_amount") or 0.0)

            if signal == "LONG":
                pnl_pct = (exit_price - entry) / entry
            else:
                pnl_pct = (entry - exit_price) / entry

            pnl_amount = round(pnl_pct * entry * shares, 2)

            hold_days, hold_hours = self._hold_time(opened_at)
            # R-multiple: realised P&L in units of the risk taken. The single
            # most important per-trade stat — expectancy is just mean(R).
            r_multiple = round(pnl_amount / risk_amount, 3) if risk_amount > 0 else None
            # MFE/MAE: best/worst the trade looked, in % from entry (direction-
            # aware). High MFE on a loser = target too far / exited too late;
            # high MAE on a winner = stop too tight, got lucky.
            mfe_pct, mae_pct = self._excursions(pos, entry, signal)

            # Atomic: INSERT then DELETE in the same connection/transaction
            conn.execute("""
                INSERT INTO closed_trades
                (signal_uuid, ticker, signal, trade_type,
                 entry_price, exit_price, shares,
                 pnl_amount, pnl_pct, exit_reason, close_date, status,
                 confidence, alignment, regime_at_entry, entry_quality,
                 regime_match, size_mult, atr_at_entry, reward_risk,
                 r_multiple, hold_hours, mfe_pct, mae_pct)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                signal_uuid, ticker, signal, trade_type,
                entry, exit_price, shares,
                pnl_amount, round(pnl_pct, 6),
                reason, str(datetime.now()), "CLOSED",
                pos.get("confidence"), pos.get("alignment"),
                pos.get("regime_at_entry"), pos.get("entry_quality"),
                pos.get("regime_match"), pos.get("size_mult"),
                pos.get("atr_at_entry"), pos.get("reward_risk"),
                r_multiple, round(hold_hours, 2), mfe_pct, mae_pct,
            ))
            conn.execute("DELETE FROM open_trades WHERE id=?", (position_id,))
            conn.commit()

        except Exception:
            conn.rollback()
            conn.close()
            raise   # let orchestrator._run_exit_checks catch and log it

        conn.close()

        result = {
            "position_id": position_id,
            "signal":      signal,
            "signal_uuid": signal_uuid,
            "ticker":      ticker,
            "trade_type":  trade_type,
            "category":    trade_type,
            "entry":       entry,
            "exit":        exit_price,
            "shares":      shares,
            "pnl_amount":  pnl_amount,
            "pnl_pct":     pnl_pct,
            "hold_days":   hold_days,
            "hold_hours":  round(hold_hours, 2),
            "r_multiple":  r_multiple,
            "mfe_pct":     mfe_pct,
            "mae_pct":     mae_pct,
            "reason":      reason,
            # Entry context carried through so OutcomeTracker needs no re-query
            "confidence":    pos.get("confidence"),
            "alignment":     pos.get("alignment"),
            "regime":        pos.get("regime_at_entry"),
            "entry_quality": pos.get("entry_quality"),
            "regime_match":  pos.get("regime_match"),
            "risk_amount":   risk_amount,
        }

        # Auto-write outcome for learning loop
        self._record_outcome(result)

        logger.info(
            f"[{ticker}] Position closed: id={position_id} {trade_type} "
            f"pnl={pnl_pct:+.2%} (₹{pnl_amount:+.2f}) reason={reason}"
        )
        return result

    def _record_outcome(self, trade: Dict) -> None:
        """Write to signal_outcomes after every close — the learning loop."""
        try:
            from risk.outcome_tracker import OutcomeTracker
            OutcomeTracker(self.db_path).record(trade)
        except Exception as e:
            logger.warning(f"outcome_tracker failed: {e}")

    @staticmethod
    def _hold_time(opened_at) -> tuple:
        """Return (hold_days, hold_hours) from the open timestamp."""
        try:
            opened_dt = datetime.strptime(str(opened_at)[:19], "%Y-%m-%d %H:%M:%S")
            delta = datetime.now() - opened_dt
            return delta.days, delta.total_seconds() / 3600.0
        except Exception:
            return 0, 0.0

    @staticmethod
    def _excursions(pos: Dict, entry: float, signal: str) -> tuple:
        """
        Max favorable / adverse excursion in % from entry, direction-aware.

        Uses the highest/lowest price tracked while the position was open
        (updated every exit-check tick). For a LONG the high is favorable and
        the low is adverse; for a SHORT it inverts. Returns (mfe_pct, mae_pct)
        as positive/negative percentages, or (None, None) if untracked.
        """
        try:
            hi = float(pos.get("highest_price") or entry)
            lo = float(pos.get("lowest_price")  or entry)
            if entry <= 0:
                return None, None
            if signal == "LONG":
                mfe = (hi - entry) / entry
                mae = (lo - entry) / entry
            else:  # SHORT — a falling price is favorable
                mfe = (entry - lo) / entry
                mae = (entry - hi) / entry
            return round(mfe, 6), round(mae, 6)
        except Exception:
            return None, None

    # ─────────────────────────────────────────────────────────────
    # EXIT CONDITION CHECKS
    # ─────────────────────────────────────────────────────────────

    def _check_position(self, pos: Dict, price: float) -> Dict:
        """Run all exit checks on a single position."""
        pos_id     = pos["id"]
        signal     = pos["signal"]
        entry      = float(pos["entry_price"])
        stop       = float(pos["stop_price"])
        target     = float(pos["target_price"])
        highest    = float(pos.get("highest_price", entry))
        lowest     = float(pos.get("lowest_price",  entry))
        opened_at  = pos.get("opened_at", str(datetime.now()))
        trade_type = pos.get("trade_type", "swing")
        shares     = int(pos.get("shares", 0))

        # Compute 1R and 2R ladder targets from entry and stop
        stop_dist  = abs(entry - stop)
        if signal == "LONG":
            target_1r = entry + stop_dist * 1.0
            target_2r = entry + stop_dist * 2.0
        else:
            target_1r = entry - stop_dist * 1.0
            target_2r = entry - stop_dist * 2.0

        # P&L
        if signal == "LONG":
            pnl_pct = (price - entry) / entry
        else:
            pnl_pct = (entry - price) / entry

        # Update high/low watermark
        new_high = max(highest, price)
        new_low  = min(lowest,  price)
        self._update_watermark(pos_id, new_high, new_low)

        # ── Check 1: Hard stop ────────────────────────────────────
        if signal == "LONG"  and price <= stop:
            return self._exit(pos_id, signal, price, pnl_pct,
                              "STOP_LOSS", "Hard stop loss triggered")
        if signal == "SHORT" and price >= stop:
            return self._exit(pos_id, signal, price, pnl_pct,
                              "STOP_LOSS", "Hard stop loss triggered")

        # ── Check 2: Exit ladder — 1R partial (50%) ──────────────
        if shares > 1:
            at_1r = (signal == "LONG" and price >= target_1r and price < target_2r)
            at_2r = (signal == "LONG" and price >= target_2r)
            if signal == "SHORT":
                at_1r = (price <= target_1r and price > target_2r)
                at_2r = (price <= target_2r)

            if at_2r:
                # Book 30% at 2R, trail final 20% with chandelier-style trail
                book_shares = max(1, int(shares * 0.30))
                trail_stop  = (new_high * (1 - 0.02)) if signal == "LONG" else (new_low * (1 + 0.02))
                if (signal == "LONG" and price <= trail_stop) or \
                   (signal == "SHORT" and price >= trail_stop):
                    return self._partial_exit(
                        pos_id, signal, price, pnl_pct,
                        "CHANDELIER_TRAIL",
                        f"2R+trail: ₹{trail_stop:.2f} broken | booking {book_shares} sh",
                        book_pct=1.0,
                    )
                return self._partial_exit(
                    pos_id, signal, price, pnl_pct,
                    "TARGET_2R",
                    f"2R target ₹{target_2r:.2f} hit — booking 30%",
                    book_pct=0.30,
                )
            elif at_1r:
                return self._partial_exit(
                    pos_id, signal, price, pnl_pct,
                    "TARGET_1R",
                    f"1R target ₹{target_1r:.2f} hit — booking 50%",
                    book_pct=0.50,
                )

        # ── Full target hit (single-share position or full ladder done) ──
        if signal == "LONG"  and price >= target:
            return self._exit(pos_id, signal, price, pnl_pct,
                              "TARGET_HIT", f"Target ₹{target:.2f} reached")
        if signal == "SHORT" and price <= target:
            return self._exit(pos_id, signal, price, pnl_pct,
                              "TARGET_HIT", f"Target ₹{target:.2f} reached")

        # ── Check 3: Trailing stop ────────────────────────────────
        if pnl_pct >= self.TRAIL_TRIGGER:
            if signal == "LONG":
                trail_stop = new_high * (1 - self.TRAIL_DISTANCE)
                if price <= trail_stop:
                    return self._exit(pos_id, signal, price, pnl_pct,
                                      "TRAILING_STOP",
                                      f"Trailing stop: ₹{trail_stop:.2f} "
                                      f"({self.TRAIL_DISTANCE:.1%} from ₹{new_high:.2f} high)")
            else:  # SHORT
                trail_stop = new_low * (1 + self.TRAIL_DISTANCE)
                if price >= trail_stop:
                    return self._exit(pos_id, signal, price, pnl_pct,
                                      "TRAILING_STOP",
                                      f"Trailing stop: ₹{trail_stop:.2f}")

        # ── Check 4: Time exit ────────────────────────────────────
        try:
            days_held = (
                datetime.now() -
                datetime.strptime(str(opened_at)[:19], "%Y-%m-%d %H:%M:%S")
            ).days
        except Exception:
            days_held = 0

        max_hold = self.MAX_HOLD.get(trade_type, 7)
        if days_held >= max_hold:
            return self._exit(pos_id, signal, price, pnl_pct,
                              "TIME_EXIT",
                              f"Max hold {max_hold} days reached (held {days_held}d)")

        # ── Check 5: Intraday must close by 15:15 ────────────────
        if trade_type == "intraday":
            now = datetime.now()
            close_time = now.replace(hour=15, minute=15, second=0)
            if now >= close_time:
                return self._exit(pos_id, signal, price, pnl_pct,
                                  "INTRADAY_CLOSE",
                                  "Intraday: must close before 15:15")

        # No exit triggered
        return {
            "should_exit":  False,
            "position_id":  pos_id,
            "signal":       signal,
            "current_price":price,
            "pnl_pct":      pnl_pct,
            "days_held":    days_held,
        }

    def _exit(
        self,
        pos_id:   int,
        signal:   str,
        price:    float,
        pnl_pct:  float,
        reason:   str,
        message:  str,
    ) -> Dict:
        return {
            "should_exit":  True,
            "partial":      False,
            "position_id":  pos_id,
            "signal":       signal,
            "exit_price":   price,
            "pnl_pct":      round(pnl_pct, 6),
            "reason":       reason,
            "message":      message,
        }

    def _partial_exit(
        self,
        pos_id:   int,
        signal:   str,
        price:    float,
        pnl_pct:  float,
        reason:   str,
        message:  str,
        book_pct: float = 0.50,
    ) -> Dict:
        return {
            "should_exit":  True,
            "partial":      True,
            "book_pct":     book_pct,
            "position_id":  pos_id,
            "signal":       signal,
            "exit_price":   price,
            "pnl_pct":      round(pnl_pct, 6),
            "reason":       reason,
            "message":      message,
        }

    def close_position_partial(
        self,
        position_id: int,
        exit_price:  float,
        reason:      str,
        book_pct:    float = 0.50,
    ) -> Dict:
        """
        Book a fraction of the position (e.g. 50% at 1R, 30% at 2R).
        Reduces shares in open_trades; writes partial row to closed_trades.

        Atomic: INSERT + UPDATE/DELETE in one transaction.
        Raises on failure so caller can log without silently losing the trade.
        """
        conn = self._connect()
        try:
            pos = conn.execute(
                "SELECT * FROM open_trades WHERE id=?", (position_id,)
            ).fetchone()
            if not pos:
                logger.warning(f"close_position_partial: id={position_id} not found")
                return {}

            pos          = dict(pos)
            entry        = float(pos["entry_price"])
            total_shares = int(pos["shares"])
            signal       = pos["signal"]
            signal_uuid  = pos.get("signal_uuid") or ""
            ticker       = pos.get("ticker")      or "HDFCBANK.NS"
            trade_type   = pos.get("trade_type")  or "swing"

            book_shares = max(1, int(total_shares * book_pct))
            remain      = max(0, total_shares - book_shares)
            risk_amount = float(pos.get("risk_amount") or 0.0)

            if signal == "LONG":
                pnl_pct = (exit_price - entry) / entry
            else:
                pnl_pct = (entry - exit_price) / entry

            pnl_amount = round(pnl_pct * entry * book_shares, 2)

            # R-multiple on the booked slice — risk scales with the fraction booked.
            booked_risk = risk_amount * book_pct
            r_multiple  = round(pnl_amount / booked_risk, 3) if booked_risk > 0 else None
            _, hold_hours = self._hold_time(pos.get("opened_at", str(datetime.now())))
            mfe_pct, mae_pct = self._excursions(pos, entry, signal)

            conn.execute("""
                INSERT INTO closed_trades
                (signal_uuid, ticker, signal, trade_type,
                 entry_price, exit_price, shares,
                 pnl_amount, pnl_pct, exit_reason, close_date, status,
                 confidence, alignment, regime_at_entry, entry_quality,
                 regime_match, size_mult, atr_at_entry, reward_risk,
                 r_multiple, hold_hours, mfe_pct, mae_pct)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                signal_uuid, ticker, signal, trade_type,
                entry, exit_price, book_shares,
                pnl_amount, round(pnl_pct, 6),
                f"{reason}_PARTIAL", str(datetime.now()), "CLOSED",
                pos.get("confidence"), pos.get("alignment"),
                pos.get("regime_at_entry"), pos.get("entry_quality"),
                pos.get("regime_match"), pos.get("size_mult"),
                pos.get("atr_at_entry"), pos.get("reward_risk"),
                r_multiple, round(hold_hours, 2), mfe_pct, mae_pct,
            ))

            if remain > 0:
                conn.execute(
                    "UPDATE open_trades SET shares=? WHERE id=?", (remain, position_id)
                )
            else:
                conn.execute("DELETE FROM open_trades WHERE id=?", (position_id,))

            conn.commit()

        except Exception:
            conn.rollback()
            conn.close()
            raise

        conn.close()

        logger.info(
            f"[{ticker}] Partial exit: id={position_id} {trade_type} "
            f"booked {book_shares}/{total_shares} sh | "
            f"pnl={pnl_pct:+.2%} (₹{pnl_amount:+.2f}) | remain={remain} | reason={reason}"
        )
        return {
            "position_id":   position_id,
            "signal":        signal,
            "signal_uuid":   signal_uuid,
            "ticker":        ticker,
            "trade_type":    trade_type,
            "entry":         entry,
            "exit":          exit_price,
            "booked_shares": book_shares,
            "remaining":     remain,
            "pnl_amount":    pnl_amount,
            "pnl_pct":       pnl_pct,
            "reason":        reason,
        }

    def _update_watermark(self, pos_id: int,
                           high: float, low: float) -> None:
        try:
            conn = self._connect()
            conn.execute("""
                UPDATE open_trades
                SET highest_price=?, lowest_price=?
                WHERE id=?
            """, (high, low, pos_id))
            conn.commit()
            conn.close()
        except Exception:
            pass

    def _get_open_positions(self, ticker: str = None) -> List[Dict]:
        """Return open positions, optionally filtered to a single ticker."""
        try:
            conn = self._connect()
            if ticker:
                rows = conn.execute(
                    "SELECT * FROM open_trades WHERE status='OPEN' AND ticker=?",
                    (ticker,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM open_trades WHERE status='OPEN'"
                ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception:
            return []

    def _connect(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        # WAL gives concurrent reads during writes; foreign_keys enforces integrity
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        c.execute("PRAGMA foreign_keys=ON")
        return c

    def _setup_db(self) -> None:
        """
        Safety-net schema for trading tables.

        `database/db_setup.py._trading_tables()` is the AUTHORITATIVE source.
        This CREATE block exists so ExitEngine works in environments where
        DatabaseSetup hasn't run yet (unit tests, ad-hoc scripts). The two
        CREATEs MUST stay byte-equivalent — when you add a column, add it in
        both files AND to `DatabaseSetup._migrate_columns()` so existing DBs
        gain the column via ALTER TABLE.
        """
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS open_trades (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_uuid    TEXT,
                ticker         TEXT    DEFAULT 'HDFCBANK.NS',
                signal         TEXT,
                entry_price    REAL,
                stop_price     REAL,
                target_price   REAL,
                shares         INTEGER,
                risk_amount    REAL,
                trade_type     TEXT    DEFAULT 'swing',
                highest_price  REAL,
                lowest_price   REAL,
                status         TEXT    DEFAULT 'OPEN',
                opened_at      TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS closed_trades (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_uuid  TEXT,
                ticker       TEXT    DEFAULT 'HDFCBANK.NS',
                signal       TEXT,
                trade_type   TEXT    DEFAULT 'swing',
                entry_price  REAL,
                exit_price   REAL,
                shares       INTEGER,
                pnl_amount   REAL,
                pnl_pct      REAL,
                exit_reason  TEXT,
                close_date   TEXT,
                status       TEXT    DEFAULT 'CLOSED'
            )
        """)
        # Idempotent migration: add any column that existing DBs might be missing
        _migrations = [
            ("open_trades",   "signal_uuid",  "TEXT"),
            ("open_trades",   "ticker",       "TEXT DEFAULT 'HDFCBANK.NS'"),
            ("closed_trades", "signal_uuid",  "TEXT"),
            ("closed_trades", "ticker",       "TEXT DEFAULT 'HDFCBANK.NS'"),
            ("closed_trades", "trade_type",   "TEXT DEFAULT 'swing'"),
        ]
        for table, col, coltype in _migrations:
            existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            if col not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")
        conn.commit()
        conn.close()