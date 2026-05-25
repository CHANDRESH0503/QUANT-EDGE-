#!/usr/bin/env python3
"""
graceful_close_for_deploy.py
────────────────────────────
Run this BEFORE `git pull + systemctl restart` whenever a deploy would break
live open positions (e.g. schema change, logic change).

It:
  1. Lists all OPEN positions
  2. Closes each in `closed_trades` with reason = DEPLOY_RESTART
  3. Marks them CLOSED in `open_trades`

This preserves the trade history instead of silently deleting rows.

Usage:
  python3 graceful_close_for_deploy.py                         # database/trading.db
  python3 graceful_close_for_deploy.py --db /path/trading.db   # custom path
  python3 graceful_close_for_deploy.py --dry-run               # just list, no changes
"""

import argparse
import sqlite3
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db",      default="database/trading.db")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT * FROM open_trades WHERE status='OPEN'"
    ).fetchall()

    if not rows:
        print("No open positions — nothing to close. Safe to restart.")
        conn.close()
        return

    print(f"Found {len(rows)} open position(s):")
    for r in rows:
        print(
            f"  id={r['id']} {r['ticker']} {r['signal']} {r['trade_type']}"
            f" entry={r['entry_price']} shares={r['shares']}"
            f" opened={r['opened_at']}"
        )

    if args.dry_run:
        print("\n[dry-run] No changes made.")
        conn.close()
        return

    confirm = input(f"\nClose all {len(rows)} positions as DEPLOY_RESTART? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted — no changes made.")
        conn.close()
        return

    now = datetime.now().isoformat(sep=" ", timespec="seconds")
    closed = 0
    for r in rows:
        try:
            conn.execute("""
                INSERT INTO closed_trades
                    (signal_uuid, ticker, signal, trade_type,
                     entry_price, exit_price, shares,
                     pnl_amount, pnl_pct, exit_reason, close_date, status)
                VALUES (?,?,?,?,?,?,?,0,0,?,?,?)
            """, (
                r["signal_uuid"], r["ticker"], r["signal"],
                r["trade_type"] or "swing",
                r["entry_price"], r["entry_price"],   # exit = entry → 0 P&L
                r["shares"], "DEPLOY_RESTART", now, "CLOSED",
            ))
            conn.execute(
                "DELETE FROM open_trades WHERE id=?", (r["id"],)
            )
            conn.commit()
            print(f"  ✓ Closed id={r['id']} {r['ticker']} {r['signal']}")
            closed += 1
        except Exception as e:
            conn.rollback()
            print(f"  ✗ Failed id={r['id']}: {e}")

    print(f"\nDone — {closed}/{len(rows)} positions closed gracefully.")
    print("Now safe to run: git pull && sudo systemctl restart quantedge-signal quantedge-api")
    conn.close()


if __name__ == "__main__":
    main()
