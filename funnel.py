#!/usr/bin/env python3
# funnel.py — Signal funnel / near-miss "tape review" (EDGE-3, 2026-06-03).
#
# A 20yr trader journals the setups they PASSED, not just the ones they took —
# that's where the leaks are. This reads the latest per-bank gate_results
# snapshot (written every cycle, one row/ticker) and prints, per bank × category,
# exactly where each signal died and HOW CLOSE it was: direction, confidence vs
# the (regime/VIX/DQ-adjusted) threshold, the gap, Gate-5 grade + R:R, regime
# match, and whether the EDGE-1 expectancy override could rescue it.
#
# Usage:  python3 funnel.py [path/to/trading.db]
# No pipeline re-run, no network — pure DB read. Safe on the live VPS.

import json
import sqlite3
import sys
from collections import Counter

DB = sys.argv[1] if len(sys.argv) > 1 else "database/trading.db"


def _f(x, d=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def main() -> None:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT ticker, gate_results, regime, written_at FROM gate_results ORDER BY ticker"
    ).fetchall()
    conn.close()

    if not rows:
        print("No gate_results yet — run at least one cycle.")
        return

    print(f"\n{'='*92}\nSIGNAL FUNNEL — near-miss tape review   (db: {DB})\n{'='*92}")
    death = Counter()           # which gate kills signals
    near  = []                  # (ticker, cat, gap) for Gate-6 misses

    for r in rows:
        try:
            g = json.loads(r["gate_results"])
        except Exception:
            continue
        tkr = r["ticker"].replace(".NS", "")
        regime = (g.get("gate1") or {}).get("regime", r["regime"] or "?")
        print(f"\n{tkr:11s} regime={regime:16s} {r['written_at']}")

        # Early gate failures (whole-bank)
        for gk, label in (("pre_check", "pre"), ("gate1", "Gate1-regime"),
                          ("gate2", "Gate2-rules"), ("gate3", "Gate3-rank"),
                          ("data_quality", "DataQuality"), ("gate4", "Gate4-ML")):
            node = g.get(gk)
            if node and node.get("passed") is False:
                death[label] += 1
                print(f"    ✗ BLOCKED @ {label}: {node.get('reason','')}")
                break
        else:
            pc = g.get("per_category", {})
            for cat in ("swing", "positional", "intraday"):
                c = pc.get(cat)
                if not c:
                    continue
                direction = c.get("direction", "FLAT")
                conf      = _f(c.get("confidence"))
                rmatch    = c.get("regime_match")
                g5        = c.get("gate5") or {}
                g6        = c.get("gate6") or {}
                grade     = g5.get("entry_quality", "—")
                rr        = _f(g5.get("reward_risk"))
                thr       = _f(g6.get("threshold"))
                passed    = c.get("passed")
                if passed:
                    tag = "✓ PASS" + (" (expectancy)" if g6.get("expectancy_pass") else "")
                    print(f"    {tag:18s} {cat:10s} {direction:5s} conf={conf:.0%} "
                          f"thr={thr:.0%} grade={grade} rr={rr:.1f} aligned={rmatch}")
                else:
                    gap = (thr - conf) if (thr and conf) else None
                    why = c.get("reason") or (g6.get("reason") if g6 else "blocked")
                    if gap is not None and gap > 0:
                        death["Gate6-conf"] += 1
                        near.append((tkr, cat, gap, direction, grade, rr))
                        # would expectancy rescue it if geometry were clean?
                        rescue = (conf >= 0.50 and rr >= 2.0 and gap <= 0.10
                                  and grade in ("A", "B"))
                        flag = "  ← within expectancy reach" if rescue else ""
                        print(f"    ✗ {cat:10s} {direction:5s} conf={conf:.0%} "
                              f"thr={thr:.0%} gap={gap:+.0%} grade={grade} rr={rr:.1f}{flag}")
                    else:
                        death[(why or "blocked").split("—")[0].strip()[:22]] += 1
                        print(f"    ✗ {cat:10s} {direction:5s}: {why}")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'-'*92}\nWHERE SIGNALS DIE (this snapshot):")
    for k, v in death.most_common():
        print(f"    {v:3d}  {k}")
    if near:
        near.sort(key=lambda x: x[2])
        print(f"\nCLOSEST near-misses (Gate-6 confidence):")
        for tkr, cat, gap, d, grade, rr in near[:8]:
            print(f"    {tkr:11s} {cat:10s} {d:5s} short by {gap:.0%}  grade={grade} rr={rr:.1f}")
    print(f"{'='*92}\n")


if __name__ == "__main__":
    main()
