"""Pause grid on the 50%+ 4% equity book. Aim DD < 25 while keeping CAGR >= 50."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from swing_strategy.run_2m_nifty_volume_hunt import (
    MIN_AGE,
    age_ok,
    attach,
    hit,
    load_nifty,
    pick,
)
from swing_strategy.run_pause_search_v2 import simulate as pause_sim

SCORED = BASE / "Reports" / "SwingLowCagrHunt" / "Scored_hgb.parquet"


def main() -> None:
    nifty = load_nifty()
    scored = attach(pd.read_parquet(SCORED), nifty)
    scored = scored[age_ok(scored["Liquidity_Date"], scored["Entry_Date"], MIN_AGE)].copy()
    sh15 = float(scored["sh_n"].quantile(0.15))
    u = scored[scored["sh_n"] >= sh15]
    picked = pick(u, 12, 0.48)
    print(f"picked {len(picked)}  sh15={sh15:.4f}  (shares >= {sh15:.3f} * Nifty)", flush=True)
    cfgs = [{"tag": "base", "pct": 0.04}]
    for k in (4, 5, 6, 7, 8):
        for d in (10, 12, 15, 18, 20, 25, 30):
            cfgs.append({"tag": f"pause{k}_{d}d", "pct": 0.04, "k": k, "pause_days": d})
    for k, skip in ((5, 3), (5, 5), (6, 3), (6, 5), (6, 8), (7, 5)):
        cfgs.append({"tag": f"skip{skip}_k{k}", "pct": 0.04, "k": k, "skip_trades": skip})
        cfgs.append({"tag": f"both_k{k}_s{skip}_d15", "pct": 0.04, "k": k, "skip_trades": skip, "pause_days": 15})
    for n, d in ((2, 5), (2, 8), (2, 10), (2, 12), (3, 8), (3, 10)):
        cfgs.append({"tag": f"dayloss{n}_{d}d", "pct": 0.04, "day_loss_n": n, "day_loss_pause": d})
        cfgs.append({"tag": f"dayloss{n}_{d}d_p6", "pct": 0.04, "day_loss_n": n, "day_loss_pause": d, "k": 6, "pause_days": 15})
    for cut, after in ((0.50, 2), (0.50, 3), (0.40, 3), (0.60, 3)):
        cfgs.append({"tag": f"half{after}_{cut}", "pct": 0.04, "cut_after": after, "size_cut": cut})
        cfgs.append({"tag": f"half{after}_{cut}_p6", "pct": 0.04, "cut_after": after, "size_cut": cut, "k": 6, "pause_days": 15})
    rows = []
    hits = []
    for cfg in cfgs:
        r = pause_sim(picked, cfg)
        row = {"tag": cfg["tag"], "CAGR": r["CAGR"], "DD": r["DD"], "N": r["N"]}
        rows.append(row)
        flag = " HIT" if hit(r) else ""
        if r["CAGR"] >= 50 or r["DD"] < 28 or cfg["tag"] == "base":
            print(f"  {cfg['tag']:22s} {r['CAGR']:+6.2f}% DD {r['DD']:5.1f} n={r['N']}{flag}", flush=True)
        if hit(r):
            hits.append(row)
    print("\nCAGR>=50 lowest DD:", flush=True)
    ok = [r for r in rows if r["CAGR"] >= 50]
    if not ok:
        print("  none", flush=True)
    else:
        for r in sorted(ok, key=lambda x: x["DD"])[:15]:
            print(f"  {r['CAGR']:+6.2f}% DD {r['DD']:5.1f} n={r['N']} {r['tag']}", flush=True)
    print("HIT count", len(hits), flush=True)
    if hits:
        for r in hits:
            print(" ", r, flush=True)


if __name__ == "__main__":
    main()
