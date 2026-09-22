"""fair-BE (bar walk) + Nifty SMA / pause at locked 2%."""
from __future__ import annotations

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from swing_strategy.run_2pct_50_25_hunt import (
    ensure_1m,
    is_hit,
    load_nifty,
    nifty_filter,
    pick,
    sim,
)
from swing_strategy.run_2pct_valid_hunt import be_fair, rewrite


def main() -> None:
    nifty = load_nifty()
    re1 = ensure_1m(nifty)
    hits = []
    for top, th in ((16, 0.48), (32, 0.42), (12, 0.48), (24, 0.42), (8, 0.50)):
        picked = pick(re1, top, th)
        print(f"rewrite re1 t{th} top{top} n={len(picked)}", flush=True)
        rw = rewrite(picked, be_fair)
        for nf in ("all", "sma50", "sma200", "ret20", "sma50_ret20", "not_crash"):
            u = nifty_filter(rw, nf)
            if u.empty:
                continue
            for name, cfg in (
                ("base", {}),
                ("p6_15", {"k": 6, "pause_days": 15}),
                ("p7_15", {"k": 7, "pause_days": 15}),
                ("dl2_10", {"day_loss_n": 2, "day_loss_pause": 10}),
                ("max6", {"max_day": 6}),
                ("max8", {"max_day": 8}),
                ("h3", {"cut_after": 3, "size_cut": 0.5}),
                ("p6_max6", {"k": 6, "pause_days": 15, "max_day": 6}),
                ("p6_h3", {"k": 6, "pause_days": 15, "cut_after": 3, "size_cut": 0.5}),
            ):
                r = sim(u, cfg)
                tag = f"fair/{nf}/t{th}/top{top}/{name}"
                flag = " HIT" if is_hit(r) else ""
                if r["CAGR"] >= 45 or r["DD"] <= 26 or flag:
                    print(f"  {tag:55s} {r['CAGR']:+6.2f}% DD {r['DD']:5.1f} n={r['N']}{flag}", flush=True)
                if is_hit(r):
                    hits.append((tag, r))
                    print("FOUND", tag, r, flush=True)
                    return
    print("no hit in fair+sma", flush=True)


if __name__ == "__main__":
    main()
