"""Bridge the 58%/30% vs 41%/24% gap at 2% + fair BE."""
from __future__ import annotations

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from swing_strategy.run_2pct_50_25_hunt import ensure_1m, is_hit, load_nifty, nifty_filter, pick, sim
from swing_strategy.run_2pct_valid_hunt import be_fair, rewrite


def main() -> None:
    nifty = load_nifty()
    re1 = ensure_1m(nifty)
    picked = pick(re1, 16, 0.48)
    print(f"rewrite n={len(picked)}", flush=True)
    rw = rewrite(picked, be_fair)
    cfgs = [
        ("base", {}),
        ("dl2_6", {"day_loss_n": 2, "day_loss_pause": 6}),
        ("dl2_8", {"day_loss_n": 2, "day_loss_pause": 8}),
        ("dl2_10", {"day_loss_n": 2, "day_loss_pause": 10}),
        ("dl2_12", {"day_loss_n": 2, "day_loss_pause": 12}),
        ("dl2_15", {"day_loss_n": 2, "day_loss_pause": 15}),
        ("dl3_8", {"day_loss_n": 3, "day_loss_pause": 8}),
        ("dl3_10", {"day_loss_n": 3, "day_loss_pause": 10}),
        ("p4_10", {"k": 4, "pause_days": 10}),
        ("p5_10", {"k": 5, "pause_days": 10}),
        ("p6_10", {"k": 6, "pause_days": 10}),
        ("max4", {"max_day": 4}),
        ("max5", {"max_day": 5}),
        ("h2", {"cut_after": 2, "size_cut": 0.5}),
        ("h3_04", {"cut_after": 3, "size_cut": 0.4}),
        ("dl2_10_h3", {"day_loss_n": 2, "day_loss_pause": 10, "cut_after": 3, "size_cut": 0.5}),
        ("dl2_8_max6", {"day_loss_n": 2, "day_loss_pause": 8, "max_day": 6}),
        ("dl2_10_max6", {"day_loss_n": 2, "day_loss_pause": 10, "max_day": 6}),
        ("dl2_10_p6", {"day_loss_n": 2, "day_loss_pause": 10, "k": 6, "pause_days": 15}),
    ]
    for nf in ("all", "sma50", "sma100", "sma200", "ret20"):
        u = nifty_filter(rw, nf)
        print(f"===== {nf} n={len(u)} =====", flush=True)
        for name, cfg in cfgs:
            r = sim(u, cfg)
            flag = " HIT" if is_hit(r) else ""
            print(f"  {name:14s} {r['CAGR']:+6.2f}% DD {r['DD']:5.1f} n={r['N']}{flag}", flush=True)
            if is_hit(r):
                print("FOUND", nf, name, r, flush=True)
                return
    print("no hit", flush=True)


if __name__ == "__main__":
    main()
