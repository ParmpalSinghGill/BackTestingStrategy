"""True 2% cap (no 1.8x vol boost, optional no 1.4x rank) + fair BE."""
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
    rw = rewrite(picked, be_fair)
    print(f"n={len(rw)}", flush=True)
    scale_his = (1.0, 1.1, 1.2, 1.3, 1.5, 1.8)
    rank_his = (1.0, 1.1, 1.2, 1.4)
    pauses = [
        {},
        {"day_loss_n": 2, "day_loss_pause": 10},
        {"day_loss_n": 2, "day_loss_pause": 8},
        {"k": 6, "pause_days": 15},
        {"cut_after": 3, "size_cut": 0.5},
        {"day_loss_n": 2, "day_loss_pause": 10, "cut_after": 3, "size_cut": 0.5},
    ]
    for nf in ("all", "sma100"):
        u = nifty_filter(rw, nf)
        for sh in scale_his:
            for rh in rank_his:
                for i, p in enumerate(pauses):
                    cfg = {"scale_hi": sh, "rank_hi": rh, "rank_lo": min(0.6, rh), **p}
                    r = sim(u, cfg)
                    tag = f"{nf}/sh{sh}/rh{rh}/p{i}"
                    flag = " HIT" if is_hit(r) else ""
                    if r["CAGR"] >= 48 or r["DD"] <= 25.5 or flag:
                        print(
                            f"  {tag:32s} {r['CAGR']:+6.2f}% DD {r['DD']:5.1f} n={r['N']}{flag}",
                            flush=True,
                        )
                    if is_hit(r):
                        print("FOUND", tag, r, flush=True)
                        return
    print("no hit", flush=True)


if __name__ == "__main__":
    main()
