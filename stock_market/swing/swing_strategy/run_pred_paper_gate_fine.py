"""Finer paper-prediction gate around 6-fail / 70% windows."""
from __future__ import annotations

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from swing_strategy.run_2pct_50_25_hunt import is_hit
from swing_strategy.run_pred_paper_gate import (
    M46,
    gate_asof,
    load_pred_list,
    sim_pred_gate,
)


def main() -> None:
    ml = load_pred_list()
    rows = []
    for k in (4, 5, 6, 7, 8, 9):
        for d in (4, 5, 6, 7, 8, 9, 10, 12, 15):
            cfg = {"tag": f"k{k}_{d}d", "k": k, "pause_days": d, **M46}
            r = sim_pred_gate(ml, cfg)
            rows.append(r)
            flag = " HIT" if is_hit(r) else ""
            if r["CAGR"] >= 48 or r["DD"] <= 26 or flag:
                print(f"  {cfg['tag']:12s} {r['CAGR']:+6.2f}% DD {r['DD']:5.1f} n={r['N']} skipd={r['skipped_days']}{flag}", flush=True)
            if is_hit(r):
                print("FOUND", r, flush=True)
    for n in (12, 16, 18, 22, 28, 35, 40, 50):
        for fm in (0.62, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76):
            cfg = {"tag": f"f{int(fm*100)}_n{n}", "roll_n": n, "fail_max": fm, **M46}
            r = sim_pred_gate(ml, cfg)
            rows.append(r)
            flag = " HIT" if is_hit(r) else ""
            if r["CAGR"] >= 48 or r["DD"] <= 26 or flag:
                print(f"  {cfg['tag']:12s} {r['CAGR']:+6.2f}% DD {r['DD']:5.1f} n={r['N']} skipd={r['skipped_days']}{flag}", flush=True)
            if is_hit(r):
                print("FOUND", r, flush=True)
    print("\nCAGR>50 lowest DD", flush=True)
    for r in sorted([x for x in rows if x["CAGR"] > 50], key=lambda x: x["DD"])[:10]:
        print(f"  {r['CAGR']:+6.2f}% DD {r['DD']:5.1f} {r.get('tag')}", flush=True)
    print("DD<25 highest CAGR", flush=True)
    for r in sorted([x for x in rows if x["DD"] < 25], key=lambda x: -x["CAGR"])[:10]:
        print(f"  {r['CAGR']:+6.2f}% DD {r['DD']:5.1f} {r.get('tag')}", flush=True)
    g = gate_asof(ml, "2026-09-08", 40, 0.70)
    print("example gate n40 fail70", g, flush=True)
    g2 = gate_asof(ml, "2026-09-08", 20, 0.75)
    print("example gate n20 fail75", g2, flush=True)


if __name__ == "__main__":
    main()
