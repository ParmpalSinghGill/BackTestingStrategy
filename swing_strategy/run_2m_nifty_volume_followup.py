"""Tighter Meta_P / size / pause follow-up on >=2m + light Nifty volume."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from swing_strategy.run_2m_nifty_volume_hunt import (
    MIN_AGE,
    NIFTY_PATH,
    OUT,
    age_ok,
    attach,
    eval_pct,
    hit,
    load_nifty,
    pick,
)
from swing_strategy.run_pause_search_v2 import simulate as pause_sim

PREV = OUT / "vol_nifty_2m.json"
SCORED = BASE / "Reports" / "SwingLowCagrHunt" / "Scored_hgb.parquet"


def main() -> None:
    nifty = load_nifty()
    scored = attach(pd.read_parquet(SCORED), nifty)
    scored = scored[age_ok(scored["Liquidity_Date"], scored["Entry_Date"], MIN_AGE)].copy()
    sh15 = float(scored["sh_n"].quantile(0.15))
    to15 = float(scored["to_n"].quantile(0.15))
    universes = {
        "2m": scored,
        "2m_sh15": scored[scored["sh_n"] >= sh15],
        "2m_to15": scored[scored["to_n"] >= to15],
    }
    print(f"sh15={sh15:.4f} to15={to15:.4f}", flush=True)
    rows = []
    hits = []
    for uname, u in universes.items():
        print(f"===== {uname} n={len(u):,} =====", flush=True)
        for thresh in (0.46, 0.48, 0.50, 0.52, 0.55):
            for top_n in (8, 12, 16, 24, 32):
                picked = pick(u, top_n, thresh)
                if picked.empty:
                    continue
                for pct in (0.02, 0.025, 0.03, 0.035, 0.04):
                    res = eval_pct(picked, pct)
                    tag = f"{uname}_t{thresh}_top{top_n}_p{pct:.3f}"
                    row = {"tag": tag, "universe": uname, "thresh": thresh, "top_n": top_n, "pct": pct, **res}
                    rows.append(row)
                    if hit(res):
                        hits.append(row)
                        print(f"  HIT {tag} {res['CAGR']:+.2f}% DD {res['DD']}", flush=True)
        best = max(rows[-25:], key=lambda r: r["CAGR"] - 0.35 * r["DD"])
        print(f"  last-block best {best['tag']} {best['CAGR']:+.2f}% DD {best['DD']}", flush=True)

    if not hits:
        print("===== overlays on CAGR>=46 books =====", flush=True)
        cands = [r for r in rows if r["CAGR"] >= 46]
        cands.sort(key=lambda r: -(r["CAGR"] - 0.35 * r["DD"]))
        seen = set()
        overlays = [
            {"tag": "half3_0.5", "cut_after": 3, "size_cut": 0.50},
            {"tag": "half2_0.5", "cut_after": 2, "size_cut": 0.50},
            {"tag": "half3_0.35", "cut_after": 3, "size_cut": 0.35},
            {"tag": "pause6_15", "k": 6, "pause_days": 15},
            {"tag": "dayloss2_10", "day_loss_n": 2, "day_loss_pause": 10},
            {"tag": "dd30_15", "dd_pause": 0.30, "dd_days": 15, "dd_resume": 0.08},
            {"tag": "dd28_10", "dd_pause": 0.28, "dd_days": 10, "dd_resume": 0.10},
        ]
        for r in cands[:8]:
            key = (r["universe"], r["thresh"], r["top_n"])
            if key in seen:
                continue
            seen.add(key)
            u = universes[r["universe"]]
            picked = pick(u, r["top_n"], r["thresh"])
            cfg_pct = {"pct": r["pct"]}
            print(f"  overlay {r['tag']} {r['CAGR']:+.2f}/{r['DD']}", flush=True)
            for cfg in overlays:
                sim = pause_sim(picked, {**cfg_pct, **cfg})
                row = {
                    "tag": f"{r['tag']}__{cfg['tag']}",
                    "CAGR": sim["CAGR"],
                    "DD": sim["DD"],
                    "N": sim["N"],
                    "pct": r["pct"],
                }
                rows.append(row)
                print(f"    {cfg['tag']:14s} {sim['CAGR']:+6.2f}% DD {sim['DD']:5.1f} n={sim['N']}", flush=True)
                if hit(sim):
                    hits.append(row)

    print("\n===== TARGET 50 / 25 =====", flush=True)
    if hits:
        for r in hits:
            print("HIT", r, flush=True)
    else:
        print("no hit", flush=True)
        for r in sorted(rows, key=lambda x: -x["CAGR"])[:8]:
            print(f"  CAGR {r['CAGR']:+6.2f} DD {r['DD']:5.1f} n={r.get('N')} {r['tag']}", flush=True)
        good = [r for r in rows if r["CAGR"] >= 40]
        for r in sorted(good, key=lambda x: x["DD"])[:8]:
            print(f"  DD   {r['CAGR']:+6.2f} DD {r['DD']:5.1f} n={r.get('N')} {r['tag']}", flush=True)

    prev = json.loads(PREV.read_text(encoding="utf-8")) if PREV.exists() else []
    PREV.write_text(json.dumps(prev + rows, indent=2, default=str), encoding="utf-8")
    print(f"appended {len(rows)} rows -> {PREV}", flush=True)


if __name__ == "__main__":
    main()
