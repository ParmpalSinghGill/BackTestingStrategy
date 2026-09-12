"""Same Swing_PP entries, fair-BE, 2% equity, paper-gate — targets 1:2 / 1:3 / 1:4 / 1:5."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from swing_strategy.run_2pct_valid_hunt import be_fair, rewrite
from swing_strategy.run_pred_paper_gate import CACHE_BE, M46, load_pred_list, sim_pred_gate

OUT = BASE / "Reports" / "SwingLow_OldLiquidity"
LOG = OUT / "swing_pp_rr_scan.json"


def rewrite_rr(base: pd.DataFrame, rr: float) -> pd.DataFrame:
    def fn(idx, entry, sl, opens, highs, lows, n):
        return be_fair(idx, entry, sl, opens, highs, lows, n, rr=rr)

    print(f"[rewrite] fair BE target 1:{rr:.0f} on {len(base):,} names ...", flush=True)
    out = rewrite(base, fn)
    out["Outcome"] = [
        "Success" if float(r) >= rr - 0.5 else ("Failure" if float(r) < -0.25 else "BE")
        for r in out["Realized_R"]
    ]
    return out


def main() -> None:
    base = load_pred_list()
    base["Entry_Date"] = pd.to_datetime(base["Entry_Date"])
    base["Exit_Date"] = pd.to_datetime(base["Exit_Date"])
    rows = []
    for rr in (2.0, 3.0, 4.0, 5.0):
        cache = OUT / f"Pred_fairBE_age1m_t048_top16_rr{int(rr)}.parquet"
        if rr == 2.0 and CACHE_BE.exists() and not cache.exists():
            rw = base.copy()
        elif cache.exists():
            rw = pd.read_parquet(cache)
            rw["Entry_Date"] = pd.to_datetime(rw["Entry_Date"])
            rw["Exit_Date"] = pd.to_datetime(rw["Exit_Date"])
        else:
            rw = rewrite_rr(base, rr)
            OUT.mkdir(parents=True, exist_ok=True)
            rw.to_parquet(cache, index=False)
        n_win = int((rw["Realized_R"] >= rr - 0.5).sum())
        n_sl = int((rw["Realized_R"] < -0.25).sum())
        n_be = len(rw) - n_win - n_sl
        win_r = rr - 0.5
        ungated = sim_pred_gate(rw, {"tag": f"1to{int(rr)}_nogate", "win_r": win_r, **M46})
        gated = sim_pred_gate(
            rw,
            {"tag": f"1to{int(rr)}_gate50_74", "roll_n": 50, "fail_max": 0.74, "win_r": win_r, **M46},
        )
        for r, kind in ((ungated, "no_gate"), (gated, "paper_gate")):
            rec = {
                "RR": f"1:{int(rr)}",
                "kind": kind,
                "paper_wins": n_win,
                "paper_SL": n_sl,
                "paper_BE": n_be,
                **r,
            }
            rows.append(rec)
            print(
                f"  1:{int(rr):.0f} {kind:11s}  CAGR {r['CAGR']:+7.2f}%  DD {r['DD']:5.1f}%  "
                f"n={r['N']:4d}  skipd={r.get('skipped_days', 0)}  "
                f"paper W/SL/BE {n_win}/{n_sl}/{n_be}",
                flush=True,
            )
    LOG.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"wrote {LOG}", flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
