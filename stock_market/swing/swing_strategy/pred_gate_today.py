"""Print TRADE or SKIP for next session from ALL predicted paper results (no fills)."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from swing_strategy.run_pred_paper_gate import gate_asof, load_pred_list

# Last 50 completed predicted names (paper 1:2 vs SL). Skip if fail% >= 74%.
ROLL_N = 50
FAIL_MAX = 0.74


def main() -> None:
    asof = sys.argv[1] if len(sys.argv) > 1 else None
    if asof is None:
        d = datetime.now().date()
        # use last complete session: before 16:30 yesterday conceptually; default today if weekday after close
        asof = str(d)
    ml = load_pred_list()
    g = gate_asof(ml, asof, ROLL_N, FAIL_MAX)
    nxt = datetime.strptime(g["asof"], "%Y-%m-%d").date() + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    print(g["rule"])
    print(f"asof {g['asof']}  last {g['ready']} paper results  fail% {g.get('fail_pct')}")
    print(f"next session {nxt}:  {'SKIP — do not enter' if g['skip'] else 'TRADE — take the prediction list'}")


if __name__ == "__main__":
    main()
