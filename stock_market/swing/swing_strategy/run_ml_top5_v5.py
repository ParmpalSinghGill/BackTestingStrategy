"""Fine grid around 39.6% to clear 40% Net Zerodha CAGR."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from swing_strategy.run_ml_top5_selector import OUT_BASE, run_portfolio, select_day
from swing_strategy.run_ml_top5_v3 import eval_risk


def main():
    idio = pd.read_parquet(OUT_BASE / "Scored_xgb_idio.parquet")
    idio["Entry_Date"] = pd.to_datetime(idio["Entry_Date"])
    board = []
    for risk in (2600, 2800, 3000, 3200, 3500, 4000):
        for top_n in (10, 11, 12, 13, 14):
            eval_risk(idio, "idio", top_n, 50_000.0, float(risk), board)
    bdf = pd.DataFrame(board).sort_values("CAGR_Pct", ascending=False)
    print(bdf.head(12).to_string(index=False), flush=True)
    bdf.to_csv(OUT_BASE / "Leaderboard_v5.csv", index=False)
    best = bdf.iloc[0].to_dict()
    print(f"\nBEST v5: {best['Experiment']} CAGR {best['CAGR_Pct']:+.2f}% DD {best['Max_DD_Pct']}", flush=True)
    print(f"Configs >= 40: {int((bdf.CAGR_Pct >= 40).sum())}", flush=True)
    (OUT_BASE / "best_v5.json").write_text(json.dumps(best, indent=2, default=str), encoding="utf-8")
    if best["CAGR_Pct"] >= 40:
        picked = select_day(idio, int(best["Top_N"]), None)
        risk = float(str(best["Experiment"]).split("_R")[1].split("_")[0])
        run_portfolio(picked, 50_000.0, risk, str(best["Experiment"]), write_files=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
