"""Round 4: higher risk caps + 1:3 idio + blended scores."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from swing_strategy.run_ml_top5_selector import OUT_BASE, run_portfolio, select_day
from swing_strategy.run_ml_top5_v2 import apply_rr3_labels
from swing_strategy.run_ml_top5_v3 import EXTRA, FEATURE_COLS, add_idio, eval_risk, walk_custom
from swing_strategy.run_ml_top5_selector import build_features


def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    board = []
    idio = pd.read_parquet(OUT_BASE / "Scored_xgb_idio.parquet")
    idio["Entry_Date"] = pd.to_datetime(idio["Entry_Date"])
    ens = pd.read_parquet(OUT_BASE / "Scored_ensemble.parquet")
    ens["Entry_Date"] = pd.to_datetime(ens["Entry_Date"])

    blend = idio.drop(columns=["ML_Score"]).merge(
        idio[["Ticker", "Entry_Date", "ML_Score"]].rename(columns={"ML_Score": "s1"}),
        on=["Ticker", "Entry_Date"],
    ).merge(
        ens[["Ticker", "Entry_Date", "ML_Score"]].rename(columns={"ML_Score": "s2"}),
        on=["Ticker", "Entry_Date"],
    )
    blend["s1p"] = blend.groupby("Entry_Date")["s1"].rank(pct=True)
    blend["s2p"] = blend.groupby("Entry_Date")["s2"].rank(pct=True)
    blend["ML_Score"] = 0.65 * blend["s1p"] + 0.35 * blend["s2p"]

    print("=== Higher risk on xgb_idio / blend ===", flush=True)
    for risk in (1200.0, 1500.0, 2000.0, 2500.0):
        for top_n in (8, 12, 16):
            eval_risk(idio, "idio", top_n, 50_000.0, risk, board)
            eval_risk(idio, "idio", top_n, 100_000.0, risk, board)
            eval_risk(blend, "blend", top_n, 50_000.0, risk, board)

    print("=== 1:3 idio classifier ===", flush=True)
    df = add_idio(apply_rr3_labels(build_features()))
    cols = FEATURE_COLS + [c for c in EXTRA if c != "y_mfe_proxy"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    sc3 = walk_custom(df, cols, "xgb_idio")
    sc3.to_parquet(OUT_BASE / "Scored_xgb_idio_rr3.parquet", index=False)
    for risk in (500.0, 1000.0, 1500.0, 2000.0):
        for top_n in (8, 12):
            eval_risk(sc3, "idio3", top_n, 50_000.0, risk, board)

    bdf = pd.DataFrame(board)
    bdf.to_csv(OUT_BASE / "Leaderboard_v4.csv", index=False)
    best = bdf.sort_values("CAGR_Pct", ascending=False).iloc[0].to_dict()
    print(f"\nBEST v4: {best.get('Experiment')} CAGR {best.get('CAGR_Pct'):+.2f}% DD {best.get('Max_DD_Pct')}", flush=True)
    print(bdf.sort_values("CAGR_Pct", ascending=False).head(15).to_string(index=False), flush=True)
    (OUT_BASE / "best_v4.json").write_text(json.dumps(best, indent=2, default=str), encoding="utf-8")

    hit40 = bdf[bdf["CAGR_Pct"] >= 40]
    print(f"\nConfigs >= 40% CAGR: {len(hit40)}", flush=True)

    model = str(best.get("Model", "idio"))
    src = {"idio": idio, "blend": blend, "idio3": sc3}.get(model, idio)
    picked = select_day(src, int(best.get("Top_N", 12)), None)
    risk = 1000.0
    name = str(best.get("Experiment", ""))
    for tok in name.split("_"):
        if tok.startswith("R") and tok[1:].isdigit():
            risk = float(tok[1:])
    run_portfolio(picked, float(best["Capital"]), risk, name, write_files=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
