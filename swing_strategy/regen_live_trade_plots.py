"""Regenerate live-statement trade PNGs after plotter changes (deletes cached files first)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from swing_strategy.plotter import plot_swing_trade_chart
from swing_strategy.run_live_account_statement import CHARTS, EXP, META, REPORTS


def main() -> None:
    CHARTS.mkdir(parents=True, exist_ok=True)
    for p in CHARTS.glob("*.png"):
        p.unlink()

    print("Loading statement ...", flush=True)
    stmt = pd.read_csv(REPORTS / "Swing_Strategy_Account_Statement.csv")
    buys = stmt[stmt["Type"] == "BUY (ENTRY)"][
        ["Trade_ID", "Date", "Ticker", "Price", "Support_Price", "Liquidity_Source"]
    ].rename(columns={"Date": "Entry_Date", "Price": "Entry_Price"})
    sells = stmt[stmt["Type"] == "SELL (EXIT)"][
        ["Trade_ID", "Date", "Price", "Net_PnL"]
    ].rename(columns={"Date": "Exit_Date", "Price": "Exit_Price"})
    trades = buys.merge(sells, on="Trade_ID", how="inner")
    trades["Entry_Date"] = pd.to_datetime(trades["Entry_Date"])
    trades["Exit_Date"] = pd.to_datetime(trades["Exit_Date"])

    print("Loading SL from scored meta ...", flush=True)
    meta = pd.read_parquet(META, columns=["Ticker", "Entry_Date", "SL_Price"])
    meta["Entry_Date"] = pd.to_datetime(meta["Entry_Date"])
    trades = trades.merge(meta, on=["Ticker", "Entry_Date"], how="left")
    trades["SL_Price"] = trades["SL_Price"].fillna(trades["Entry_Price"] * 0.97)
    trades["Target_Price"] = (trades["Entry_Price"] + 2.0 * (trades["Entry_Price"] - trades["SL_Price"])).round(2)

    jobs = trades.to_dict("records")
    pick = jobs[:40] + jobs[-20:]
    mid = sorted(jobs[40:-20], key=lambda x: abs(float(x["Net_PnL"])), reverse=True)[:60]
    uniq, seen = [], set()
    for p in pick + mid:
        tid = int(p["Trade_ID"])
        if tid in seen:
            continue
        seen.add(tid)
        uniq.append({
            "Trade_ID": tid,
            "Ticker": p["Ticker"],
            "C2_Date": pd.Timestamp(p["Entry_Date"]),
            "Exit_Date": pd.Timestamp(p["Exit_Date"]),
            "Support_Price": float(p["Support_Price"]),
            "Entry_Price": float(p["Entry_Price"]),
            "SL_Price": float(p["SL_Price"]),
            "Target_Price": float(p["Target_Price"]),
            "Exit_Price": float(p["Exit_Price"]),
            "Net_PnL": float(p["Net_PnL"]),
            "Liquidity_Type": p["Liquidity_Source"],
            "ML_RR_Choice": "1:2",
        })

    print(f"Replotting {len(uniq)} {EXP} trade charts ...", flush=True)
    ok = 0
    for i, rec in enumerate(uniq, 1):
        uri = plot_swing_trade_chart(rec, CHARTS)
        if uri and uri != "N/A":
            ok += 1
        if i % 20 == 0:
            print(f"  plots {i}/{len(uniq)}", flush=True)
    print(f"DONE plots={ok}/{len(uniq)} -> {CHARTS}", flush=True)


if __name__ == "__main__":
    main()
