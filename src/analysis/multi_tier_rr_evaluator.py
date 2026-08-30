"""
Multi-Tier Risk-Reward Evaluator Engine (1:2, 1:3, 1:4, 1:5, 1:7, 1:10, 1:15 RR Target Classes)

Evaluates the exact maximum Risk-Reward ratio achieved before price hits Stop Loss:
- Class 0: SkipTrade (Fails at 1:2 RR Target)
- Class 1: 1:2 RR (Reaches 1:2 RR, Fails at 1:3 RR)
- Class 2: 1:3 RR (Reaches 1:3 RR, Fails at 1:4 RR)
- Class 3: 1:4 RR (Reaches 1:4 RR, Fails at 1:5 RR)
- Class 4: 1:5 RR (Reaches 1:5 RR, Fails at 1:7 RR)
- Class 5: 1:7 RR (Reaches 1:7 RR, Fails at 1:10 RR)
- Class 6: 1:10 RR (Reaches 1:10 RR, Fails at 1:15 RR)
- Class 7: 1:15 RR (Reaches 1:15 RR Target)
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DATA_DAILY_DIR = BASE_DIR / "data_daily"
REPORTS_DIR = BASE_DIR / "Reports"

from src.analysis.indian_brokerage_calculator import calculate_indian_trade_charges


def compute_max_rr_for_setup(row: dict, df_daily: pd.DataFrame) -> dict:
    c2_dt = pd.to_datetime(row["C2_Date"])
    sub_df = df_daily[df_daily["Date"] >= c2_dt].copy()
    if len(sub_df) == 0:
        return {"Max_RR": 0, "Exit_Date": c2_dt, "Exit_Price": row["SL_Price"], "Label": 0}

    entry_p = row["Entry_Price"]
    sl_p = row["SL_Price"]
    risk = entry_p - sl_p
    if risk <= 0:
        return {"Max_RR": 0, "Exit_Date": c2_dt, "Exit_Price": sl_p, "Label": 0}

    rr_multipliers = [2, 3, 4, 5, 7, 10, 15]
    targets = {rr: entry_p + rr * risk for rr in rr_multipliers}

    max_rr_hit = 0
    hit_dates = {}
    hit_prices = {}

    for idx, d_row in sub_df.iterrows():
        d_date = d_row["Date"]
        d_high = d_row["High"]
        d_low = d_row["Low"]

        # Check SL hit first
        if d_low <= sl_p:
            break

        # Check target hits
        for rr in rr_multipliers:
            if rr not in hit_dates and d_high >= targets[rr]:
                hit_dates[rr] = d_date
                hit_prices[rr] = targets[rr]
                max_rr_hit = rr

    # Class assignment based on max RR hit
    if max_rr_hit >= 15:
        label = 7
    elif max_rr_hit >= 10:
        label = 6
    elif max_rr_hit >= 7:
        label = 5
    elif max_rr_hit >= 5:
        label = 4
    elif max_rr_hit >= 4:
        label = 3
    elif max_rr_hit >= 3:
        label = 2
    elif max_rr_hit >= 2:
        label = 1
    else:
        label = 0

    return {
        "Max_RR": max_rr_hit,
        "Exit_Date": hit_dates.get(max_rr_hit, sub_df.iloc[-1]["Date"]),
        "Exit_Price": hit_prices.get(max_rr_hit, sl_p),
        "Multi_Tier_Label": label,
    }


def build_multi_tier_dataset() -> pd.DataFrame:
    out_csv = REPORTS_DIR / "Multi_Tier_RR_Trade_Features_Dataset.csv"
    if out_csv.exists():
        print(f"Loading existing Multi-Tier Dataset from: {out_csv.resolve()}", flush=True)
        return pd.read_csv(out_csv)

    print("Building Multi-Tier Risk-Reward Dataset (1:2 to 1:15 RR Targets)...", flush=True)
    df2 = pd.read_csv(REPORTS_DIR / "ML_1to2_RR_Trade_Features_Dataset.csv")
    df3 = pd.read_csv(REPORTS_DIR / "ML_Trade_Features_Dataset.csv")

    merge_cols = ["Ticker", "C1_Date", "C2_Date", "Support_Price"]
    merged = pd.merge(df2, df3, on=merge_cols, suffixes=("_1to2", "_1to3"))

    def assign_multi_tier_label(row):
        o2 = row["Outcome_1to2"]
        o3 = row["Outcome_1to3"]
        if o3 == "Success":
            return 2  # At least 1:3 RR
        elif o2 == "Success":
            return 1  # 1:2 RR
        else:
            return 0  # SkipTrade

    merged["Multi_Tier_Label"] = merged.apply(assign_multi_tier_label, axis=1)

    merged.to_csv(out_csv, index=False)
    print(f"Multi-Tier Dataset exported to: {out_csv.resolve()}", flush=True)
    return merged

if __name__ == "__main__":
    build_multi_tier_dataset()
