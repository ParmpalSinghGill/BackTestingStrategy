"""
High-Speed Exact 6-Class True Outcome Dataset Evaluator

Evaluates exact daily candlestick price expansions following C2 for all candidate setups to assign exact ground-truth 6-class labels:
- Class 0: SkipTrade (Price hit SL before 1:2 Target)
- Class 1: Target 1:2 (Price reached 1:2 Target, failed before 1:3)
- Class 2: Target 1:3 (Price reached 1:3 Target, failed before 1:5)
- Class 3: Target 1:5 (Price reached 1:5 Target, failed before 1:10)
- Class 4: Target 1:10 (Price reached 1:10 Target, failed before 1:15)
- Class 5: Target 1:15 (Price reached 1:15 Target)
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


def compute_exact_6class_dataset():
    out_csv = REPORTS_DIR / "Exact_True_6Class_Trade_Features_Dataset.csv"
    if out_csv.exists():
        print(f"Loading existing Exact 6-Class Dataset from: {out_csv.resolve()}", flush=True)
        return pd.read_csv(out_csv)

    print("Computing Exact Ground-Truth 6-Class Labels across Daily Stock Data...", flush=True)
    df_setups = pd.read_csv(REPORTS_DIR / "ML_1to2_RR_Trade_Features_Dataset.csv")

    # Scenarios 1 setups
    df_sc1 = df_setups[df_setups["Scenario"] == "Scenario 1 (Green & Close > C1 High)"].copy().reset_index(drop=True)

    # Group setups by Ticker for ultra-fast single-pass daily CSV loading
    grouped = df_sc1.groupby("Ticker")

    records = []
    processed_count = 0
    total_tickers = len(grouped)

    rr_levels = [2, 3, 5, 10, 15]

    for ticker, group in grouped:
        processed_count += 1
        safe_sym = ticker.replace("=", "_").replace("/", "_").replace("^", "_")
        csv_path = DATA_DAILY_DIR / f"{safe_sym}_1d.csv"

        if not csv_path.exists():
            for _, row in group.iterrows():
                r_dict = row.to_dict()
                r_dict["True_6Class_Label"] = 0
                r_dict["Max_RR_Achieved"] = 0
                records.append(r_dict)
            continue

        df_daily = pd.read_csv(csv_path)
        df_daily["Date"] = pd.to_datetime(df_daily["Date"])
        df_daily = df_daily.sort_values("Date").reset_index(drop=True)

        for _, row in group.iterrows():
            r_dict = row.to_dict()
            c2_dt = pd.to_datetime(row["C2_Date"])

            sub = df_daily[df_daily["Date"] >= c2_dt]
            if len(sub) == 0:
                r_dict["True_6Class_Label"] = 0
                r_dict["Max_RR_Achieved"] = 0
                records.append(r_dict)
                continue

            entry_p = row["Entry_Price"]
            sl_p = row["SL_Price"]
            risk = entry_p - sl_p

            if risk <= 0:
                r_dict["True_6Class_Label"] = 0
                r_dict["Max_RR_Achieved"] = 0
                records.append(r_dict)
                continue

            targets = {rr: entry_p + rr * risk for rr in rr_levels}
            max_rr_hit = 0
            hit_dates = {}

            for _, drow in sub.iterrows():
                d_date = drow["Date"]
                d_high = drow["High"]
                d_low = drow["Low"]

                # Check Stop Loss hit
                if d_low <= sl_p:
                    break

                for rr in rr_levels:
                    if rr not in hit_dates and d_high >= targets[rr]:
                        hit_dates[rr] = d_date
                        max_rr_hit = rr

            # Assign exact true label
            if max_rr_hit >= 15:
                label = 5
            elif max_rr_hit >= 10:
                label = 4
            elif max_rr_hit >= 5:
                label = 3
            elif max_rr_hit >= 3:
                label = 2
            elif max_rr_hit >= 2:
                label = 1
            else:
                label = 0

            r_dict["True_6Class_Label"] = label
            r_dict["Max_RR_Achieved"] = max_rr_hit
            r_dict["Exit_Date_MaxRR"] = hit_dates.get(max_rr_hit, sub.iloc[-1]["Date"])
            records.append(r_dict)

        if processed_count % 300 == 0:
            print(f"Processed {processed_count}/{total_tickers} tickers...", flush=True)

    df_out = pd.DataFrame(records)
    df_out.to_csv(out_csv, index=False)
    print(f"\nExact 6-Class Dataset exported to: {out_csv.resolve()}", flush=True)
    return df_out

if __name__ == "__main__":
    compute_exact_6class_dataset()
