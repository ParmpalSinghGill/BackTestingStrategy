"""
Analyze Trade Selection: Taking All Trades vs Taking Just a Few Selected Trades Per Day
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "Reports"

def main():
    opt_csv = REPORTS_DIR / "quick_reversal_optimized_trades.csv"
    if not opt_csv.exists():
        print(f"Error: {opt_csv} not found.")
        return
        
    df = pd.read_csv(opt_csv)
    
    # Filter 15m timeframe (best performing)
    df15 = df[df['Timeframe'] == '15m'].copy()
    df15['Date'] = pd.to_datetime(df15['Entry_Time']).dt.date
    
    total_days = df15['Date'].nunique()
    total_trades = len(df15)
    avg_trades_per_day = total_trades / total_days if total_days > 0 else 0
    
    print(f"Total Trading Days in Dataset: {total_days}")
    print(f"Total 15m Confirmed Setup Triggers: {total_trades}")
    print(f"Average Setup Triggers Per Day across 148 stocks: {avg_trades_per_day:.1f} trades/day\n")
    
    # Compare:
    # 1. Taking ALL trades
    all_win_rate = df15['Is_Winner'].mean() * 100
    all_avg_pnl = df15['Net_PnL_Pct'].mean()
    
    # 2. Taking FIRST 2 trades per day
    df_first2 = df15.groupby('Date').head(2)
    f2_win_rate = df_first2['Is_Winner'].mean() * 100
    f2_avg_pnl = df_first2['Net_PnL_Pct'].mean()
    f2_trades_per_day = len(df_first2) / total_days
    
    # 3. Taking FIRST 5 trades per day
    df_first5 = df15.groupby('Date').head(5)
    f5_win_rate = df_first5['Is_Winner'].mean() * 100
    f5_avg_pnl = df_first5['Net_PnL_Pct'].mean()
    f5_trades_per_day = len(df_first5) / total_days
    
    print("==========================================================================")
    print("        TRADE SELECTION ANALYSIS: TAKING ALL VS TAKING A FEW TRADES       ")
    print("==========================================================================")
    print(f"1. TAKING ALL TRADES ({total_trades} total trades):")
    print(f"   - Win Rate: {all_win_rate:.1f}%")
    print(f"   - Avg PnL / Trade: {all_avg_pnl:+.2f}%\n")
    
    print(f"2. TAKING JUST FIRST 2 TRADES PER DAY ({len(df_first2)} total trades, {f2_trades_per_day:.1f} trades/day):")
    print(f"   - Win Rate: {f2_win_rate:.1f}%")
    print(f"   - Avg PnL / Trade: {f2_avg_pnl:+.2f}%\n")
    
    print(f"3. TAKING JUST FIRST 5 TRADES PER DAY ({len(df_first5)} total trades, {f5_trades_per_day:.1f} trades/day):")
    print(f"   - Win Rate: {f5_win_rate:.1f}%")
    print(f"   - Avg PnL / Trade: {f5_avg_pnl:+.2f}%")
    print("==========================================================================")

if __name__ == "__main__":
    main()
