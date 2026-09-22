"""
Calculate Average % of Loss and Risk-Reward Metrics
"""

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
    df['entry_dt'] = pd.to_datetime(df['Entry_Time'])
    df['Date'] = df['entry_dt'].dt.date

    print("==========================================================================")
    print("                 AVERAGE LOSS % & RISK-REWARD ANALYSIS                    ")
    print("==========================================================================")
    
    # 1st Attempt Only Subset
    df_1st = df.groupby(['Ticker', 'Date', 'Side', 'Timeframe']).head(1).copy()
    
    for tf in ['15m', '5m', '1m']:
        sub = df_1st[df_1st['Timeframe'] == tf]
        
        wins = sub[sub['Net_PnL_Pct'] > 0]
        losses = sub[sub['Net_PnL_Pct'] < 0]
        
        avg_win_pct = wins['Net_PnL_Pct'].mean()
        avg_loss_pct = losses['Net_PnL_Pct'].mean() # negative number
        max_loss_pct = losses['Net_PnL_Pct'].min() # worst single loss
        
        win_rate = (len(wins) / len(sub)) * 100
        rr_ratio = abs(avg_win_pct / avg_loss_pct) if avg_loss_pct != 0 else np.nan
        
        print(f"--- Timeframe: {tf} (1st Attempt Only) ---")
        print(f"  - Total Trades: {len(sub):,}")
        print(f"  - Win Rate: {win_rate:.1f}%")
        print(f"  - AVERAGE LOSS PER LOSING TRADE: {avg_loss_pct:+.2f}%  (Abs: {abs(avg_loss_pct):.2f}%)")
        print(f"  - AVERAGE GAIN PER WINNING TRADE: {avg_win_pct:+.2f}%")
        print(f"  - Realized Reward-to-Risk Ratio: {rr_ratio:.2f}:1")
        print(f"  - Worst Single Trade Loss: {max_loss_pct:+.2f}%\n")

    print("==========================================================================")

if __name__ == "__main__":
    main()
