"""
Analyze Exact Factors Making Difference Between 5M and 15M Timeframes
----------------------------------------------------------------------
Compares SL hits, TP1 hits, TP2 hits, and Noise Wicks on 5M vs 15M.
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "Reports"

df5 = pd.read_csv(REPORTS_DIR / "5M_Top5_Trades_Detailed.csv")
df15 = pd.read_csv(REPORTS_DIR / "Optimal_Top5_Trades_Detailed.csv")

print("==========================================================================")
print("             5M vs 15M QUANTITATIVE FACTOR BREAKDOWN                      ")
print("==========================================================================")

sl_hits_5 = (df5['Exit1_PnL_Pct'] < 0).sum()
sl_hits_15 = (df15['Exit1_PnL_Pct'] < 0).sum()

tp1_hits_5 = (df5['Exit1_PnL_Pct'] > 0).sum()
tp1_hits_15 = (df15['Exit1_PnL_Pct'] > 0).sum()

tp2_hits_5 = (df5['Exit2_PnL_Pct'] >= 2.0).sum()
tp2_hits_15 = (df15['Exit2_PnL_Pct'] >= 2.0).sum()

avg_win_5 = df5[df5['Net_Trade_PnL_Amount'] > 0]['Net_Trade_PnL_Amount'].mean()
avg_win_15 = df15[df15['Net_Trade_PnL_Amount'] > 0]['Net_Trade_PnL_Amount'].mean()

avg_loss_5 = df5[df5['Net_Trade_PnL_Amount'] < 0]['Net_Trade_PnL_Amount'].mean()
avg_loss_15 = df15[df15['Net_Trade_PnL_Amount'] < 0]['Net_Trade_PnL_Amount'].mean()

print(f"1. STOP LOSS HITS (Exit 1 Loss):")
print(f"   - 5M Timeframe:  {sl_hits_5} Trades ({sl_hits_5/len(df5)*100:.1f}% of trades stopped out)")
print(f"   - 15M Timeframe: {sl_hits_15} Trades ({sl_hits_15/len(df15)*100:.1f}% of trades stopped out)")
print(f"   --> 15M saved {sl_hits_5 - sl_hits_15} trades from getting prematurely stopped out by noise!\n")

print(f"2. TARGET 1 REACH RATE (Exit 1 Profit):")
print(f"   - 5M Timeframe:  {tp1_hits_5} Trades ({tp1_hits_5/len(df5)*100:.1f}% hit Target 1)")
print(f"   - 15M Timeframe: {tp1_hits_15} Trades ({tp1_hits_15/len(df15)*100:.1f}% hit Target 1)\n")

print(f"3. AVERAGE WIN VS AVERAGE LOSS:")
print(f"   - 5M Avg Winner:  +${avg_win_5:.2f} | 5M Avg Loser: -${abs(avg_loss_5):.2f}")
print(f"   - 15M Avg Winner: +${avg_win_15:.2f} | 15M Avg Loser: -${abs(avg_loss_15):.2f}")
print("==========================================================================")
