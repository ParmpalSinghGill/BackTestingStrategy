"""
Analyze 1st Attempt Only at High Level and Low Level Per Stock Per Day
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "Reports"

START_CAPITAL = 1000.0
NOTIONAL_PER_TRADE = 2500.0


def fyers_trade_cost(price: float, qty: int, side: str) -> float:
    turnover = price * qty
    brokerage = min(20.0, turnover * 0.0003)
    stt = turnover * 0.00025 if side == "sell" else 0.0
    exchange = turnover * 0.0000325
    gst = (brokerage + exchange) * 0.18
    sebi = turnover * 0.0000005
    stamp = turnover * 0.00003
    return brokerage + stt + exchange + gst + sebi + stamp


def run_portfolio_simulation(trades_df: pd.DataFrame, max_open: int = 5) -> dict:
    if trades_df.empty:
        return {'Accepted_Trades': 0, 'Net_PnL': 0.0, 'Return_Pct': 0.0}
        
    sorted_trades = trades_df.sort_values("entry_dt").copy()
    active_trades = []
    accepted_trades = []
    
    for idx, trade in sorted_trades.iterrows():
        entry_t = trade["entry_dt"]
        active_trades = [t for t in active_trades if t["exit_dt"] > entry_t]
        
        if len(active_trades) < max_open:
            entry_price = trade["Entry_Price"]
            exit_price = trade["exit_price"]
            side = trade["Side"].lower()
            
            qty = max(1, int(NOTIONAL_PER_TRADE / entry_price))
            pnl = (entry_price - exit_price) * qty if side in ["short", "sell"] else (exit_price - entry_price) * qty
            
            entry_fee = fyers_trade_cost(entry_price, qty, "buy")
            exit_fee = fyers_trade_cost(exit_price, qty, "sell")
            net_pnl = pnl - entry_fee - exit_fee
            
            trade_record = {"exit_dt": trade["exit_dt"], "net_pnl": net_pnl}
            active_trades.append(trade_record)
            accepted_trades.append(trade_record)
            
    if not accepted_trades:
        return {'Accepted_Trades': 0, 'Net_PnL': 0.0, 'Return_Pct': 0.0}
        
    accepted_df = pd.DataFrame(accepted_trades)
    net_pnl = accepted_df["net_pnl"].sum()
    ret_pct = (net_pnl / START_CAPITAL) * 100.0
    return {'Accepted_Trades': len(accepted_df), 'Net_PnL': round(net_pnl, 2), 'Return_Pct': round(ret_pct, 2)}


def main():
    opt_csv = REPORTS_DIR / "quick_reversal_optimized_trades.csv"
    if not opt_csv.exists():
        print(f"Error: {opt_csv} not found.")
        return
        
    df = pd.read_csv(opt_csv)
    df['entry_dt'] = pd.to_datetime(df['Entry_Time'])
    df['Date'] = df['entry_dt'].dt.date
    df['exit_dt'] = df['entry_dt'] + pd.Timedelta(minutes=15)
    
    df['exit_price'] = np.where(
        df['Side'] == 'LONG',
        df['Entry_Price'] * (1.0 + df['Net_PnL_Pct'] / 100.0),
        df['Entry_Price'] * (1.0 - df['Net_PnL_Pct'] / 100.0)
    )

    print("==========================================================================")
    print("      FIRST ATTEMPT ONLY (1st Low Test & 1st High Test per Stock/Day)     ")
    print("==========================================================================")
    
    for tf in ['1m', '5m', '15m']:
        tf_df = df[df['Timeframe'] == tf].copy()
        
        # Filter ONLY the 1st attempt for LONG (Low level) and 1st attempt for SHORT (High level)
        # per (Ticker, Date)
        df_1st = tf_df.groupby(['Ticker', 'Date', 'Side']).head(1).copy()
        
        total_trades = len(df_1st)
        win_rate = df_1st['Is_Winner'].mean() * 100
        avg_pnl = df_1st['Net_PnL_Pct'].mean()
        
        gains = df_1st[df_1st['Net_PnL_Pct'] > 0]['Net_PnL_Pct'].sum()
        losses = abs(df_1st[df_1st['Net_PnL_Pct'] < 0]['Net_PnL_Pct'].sum())
        profit_factor = gains / losses if losses > 0 else np.nan
        
        port_2 = run_portfolio_simulation(df_1st, max_open=2)
        port_5 = run_portfolio_simulation(df_1st, max_open=5)
        
        print(f"\nTimeframe: {tf:<4}")
        print(f"  - Total Trades (1st Attempt Only): {total_trades}")
        print(f"  - Win Rate: {win_rate:.1f}%")
        print(f"  - Avg PnL / Trade: {avg_pnl:+.2f}%")
        print(f"  - Profit Factor: {profit_factor:.2f}")
        print(f"  - Portfolio Net Return (Max 2 Open Trades): {port_2['Return_Pct']:+.2f}% (${port_2['Net_PnL']:+,.2f})")
        print(f"  - Portfolio Net Return (Max 5 Open Trades): {port_5['Return_Pct']:+.2f}% (${port_5['Net_PnL']:+,.2f})")

    print("\n==========================================================================")

if __name__ == "__main__":
    main()
