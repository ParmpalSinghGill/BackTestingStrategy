"""
Calculate Strategy Returns and Portfolio PnL
---------------------------------------------
Calculates Total Net Return %, Average Trade Return %, Gross Profit %, Gross Loss %,
and Realistic Portfolio Simulation Net Return % (enforcing max open trade limits & FYERS brokerage fees).

Compares:
1. User's Strict Strategy (Touch Entry, 0.01% SL, Partial 50% TP1 @ 1%/Opposite, 50% TP2 @ 2%).
2. Optimized Strategy (Wick Confirmation, 0.3% SL Buffer, 50% TP1 @ 0.8%/Midpoint, 50% TP2 @ 1.5%).
"""

import os
import glob
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from stocks_parser import parse_stocks_file

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "Reports"

START_CAPITAL = 1000.0 # $1,000 Starting Capital
MAX_OPEN_TRADES = 2
NOTIONAL_PER_TRADE = 2500.0 # $2,500 per trade (5x intraday margin)


def fyers_trade_cost(price: float, qty: int, side: str) -> float:
    """Approximate intraday FYERS charges on one leg of a trade."""
    turnover = price * qty
    brokerage = min(20.0, turnover * 0.0003)
    stt = turnover * 0.00025 if side == "sell" else 0.0
    exchange = turnover * 0.0000325
    gst = (brokerage + exchange) * 0.18
    sebi = turnover * 0.0000005
    stamp = turnover * 0.00003
    return brokerage + stt + exchange + gst + sebi + stamp


def run_portfolio_simulation(trades_df: pd.DataFrame, max_open: int = 2) -> dict:
    """Run chronological portfolio simulation enforcing max open trade limit and brokerage fees."""
    if trades_df.empty:
        return {'Accepted_Trades': 0, 'Net_PnL': 0.0, 'Return_Pct': 0.0}
        
    sorted_trades = trades_df.sort_values("entry_dt").copy()
    active_trades = []
    accepted_trades = []
    
    for idx, trade in sorted_trades.iterrows():
        entry_t = trade["entry_dt"]
        active_trades = [t for t in active_trades if t["exit_dt"] > entry_t]
        
        if len(active_trades) < max_open:
            entry_price = trade["entry_price"]
            exit_price = trade["exit_price"]
            side = trade["side"].lower()
            
            qty = max(1, int(NOTIONAL_PER_TRADE / entry_price))
            
            if side in ["short", "sell"]:
                pnl = (entry_price - exit_price) * qty
            else:
                pnl = (exit_price - entry_price) * qty
                
            entry_fee = fyers_trade_cost(entry_price, qty, "buy")
            exit_fee = fyers_trade_cost(exit_price, qty, "sell")
            net_pnl = pnl - entry_fee - exit_fee
            
            trade_record = {
                "exit_dt": trade["exit_dt"],
                "net_pnl": net_pnl
            }
            active_trades.append(trade_record)
            accepted_trades.append(trade_record)
            
    if not accepted_trades:
        return {'Accepted_Trades': 0, 'Net_PnL': 0.0, 'Return_Pct': 0.0}
        
    accepted_df = pd.DataFrame(accepted_trades)
    net_pnl = accepted_df["net_pnl"].sum()
    ret_pct = (net_pnl / START_CAPITAL) * 100.0
    return {
        'Accepted_Trades': len(accepted_df),
        'Net_PnL': round(net_pnl, 2),
        'Return_Pct': round(ret_pct, 2)
    }


def main():
    print("==========================================================================")
    print("            STRATEGY CUMULATIVE RETURN & PORTFOLIO PnL ANALYSIS           ")
    print("==========================================================================")
    
    # 1. Load User's Strict Strategy Trades Log
    user_csv = REPORTS_DIR / "quick_reversal_trades.csv"
    if not user_csv.exists():
        print(f"Error: {user_csv} not found.")
        return
        
    df_user = pd.read_csv(user_csv)
    df_user['entry_dt'] = pd.to_datetime(df_user['Entry_Time'])
    # Estimate exit time roughly as entry_time + 15 mins for chronological sorting
    df_user['exit_dt'] = df_user['entry_dt'] + pd.Timedelta(minutes=15)
    df_user.rename(columns={'Entry_Price': 'entry_price', 'Side': 'side'}, inplace=True)
    
    # Approximate exit price from entry_price and Net_PnL_Pct
    df_user['exit_price'] = np.where(
        df_user['side'] == 'LONG',
        df_user['entry_price'] * (1.0 + df_user['Net_PnL_Pct'] / 100.0),
        df_user['entry_price'] * (1.0 - df_user['Net_PnL_Pct'] / 100.0)
    )

    # 2. Compute Return Metrics for User's Strategy
    user_results = []
    for tf in ['1m', '5m', '15m']:
        sub = df_user[df_user['Timeframe'] == tf]
        total_trades = len(sub)
        if total_trades == 0: continue
        
        cum_ret_pct = sub['Net_PnL_Pct'].sum()
        avg_trade_ret = sub['Net_PnL_Pct'].mean()
        
        gains = sub[sub['Net_PnL_Pct'] > 0]['Net_PnL_Pct'].sum()
        losses = abs(sub[sub['Net_PnL_Pct'] < 0]['Net_PnL_Pct'].sum())
        profit_factor = gains / losses if losses > 0 else np.nan
        
        port_sim = run_portfolio_simulation(sub, max_open=2)
        port_sim_5 = run_portfolio_simulation(sub, max_open=5)
        
        user_results.append({
            'Strategy': "User Strict (0.01% SL, Touch)",
            'Timeframe': tf,
            'Total Trades': total_trades,
            'Sum of Trade Returns %': f"{cum_ret_pct:+.1f}%",
            'Avg Trade Return %': f"{avg_trade_ret:+.2f}%",
            'Gross Gains %': f"+{gains:.1f}%",
            'Gross Losses %': f"-{losses:.1f}%",
            'Profit Factor': f"{profit_factor:.2f}",
            'Portfolio Net Return (Max 2 Open Trades)': f"{port_sim['Return_Pct']:+.2f}% (${port_sim['Net_PnL']:+,.2f})",
            'Portfolio Net Return (Max 5 Open Trades)': f"{port_sim_5['Return_Pct']:+.2f}% (${port_sim_5['Net_PnL']:+,.2f})"
        })

    # Print User Strategy Results
    df_user_res = pd.DataFrame(user_results)
    print("\n--- USER STRICT STRATEGY RETURNS (0.01% SL, Touch Entry) ---")
    print(df_user_res.to_string(index=False))

    # ---------------------------------------------------------------------------
    # Load / Simulate Optimized Confirmation Strategy Trades for Returns
    # ---------------------------------------------------------------------------
    opt_csv = REPORTS_DIR / "quick_reversal_optimized_trades.csv"
    if opt_csv.exists():
        df_opt = pd.read_csv(opt_csv)
        df_opt['entry_dt'] = pd.to_datetime(df_opt['Entry_Time'])
        df_opt['exit_dt'] = df_opt['entry_dt'] + pd.Timedelta(minutes=15)
        df_opt.rename(columns={'Entry_Price': 'entry_price', 'Side': 'side'}, inplace=True)
        df_opt['exit_price'] = np.where(
            df_opt['side'] == 'LONG',
            df_opt['entry_price'] * (1.0 + df_opt['Net_PnL_Pct'] / 100.0),
            df_opt['entry_price'] * (1.0 - df_opt['Net_PnL_Pct'] / 100.0)
        )

        opt_results = []
        for tf in ['1m', '5m', '15m']:
            sub = df_opt[df_opt['Timeframe'] == tf]
            total_trades = len(sub)
            if total_trades == 0: continue
            
            cum_ret_pct = sub['Net_PnL_Pct'].sum()
            avg_trade_ret = sub['Net_PnL_Pct'].mean()
            
            gains = sub[sub['Net_PnL_Pct'] > 0]['Net_PnL_Pct'].sum()
            losses = abs(sub[sub['Net_PnL_Pct'] < 0]['Net_PnL_Pct'].sum())
            profit_factor = gains / losses if losses > 0 else np.nan
            
            port_sim = run_portfolio_simulation(sub, max_open=2)
            port_sim_5 = run_portfolio_simulation(sub, max_open=5)
            
            opt_results.append({
                'Strategy': "Optimized (0.3% SL, Confirmation)",
                'Timeframe': tf,
                'Total Trades': total_trades,
                'Sum of Trade Returns %': f"{cum_ret_pct:+.1f}%",
                'Avg Trade Return %': f"{avg_trade_ret:+.2f}%",
                'Gross Gains %': f"+{gains:.1f}%",
                'Gross Losses %': f"-{losses:.1f}%",
                'Profit Factor': f"{profit_factor:.2f}",
                'Portfolio Net Return (Max 2 Open Trades)': f"{port_sim['Return_Pct']:+.2f}% (${port_sim['Net_PnL']:+,.2f})",
                'Portfolio Net Return (Max 5 Open Trades)': f"{port_sim_5['Return_Pct']:+.2f}% (${port_sim_5['Net_PnL']:+,.2f})"
            })

        df_opt_res = pd.DataFrame(opt_results)
        print("\n--- OPTIMIZED CONFIRMATION STRATEGY RETURNS (0.3% SL Buffer, Confirmation Entry) ---")
        print(df_opt_res.to_string(index=False))

    # Save summary report
    summary_path = REPORTS_DIR / "strategy_returns_comparison.csv"
    pd.concat([df_user_res], ignore_index=True).to_csv(summary_path, index=False)
    print(f"\nSaved returns comparison report to {summary_path}")

if __name__ == "__main__":
    main()
