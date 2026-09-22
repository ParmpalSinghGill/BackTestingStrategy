"""
Verify Impact of Trading Charges (Brokerage, STT, Exchange Fees, GST) on Returns
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "Reports"

START_CAPITAL = 1000.0
NOTIONAL_PER_TRADE = 2500.0


def fyers_trade_cost(price: float, qty: int, side: str) -> float:
    """Intraday FYERS charges on one leg of a trade."""
    turnover = price * qty
    brokerage = min(20.0, turnover * 0.0003)
    stt = turnover * 0.00025 if side == "sell" else 0.0
    exchange = turnover * 0.0000325
    gst = (brokerage + exchange) * 0.18
    sebi = turnover * 0.0000005
    stamp = turnover * 0.00003
    return brokerage + stt + exchange + gst + sebi + stamp


def analyze_charges(tf: str = '15m', max_open: int = 5):
    opt_csv = REPORTS_DIR / "quick_reversal_optimized_trades.csv"
    if not opt_csv.exists():
        return
        
    df = pd.read_csv(opt_csv)
    df['entry_dt'] = pd.to_datetime(df['Entry_Time'])
    df['Date'] = df['entry_dt'].dt.date
    df['exit_dt'] = df['entry_dt'] + pd.Timedelta(minutes=15)
    
    tf_df = df[df['Timeframe'] == tf].copy()
    # 1st Attempt Only
    df_1st = tf_df.groupby(['Ticker', 'Date', 'Side']).head(1).copy()
    sorted_trades = df_1st.sort_values("entry_dt").copy()
    
    active_trades = []
    
    total_gross_pnl = 0.0
    total_charges = 0.0
    total_net_pnl = 0.0
    accepted_count = 0
    
    for idx, trade in sorted_trades.iterrows():
        entry_t = trade["entry_dt"]
        active_trades = [t for t in active_trades if t["exit_dt"] > entry_t]
        
        if max_open is None or len(active_trades) < max_open:
            entry_price = trade["Entry_Price"]
            pnl_pct = trade["Net_PnL_Pct"] / 100.0
            side = trade["Side"].lower()
            
            qty = max(1, int(NOTIONAL_PER_TRADE / entry_price))
            
            if side in ["short", "sell"]:
                exit_price = entry_price * (1.0 - pnl_pct)
                gross_pnl = (entry_price - exit_price) * qty
            else:
                exit_price = entry_price * (1.0 + pnl_pct)
                gross_pnl = (exit_price - entry_price) * qty
                
            entry_fee = fyers_trade_cost(entry_price, qty, "buy")
            exit_fee = fyers_trade_cost(exit_price, qty, "sell")
            trade_fee = entry_fee + exit_fee
            
            net_pnl = gross_pnl - trade_fee
            
            total_gross_pnl += gross_pnl
            total_charges += trade_fee
            total_net_pnl += net_pnl
            accepted_count += 1
            
            active_trades.append({"exit_dt": trade["exit_dt"]})
            
    gross_ret_pct = (total_gross_pnl / START_CAPITAL) * 100.0
    charges_ret_pct = (total_charges / START_CAPITAL) * 100.0
    net_ret_pct = (total_net_pnl / START_CAPITAL) * 100.0
    avg_charge_per_trade = total_charges / accepted_count if accepted_count > 0 else 0.0
    
    print(f"==========================================================================")
    print(f"     DETAILED CHARGES BREAKDOWN ({tf} Timeframe, Max {max_open} Open Trades)   ")
    print(f"==========================================================================")
    print(f"Total Accepted Trades: {accepted_count:,}")
    print(f"Starting Capital: ${START_CAPITAL:,.2f}  (Allocated Notional: ${NOTIONAL_PER_TRADE:,.2f}/trade)")
    print(f"Gross PnL (Before Charges):  ${total_gross_pnl:+,.2f}  ({gross_ret_pct:+.2f}%)")
    print(f"Total Trading Charges Paid:  ${total_charges:,.2f}  (-{charges_ret_pct:.2f}%)")
    print(f"NET PnL (AFTER ALL CHARGES): ${total_net_pnl:+,.2f}  ({net_ret_pct:+.2f}%)")
    print(f"Average Charges per Trade:  ${avg_charge_per_trade:.2f} per trade")
    print(f"==========================================================================\n")

def main():
    analyze_charges(tf='15m', max_open=5)
    analyze_charges(tf='15m', max_open=None)

if __name__ == "__main__":
    main()
