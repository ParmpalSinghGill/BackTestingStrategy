import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "backtest_trades.csv"

START_CAPITAL = 1000.0
MAX_OPEN_TRADES = 2
ALLOCATED_NOTIONAL_PER_TRADE = 2500.0  # Max notional 5000 / 2 open trades

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

def run_simulation():
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} not found. Please run backtest.py first.")
        return

    df = pd.read_csv(CSV_PATH)
    df["entry_dt"] = pd.to_datetime(df["entry_dt"])
    df["exit_dt"] = pd.to_datetime(df["exit_dt"])

    print(f"Loaded {len(df)} candidate trades.")
    print("Enforcing max 2 open trades at a time (chronological simulation).")
    print(f"Starting capital: ${START_CAPITAL:.2f} (with 5x leverage, total $5000 notional, $2500 max per trade)\n")

    results = []

    # Run simulation separately for each timeframe
    for tf, tf_df in df.groupby("timeframe"):
        # Sort trades chronologically by entry time
        sorted_trades = tf_df.sort_values("entry_dt").copy()
        
        active_trades = []
        accepted_trades = []

        for idx, trade in sorted_trades.iterrows():
            entry_t = trade["entry_dt"]
            
            # Remove any active trades that have already exited by the current entry time
            active_trades = [t for t in active_trades if t["exit_dt"] > entry_t]
            
            # If we have space for a new trade, take it
            if len(active_trades) < MAX_OPEN_TRADES:
                entry_price = trade["entry"]
                exit_price = trade["exit"]
                side = trade["side"]
                
                # Dynamic position sizing: allocate up to $2500 per trade
                qty = int(ALLOCATED_NOTIONAL_PER_TRADE / entry_price)
                if qty < 1:
                    qty = 1  # Buy at least 1 share
                    
                # Calculate PnL for dynamic sizing
                if side == "short":
                    pnl = (entry_price - exit_price) * qty
                    entry_fee = fyers_trade_cost(entry_price, qty, "buy")
                    exit_fee = fyers_trade_cost(exit_price, qty, "sell")
                else:
                    pnl = (exit_price - entry_price) * qty
                    entry_fee = fyers_trade_cost(entry_price, qty, "buy")
                    exit_fee = fyers_trade_cost(exit_price, qty, "sell")
                    
                net_pnl = pnl - entry_fee - exit_fee
                
                trade_record = {
                    "stock": trade["stock"],
                    "side": side,
                    "entry_dt": entry_t,
                    "exit_dt": trade["exit_dt"],
                    "entry": entry_price,
                    "exit": exit_price,
                    "qty": qty,
                    "original_net_pnl": trade["net_pnl"], # qty = 1
                    "dynamic_net_pnl": net_pnl
                }
                
                active_trades.append(trade_record)
                accepted_trades.append(trade_record)

        # Aggregate metrics
        accepted_df = pd.DataFrame(accepted_trades)
        if accepted_df.empty:
            print(f"Timeframe: {tf:<5} | No trades accepted.")
            continue
            
        total_pnl_qty_1 = accepted_df["original_net_pnl"].sum()
        total_pnl_dynamic = accepted_df["dynamic_net_pnl"].sum()
        return_pct = (total_pnl_dynamic / START_CAPITAL) * 100.0
        
        results.append({
            "Timeframe": tf,
            "Total Candidate Trades": len(tf_df),
            "Accepted Trades": len(accepted_df),
            "ignored_trades": len(tf_df) - len(accepted_df),
            "Net PnL (Qty=1)": total_pnl_qty_1,
            "Net PnL (Leveraged)": total_pnl_dynamic,
            "Return % (on $1000 Capital)": return_pct
        })

    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    run_simulation()
