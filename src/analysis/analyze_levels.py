import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "backtest_trades.csv"

START_CAPITAL = 1000.0
MAX_OPEN_TRADES = 2
ALLOCATED_NOTIONAL_PER_TRADE = 2500.0

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

def run_portfolio_sim(trades_df):
    """Enforce max 2 open trades and calculate net PnL (leveraged)."""
    if trades_df.empty:
        return 0.0, 0
    sorted_trades = trades_df.sort_values("entry_dt").copy()
    active_trades = []
    accepted_trades = []
    
    for idx, trade in sorted_trades.iterrows():
        entry_t = trade["entry_dt"]
        active_trades = [t for t in active_trades if t["exit_dt"] > entry_t]
        
        if len(active_trades) < MAX_OPEN_TRADES:
            entry_price = trade["entry"]
            exit_price = trade["exit"]
            side = trade["side"]
            
            qty = int(ALLOCATED_NOTIONAL_PER_TRADE / entry_price)
            if qty < 1:
                qty = 1
                
            if side == "short":
                pnl = (entry_price - exit_price) * qty
            else:
                pnl = (exit_price - entry_price) * qty
                
            entry_fee = fyers_trade_cost(entry_price, qty, "buy")
            exit_fee = fyers_trade_cost(exit_price, qty, "sell")
            net_pnl = pnl - entry_fee - exit_fee
            
            trade_record = {
                "exit_dt": trade["exit_dt"],
                "dynamic_net_pnl": net_pnl
            }
            active_trades.append(trade_record)
            accepted_trades.append(trade_record)
            
    if not accepted_trades:
        return 0.0, 0
    accepted_df = pd.DataFrame(accepted_trades)
    return accepted_df["dynamic_net_pnl"].sum(), len(accepted_df)

def main():
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} not found.")
        return

    df = pd.read_csv(CSV_PATH)
    df["entry_dt"] = pd.to_datetime(df["entry_dt"])
    df["exit_dt"] = pd.to_datetime(df["exit_dt"])

    # Categorize trades by source:
    # 'prev_day' = contains prev_day (e.g. prev_day_high, prev_day_low)
    # 'pivot' = contains pivot (e.g. pivot_support, pivot_resistance)
    # Note: coinciding levels will be captured under both for individual comparisons
    
    print("=================================================================")
    print("      RAW SETUP PERFORMANCE (NO PORTFOLIO LIMITS)               ")
    print("=================================================================")
    
    timeframes = sorted(df["timeframe"].unique())
    raw_results = []
    
    for tf in timeframes:
        tf_df = df[df["timeframe"] == tf]
        
        for source in ["prev_day", "pivot"]:
            # Check if source string is in level_sources (handles "prev_day", "pivot", "pivot,prev_day")
            source_df = tf_df[tf_df["level_sources"].str.contains(source, na=False)]
            
            if source_df.empty:
                continue
                
            win_1r = source_df["win_1r"].mean() * 100
            avg_rr = source_df["rr"].mean()
            expectancy = source_df["r_eod"].mean()
            
            raw_results.append({
                "Timeframe": tf,
                "Level Source": source,
                "Candidate Trades": len(source_df),
                "Win Rate (>=1R)": f"{win_1r:.1f}%",
                "Avg R/R": f"{avg_rr:.2f}",
                "Expectancy (R)": f"{expectancy:.2f}"
            })
            
    raw_summary = pd.DataFrame(raw_results)
    print(raw_summary.to_string(index=False))
    print("\n")

    print("=================================================================")
    print("      PORTFOLIO PERFORMANCE (ENFORCING MAX 2 OPEN TRADES)       ")
    print("=================================================================")
    
    portfolio_results = []
    
    for tf in timeframes:
        tf_df = df[df["timeframe"] == tf]
        
        for source in ["prev_day", "pivot"]:
            source_df = tf_df[tf_df["level_sources"].str.contains(source, na=False)]
            
            pnl, accepted_count = run_portfolio_sim(source_df)
            return_pct = (pnl / START_CAPITAL) * 100
            
            portfolio_results.append({
                "Timeframe": tf,
                "Level Source": source,
                "Candidate Trades": len(source_df),
                "Accepted Trades": accepted_count,
                "Net PnL (Leveraged)": f"${pnl:.2f}",
                "Return % (on $1000 Capital)": f"{return_pct:.2f}%"
            })
            
    portfolio_summary = pd.DataFrame(portfolio_results)
    print(portfolio_summary.to_string(index=False))

if __name__ == "__main__":
    main()
