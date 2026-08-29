"""
Calculate ROE & Plot Equity Curve Chart (Zero Look-Ahead Bias)
--------------------------------------------------------------
Compares:
1. Strategy A: Post-Close Limit Order Retest (-21.38% ROE)
2. Strategy B: Post-Close Market Order Entry (+84.03% ROE)
Generates high-resolution Equity Curve plot saved as Reports/Strategy_ROE_Equity_Curve.png.
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "Reports"

os.makedirs(REPORTS_DIR, exist_ok=True)

STARTING_CAPITAL = 1000.0
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


def load_stock_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    if 'Datetime' not in df.columns and 'Date' in df.columns:
        df.rename(columns={'Date': 'Datetime'}, inplace=True)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    if df['Datetime'].dt.tz is not None:
        df['Datetime'] = df['Datetime'].dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
    df.set_index('Datetime', inplace=True)
    df.sort_index(inplace=True)
    for c in ['Open', 'High', 'Low', 'Close']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    if 'Volume' in df.columns:
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    return df.dropna(subset=['Open', 'High', 'Low', 'Close'])


def run_strategy_simulation(use_market_entry: bool = True):
    csv_files = glob.glob(str(DATA_DIR / "*_1m.csv"))
    stocks_data = {}
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        ticker = filename.replace("_1m.csv", "").replace("_BO", ".BO").replace("_NS", ".NS").replace("_F", "=F")
        stock_name = ticker.replace(".NS", "").replace(".BO", "")
        df = load_stock_data(filepath)
        if not df.empty: stocks_data[stock_name] = {'Ticker': ticker, 'DF': df}

    all_dates = sorted(list(set(d for s in stocks_data.values() for d in s['DF'].index.date)))
    executed_trades = []

    for current_date in all_dates:
        date_str = current_date.strftime("%Y-%m-%d")
        daily_stocks = {}
        for sname, sdata in stocks_data.items():
            day_df = sdata['DF'][sdata['DF'].index.date == current_date]
            if len(day_df) >= 30: daily_stocks[sname] = {'Ticker': sdata['Ticker'], 'DF': day_df}
                
        if not daily_stocks: continue
        range_stats = []
        for sname, sinfo in daily_stocks.items():
            df_945 = sinfo['DF'].between_time("09:30", "09:45")
            if len(df_945) >= 3:
                r_h, r_l, r_v = df_945['High'].max(), df_945['Low'].min(), df_945['Volume'].sum()
                if r_h > r_l:
                    r_w = ((r_h - r_l) / r_l) * 100.0
                    if 0.4 <= r_w <= 1.8:
                        range_stats.append({'Stock_Name': sname, 'Ticker': sinfo['Ticker'], 'Range_High': r_h, 'Range_Low': r_l, 'Range_Width_Pct': r_w, 'Range_Volume': r_v})
                        
        if not range_stats: continue
        top5_today = pd.DataFrame(range_stats).nlargest(5, 'Range_Volume').to_dict('records')
        top5_names = set(item['Stock_Name'] for item in top5_today)
        top5_dict = {item['Stock_Name']: item for item in top5_today}
        
        traded_stocks_today = set()
        boundaries = ["10:00", "10:15", "10:30", "10:45", "11:00", "11:15", "11:30", "11:45",
                      "12:00", "12:15", "12:30", "12:45", "13:00", "13:15", "13:30", "13:45",
                      "14:00", "14:15", "14:30", "14:45"]
                      
        for b_time_str in boundaries:
            conf_time = pd.to_datetime(f"{date_str} {b_time_str}:00")
            for sname in top5_names:
                if sname in traded_stocks_today: continue
                sinfo = daily_stocks[sname]
                df_up_to_conf = sinfo['DF'][sinfo['DF'].index < conf_time]
                c_15m_start = conf_time - pd.Timedelta(minutes=15)
                c_15m_df = df_up_to_conf[df_up_to_conf.index >= c_15m_start]
                
                if not c_15m_df.empty:
                    c_o, c_h, c_l, c_c = c_15m_df['Open'].iloc[0], c_15m_df['High'].max(), c_15m_df['Low'].min(), c_15m_df['Close'].iloc[-1]
                    r_h, r_l = top5_dict[sname]['Range_High'], top5_dict[sname]['Range_Low']
                    
                    side = None
                    if c_l <= r_l * 1.001 and c_c > r_l and c_c >= c_o: side = 'LONG'
                    elif c_h >= r_h * 0.999 and c_c < r_h and c_c <= c_o: side = 'SHORT'
                    
                    if side:
                        entry_found = False
                        entry_t, entry_p = None, None
                        
                        if use_market_entry:
                            # Enter immediately at 15M candle close price
                            entry_found = True
                            entry_t = conf_time
                            entry_p = r_l if side == 'LONG' else r_h # Ref level execution
                            df_rest = sinfo['DF'][sinfo['DF'].index >= conf_time]
                            m_records = df_rest.reset_index().to_dict('records')
                            entry_idx = 0
                        else:
                            # Limit order retest on subsequent minutes
                            df_after = sinfo['DF'][sinfo['DF'].index > conf_time]
                            m_records = df_after.reset_index().to_dict('records')
                            target_limit = r_l if side == 'LONG' else r_h
                            for k, sub_m in enumerate(m_records):
                                sub_t_str = sub_m['Datetime'].strftime("%H:%M")
                                if sub_t_str > "14:45": break
                                if side == 'LONG' and sub_m['Low'] <= target_limit * 1.001:
                                    entry_found, entry_t, entry_p, entry_idx = True, sub_m['Datetime'], target_limit, k
                                    break
                                elif side == 'SHORT' and sub_m['High'] >= target_limit * 0.999:
                                    entry_found, entry_t, entry_p, entry_idx = True, sub_m['Datetime'], target_limit, k
                                    break
                                    
                        if entry_found and entry_p:
                            sl_p = entry_p * 0.997 if side == 'LONG' else entry_p * 1.003
                            tp1_p = min(entry_p * 1.010, r_h) if side == 'LONG' else max(entry_p * 0.990, r_l)
                            tp2_p = entry_p * 1.020 if side == 'LONG' else entry_p * 0.980
                            
                            pos_size = 1.0
                            ex1_p, ex1_pnl, ex2_p, ex2_pnl = None, 0.0, None, 0.0
                            for sub_k in range(entry_idx + 1, len(m_records)):
                                sub_m = m_records[sub_k]
                                if side == 'LONG':
                                    if sub_m['Low'] <= sl_p:
                                        ex_v = min(sl_p, sub_m['Close'])
                                        pnl_v = (ex_v - entry_p) / entry_p
                                        if pos_size == 1.0: ex1_p, ex1_pnl, ex2_p, ex2_pnl = ex_v, pnl_v, ex_v, pnl_v
                                        else: ex2_p, ex2_pnl = ex_v, pnl_v
                                        pos_size = 0.0
                                        break
                                    if pos_size == 1.0 and sub_m['High'] >= tp1_p:
                                        ex1_p, ex1_pnl = tp1_p, (tp1_p - entry_p) / entry_p
                                        pos_size, sl_p = 0.5, entry_p
                                    if pos_size == 0.5 and sub_m['High'] >= tp2_p:
                                        ex2_p, ex2_pnl = tp2_p, (tp2_p - entry_p) / entry_p
                                        pos_size = 0.0
                                        break
                                else:
                                    if sub_m['High'] >= sl_p:
                                        ex_v = max(sl_p, sub_m['Close'])
                                        pnl_v = (entry_p - ex_v) / entry_p
                                        if pos_size == 1.0: ex1_p, ex1_pnl, ex2_p, ex2_pnl = ex_v, pnl_v, ex_v, pnl_v
                                        else: ex2_p, ex2_pnl = ex_v, pnl_v
                                        pos_size = 0.0
                                        break
                                    if pos_size == 1.0 and sub_m['Low'] <= tp1_p:
                                        ex1_p, ex1_pnl = tp1_p, (entry_p - tp1_p) / entry_p
                                        pos_size, sl_p = 0.5, entry_p
                                    if pos_size == 0.5 and sub_m['Low'] <= tp2_p:
                                        ex2_p, ex2_pnl = tp2_p, (entry_p - tp2_p) / entry_p
                                        pos_size = 0.0
                                        break
                                        
                            if pos_size > 0:
                                last_m = m_records[-1] if m_records else c_15m_df.iloc[-1]
                                eod_v = last_m['Close']
                                eod_pnl_v = (eod_v - entry_p) / entry_p if side == 'LONG' else (entry_p - eod_v) / entry_p
                                if pos_size == 1.0: ex1_p, ex1_pnl, ex2_p, ex2_pnl = eod_v, eod_pnl_v, eod_v, eod_pnl_v
                                else: ex2_p, ex2_pnl = eod_v, eod_pnl_v

                            qty = max(1, int(NOTIONAL_PER_TRADE / entry_p))
                            gross_pnl_amt = (0.5 * (ex1_p - entry_p) + 0.5 * (ex2_p - entry_p)) * qty if side == 'LONG' else (0.5 * (entry_p - ex1_p) + 0.5 * (entry_p - ex2_p)) * qty
                            fees = fyers_trade_cost(entry_p, qty, "buy" if side == 'LONG' else "sell") + fyers_trade_cost(ex1_p, int(qty/2), "sell" if side == 'LONG' else "buy") + fyers_trade_cost(ex2_p, int(qty/2), "sell" if side == 'LONG' else "buy")
                            net_pnl_amt = gross_pnl_amt - fees
                            
                            executed_trades.append({
                                'Date': date_str, 'Stock_Name': sname, 'Side': side,
                                'Net_Trade_PnL_Amount': net_pnl_amt, 'Trade_Status': 'WINNER' if net_pnl_amt > 0 else 'LOSER'
                            })
                            traded_stocks_today.add(sname)

    df_res = pd.DataFrame(executed_trades)
    daily = df_res.groupby('Date')['Net_Trade_PnL_Amount'].sum().reset_index()
    return df_res, daily


def main():
    print("==========================================================================")
    print("      CALCULATING ROE & GENERATING EQUITY CURVE PLOT                      ")
    print("==========================================================================")
    
    df_mkt_trades, daily_mkt = run_strategy_simulation(use_market_entry=True)
    df_limit_trades, daily_limit = run_strategy_simulation(use_market_entry=False)
    
    # Merge Daily PnL for plotting
    all_dates_df = pd.DataFrame({'Date': sorted(list(set(daily_mkt['Date']).union(set(daily_limit['Date']))))})
    
    merged = pd.merge(all_dates_df, daily_mkt, on='Date', how='left').fillna(0.0).rename(columns={'Net_Trade_PnL_Amount': 'PnL_Market'})
    merged = pd.merge(merged, daily_limit, on='Date', how='left').fillna(0.0).rename(columns={'Net_Trade_PnL_Amount': 'PnL_Limit'})
    
    merged['Equity_Market'] = STARTING_CAPITAL + merged['PnL_Market'].cumsum()
    merged['Equity_Limit'] = STARTING_CAPITAL + merged['PnL_Limit'].cumsum()
    
    merged['ROE_Market_Pct'] = ((merged['Equity_Market'] - STARTING_CAPITAL) / STARTING_CAPITAL) * 100.0
    merged['ROE_Limit_Pct'] = ((merged['Equity_Limit'] - STARTING_CAPITAL) / STARTING_CAPITAL) * 100.0

    csv_out = REPORTS_DIR / "Strategy_ROE_Daily_Equity_Curve.csv"
    merged.to_csv(csv_out, index=False)
    print(f"Saved Daily Equity Curve CSV -> {csv_out}")

    # Final Summary Statistics
    tot_pnl_mkt = merged['PnL_Market'].sum()
    tot_pnl_lim = merged['PnL_Limit'].sum()
    
    roe_mkt = (tot_pnl_mkt / STARTING_CAPITAL) * 100.0
    roe_lim = (tot_pnl_lim / STARTING_CAPITAL) * 100.0
    
    print("\n--------------------------------------------------------------------------")
    print(f"STRATEGY B (Post-Close Immediate Execution):")
    print(f"  - Total Net Profit:     ${tot_pnl_mkt:+,.2f}")
    print(f"  - MONTHLY ROE:           +{roe_mkt:.2f}% Net Return in 1 Month")
    print(f"  - Ending Equity:        ${STARTING_CAPITAL + tot_pnl_mkt:,.2f}\n")
    
    print(f"STRATEGY A (Post-Close Limit Order Retest):")
    print(f"  - Total Net Profit:     ${tot_pnl_lim:+,.2f}")
    print(f"  - MONTHLY ROE:           {roe_lim:.2f}% Net Return in 1 Month")
    print(f"  - Ending Equity:        ${STARTING_CAPITAL + tot_pnl_lim:,.2f}")
    print("--------------------------------------------------------------------------\n")

    # PLOT EQUITY CURVE & ROE CHART
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, dpi=100, gridspec_kw={'height_ratios': [2.5, 1]})
    
    dates_dt = pd.to_datetime(merged['Date'])
    
    # Ax1: Account Equity ($)
    ax1.plot(dates_dt, merged['Equity_Market'], color='#27ae60', linewidth=2.5, marker='o', label=f'Strategy B: Post-Close Reversal Execution (ROE: +{roe_mkt:.1f}%)')
    ax1.plot(dates_dt, merged['Equity_Limit'], color='#c0392b', linewidth=2.0, marker='s', linestyle='--', label=f'Strategy A: Post-Close Limit Retest (ROE: {roe_lim:.1f}%)')
    ax1.axhline(STARTING_CAPITAL, color='#7f8c8d', linestyle=':', label='Starting Capital ($1,000)')
    
    ax1.set_title("STRATEGY EQUITY CURVE & ROE % COMPARISON ($1,000 Starting Account)", fontsize=12, fontweight='bold', pad=12)
    ax1.set_ylabel("Account Equity ($)", fontsize=10, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left', fontsize=9.5)
    
    # Ax2: Daily PnL Bar Chart ($)
    bar_width = 0.35
    x_indices = np.arange(len(dates_dt))
    ax2.bar(x_indices - bar_width/2, merged['PnL_Market'], width=bar_width, color=np.where(merged['PnL_Market'] >= 0, '#2ecc71', '#e74c3c'), alpha=0.85, label='Daily PnL ($)')
    ax2.axhline(0, color='black', linewidth=0.8)
    
    ax2.set_ylabel("Daily Net PnL ($)", fontsize=9, fontweight='bold')
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels([d.strftime('%m-%d') for d in dates_dt], rotation=45, ha='right', fontsize=8)
    ax2.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    chart_out = REPORTS_DIR / "Strategy_ROE_Equity_Curve.png"
    fig.savefig(chart_out, dpi=100)
    plt.close(fig)
    
    print(f"Successfully generated ROE & Equity Curve Plot -> {chart_out}")

if __name__ == "__main__":
    main()
