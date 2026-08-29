"""
True Point-in-Time Minute-by-Minute System: Limit Orders Placed at 09:45 AM
-------------------------------------------------------------------------
No forward-looking bias:
1. At 09:45 AM, calculate 9:30-9:45 Range High/Low & Volume.
2. Select Top 5 Volume Stocks (Width 0.4%-1.8%).
3. Place Limit Order @ 9:30 Low (Long) & 9:30 High (Short) at 09:45 AM.
4. Minute-by-minute from 09:50 AM to 14:45 PM:
   - Fills limit order exact second price retests H or L.
   - Monitors 0.3% SL, TP1 (+1.0%/opposite), TP2 (+2.0%), EOD Exit (15:10).
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
PLOTS_DIR = BASE_DIR / "plot" / "Plot_15M"

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

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


def resample_ohlc(df: pd.DataFrame, timeframe: str = '15min') -> pd.DataFrame:
    return df.resample(timeframe, closed='left', label='left').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()


def draw_candlesticks(ax, df_candles):
    width = 0.006
    for idx, row in df_candles.iterrows():
        t = idx
        o, h, l, c = row['Open'], row['High'], row['Low'], row['Close']
        color = '#2ecc71' if c >= o else '#e74c3c'
        ax.vlines(t, l, h, color=color, linewidth=1.2, zorder=3)
        body_bottom = min(o, c)
        body_height = max(abs(c - o), 0.01)
        ax.bar(t, body_height, bottom=body_bottom, width=width, color=color, edgecolor=color, zorder=4, alpha=0.85)


def plot_candlestick_trade_chart(day_df: pd.DataFrame, stock_name: str, trade: dict, range_high: float, range_low: float):
    fig, ax = plt.subplots(figsize=(14, 7), dpi=100)
    plot_15m = resample_ohlc(day_df.between_time("09:15", "15:30"), '15min')
    if plot_15m.empty:
        plt.close(fig)
        return
        
    draw_candlesticks(ax, plot_15m)
    ax.axhline(range_high, color='#e74c3c', linestyle='--', linewidth=1.5, label=f'9:30 High ({range_high:.2f})')
    ax.axhline(range_low, color='#2ecc71', linestyle='--', linewidth=1.5, label=f'9:30 Low ({range_low:.2f})')
    
    range_start = plot_15m.index[0].replace(hour=9, minute=30, second=0)
    range_end = plot_15m.index[0].replace(hour=9, minute=45, second=0)
    ax.axvspan(range_start, range_end, color='#f39c12', alpha=0.15, label='9:30-9:45 Range Window')
    
    side = trade['Side']
    color_side = '#27ae60' if side == 'LONG' else '#c0392b'
    marker_side = '^' if side == 'LONG' else 'v'

    # RETEST ENTRY MARKER & LABEL
    entry_t = pd.to_datetime(trade['Retest_Entry_Time'])
    entry_p = trade['Entry_Price']
    entry_time_str = entry_t.strftime("%H:%M")
    
    ax.scatter(entry_t, entry_p, color=color_side, s=160, zorder=7, marker=marker_side)
    ax.annotate(f"RETEST ENTRY ({side})\nTime: {entry_time_str}\nPrice: @{entry_p:.2f}", (entry_t, entry_p),
                textcoords="offset points", xytext=(0, 22 if side == 'LONG' else -35),
                ha='center', fontsize=8.5, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="#e8f8f5" if side == 'LONG' else "#fadbd8", ec=color_side, lw=1.2))
                
    # EXIT 1 MARKER
    ex1_t = pd.to_datetime(trade['Exit1_Time'])
    ex1_p = trade['Exit1_Price']
    ex1_time_str = ex1_t.strftime("%H:%M")
    
    ax.scatter(ex1_t, ex1_p, color='#f39c12', s=110, zorder=6, marker='o')
    ax.annotate(f"EXIT 1 (50%)\nTime: {ex1_time_str}\nPrice: @{ex1_p:.2f} ({trade['Exit1_PnL_Pct']:+.2f}%)", (ex1_t, ex1_p),
                textcoords="offset points", xytext=(0, 25),
                ha='center', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="#fef9e7", ec="#f39c12", lw=1.2))
                
    # EXIT 2 MARKER
    ex2_t = pd.to_datetime(trade['Exit2_Time'])
    ex2_p = trade['Exit2_Price']
    ex2_time_str = ex2_t.strftime("%H:%M")
    
    ax.scatter(ex2_t, ex2_p, color='#8e44ad', s=110, zorder=6, marker='s')
    ax.annotate(f"EXIT 2 (50%)\nTime: {ex2_time_str}\nPrice: @{ex2_p:.2f} ({trade['Exit2_PnL_Pct']:+.2f}%)", (ex2_t, ex2_p),
                textcoords="offset points", xytext=(0, -32),
                ha='center', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="#f4ecf7", ec="#8e44ad", lw=1.2))
                
    date_str = trade['Date']
    pnl_str = f"+${trade['Net_Trade_PnL_Amount']:,.2f}" if trade['Net_Trade_PnL_Amount'] >= 0 else f"-${abs(trade['Net_Trade_PnL_Amount']):,.2f}"
    status_color = '#27ae60' if trade['Trade_Status'] == 'WINNER' else '#c0392b'
    
    title_text = f"[LIMIT SYSTEM] {stock_name} ({date_str}) | Side: {side} | Retest Entry @ {entry_time_str} | Net PnL: {pnl_str} ({trade['Net_Trade_PnL_Pct']:+.2f}%) [{trade['Trade_Status']}]"
    ax.set_title(title_text, fontsize=11, fontweight='bold', color=status_color, pad=12)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_xlabel("Time (IST)", fontsize=9, fontweight='bold')
    ax.set_ylabel("Price (INR)", fontsize=9, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    safe_stock = stock_name.replace(" ", "_").replace(".", "_").replace("'", "")
    time_str = entry_t.strftime("%Y-%m-%d_%H-%M")
    filename = f"P_{safe_stock}_{time_str}.png"
    save_path = PLOTS_DIR / filename
    
    try:
        fig.savefig(save_path, dpi=100)
    except Exception as e:
        print(f"Warning saving plot {filename}: {e}")
    finally:
        plt.close(fig)


def run_limit_order_system():
    csv_files = glob.glob(str(DATA_DIR / "*_1m.csv"))
    daily_candidates = []

    for filepath in csv_files:
        filename = os.path.basename(filepath)
        ticker = filename.replace("_1m.csv", "").replace("_BO", ".BO").replace("_NS", ".NS").replace("_F", "=F")
        stock_name = ticker.replace(".NS", "").replace(".BO", "")
        
        df = load_stock_data(filepath)
        if df.empty: continue
        
        for day_date, day_df in df.groupby(df.index.date):
            date_str = day_date.strftime("%Y-%m-%d")
            
            df_945 = day_df.between_time("09:30", "09:45")
            if len(df_945) < 3: continue
            
            range_high = df_945['High'].max()
            range_low = df_945['Low'].min()
            range_volume = df_945['Volume'].sum()
            if range_high <= range_low: continue
            
            range_width_pct = ((range_high - range_low) / range_low) * 100.0
            
            daily_candidates.append({
                'Date': date_str, 'Stock_Name': stock_name, 'Ticker': ticker,
                'Range_High': range_high, 'Range_Low': range_low,
                'Range_Width_Pct': range_width_pct, 'Range_Volume': range_volume,
                'Day_DF': day_df
            })

    cand_df = pd.DataFrame(daily_candidates)
    filtered_cand = cand_df[(cand_df['Range_Width_Pct'] >= 0.4) & (cand_df['Range_Width_Pct'] <= 1.8)]
    top5_per_day = filtered_cand.groupby('Date', group_keys=False).apply(lambda g: g.nlargest(5, 'Range_Volume')).reset_index(drop=True)
    
    executed_trades = []

    for _, item in top5_per_day.iterrows():
        date_str, stock_name, ticker, r_h, r_l, day_df = item['Date'], item['Stock_Name'], item['Ticker'], item['Range_High'], item['Range_Low'], item['Day_DF']
        
        # 1-minute data from 09:50 AM to 14:45 PM
        df_eval = day_df.between_time("09:50", "15:10")
        if df_eval.empty: continue
        
        minute_records = df_eval.reset_index().to_dict('records')
        
        long_done, short_done = False, False
        
        for idx, m_row in enumerate(minute_records):
            m_time = m_row['Datetime']
            m_time_str = m_time.strftime("%H:%M")
            
            if m_time_str <= "14:45":
                # Check Retest at Low (Long Limit Order)
                if not long_done and m_row['Low'] <= r_l * 1.001:
                    entry_p = r_l
                    entry_t = m_time
                    sl_p = entry_p * 0.997
                    tp1_p = min(entry_p * 1.010, r_h)
                    tp2_p = entry_p * 1.020
                    
                    pos_size = 1.0
                    ex1_t, ex1_p, ex1_pnl = None, None, 0.0
                    ex2_t, ex2_p, ex2_pnl = None, None, 0.0
                    
                    for sub_k in range(idx + 1, len(minute_records)):
                        sub_m = minute_records[sub_k]
                        if sub_m['Low'] <= sl_p:
                            ex_val = min(sl_p, sub_m['Close'])
                            pnl_val = (ex_val - entry_p) / entry_p
                            if pos_size == 1.0: ex1_t, ex1_p, ex1_pnl, ex2_t, ex2_p, ex2_pnl = sub_m['Datetime'], ex_val, pnl_val, sub_m['Datetime'], ex_val, pnl_val
                            else: ex2_t, ex2_p, ex2_pnl = sub_m['Datetime'], ex_val, pnl_val
                            pos_size = 0.0
                            break
                        if pos_size == 1.0 and sub_m['High'] >= tp1_p:
                            ex1_t, ex1_p = sub_m['Datetime'], tp1_p
                            ex1_pnl = (tp1_p - entry_p) / entry_p
                            pos_size, sl_p = 0.5, entry_p
                        if pos_size == 0.5 and sub_m['High'] >= tp2_p:
                            ex2_t, ex2_p = sub_m['Datetime'], tp2_p
                            ex2_pnl = (tp2_p - entry_p) / entry_p
                            pos_size = 0.0
                            break
                            
                    if pos_size > 0:
                        last_m = minute_records[-1]
                        eod_v = last_m['Close']
                        eod_pnl_v = (eod_v - entry_p) / entry_p
                        if pos_size == 1.0: ex1_t, ex1_p, ex1_pnl, ex2_t, ex2_p, ex2_pnl = last_m['Datetime'], eod_v, eod_pnl_v, last_m['Datetime'], eod_v, eod_pnl_v
                        else: ex2_t, ex2_p, ex2_pnl = last_m['Datetime'], eod_v, eod_pnl_v

                    qty = max(1, int(NOTIONAL_PER_TRADE / entry_p))
                    gross_pnl_amount = (0.5 * (ex1_p - entry_p) + 0.5 * (ex2_p - entry_p)) * qty
                    fees = fyers_trade_cost(entry_p, qty, "buy") + fyers_trade_cost(ex1_p, int(qty/2), "sell") + fyers_trade_cost(ex2_p, int(qty/2), "sell")
                    net_pnl_amount = gross_pnl_amount - fees
                    net_pnl_pct = (net_pnl_amount / NOTIONAL_PER_TRADE) * 100.0

                    tr_record = {
                        'Date': date_str, 'Stock_Name': stock_name, 'Ticker': ticker, 'Side': 'LONG',
                        'Range_930_High': round(r_h, 2), 'Range_930_Low': round(r_l, 2),
                        'Confirmation_Time': entry_t, 'Retest_Entry_Time': entry_t,
                        'Entry_Price': round(entry_p, 2),
                        'Exit1_Time': ex1_t, 'Exit1_Price': round(ex1_p, 2), 'Exit1_PnL_Pct': round(ex1_pnl * 100, 2),
                        'Exit2_Time': ex2_t, 'Exit2_Price': round(ex2_p, 2), 'Exit2_PnL_Pct': round(ex2_pnl * 100, 2),
                        'Gross_PnL_Pct': round((0.5 * ex1_pnl + 0.5 * ex2_pnl) * 100, 2),
                        'Trade_Fee_Amount': round(fees, 2), 'Net_Trade_PnL_Pct': round(net_pnl_pct, 2),
                        'Net_Trade_PnL_Amount': round(net_pnl_amount, 2),
                        'Trade_Status': 'WINNER' if net_pnl_amount > 0 else 'LOSER'
                    }
                    executed_trades.append((tr_record, day_df, stock_name, r_h, r_l))
                    long_done = True

                # Check Retest at High (Short Limit Order)
                if not short_done and m_row['High'] >= r_h * 0.999:
                    entry_p = r_h
                    entry_t = m_time
                    sl_p = entry_p * 1.003
                    tp1_p = max(entry_p * 0.990, r_l)
                    tp2_p = entry_p * 0.980
                    
                    pos_size = 1.0
                    ex1_t, ex1_p, ex1_pnl = None, None, 0.0
                    ex2_t, ex2_p, ex2_pnl = None, None, 0.0
                    
                    for sub_k in range(idx + 1, len(minute_records)):
                        sub_m = minute_records[sub_k]
                        if sub_m['High'] >= sl_p:
                            ex_val = max(sl_p, sub_m['Close'])
                            pnl_val = (entry_p - ex_val) / entry_p
                            if pos_size == 1.0: ex1_t, ex1_p, ex1_pnl, ex2_t, ex2_p, ex2_pnl = sub_m['Datetime'], ex_val, pnl_val, sub_m['Datetime'], ex_val, pnl_val
                            else: ex2_t, ex2_p, ex2_pnl = sub_m['Datetime'], ex_val, pnl_val
                            pos_size = 0.0
                            break
                        if pos_size == 1.0 and sub_m['Low'] <= tp1_p:
                            ex1_t, ex1_p = sub_m['Datetime'], tp1_p
                            ex1_pnl = (entry_p - tp1_p) / entry_p
                            pos_size, sl_p = entry_p, entry_p
                        if pos_size == 0.5 and sub_m['Low'] <= tp2_p:
                            ex2_t, ex2_p = sub_m['Datetime'], tp2_p
                            ex2_pnl = (entry_p - tp2_p) / entry_p
                            pos_size = 0.0
                            break
                            
                    if pos_size > 0:
                        last_m = minute_records[-1]
                        eod_v = last_m['Close']
                        eod_pnl_v = (entry_p - eod_v) / entry_p
                        if pos_size == 1.0: ex1_t, ex1_p, ex1_pnl, ex2_t, ex2_p, ex2_pnl = last_m['Datetime'], eod_v, eod_pnl_v, last_m['Datetime'], eod_v, eod_pnl_v
                        else: ex2_t, ex2_p, ex2_pnl = last_m['Datetime'], eod_v, eod_pnl_v

                    qty = max(1, int(NOTIONAL_PER_TRADE / entry_p))
                    gross_pnl_amount = (0.5 * (entry_p - ex1_p) + 0.5 * (entry_p - ex2_p)) * qty
                    fees = fyers_trade_cost(entry_p, qty, "sell") + fyers_trade_cost(ex1_p, int(qty/2), "buy") + fyers_trade_cost(ex2_p, int(qty/2), "buy")
                    net_pnl_amount = gross_pnl_amount - fees
                    net_pnl_pct = (net_pnl_amount / NOTIONAL_PER_TRADE) * 100.0

                    tr_record = {
                        'Date': date_str, 'Stock_Name': stock_name, 'Ticker': ticker, 'Side': 'SHORT',
                        'Range_930_High': round(r_h, 2), 'Range_930_Low': round(r_l, 2),
                        'Confirmation_Time': entry_t, 'Retest_Entry_Time': entry_t,
                        'Entry_Price': round(entry_p, 2),
                        'Exit1_Time': ex1_t, 'Exit1_Price': round(ex1_p, 2), 'Exit1_PnL_Pct': round(ex1_pnl * 100, 2),
                        'Exit2_Time': ex2_t, 'Exit2_Price': round(ex2_p, 2), 'Exit2_PnL_Pct': round(ex2_pnl * 100, 2),
                        'Gross_PnL_Pct': round((0.5 * ex1_pnl + 0.5 * ex2_pnl) * 100, 2),
                        'Trade_Fee_Amount': round(fees, 2), 'Net_Trade_PnL_Pct': round(net_pnl_pct, 2),
                        'Net_Trade_PnL_Amount': round(net_pnl_amount, 2),
                        'Trade_Status': 'WINNER' if net_pnl_amount > 0 else 'LOSER'
                    }
                    executed_trades.append((tr_record, day_df, stock_name, r_h, r_l))
                    short_done = True

    trades_list = [t[0] for t in executed_trades]
    trades_df = pd.DataFrame(trades_list).sort_values(['Date', 'Retest_Entry_Time'])
    
    cols_order = [
        'Date', 'Stock_Name', 'Ticker', 'Side', 'Range_930_High', 'Range_930_Low',
        'Confirmation_Time', 'Retest_Entry_Time', 'Entry_Price',
        'Exit1_Time', 'Exit1_Price', 'Exit1_PnL_Pct',
        'Exit2_Time', 'Exit2_Price', 'Exit2_PnL_Pct',
        'Gross_PnL_Pct', 'Trade_Fee_Amount', 'Net_Trade_PnL_Pct', 'Net_Trade_PnL_Amount', 'Trade_Status'
    ]
    trades_df = trades_df[cols_order]
    
    detailed_csv = REPORTS_DIR / "Optimal_Top5_Trades_Detailed.csv"
    trades_df.to_csv(detailed_csv, index=False)
    print(f"Saved True Point-In-Time Limit Order Trade Log ({len(trades_df)} trades) -> {detailed_csv}")

    # Daily PnL Summary
    daily_summary = []
    cumulative_pnl = 0.0
    for date_val, group in trades_df.groupby('Date'):
        t_trades = len(group)
        w_trades = len(group[group['Trade_Status'] == 'WINNER'])
        l_trades = len(group[group['Trade_Status'] == 'LOSER'])
        win_rate = (w_trades / t_trades) * 100.0 if t_trades > 0 else 0.0
        
        gross_pnl_amt = group['Net_Trade_PnL_Amount'].sum() + group['Trade_Fee_Amount'].sum()
        total_fees_amt = group['Trade_Fee_Amount'].sum()
        net_pnl_amt = group['Net_Trade_PnL_Amount'].sum()
        cumulative_pnl += net_pnl_amt
        
        daily_summary.append({
            'Date': date_val, 'Total_Trades': t_trades, 'Winning_Trades': w_trades, 'Losing_Trades': l_trades,
            'Win_Rate_Pct': round(win_rate, 1), 'Gross_PnL_Amount': round(gross_pnl_amt, 2),
            'Total_Charges_Amount': round(total_fees_amt, 2), 'Net_PnL_Amount': round(net_pnl_amt, 2),
            'Cumulative_Net_PnL_Amount': round(cumulative_pnl, 2)
        })
        
    daily_df = pd.DataFrame(daily_summary).sort_values('Date')
    daily_csv = REPORTS_DIR / "Optimal_Top5_Daily_PnL_Summary.csv"
    daily_df.to_csv(daily_csv, index=False)
    print(f"Saved True Point-In-Time Daily PnL Summary ({len(daily_df)} days) -> {daily_csv}\n")

    # Generate Candlestick Plots
    print(f"Generating Candlestick Trade Plots in {PLOTS_DIR}...")
    for f in glob.glob(str(PLOTS_DIR / "*.png")):
        try: os.remove(f)
        except Exception: pass
        
    plot_count = 0
    for item in executed_trades:
        tr_dict, day_df, sname, r_h, r_l = item
        plot_candlestick_trade_chart(day_df, sname, tr_dict, r_h, r_l)
        plot_count += 1
        
    print(f"Successfully generated {plot_count} Retest Candlestick trade plots in {PLOTS_DIR}!")

    print("\n==========================================================================")
    print("      TRUE POINT-IN-TIME MINUTE-BY-MINUTE LIMIT ORDER RESULTS              ")
    print("==========================================================================")
    print(f"Total Retest Trades Executed: {len(trades_df):,}")
    print(f"Trade Win Rate: {trades_df['Trade_Status'].eq('WINNER').mean()*100:.1f}%")
    print(f"Profitable Trading Days: {daily_df['Net_PnL_Amount'].gt(0).sum()} / {len(daily_df)} Days ({(daily_df['Net_PnL_Amount'].gt(0).sum()/len(daily_df))*100:.1f}% Win Days)")
    print(f"Total Net PnL (After Charges): ${daily_df['Net_PnL_Amount'].sum():+,.2f}")
    print("==========================================================================")

if __name__ == "__main__":
    run_limit_order_system()
