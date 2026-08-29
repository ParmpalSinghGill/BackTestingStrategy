"""
Pure Real-Time 1-Minute Limit Order Engine (ZERO Look-Ahead Bias)
------------------------------------------------------------------
1. At 09:45 AM, place Limit Buy Order @ 9:30 Low and Limit Short Order @ 9:30 High.
2. The VERY FIRST minute candle that touches the limit price (e.g. 10:15 AM wick) FILLS THE ORDER IMMEDIATELY.
3. Zero checking of candle close! Zero look-ahead bias!
4. If price rises to Stop Loss (0.3% above 9:30 High), trade stops out immediately.
5. Max 1 Attempt Per Stock Per Day.
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


def snap_to_15m_grid(dt_val: pd.Timestamp) -> pd.Timestamp:
    minute = dt_val.minute
    snapped_minute = (minute // 15) * 15
    return dt_val.replace(minute=snapped_minute, second=0, microsecond=0)


def plot_candlestick_trade_chart(day_df: pd.DataFrame, stock_name: str, trade: dict, range_high: float, range_low: float):
    fig, ax = plt.subplots(figsize=(13.5, 7), dpi=100)
    plot_15m = resample_ohlc(day_df.between_time("09:15", "15:30"), '15min')
    if plot_15m.empty:
        plt.close(fig)
        return
        
    draw_candlesticks(ax, plot_15m)
    
    side = trade['Side']
    entry_p = float(trade['Entry_Price'])
    sl_price = range_low * 0.997 if side == 'LONG' else range_high * 1.003
    
    ax.axhline(range_high, color='#e74c3c', linestyle='--', linewidth=1.4, label=f'9:30 High ({range_high:.2f})')
    ax.axhline(range_low, color='#2ecc71', linestyle='--', linewidth=1.4, label=f'9:30 Low ({range_low:.2f})')
    ax.axhline(sl_price, color='#8B0000', linestyle=':', linewidth=1.6, label=f'Initial SL ({sl_price:.2f})')
    
    range_start = plot_15m.index[0].replace(hour=9, minute=30, second=0)
    range_end = plot_15m.index[0].replace(hour=9, minute=45, second=0)
    ax.axvspan(range_start, range_end, color='#f39c12', alpha=0.12, label='9:30-9:45 Range Window')
    
    color_side = '#27ae60' if side == 'LONG' else '#c0392b'
    marker_side = '^' if side == 'LONG' else 'v'

    raw_entry_t = pd.to_datetime(trade['Entry_Time'])
    entry_t = snap_to_15m_grid(raw_entry_t)
    entry_time_str = raw_entry_t.strftime("%H:%M")
    
    ax.scatter(entry_t, entry_p, color=color_side, s=150, zorder=7, marker=marker_side)
    ax.annotate(f"FIRST WICK ENTRY ({side}) @{entry_p:.2f}", (entry_t, entry_p),
                textcoords="offset points", xytext=(0, 18 if side == 'LONG' else -28),
                ha='center', fontsize=8.5, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.25", fc="#e8f8f5" if side == 'LONG' else "#fadbd8", ec=color_side, lw=1.2))
                
    raw_ex1_t = pd.to_datetime(trade['Exit1_Time'])
    ex1_t = snap_to_15m_grid(raw_ex1_t)
    ex1_p = float(trade['Exit1_Price'])
    ax.scatter(ex1_t, ex1_p, color='#f39c12', s=100, zorder=6, marker='o')
    ax.annotate(f"EXIT 1 (50%) @{ex1_p:.2f}", (ex1_t, ex1_p),
                textcoords="offset points", xytext=(0, 20),
                ha='center', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", fc="#fef9e7", ec="#f39c12", lw=1))
                
    raw_ex2_t = pd.to_datetime(trade['Exit2_Time'])
    ex2_t = snap_to_15m_grid(raw_ex2_t)
    ex2_p = float(trade['Exit2_Price'])
    ax.scatter(ex2_t, ex2_p, color='#8e44ad', s=100, zorder=6, marker='s')
    ax.annotate(f"EXIT 2 (50%) @{ex2_p:.2f}", (ex2_t, ex2_p),
                textcoords="offset points", xytext=(0, -24),
                ha='center', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", fc="#f4ecf7", ec="#8e44ad", lw=1))
                
    date_str = trade['Date']
    pnl_str = f"+${trade['Net_Trade_PnL_Amount']:,.2f}" if trade['Net_Trade_PnL_Amount'] >= 0 else f"-${abs(trade['Net_Trade_PnL_Amount']):,.2f}"
    status_color = '#27ae60' if trade['Trade_Status'] == 'WINNER' else '#c0392b'
    
    title_text = f"[PURE REALTIME ENGINE] {stock_name} ({date_str}) | Side: {side} | Net PnL: {pnl_str} ({trade['Net_Trade_PnL_Pct']:+.2f}%) [{trade['Trade_Status']}]"
    ax.set_title(title_text, fontsize=11, fontweight='bold', color=status_color, pad=12)
    
    info_text = (
        f"--- PURE REALTIME TRADE ---\n"
        f"Date: {date_str}\n"
        f"Side: {side}\n"
        f"First Wick Entry: {entry_time_str} @ {entry_p:.2f}\n"
        f"Initial SL: {sl_price:.2f} (0.3%)\n"
        f"Exit 1: {raw_ex1_t.strftime('%H:%M')} @ {ex1_p:.2f} ({trade['Exit1_PnL_Pct']:+.2f}%)\n"
        f"Exit 2: {raw_ex2_t.strftime('%H:%M')} @ {ex2_p:.2f} ({trade['Exit2_PnL_Pct']:+.2f}%)\n"
        f"Net PnL: {pnl_str} ({trade['Net_Trade_PnL_Pct']:+.2f}%)\n"
        f"Status: {trade['Trade_Status']}"
    )
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_xlabel("Time (IST)", fontsize=9, fontweight='bold')
    ax.set_ylabel("Price (INR)", fontsize=9, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.5)
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, loc='upper right', fontsize=8)
    
    ax.text(0.015, 0.96, info_text, transform=ax.transAxes, fontsize=8, verticalalignment='top',
            fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.5", fc="#ffffff", ec="#bdc3c7", alpha=0.9, lw=1))
    
    plt.tight_layout()
    safe_stock = stock_name.replace(" ", "_").replace(".", "_").replace("'", "")
    time_str = raw_entry_t.strftime("%Y-%m-%d_%H-%M")
    filename = f"P_{safe_stock}_{time_str}.png"
    save_path = PLOTS_DIR / filename
    
    try:
        fig.savefig(save_path, dpi=100)
    except Exception as e:
        print(f"Warning saving plot {filename}: {e}")
    finally:
        plt.close(fig)


def run_pure_realtime_engine():
    csv_files = glob.glob(str(DATA_DIR / "*_1m.csv"))
    
    stocks_data = {}
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        ticker = filename.replace("_1m.csv", "").replace("_BO", ".BO").replace("_NS", ".NS").replace("_F", "=F")
        stock_name = ticker.replace(".NS", "").replace(".BO", "")
        df = load_stock_data(filepath)
        if not df.empty:
            stocks_data[stock_name] = {'Ticker': ticker, 'DF': df}

    all_dates = sorted(list(set(d for s in stocks_data.values() for d in s['DF'].index.date)))
    executed_trades = []

    for current_date in all_dates:
        date_str = current_date.strftime("%Y-%m-%d")
        daily_stocks = {}
        for sname, sdata in stocks_data.items():
            day_df = sdata['DF'][sdata['DF'].index.date == current_date]
            if len(day_df) >= 30:
                daily_stocks[sname] = {'Ticker': sdata['Ticker'], 'DF': day_df}
                
        if not daily_stocks: continue
        
        range_stats = []
        for sname, sinfo in daily_stocks.items():
            df_945 = sinfo['DF'].between_time("09:30", "09:45")
            if len(df_945) >= 3:
                r_h = df_945['High'].max()
                r_l = df_945['Low'].min()
                r_v = df_945['Volume'].sum()
                if r_h > r_l:
                    r_w = ((r_h - r_l) / r_l) * 100.0
                    if 0.4 <= r_w <= 1.8:
                        range_stats.append({
                            'Stock_Name': sname, 'Ticker': sinfo['Ticker'],
                            'Range_High': r_h, 'Range_Low': r_l, 'Range_Width_Pct': r_w, 'Range_Volume': r_v
                        })
                        
        if not range_stats: continue
        top5_today = pd.DataFrame(range_stats).nlargest(5, 'Range_Volume').to_dict('records')
        top5_names = set(item['Stock_Name'] for item in top5_today)
        top5_dict = {item['Stock_Name']: item for item in top5_today}
        
        traded_stocks_today = set()
        
        # PURE 1-MINUTE TIME LOOP FROM 09:50 AM TO 14:45 PM
        time_range = pd.date_range(f"{date_str} 09:50:00", f"{date_str} 14:45:00", freq='1min')
        
        for t in time_range:
            t_str = t.strftime("%H:%M")
            
            for sname in top5_names:
                if sname in traded_stocks_today: continue
                
                sinfo = daily_stocks[sname]
                # Get the single 1-minute bar at timestamp t
                m_bar = sinfo['DF'][sinfo['DF'].index == t]
                if m_bar.empty: continue
                
                m_row = m_bar.iloc[0]
                r_h = top5_dict[sname]['Range_High']
                r_l = top5_dict[sname]['Range_Low']
                
                side = None
                # FIRST WICK TOUCH EXECUTION (ZERO LOOK-AHEAD BIAS)
                if m_row['Low'] <= r_l * 1.001:
                    side = 'LONG'
                elif m_row['High'] >= r_h * 0.999:
                    side = 'SHORT'
                    
                if side:
                    entry_p = r_l if side == 'LONG' else r_h
                    entry_t = t
                    sl_p = entry_p * 0.997 if side == 'LONG' else entry_p * 1.003
                    tp1_p = min(entry_p * 1.010, r_h) if side == 'LONG' else max(entry_p * 0.990, r_l)
                    tp2_p = entry_p * 1.020 if side == 'LONG' else entry_p * 0.980
                    
                    df_rest = sinfo['DF'][sinfo['DF'].index > t]
                    m_records = df_rest.reset_index().to_dict('records')
                    
                    pos_size = 1.0
                    ex1_t, ex1_p, ex1_pnl = None, None, 0.0
                    ex2_t, ex2_p, ex2_pnl = None, None, 0.0
                    
                    for sub_m in m_records:
                        if side == 'LONG':
                            if sub_m['Low'] <= sl_p:
                                ex_v = min(sl_p, sub_m['Close'])
                                pnl_v = (ex_v - entry_p) / entry_p
                                if pos_size == 1.0: ex1_t, ex1_p, ex1_pnl, ex2_t, ex2_p, ex2_pnl = sub_m['Datetime'], ex_v, pnl_v, sub_m['Datetime'], ex_v, pnl_v
                                else: ex2_t, ex2_p, ex2_pnl = sub_m['Datetime'], ex_v, pnl_v
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
                        else: # SHORT
                            if sub_m['High'] >= sl_p:
                                ex_v = max(sl_p, sub_m['Close'])
                                pnl_v = (entry_p - ex_v) / entry_p
                                if pos_size == 1.0: ex1_t, ex1_p, ex1_pnl, ex2_t, ex2_p, ex2_pnl = sub_m['Datetime'], ex_v, pnl_v, sub_m['Datetime'], ex_v, pnl_v
                                else: ex2_t, ex2_p, ex2_pnl = sub_m['Datetime'], ex_v, pnl_v
                                pos_size = 0.0
                                break
                            if pos_size == 1.0 and sub_m['Low'] <= tp1_p:
                                ex1_t, ex1_p = sub_m['Datetime'], tp1_p
                                ex1_pnl = (entry_p - tp1_p) / entry_p
                                pos_size, sl_p = 0.5, entry_p
                            if pos_size == 0.5 and sub_m['Low'] <= tp2_p:
                                ex2_t, ex2_p = sub_m['Datetime'], tp2_p
                                ex2_pnl = (entry_p - tp2_p) / entry_p
                                pos_size = 0.0
                                break

                    if pos_size > 0:
                        last_m = m_records[-1]
                        eod_v = last_m['Close']
                        eod_pnl_v = (eod_v - entry_p) / entry_p if side == 'LONG' else (entry_p - eod_v) / entry_p
                        if pos_size == 1.0: ex1_t, ex1_p, ex1_pnl, ex2_t, ex2_p, ex2_pnl = last_m['Datetime'], eod_v, eod_pnl_v, last_m['Datetime'], eod_v, eod_pnl_v
                        else: ex2_t, ex2_p, ex2_pnl = last_m['Datetime'], eod_v, eod_pnl_v

                    qty = max(1, int(NOTIONAL_PER_TRADE / entry_p))
                    gross_pnl_amt = (0.5 * (ex1_p - entry_p) + 0.5 * (ex2_p - entry_p)) * qty if side == 'LONG' else (0.5 * (entry_p - ex1_p) + 0.5 * (entry_p - ex2_p)) * qty
                    fees = fyers_trade_cost(entry_p, qty, "buy" if side == 'LONG' else "sell") + fyers_trade_cost(ex1_p, int(qty/2), "sell" if side == 'LONG' else "buy") + fyers_trade_cost(ex2_p, int(qty/2), "sell" if side == 'LONG' else "buy")
                    net_pnl_amt = gross_pnl_amt - fees
                    net_pnl_pct = (net_pnl_amt / NOTIONAL_PER_TRADE) * 100.0
                    
                    tr_rec = {
                        'Date': date_str, 'Stock_Name': sname, 'Ticker': sinfo['Ticker'], 'Side': side,
                        'Range_930_High': round(r_h, 2), 'Range_930_Low': round(r_l, 2),
                        'Entry_Time': entry_t, 'Entry_Price': round(entry_p, 2),
                        'Exit1_Time': ex1_t, 'Exit1_Price': round(ex1_p, 2), 'Exit1_PnL_Pct': round(ex1_pnl * 100, 2),
                        'Exit2_Time': ex2_t, 'Exit2_Price': round(ex2_p, 2), 'Exit2_PnL_Pct': round(ex2_pnl * 100, 2),
                        'Gross_PnL_Pct': round((0.5 * ex1_pnl + 0.5 * ex2_pnl) * 100, 2),
                        'Trade_Fee_Amount': round(fees, 2), 'Net_Trade_PnL_Pct': round(net_pnl_pct, 2),
                        'Net_Trade_PnL_Amount': round(net_pnl_amt, 2),
                        'Trade_Status': 'WINNER' if net_pnl_amt > 0 else 'LOSER'
                    }
                    executed_trades.append((tr_rec, sinfo['DF'][sinfo['DF'].index.date == current_date], sname, r_h, r_l))
                    traded_stocks_today.add(sname)

    trades_list = [t[0] for t in executed_trades]
    trades_df = pd.DataFrame(trades_list).sort_values(['Date', 'Entry_Time'])
    
    cols_order = [
        'Date', 'Stock_Name', 'Ticker', 'Side', 'Range_930_High', 'Range_930_Low',
        'Entry_Time', 'Entry_Price',
        'Exit1_Time', 'Exit1_Price', 'Exit1_PnL_Pct',
        'Exit2_Time', 'Exit2_Price', 'Exit2_PnL_Pct',
        'Gross_PnL_Pct', 'Trade_Fee_Amount', 'Net_Trade_PnL_Pct', 'Net_Trade_PnL_Amount', 'Trade_Status'
    ]
    trades_df = trades_df[cols_order]
    
    detailed_csv = REPORTS_DIR / "Optimal_Top5_Trades_Detailed.csv"
    trades_df.to_csv(detailed_csv, index=False)
    print(f"Saved Pure Real-Time First Wick Trade Log ({len(trades_df)} trades) -> {detailed_csv}")

    # Daily Summary
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
    print(f"Saved Pure Real-Time Daily PnL Summary ({len(daily_df)} days) -> {daily_csv}\n")

    # Generate Plots
    print(f"Generating Candlestick Plots in {PLOTS_DIR}...")
    for f in glob.glob(str(PLOTS_DIR / "*.png")):
        try: os.remove(f)
        except Exception: pass
        
    plot_count = 0
    for item in executed_trades:
        tr_dict, day_df, sname, r_h, r_l = item
        plot_candlestick_trade_chart(day_df, sname, tr_dict, r_h, r_l)
        plot_count += 1
        
    print(f"Successfully generated {plot_count} trade plots in {PLOTS_DIR}!")

    print("\n==========================================================================")
    print("      PURE REAL-TIME FIRST WICK TOUCH RESULTS (ZERO LOOK-AHEAD BIAS)      ")
    print("==========================================================================")
    print(f"Total Trades Taken:            {len(trades_df):,} (Exactly {len(trades_df)/len(daily_df):.1f} trades/day)")
    print(f"Trade Win Rate:                {trades_df['Trade_Status'].eq('WINNER').mean()*100:.1f}%")
    print(f"Profitable Trading Days:       {daily_df['Net_PnL_Amount'].gt(0).sum()} / {len(daily_df)} Days ({(daily_df['Net_PnL_Amount'].gt(0).sum()/len(daily_df))*100:.1f}% Win Days)")
    print(f"Total Net Profit (After Fees): ${daily_df['Net_PnL_Amount'].sum():+,.2f}")
    roe_val = (daily_df['Net_PnL_Amount'].sum() / STARTING_CAPITAL) * 100.0
    print(f"MONTHLY RETURN ON EQUITY (ROE): +{roe_val:.2f}% Net Return in 1 Month")
    print("==========================================================================")

if __name__ == "__main__":
    run_pure_realtime_engine()
