"""
Fix Point-In-Time Retest Timestamps & Execution Realism
------------------------------------------------------
1. Confirmation Time = 15M Candle Close Boundary (e.g. 10:00 AM).
2. Retest Entry Time = Exact Minute AFTER Confirmation when price retests 9:30 Level.
3. Market Entry Time = 10:00 AM (Entry Price = 10:00 AM Close Price).
4. Generates updated CSVs and clear OHLC Candlestick charts in plot/Plot_15M/.
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


def plot_candlestick_trade_chart(day_df: pd.DataFrame, stock_name: str, trade: dict, range_high: float, range_low: float):
    fig, ax = plt.subplots(figsize=(14, 7.2), dpi=100)
    
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

    # 1. CONFIRMATION CANDLE MARKER
    conf_t = pd.to_datetime(trade['Confirmation_Time'])
    conf_p = range_low if side == 'LONG' else range_high
    conf_time_str = conf_t.strftime("%H:%M")
    
    ax.scatter(conf_t, conf_p, color='#3498db', s=130, zorder=6, marker='d')
    ax.annotate(f"CONFIRMATION (15M Close)\nTime: {conf_time_str}\nSignal: {side}", (conf_t, conf_p),
                textcoords="offset points", xytext=(0, 25 if side == 'SHORT' else -38),
                ha='center', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="#ebf5fb", ec="#3498db", lw=1.2))

    # 2. ENTRY MARKER
    entry_t = pd.to_datetime(trade['Entry_Time'])
    entry_p = trade['Entry_Price']
    entry_time_str = entry_t.strftime("%H:%M")
    
    ax.scatter(entry_t, entry_p, color=color_side, s=160, zorder=7, marker=marker_side)
    ax.annotate(f"ENTRY ({side})\nTime: {entry_time_str}\nPrice: @{entry_p:.2f}", (entry_t, entry_p),
                textcoords="offset points", xytext=(0, 22 if side == 'LONG' else -35),
                ha='center', fontsize=8.5, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="#e8f8f5" if side == 'LONG' else "#fadbd8", ec=color_side, lw=1.2))
                
    # 3. EXIT 1 MARKER
    ex1_t = pd.to_datetime(trade['Exit1_Time'])
    ex1_p = trade['Exit1_Price']
    ex1_time_str = ex1_t.strftime("%H:%M")
    
    ax.scatter(ex1_t, ex1_p, color='#f39c12', s=110, zorder=6, marker='o')
    ax.annotate(f"EXIT 1 (50%)\nTime: {ex1_time_str}\nPrice: @{ex1_p:.2f} ({trade['Exit1_PnL_Pct']:+.2f}%)", (ex1_t, ex1_p),
                textcoords="offset points", xytext=(0, 25),
                ha='center', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="#fef9e7", ec="#f39c12", lw=1.2))
                
    # 4. EXIT 2 MARKER
    ex2_t = pd.to_datetime(trade['Exit2_Time'])
    ex2_p = trade['Exit2_Price']
    ex2_time_str = ex2_t.strftime("%H:%M")
    
    ax.scatter(ex2_t, ex2_p, color='#8e44ad', s=110, zorder=6, marker='s')
    ax.annotate(f"EXIT 2 (50%)\nTime: {ex2_time_str}\nPrice: @{ex2_p:.2f} ({trade['Exit2_PnL_Pct']:+.2f}%)", (ex2_t, ex2_p),
                textcoords="offset points", xytext=(0, -32),
                ha='center', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", fc="#f4ecf7", ec="#8e44ad", lw=1.2))
                
    date_str = trade['Date']
    pnl_str = f"+${trade['Net_Trade_PnL_Amount']:,.2f}" if trade['Net_Trade_PnL_Amount'] >= 0 else f"-${abs(trade['Net_Trade_PnL_Amount']):,.2f}"
    status_color = '#27ae60' if trade['Trade_Status'] == 'WINNER' else '#c0392b'
    
    title_text = f"[REAL TIME ENGINE] {stock_name} ({date_str}) | Side: {side} | Conf: {conf_time_str} | Entry: {entry_time_str} | Net PnL: {pnl_str} ({trade['Net_Trade_PnL_Pct']:+.2f}%) [{trade['Trade_Status']}]"
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


def run_realistic_retest_engine():
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
        
        # 15-minute boundaries to check confirmation
        boundaries = ["10:00", "10:15", "10:30", "10:45", "11:00", "11:15", "11:30", "11:45",
                      "12:00", "12:15", "12:30", "12:45", "13:00", "13:15", "13:30", "13:45",
                      "14:00", "14:15", "14:30", "14:45"]
                      
        for b_time_str in boundaries:
            conf_time = pd.to_datetime(f"{date_str} {b_time_str}:00")
            
            for sname in top5_names:
                if sname in traded_stocks_today: continue
                
                sinfo = daily_stocks[sname]
                df_up_to_conf = sinfo['DF'][sinfo['DF'].index <= conf_time]
                c_15m_start = conf_time - pd.Timedelta(minutes=15)
                c_15m_df = df_up_to_conf[df_up_to_conf.index >= c_15m_start]
                
                if not c_15m_df.empty:
                    c_o, c_h, c_l, c_c = c_15m_df['Open'].iloc[0], c_15m_df['High'].max(), c_15m_df['Low'].min(), c_15m_df['Close'].iloc[-1]
                    r_h = top5_dict[sname]['Range_High']
                    r_l = top5_dict[sname]['Range_Low']
                    
                    side = None
                    if c_l <= r_l * 1.001 and c_c > r_l and c_c >= c_o:
                        side = 'LONG'
                    elif c_h >= r_h * 0.999 and c_c < r_h and c_c <= c_o:
                        side = 'SHORT'
                        
                    if side:
                        # NOW SEARCH SUBSEQUENT MINUTES AFTER CONFIRMATION FOR RETEST AT 9:30 LEVEL
                        df_after_conf = sinfo['DF'][sinfo['DF'].index > conf_time]
                        m_records = df_after_conf.reset_index().to_dict('records')
                        
                        entry_found = False
                        entry_t = None
                        entry_p = r_l if side == 'LONG' else r_h
                        entry_idx_in_records = 0
                        
                        for k, sub_m in enumerate(m_records):
                            sub_t_str = sub_m['Datetime'].strftime("%H:%M")
                            if sub_t_str > "14:45": break
                            
                            # RETEST CONDITION
                            if side == 'LONG' and sub_m['Low'] <= entry_p * 1.001:
                                entry_found = True
                                entry_t = sub_m['Datetime']
                                entry_idx_in_records = k
                                break
                            elif side == 'SHORT' and sub_m['High'] >= entry_p * 0.999:
                                entry_found = True
                                entry_t = sub_m['Datetime']
                                entry_idx_in_records = k
                                break
                                
                        if entry_found:
                            sl_p = entry_p * 0.997 if side == 'LONG' else entry_p * 1.003
                            tp1_p = min(entry_p * 1.010, r_h) if side == 'LONG' else max(entry_p * 0.990, r_l)
                            tp2_p = entry_p * 1.020 if side == 'LONG' else entry_p * 0.980
                            
                            pos_size = 1.0
                            ex1_t, ex1_p, ex1_pnl = None, None, 0.0
                            ex2_t, ex2_p, ex2_pnl = None, None, 0.0
                            
                            for sub_k in range(entry_idx_in_records + 1, len(m_records)):
                                sub_m = m_records[sub_k]
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
                                else:
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
                                'Confirmation_Time': conf_time, 'Entry_Time': entry_t, 'Entry_Price': round(entry_p, 2),
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
        'Confirmation_Time', 'Entry_Time', 'Entry_Price',
        'Exit1_Time', 'Exit1_Price', 'Exit1_PnL_Pct',
        'Exit2_Time', 'Exit2_Price', 'Exit2_PnL_Pct',
        'Gross_PnL_Pct', 'Trade_Fee_Amount', 'Net_Trade_PnL_Pct', 'Net_Trade_PnL_Amount', 'Trade_Status'
    ]
    trades_df = trades_df[cols_order]
    
    detailed_csv = REPORTS_DIR / "Optimal_Top5_Trades_Detailed.csv"
    trades_df.to_csv(detailed_csv, index=False)
    print(f"Saved Strict Separate Timestamp Retest Trade Log ({len(trades_df)} trades) -> {detailed_csv}")

    # Daily Summary & ROE Calculation
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
    print(f"Saved Daily PnL Summary ({len(daily_df)} days) -> {daily_csv}\n")

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
    print("      REALISTIC POINT-IN-TIME RETEST ENGINE RESULTS                        ")
    print("==========================================================================")
    print(f"Total Retest Trades Executed: {len(trades_df):,}")
    print(f"Trade Win Rate: {trades_df['Trade_Status'].eq('WINNER').mean()*100:.1f}%")
    print(f"Profitable Trading Days: {daily_df['Net_PnL_Amount'].gt(0).sum()} / {len(daily_df)} Days ({(daily_df['Net_PnL_Amount'].gt(0).sum()/len(daily_df))*100:.1f}% Win Days)")
    print(f"Total Net Profit (After Fees): ${daily_df['Net_PnL_Amount'].sum():+,.2f}")
    roe_val = (daily_df['Net_PnL_Amount'].sum() / STARTING_CAPITAL) * 100.0
    print(f"MONTHLY RETURN ON EQUITY (ROE): +{roe_val:.2f}% Net Return in 1 Month")
    print("==========================================================================")

if __name__ == "__main__":
    run_realistic_retest_engine()
