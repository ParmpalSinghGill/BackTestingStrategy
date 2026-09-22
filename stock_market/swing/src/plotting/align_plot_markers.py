"""
Align Trade Plot Markers Cleanly on 15M Candlestick Time Grid
------------------------------------------------------------
Snaps Entry, Exit 1, and Exit 2 markers to exact 15m candle bar timestamps
so markers align 100% cleanly on top of the candles!
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

os.makedirs(PLOTS_DIR, exist_ok=True)


def resample_ohlc(df: pd.DataFrame, timeframe: str = '15min') -> pd.DataFrame:
    return df.resample(timeframe, closed='left', label='left').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()


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
    return df.dropna(subset=['Open', 'High', 'Low', 'Close'])


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

    # SNAP MARKERS TO 15M CANDLE TIMESTAMP GRID FOR PERFECT ALIGNMENT
    raw_entry_t = pd.to_datetime(trade['Entry_Time'])
    entry_t = snap_to_15m_grid(raw_entry_t)
    entry_time_str = raw_entry_t.strftime("%H:%M")
    
    ax.scatter(entry_t, entry_p, color=color_side, s=150, zorder=7, marker=marker_side)
    ax.annotate(f"ENTRY ({side}) @{entry_p:.2f}", (entry_t, entry_p),
                textcoords="offset points", xytext=(0, 18 if side == 'LONG' else -28),
                ha='center', fontsize=8.5, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.25", fc="#e8f8f5" if side == 'LONG' else "#fadbd8", ec=color_side, lw=1.2))
                
    # EXIT 1 MARKER
    raw_ex1_t = pd.to_datetime(trade['Exit1_Time'])
    ex1_t = snap_to_15m_grid(raw_ex1_t)
    ex1_p = float(trade['Exit1_Price'])
    ax.scatter(ex1_t, ex1_p, color='#f39c12', s=110, zorder=6, marker='o')
    ax.annotate(f"EXIT 1 (50%) @{ex1_p:.2f}", (ex1_t, ex1_p),
                textcoords="offset points", xytext=(0, 20),
                ha='center', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", fc="#fef9e7", ec="#f39c12", lw=1))
                
    # EXIT 2 MARKER
    raw_ex2_t = pd.to_datetime(trade['Exit2_Time'])
    ex2_t = snap_to_15m_grid(raw_ex2_t)
    ex2_p = float(trade['Exit2_Price'])
    ax.scatter(ex2_t, ex2_p, color='#8e44ad', s=110, zorder=6, marker='s')
    ax.annotate(f"EXIT 2 (50%) @{ex2_p:.2f}", (ex2_t, ex2_p),
                textcoords="offset points", xytext=(0, -24),
                ha='center', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", fc="#f4ecf7", ec="#8e44ad", lw=1))
                
    date_str = trade['Date']
    pnl_str = f"+${trade['Net_Trade_PnL_Amount']:,.2f}" if trade['Net_Trade_PnL_Amount'] >= 0 else f"-${abs(trade['Net_Trade_PnL_Amount']):,.2f}"
    status_color = '#27ae60' if trade['Trade_Status'] == 'WINNER' else '#c0392b'
    
    title_text = f"{stock_name} ({date_str}) | Side: {side} | Net PnL: {pnl_str} ({trade['Net_Trade_PnL_Pct']:+.2f}%) [{trade['Trade_Status']}]"
    ax.set_title(title_text, fontsize=11, fontweight='bold', color=status_color, pad=12)
    
    info_text = (
        f"--- TRADE DETAILS ---\n"
        f"Date: {date_str}\n"
        f"Side: {side}\n"
        f"Entry Time: {entry_time_str} @ {entry_p:.2f}\n"
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


def main():
    trades_csv = REPORTS_DIR / "Optimal_Top5_Trades_Detailed.csv"
    if not trades_csv.exists():
        return
        
    trades_df = pd.read_csv(trades_csv)
    
    for f in glob.glob(str(PLOTS_DIR / "*.png")):
        try: os.remove(f)
        except Exception: pass
        
    count = 0
    for idx, row in trades_df.iterrows():
        ticker = str(row['Ticker'])
        stock_name = str(row['Stock_Name'])
        date_str = str(row['Date'])
        
        file_pattern = str(DATA_DIR / f"*{ticker.replace('.NS', '').replace('.BO', '')}*_1m.csv")
        matching_files = glob.glob(file_pattern)
        if not matching_files: continue
            
        day_df_all = load_stock_data(matching_files[0])
        day_date = pd.to_datetime(date_str).date()
        day_df = day_df_all[day_df_all.index.date == day_date]
        if day_df.empty: continue
            
        r_h = float(row['Range_930_High'])
        r_l = float(row['Range_930_Low'])
        
        tr_dict = row.to_dict()
        plot_candlestick_trade_chart(day_df, stock_name, tr_dict, r_h, r_l)
        count += 1
        
    print(f"Successfully re-aligned {count} trade plots on 15M candle grid in {PLOTS_DIR}!")

if __name__ == "__main__":
    main()
