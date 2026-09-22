"""
Generate 15M Detailed Trade Log CSV, Daily PnL CSV, and Trade Plots
----------------------------------------------------------------------
Produces:
1. Reports/15M_Trades_Detailed.csv
2. Reports/15M_Daily_PnL_Summary.csv
3. plot/Plot_15M/P_{StockName}_{Date}_{Time}.png for trade charts matching CSV 100%.
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
from pathlib import Path
from stocks_parser import parse_stocks_file

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "Reports"
PLOTS_DIR = BASE_DIR / "plot" / "Plot_15M"

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

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


def resample_ohlc(df: pd.DataFrame, timeframe: str = '15min') -> pd.DataFrame:
    resampled = df.resample(timeframe, closed='left', label='left').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    return resampled


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


def simulate_trade_details(candles: list, trigger_idx: int, side: str, range_high: float, range_low: float):
    num_candles = len(candles)
    signal_candle = candles[trigger_idx]
    entry_time = signal_candle['Datetime']
    
    if side == 'LONG':
        entry_price = range_low
        sl_price = range_low * 0.997 # 0.3% SL
        tp1_target = min(entry_price * 1.008, range_high)
        tp2_target = entry_price * 1.015
    else: # SHORT
        entry_price = range_high
        sl_price = range_high * 1.003
        tp1_target = max(entry_price * 0.992, range_low)
        tp2_target = entry_price * 0.985

    pos_size = 1.0
    exit1_time = None
    exit1_price = None
    exit1_pnl_pct = 0.0
    
    exit2_time = None
    exit2_price = None
    exit2_pnl_pct = 0.0
    
    hit_tp1 = False
    
    for j in range(trigger_idx + 1, num_candles):
        c = candles[j]
        
        if side == 'LONG':
            # Check Stop Loss
            if c['Low'] <= sl_price:
                ex_p = min(sl_price, c['Close'])
                pnl_leg = (ex_p - entry_price) / entry_price
                if pos_size == 1.0:
                    exit1_time = c['Datetime']
                    exit1_price = ex_p
                    exit1_pnl_pct = pnl_leg
                    
                    exit2_time = c['Datetime']
                    exit2_price = ex_p
                    exit2_pnl_pct = pnl_leg
                    pos_size = 0.0
                    break
                else: # 50% already exited at TP1
                    exit2_time = c['Datetime']
                    exit2_price = ex_p
                    exit2_pnl_pct = pnl_leg
                    pos_size = 0.0
                    break
                    
            # Check TP1
            if pos_size == 1.0 and c['High'] >= tp1_target:
                exit1_time = c['Datetime']
                exit1_price = tp1_target
                exit1_pnl_pct = (tp1_target - entry_price) / entry_price
                pos_size = 0.5
                hit_tp1 = True
                sl_price = entry_price # Shift SL to Break-Even
                
            # Check TP2
            if pos_size == 0.5 and c['High'] >= tp2_target:
                exit2_time = c['Datetime']
                exit2_price = tp2_target
                exit2_pnl_pct = (tp2_target - entry_price) / entry_price
                pos_size = 0.0
                break
                
        else: # SHORT
            if c['High'] >= sl_price:
                ex_p = max(sl_price, c['Close'])
                pnl_leg = (entry_price - ex_p) / entry_price
                if pos_size == 1.0:
                    exit1_time = c['Datetime']
                    exit1_price = ex_p
                    exit1_pnl_pct = pnl_leg
                    
                    exit2_time = c['Datetime']
                    exit2_price = ex_p
                    exit2_pnl_pct = pnl_leg
                    pos_size = 0.0
                    break
                else:
                    exit2_time = c['Datetime']
                    exit2_price = ex_p
                    exit2_pnl_pct = pnl_leg
                    pos_size = 0.0
                    break
                    
            if pos_size == 1.0 and c['Low'] <= tp1_target:
                exit1_time = c['Datetime']
                exit1_price = tp1_target
                exit1_pnl_pct = (entry_price - tp1_target) / entry_price
                pos_size = 0.5
                hit_tp1 = True
                sl_price = entry_price
                
            if pos_size == 0.5 and c['Low'] <= tp2_target:
                exit2_time = c['Datetime']
                exit2_price = tp2_target
                exit2_pnl_pct = (entry_price - tp2_target) / entry_price
                pos_size = 0.0
                break

    # EOD Exit if position still open at 14:50
    if pos_size > 0:
        c_last = candles[-1]
        eod_p = c_last['Close']
        eod_pnl_leg = (eod_p - entry_price) / entry_price if side == 'LONG' else (entry_price - eod_p) / entry_price
        
        if pos_size == 1.0:
            exit1_time = c_last['Datetime']
            exit1_price = eod_p
            exit1_pnl_pct = eod_pnl_leg
            
            exit2_time = c_last['Datetime']
            exit2_price = eod_p
            exit2_pnl_pct = eod_pnl_leg
        else:
            exit2_time = c_last['Datetime']
            exit2_price = eod_p
            exit2_pnl_pct = eod_pnl_leg

    # Calculate combined gross PnL %
    gross_pnl_pct = 0.5 * exit1_pnl_pct + 0.5 * exit2_pnl_pct
    
    qty = max(1, int(NOTIONAL_PER_TRADE / entry_price))
    if side == 'LONG':
        gross_pnl_amount = (0.5 * (exit1_price - entry_price) + 0.5 * (exit2_price - entry_price)) * qty
    else:
        gross_pnl_amount = (0.5 * (entry_price - exit1_price) + 0.5 * (entry_price - exit2_price)) * qty
        
    fee1 = fyers_trade_cost(entry_price, qty, "buy" if side == 'LONG' else "sell")
    fee2 = fyers_trade_cost(exit1_price, int(qty/2), "sell" if side == 'LONG' else "buy")
    fee3 = fyers_trade_cost(exit2_price, int(qty/2), "sell" if side == 'LONG' else "buy")
    total_trade_fee = fee1 + fee2 + fee3
    
    net_pnl_amount = gross_pnl_amount - total_trade_fee
    net_pnl_pct = (net_pnl_amount / NOTIONAL_PER_TRADE) * 100.0
    
    return {
        'Entry_Time': entry_time,
        'Entry_Price': round(entry_price, 2),
        'Exit1_Time': exit1_time,
        'Exit1_Price': round(exit1_price, 2),
        'Exit1_PnL_Pct': round(exit1_pnl_pct * 100, 2),
        'Exit2_Time': exit2_time,
        'Exit2_Price': round(exit2_price, 2),
        'Exit2_PnL_Pct': round(exit2_pnl_pct * 100, 2),
        'Gross_PnL_Pct': round(gross_pnl_pct * 100, 2),
        'Trade_Fee_Amount': round(total_trade_fee, 2),
        'Net_Trade_PnL_Pct': round(net_pnl_pct, 2),
        'Net_Trade_PnL_Amount': round(net_pnl_amount, 2),
        'Trade_Status': 'WINNER' if net_pnl_amount > 0 else 'LOSER'
    }


def plot_trade_chart(day_df: pd.DataFrame, stock_name: str, trade: dict, range_high: float, range_low: float):
    """Generate high quality plot for trade matching CSV 100%."""
    fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
    
    plot_df = day_df.between_time("09:15", "15:30")
    if plot_df.empty:
        plt.close(fig)
        return
        
    # Plot price line / candles
    ax.plot(plot_df.index, plot_df['Close'], color='#2b5c8f', linewidth=1.5, label='Price (Close)')
    
    # 9:30-9:45 High and Low Lines
    ax.axhline(range_high, color='#e74c3c', linestyle='--', linewidth=1.5, label=f'9:30 High ({range_high:.2f})')
    ax.axhline(range_low, color='#2ecc71', linestyle='--', linewidth=1.5, label=f'9:30 Low ({range_low:.2f})')
    
    # Shade 9:30-9:45 Opening Range Window
    range_start = plot_df.index[0].replace(hour=9, minute=30, second=0)
    range_end = plot_df.index[0].replace(hour=9, minute=45, second=0)
    ax.axvspan(range_start, range_end, color='#f39c12', alpha=0.15, label='9:30-9:45 Range Window')
    
    # Mark ENTRY Point
    entry_t = pd.to_datetime(trade['Entry_Time'])
    entry_p = trade['Entry_Price']
    side = trade['Side']
    marker_side = '^' if side == 'LONG' else 'v'
    color_side = '#27ae60' if side == 'LONG' else '#c0392b'
    
    ax.scatter(entry_t, entry_p, color=color_side, s=120, zorder=5, marker=marker_side)
    ax.annotate(f"ENTRY ({side})\n@{entry_p:.2f}", (entry_t, entry_p),
                textcoords="offset points", xytext=(0, 15 if side == 'LONG' else -25),
                ha='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="#e8f8f5" if side == 'LONG' else "#fadbd8", ec=color_side, lw=1))
                
    # Mark EXIT 1 Point
    ex1_t = pd.to_datetime(trade['Exit1_Time'])
    ex1_p = trade['Exit1_Price']
    ax.scatter(ex1_t, ex1_p, color='#f39c12', s=90, zorder=5, marker='o')
    ax.annotate(f"EXIT 1 (50%)\n@{ex1_p:.2f} ({trade['Exit1_PnL_Pct']:+.2f}%)", (ex1_t, ex1_p),
                textcoords="offset points", xytext=(0, 20),
                ha='center', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="#fef9e7", ec="#f39c12", lw=1))
                
    # Mark EXIT 2 Point
    ex2_t = pd.to_datetime(trade['Exit2_Time'])
    ex2_p = trade['Exit2_Price']
    ax.scatter(ex2_t, ex2_p, color='#8e44ad', s=90, zorder=5, marker='s')
    ax.annotate(f"EXIT 2 (50%)\n@{ex2_p:.2f} ({trade['Exit2_PnL_Pct']:+.2f}%)", (ex2_t, ex2_p),
                textcoords="offset points", xytext=(0, -25),
                ha='center', fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="#f4ecf7", ec="#8e44ad", lw=1))
                
    date_str = trade['Date']
    pnl_str = f"+${trade['Net_Trade_PnL_Amount']:,.2f}" if trade['Net_Trade_PnL_Amount'] >= 0 else f"-${abs(trade['Net_Trade_PnL_Amount']):,.2f}"
    status_color = '#27ae60' if trade['Trade_Status'] == 'WINNER' else '#c0392b'
    
    title_text = f"{stock_name} ({date_str}) | Side: {side} | Net PnL: {pnl_str} ({trade['Net_Trade_PnL_Pct']:+.2f}%) [{trade['Trade_Status']}]"
    ax.set_title(title_text, fontsize=11, fontweight='bold', color=status_color, pad=12)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_xlabel("Time (IST)", fontsize=9, fontweight='bold')
    ax.set_ylabel("Price (INR)", fontsize=9, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    # Save plot file matching name convention P_stockname_entrytime.png
    safe_stock = stock_name.replace(" ", "_").replace(".", "_")
    time_str = entry_t.strftime("%Y-%m-%d_%H-%M")
    filename = f"P_{safe_stock}_{time_str}.png"
    save_path = PLOTS_DIR / filename
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


def main():
    print("==========================================================================")
    print(" GENERATING DETAILED TRADE LOG CSV, DAILY PnL SUMMARY & 15M TRADE PLOTS ")
    print("==========================================================================")
    
    mc_watchlist = parse_stocks_file()
    stock_name_map = {}
    for date_str, smap in mc_watchlist.items():
        for name, ticker in smap.items():
            if ticker:
                stock_name_map[ticker] = name

    csv_files = glob.glob(str(DATA_DIR / "*_1m.csv"))
    print(f"Loaded {len(csv_files)} stock data CSV files.\n")
    
    detailed_trades = []
    
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        ticker = filename.replace("_1m.csv", "").replace("_BO", ".BO").replace("_NS", ".NS").replace("_F", "=F")
        stock_name = stock_name_map.get(ticker, ticker.replace(".NS", "").replace(".BO", ""))
        
        df = load_stock_data(filepath)
        if df.empty: continue
        
        for day_date, day_df in df.groupby(df.index.date):
            date_str = day_date.strftime("%Y-%m-%d")
            ref_df = day_df.between_time("09:30", "09:45")
            if len(ref_df) < 3: continue
            
            range_high = ref_df['High'].max()
            range_low = ref_df['Low'].min()
            if range_high <= range_low: continue
            
            eval_df = resample_ohlc(day_df.between_time("09:46", "14:50"), '15min')
            if eval_df.empty: continue
            candles = eval_df.reset_index().to_dict('records')
            
            # 1st Attempt Only per day per side
            long_done = False
            short_done = False
            
            i = 0
            while i < len(candles):
                c = candles[i]
                
                # 1st Attempt Long at Low
                if not long_done and c['Low'] <= range_low * 1.001 and c['Close'] > range_low:
                    tr = simulate_trade_details(candles, i, 'LONG', range_high, range_low)
                    tr['Date'] = date_str
                    tr['Stock_Name'] = stock_name
                    tr['Ticker'] = ticker
                    tr['Side'] = 'LONG'
                    tr['Range_930_High'] = round(range_high, 2)
                    tr['Range_930_Low'] = round(range_low, 2)
                    
                    detailed_trades.append((tr, day_df, stock_name, range_high, range_low))
                    long_done = True
                    i += 2
                    continue
                    
                # 1st Attempt Short at High
                if not short_done and c['High'] >= range_high * 0.999 and c['Close'] < range_high:
                    tr = simulate_trade_details(candles, i, 'SHORT', range_high, range_low)
                    tr['Date'] = date_str
                    tr['Stock_Name'] = stock_name
                    tr['Ticker'] = ticker
                    tr['Side'] = 'SHORT'
                    tr['Range_930_High'] = round(range_high, 2)
                    tr['Range_930_Low'] = round(range_low, 2)
                    
                    detailed_trades.append((tr, day_df, stock_name, range_high, range_low))
                    short_done = True
                    i += 2
                    continue
                    
                i += 1

    if not detailed_trades:
        print("No detailed trades found.")
        return

    # 1. Create Detailed Trades DataFrame & Export CSV
    trades_list = [item[0] for item in detailed_trades]
    trades_df = pd.DataFrame(trades_list)
    
    # Reorder columns cleanly
    cols_order = [
        'Date', 'Stock_Name', 'Ticker', 'Side', 'Range_930_High', 'Range_930_Low',
        'Entry_Time', 'Entry_Price',
        'Exit1_Time', 'Exit1_Price', 'Exit1_PnL_Pct',
        'Exit2_Time', 'Exit2_Price', 'Exit2_PnL_Pct',
        'Gross_PnL_Pct', 'Trade_Fee_Amount', 'Net_Trade_PnL_Pct', 'Net_Trade_PnL_Amount', 'Trade_Status'
    ]
    trades_df = trades_df[cols_order].sort_values(['Date', 'Entry_Time'])
    
    detailed_csv_path = REPORTS_DIR / "15M_Trades_Detailed.csv"
    trades_df.to_csv(detailed_csv_path, index=False)
    print(f"Saved detailed trade log ({len(trades_df):,} trades) -> {detailed_csv_path}")

    # 2. Create Daily Summary PnL DataFrame & Export CSV
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
            'Date': date_val,
            'Total_Trades': t_trades,
            'Winning_Trades': w_trades,
            'Losing_Trades': l_trades,
            'Win_Rate_Pct': round(win_rate, 1),
            'Gross_PnL_Amount': round(gross_pnl_amt, 2),
            'Total_Charges_Amount': round(total_fees_amt, 2),
            'Net_PnL_Amount': round(net_pnl_amt, 2),
            'Cumulative_Net_PnL_Amount': round(cumulative_pnl, 2)
        })
        
    daily_df = pd.DataFrame(daily_summary).sort_values('Date')
    daily_csv_path = REPORTS_DIR / "15M_Daily_PnL_Summary.csv"
    daily_df.to_csv(daily_csv_path, index=False)
    print(f"Saved daily PnL summary ({len(daily_df)} trading days) -> {daily_csv_path}")

    # 3. Generate Trade Plots in plot/Plot_15M/
    print(f"\nGenerating high-resolution trade plots in {PLOTS_DIR}...")
    plot_count = 0
    # Plot top 100 representative trade charts to ensure fast execution and clean storage
    for item in detailed_trades[:150]:
        tr_dict, day_df, sname, r_h, r_l = item
        plot_trade_chart(day_df, sname, tr_dict, r_h, r_l)
        plot_count += 1
        
    print(f"Successfully generated {plot_count} trade plots in {PLOTS_DIR}!")

if __name__ == "__main__":
    main()
