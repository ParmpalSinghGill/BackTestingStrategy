"""
Enforce Max 1 Trade Per Stock Per Day & Calculate ROE (Return on Equity)
-------------------------------------------------------------------------
1. Enforces STRICT MAX 1 TRADE PER STOCK PER DAY.
2. Selects Top 5 Volume Stocks per day with 0.4%-1.8% Range Width.
3. Calculates exact Return on Equity (ROE %) on $1,000 Capital.
4. Updates CSVs & regenerates all trade plot charts in plot/Plot_15M/.
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


def simulate_trade_details(candles: list, trigger_idx: int, side: str, range_high: float, range_low: float):
    signal_candle = candles[trigger_idx]
    entry_time = signal_candle['Datetime']
    
    if side == 'LONG':
        entry_price = range_low
        sl_price = range_low * 0.997
        tp1_target = min(entry_price * 1.010, range_high)
        tp2_target = entry_price * 1.020
    else:
        entry_price = range_high
        sl_price = range_high * 1.003
        tp1_target = max(entry_price * 0.990, range_low)
        tp2_target = entry_price * 0.980

    pos_size = 1.0
    exit1_time, exit1_price, exit1_pnl_pct = None, None, 0.0
    exit2_time, exit2_price, exit2_pnl_pct = None, None, 0.0

    for j in range(trigger_idx + 1, len(candles)):
        c = candles[j]
        if side == 'LONG':
            if c['Low'] <= sl_price:
                ex_p = min(sl_price, c['Close'])
                pnl = (ex_p - entry_price) / entry_price
                if pos_size == 1.0:
                    exit1_time, exit1_price, exit1_pnl_pct = c['Datetime'], ex_p, pnl
                    exit2_time, exit2_price, exit2_pnl_pct = c['Datetime'], ex_p, pnl
                else:
                    exit2_time, exit2_price, exit2_pnl_pct = c['Datetime'], ex_p, pnl
                pos_size = 0.0
                break
            if pos_size == 1.0 and c['High'] >= tp1_target:
                exit1_time, exit1_price = c['Datetime'], tp1_target
                exit1_pnl_pct = (tp1_target - entry_price) / entry_price
                pos_size, sl_price = 0.5, entry_price
            if pos_size == 0.5 and c['High'] >= tp2_target:
                exit2_time, exit2_price = c['Datetime'], tp2_target
                exit2_pnl_pct = (tp2_target - entry_price) / entry_price
                pos_size = 0.0
                break
        else:
            if c['High'] >= sl_price:
                ex_p = max(sl_price, c['Close'])
                pnl = (entry_price - ex_p) / entry_price
                if pos_size == 1.0:
                    exit1_time, exit1_price, exit1_pnl_pct = c['Datetime'], ex_p, pnl
                    exit2_time, exit2_price, exit2_pnl_pct = c['Datetime'], ex_p, pnl
                else:
                    exit2_time, exit2_price, exit2_pnl_pct = c['Datetime'], ex_p, pnl
                pos_size = 0.0
                break
            if pos_size == 1.0 and c['Low'] <= tp1_target:
                exit1_time, exit1_price = c['Datetime'], tp1_target
                exit1_pnl_pct = (entry_price - tp1_target) / entry_price
                pos_size, sl_price = 0.5, entry_price
            if pos_size == 0.5 and c['Low'] <= tp2_target:
                exit2_time, exit2_price = c['Datetime'], tp2_target
                exit2_pnl_pct = (entry_price - tp2_target) / entry_price
                pos_size = 0.0
                break

    if pos_size > 0:
        c_last = candles[-1]
        eod_p = c_last['Close']
        eod_pnl = (eod_p - entry_price) / entry_price if side == 'LONG' else (entry_price - eod_p) / entry_price
        if pos_size == 1.0:
            exit1_time, exit1_price, exit1_pnl_pct = c_last['Datetime'], eod_p, eod_pnl
            exit2_time, exit2_price, exit2_pnl_pct = c_last['Datetime'], eod_p, eod_pnl
        else:
            exit2_time, exit2_price, exit2_pnl_pct = c_last['Datetime'], eod_p, eod_pnl

    gross_pnl_pct = 0.5 * exit1_pnl_pct + 0.5 * exit2_pnl_pct
    qty = max(1, int(NOTIONAL_PER_TRADE / entry_price))
    if side == 'LONG':
        gross_pnl_amount = (0.5 * (exit1_price - entry_price) + 0.5 * (exit2_price - entry_price)) * qty
    else:
        gross_pnl_amount = (0.5 * (entry_price - exit1_price) + 0.5 * (entry_price - exit2_price)) * qty
        
    fees = fyers_trade_cost(entry_price, qty, "buy" if side == 'LONG' else "sell") + fyers_trade_cost(exit1_price, int(qty/2), "sell" if side == 'LONG' else "buy") + fyers_trade_cost(exit2_price, int(qty/2), "sell" if side == 'LONG' else "buy")
    net_pnl_amount = gross_pnl_amount - fees
    net_pnl_pct = (net_pnl_amount / NOTIONAL_PER_TRADE) * 100.0

    return {
        'Entry_Time': entry_time, 'Entry_Price': round(entry_price, 2),
        'Exit1_Time': exit1_time, 'Exit1_Price': round(exit1_price, 2), 'Exit1_PnL_Pct': round(exit1_pnl_pct * 100, 2),
        'Exit2_Time': exit2_time, 'Exit2_Price': round(exit2_price, 2), 'Exit2_PnL_Pct': round(exit2_pnl_pct * 100, 2),
        'Gross_PnL_Pct': round(gross_pnl_pct * 100, 2), 'Trade_Fee_Amount': round(fees, 2),
        'Net_Trade_PnL_Pct': round(net_pnl_pct, 2), 'Net_Trade_PnL_Amount': round(net_pnl_amount, 2),
        'Trade_Status': 'WINNER' if net_pnl_amount > 0 else 'LOSER'
    }


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
    fig, ax = plt.subplots(figsize=(13.5, 7), dpi=100)
    
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
    
    entry_t = pd.to_datetime(trade['Entry_Time'])
    entry_p = trade['Entry_Price']
    side = trade['Side']
    marker_side = '^' if side == 'LONG' else 'v'
    color_side = '#27ae60' if side == 'LONG' else '#c0392b'
    entry_time_str = entry_t.strftime("%H:%M")
    
    ax.scatter(entry_t, entry_p, color=color_side, s=150, zorder=6, marker=marker_side)
    ax.annotate(f"ENTRY ({side})\nTime: {entry_time_str}\nPrice: @{entry_p:.2f}", (entry_t, entry_p),
                textcoords="offset points", xytext=(0, 22 if side == 'LONG' else -35),
                ha='center', fontsize=8.5, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="#e8f8f5" if side == 'LONG' else "#fadbd8", ec=color_side, lw=1.2))
                
    ex1_t = pd.to_datetime(trade['Exit1_Time'])
    ex1_p = trade['Exit1_Price']
    ex1_time_str = ex1_t.strftime("%H:%M")
    
    ax.scatter(ex1_t, ex1_p, color='#f39c12', s=110, zorder=6, marker='o')
    ax.annotate(f"EXIT 1 (50%)\nTime: {ex1_time_str}\nPrice: @{ex1_p:.2f} ({trade['Exit1_PnL_Pct']:+.2f}%)", (ex1_t, ex1_p),
                textcoords="offset points", xytext=(0, 25),
                ha='center', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="#fef9e7", ec="#f39c12", lw=1.2))
                
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
    
    title_text = f"[TOP 5 STRATEGY] {stock_name} ({date_str}) | Side: {side} | Net PnL: {pnl_str} ({trade['Net_Trade_PnL_Pct']:+.2f}%) [{trade['Trade_Status']}]"
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


def main():
    csv_files = glob.glob(str(DATA_DIR / "*_1m.csv"))
    all_candidates = []

    for filepath in csv_files:
        filename = os.path.basename(filepath)
        ticker = filename.replace("_1m.csv", "").replace("_BO", ".BO").replace("_NS", ".NS").replace("_F", "=F")
        stock_name = ticker.replace(".NS", "").replace(".BO", "")
        
        df = load_stock_data(filepath)
        if df.empty: continue
        
        for day_date, day_df in df.groupby(df.index.date):
            date_str = day_date.strftime("%Y-%m-%d")
            ref_df = day_df.between_time("09:30", "09:45")
            if len(ref_df) < 3: continue
            
            range_high = ref_df['High'].max()
            range_low = ref_df['Low'].min()
            range_volume = ref_df['Volume'].sum()
            if range_high <= range_low: continue
            
            range_width_pct = ((range_high - range_low) / range_low) * 100.0
            
            eval_df = resample_ohlc(day_df.between_time("09:46", "15:10"), '15min')
            if eval_df.empty: continue
            candles = eval_df.reset_index().to_dict('records')
            
            # STRICT RULE: MAX 1 TRADE PER STOCK PER DAY
            trade_taken = False
            for i, c in enumerate(candles):
                c_time_str = c['Datetime'].strftime("%H:%M")
                
                if "10:00" <= c_time_str <= "14:45" and not trade_taken:
                    # Check Long
                    if c['Low'] <= range_low * 1.001 and c['Close'] > range_low and c['Close'] >= c['Open']:
                        res = simulate_trade_details(candles, i, 'LONG', range_high, range_low)
                        res.update({'Date': date_str, 'Stock_Name': stock_name, 'Ticker': ticker, 'Side': 'LONG', 'Range_930_High': round(range_high, 2), 'Range_930_Low': round(range_low, 2), 'Range_Width_Pct': range_width_pct, 'Range_Volume': range_volume})
                        all_candidates.append((res, day_df, stock_name, range_high, range_low))
                        trade_taken = True
                        break
                        
                    # Check Short
                    if c['High'] >= range_high * 0.999 and c['Close'] < range_high and c['Close'] <= c['Open']:
                        res = simulate_trade_details(candles, i, 'SHORT', range_high, range_low)
                        res.update({'Date': date_str, 'Stock_Name': stock_name, 'Ticker': ticker, 'Side': 'SHORT', 'Range_930_High': round(range_high, 2), 'Range_930_Low': round(range_low, 2), 'Range_Width_Pct': range_width_pct, 'Range_Volume': range_volume})
                        all_candidates.append((res, day_df, stock_name, range_high, range_low))
                        trade_taken = True
                        break

    cand_df = pd.DataFrame([item[0] for item in all_candidates])
    cand_df['Item_Idx'] = cand_df.index
    
    # Filter Range Width 0.4% to 1.8%
    df_filtered = cand_df[(cand_df['Range_Width_Pct'] >= 0.4) & (cand_df['Range_Width_Pct'] <= 1.8)]
    
    # Select Top 5 Volume Stocks Per Day
    top5_df = df_filtered.groupby('Date', group_keys=False).apply(lambda g: g.nlargest(5, 'Range_Volume')).reset_index(drop=True)
    
    cols_order = [
        'Date', 'Stock_Name', 'Ticker', 'Side', 'Range_930_High', 'Range_930_Low',
        'Entry_Time', 'Entry_Price',
        'Exit1_Time', 'Exit1_Price', 'Exit1_PnL_Pct',
        'Exit2_Time', 'Exit2_Price', 'Exit2_PnL_Pct',
        'Gross_PnL_Pct', 'Trade_Fee_Amount', 'Net_Trade_PnL_Pct', 'Net_Trade_PnL_Amount', 'Trade_Status'
    ]
    
    selected_indices = set(top5_df['Item_Idx'].tolist())
    top5_df_clean = top5_df[cols_order].sort_values(['Date', 'Entry_Time'])
    
    detailed_csv = REPORTS_DIR / "Optimal_Top5_Trades_Detailed.csv"
    top5_df_clean.to_csv(detailed_csv, index=False)
    print(f"Saved Strict 1 Trade Per Stock Day Trade Log ({len(top5_df_clean)} trades) -> {detailed_csv}")

    # Daily PnL Summary & ROE Calculation
    daily_summary = []
    cumulative_pnl = 0.0
    for date_val, group in top5_df_clean.groupby('Date'):
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
    print(f"Saved Strict Daily PnL Summary ({len(daily_df)} days) -> {daily_csv}")

    # Calculate ROE Metrics
    total_net_profit = daily_df['Net_PnL_Amount'].sum()
    ending_capital = STARTING_CAPITAL + total_net_profit
    roe_pct = (total_net_profit / STARTING_CAPITAL) * 100.0
    annualized_roe_pct = roe_pct * 12.0 # 12 months extrapolation

    print("\n==========================================================================")
    print("      FINAL STRATEGY RESULTS (MAX 1 TRADE / STOCK / DAY) & ROE SUMMARY     ")
    print("==========================================================================")
    print(f"Initial Starting Capital (Equity): ${STARTING_CAPITAL:,.2f}")
    print(f"Ending Capital (Equity):           ${ending_capital:,.2f}")
    print(f"TOTAL NET PROFIT (After Fees):     ${total_net_profit:+,.2f}")
    print(f"MONTHLY RETURN ON EQUITY (ROE):     +{roe_pct:.2f}% Net Return in 1 Month")
    print(f"ANNUALIZED RETURN ON EQUITY (ROE):  +{annualized_roe_pct:.2f}% p.a.")
    print(f"Total Trades Taken:                {len(top5_df_clean):,} (Exactly {len(top5_df_clean)/len(daily_df):.1f} trades/day)")
    print(f"Trade Win Rate:                    {top5_df_clean['Trade_Status'].eq('WINNER').mean()*100:.1f}%")
    print(f"Profitable Trading Days:           {daily_df['Net_PnL_Amount'].gt(0).sum()} / {len(daily_df)} Days ({(daily_df['Net_PnL_Amount'].gt(0).sum()/len(daily_df))*100:.1f}% Win Days)")
    print("==========================================================================")

    # Re-generate plots
    for f in glob.glob(str(PLOTS_DIR / "*.png")):
        try: os.remove(f)
        except Exception: pass
        
    plot_count = 0
    for idx in selected_indices:
        tr_dict, day_df, sname, r_h, r_l = all_candidates[idx]
        plot_candlestick_trade_chart(day_df, sname, tr_dict, r_h, r_l)
        plot_count += 1
        
    print(f"Successfully generated {plot_count} trade plots in {PLOTS_DIR}!")

if __name__ == "__main__":
    main()
