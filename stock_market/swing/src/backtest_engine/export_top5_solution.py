"""
Export Complete CSV Reports for the Optimal Top 5 Volume + Range Width Filter Strategy
---------------------------------------------------------------------------------------
Exports:
1. Reports/Optimal_Top5_Trades_Detailed.csv
2. Reports/Optimal_Top5_Daily_PnL_Summary.csv
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "Reports"

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
    if 'Volume' in df.columns:
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    return df.dropna(subset=['Open', 'High', 'Low', 'Close'])


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
            
            eval_df = resample_ohlc(day_df.between_time("09:50", "15:10"), '15min')
            if eval_df.empty: continue
            candles = eval_df.reset_index().to_dict('records')
            
            long_done, short_done = False, False
            for i, c in enumerate(candles):
                c_time_str = c['Datetime'].strftime("%H:%M")
                if c_time_str <= "14:45":
                    if not long_done and c['Low'] <= range_low * 1.001 and c['Close'] > range_low and c['Close'] >= c['Open']:
                        res = simulate_trade_details(candles, i, 'LONG', range_high, range_low)
                        res.update({'Date': date_str, 'Stock_Name': stock_name, 'Ticker': ticker, 'Side': 'LONG', 'Range_930_High': round(range_high, 2), 'Range_930_Low': round(range_low, 2), 'Range_Width_Pct': range_width_pct, 'Range_Volume': range_volume})
                        all_candidates.append(res)
                        long_done = True
                        
                    if not short_done and c['High'] >= range_high * 0.999 and c['Close'] < range_high and c['Close'] <= c['Open']:
                        res = simulate_trade_details(candles, i, 'SHORT', range_high, range_low)
                        res.update({'Date': date_str, 'Stock_Name': stock_name, 'Ticker': ticker, 'Side': 'SHORT', 'Range_930_High': round(range_high, 2), 'Range_930_Low': round(range_low, 2), 'Range_Width_Pct': range_width_pct, 'Range_Volume': range_volume})
                        all_candidates.append(res)
                        short_done = True

    df_cand = pd.DataFrame(all_candidates)
    df_filtered = df_cand[(df_cand['Range_Width_Pct'] >= 0.4) & (df_cand['Range_Width_Pct'] <= 1.8)]
    
    # Pick Top 5 Highest Volume Stocks Per Day
    top5_df = df_filtered.groupby('Date', group_keys=False).apply(lambda g: g.nlargest(5, 'Range_Volume')).reset_index(drop=True)
    
    cols_order = [
        'Date', 'Stock_Name', 'Ticker', 'Side', 'Range_930_High', 'Range_930_Low',
        'Entry_Time', 'Entry_Price',
        'Exit1_Time', 'Exit1_Price', 'Exit1_PnL_Pct',
        'Exit2_Time', 'Exit2_Price', 'Exit2_PnL_Pct',
        'Gross_PnL_Pct', 'Trade_Fee_Amount', 'Net_Trade_PnL_Pct', 'Net_Trade_PnL_Amount', 'Trade_Status'
    ]
    top5_df = top5_df[cols_order].sort_values(['Date', 'Entry_Time'])
    
    detailed_csv = REPORTS_DIR / "Optimal_Top5_Trades_Detailed.csv"
    top5_df.to_csv(detailed_csv, index=False)
    print(f"Saved Optimal Top 5 Detailed Trade Log ({len(top5_df)} trades) -> {detailed_csv}")

    # Daily PnL Summary
    daily_summary = []
    cumulative_pnl = 0.0
    
    for date_val, group in top5_df.groupby('Date'):
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
    print(f"Saved Optimal Top 5 Daily PnL Summary ({len(daily_df)} days) -> {daily_csv}")

if __name__ == "__main__":
    main()
