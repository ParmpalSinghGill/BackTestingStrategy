"""
Find Optimal Stock Selection Filters for Daily Consistency & Profitability After Charges
---------------------------------------------------------------------------------------
Objective: Filter to 2-5 high-probability trades per day that maximize net PnL after charges and maximize profitable trading days.
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from stocks_parser import parse_stocks_file

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


def simulate_trade(candles: list, trigger_idx: int, side: str, range_high: float, range_low: float):
    num_candles = len(candles)
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
    exit1_price, exit1_pnl = None, 0.0
    exit2_price, exit2_pnl = None, 0.0

    for j in range(trigger_idx + 1, num_candles):
        c = candles[j]
        if side == 'LONG':
            if c['Low'] <= sl_price:
                ex_p = min(sl_price, c['Close'])
                pnl = (ex_p - entry_price) / entry_price
                if pos_size == 1.0: exit1_price, exit1_pnl, exit2_price, exit2_pnl = ex_p, pnl, ex_p, pnl
                else: exit2_price, exit2_pnl = ex_p, pnl
                pos_size = 0.0
                break
            if pos_size == 1.0 and c['High'] >= tp1_target:
                exit1_price, exit1_pnl = tp1_target, (tp1_target - entry_price) / entry_price
                pos_size, sl_price = 0.5, entry_price
            if pos_size == 0.5 and c['High'] >= tp2_target:
                exit2_price, exit2_pnl = tp2_target, (tp2_target - entry_price) / entry_price
                pos_size = 0.0
                break
        else:
            if c['High'] >= sl_price:
                ex_p = max(sl_price, c['Close'])
                pnl = (entry_price - ex_p) / entry_price
                if pos_size == 1.0: exit1_price, exit1_pnl, exit2_price, exit2_pnl = ex_p, pnl, ex_p, pnl
                else: exit2_price, exit2_pnl = ex_p, pnl
                pos_size = 0.0
                break
            if pos_size == 1.0 and c['Low'] <= tp1_target:
                exit1_price, exit1_pnl = tp1_target, (entry_price - tp1_target) / entry_price
                pos_size, sl_price = 0.5, entry_price
            if pos_size == 0.5 and c['Low'] <= tp2_target:
                exit2_price, exit2_pnl = tp2_target, (entry_price - tp2_target) / entry_price
                pos_size = 0.0
                break

    if pos_size > 0:
        eod_p = candles[-1]['Close']
        eod_pnl = (eod_p - entry_price) / entry_price if side == 'LONG' else (entry_price - eod_p) / entry_price
        if pos_size == 1.0: exit1_price, exit1_pnl, exit2_price, exit2_pnl = eod_p, eod_pnl, eod_p, eod_pnl
        else: exit2_price, exit2_pnl = eod_p, eod_pnl

    qty = max(1, int(NOTIONAL_PER_TRADE / entry_price))
    gross_pnl_amount = (0.5 * (exit1_price - entry_price) + 0.5 * (exit2_price - entry_price)) * qty if side == 'LONG' else (0.5 * (entry_price - exit1_price) + 0.5 * (entry_price - exit2_price)) * qty
    fees = fyers_trade_cost(entry_price, qty, "buy") + fyers_trade_cost(exit1_price, int(qty/2), "sell") + fyers_trade_cost(exit2_price, int(qty/2), "sell")
    net_pnl_amount = gross_pnl_amount - fees

    return {
        'Entry_Time': entry_time, 'Net_PnL_Amount': net_pnl_amount, 'Fees': fees, 'Is_Winner': net_pnl_amount > 0
    }


def main():
    mc_watchlist = parse_stocks_file()
    mc_focus_days = set()
    stock_name_map = {}
    for date_str, smap in mc_watchlist.items():
        for name, ticker in smap.items():
            if ticker:
                mc_focus_days.add((ticker, date_str))
                stock_name_map[ticker] = name

    csv_files = glob.glob(str(DATA_DIR / "*_1m.csv"))
    
    all_candidates = []

    for filepath in csv_files:
        filename = os.path.basename(filepath)
        ticker = filename.replace("_1m.csv", "").replace("_BO", ".BO").replace("_NS", ".NS").replace("_F", "=F")
        stock_name = stock_name_map.get(ticker, ticker.replace(".NS", "").replace(".BO", ""))
        
        df = load_stock_data(filepath)
        if df.empty: continue
        
        for day_date, day_df in df.groupby(df.index.date):
            date_str = day_date.strftime("%Y-%m-%d")
            is_mc_focus = any((ticker, d) in mc_focus_days for d in [day_date.strftime("%d %B"), day_date.strftime("%d %b"), date_str])
            
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
                    if not long_done and c['Low'] <= range_low * 1.001 and c['Close'] > range_low:
                        wick_rejection = (range_low - c['Low']) / range_low * 100.0
                        res = simulate_trade(candles, i, 'LONG', range_high, range_low)
                        res.update({
                            'Date': date_str, 'Stock_Name': stock_name, 'Ticker': ticker, 'Is_MC_Focus': is_mc_focus,
                            'Range_Width_Pct': range_width_pct, 'Range_Volume': range_volume, 'Wick_Rejection': wick_rejection
                        })
                        all_candidates.append(res)
                        long_done = True
                        
                    if not short_done and c['High'] >= range_high * 0.999 and c['Close'] < range_high:
                        wick_rejection = (c['High'] - range_high) / range_high * 100.0
                        res = simulate_trade(candles, i, 'SHORT', range_high, range_low)
                        res.update({
                            'Date': date_str, 'Stock_Name': stock_name, 'Ticker': ticker, 'Is_MC_Focus': is_mc_focus,
                            'Range_Width_Pct': range_width_pct, 'Range_Volume': range_volume, 'Wick_Rejection': wick_rejection
                        })
                        all_candidates.append(res)
                        short_done = True

    df_cand = pd.DataFrame(all_candidates)
    print("==========================================================================")
    print(f" TOTAL CANDIDATE TRADES EVALUATED: {len(df_cand):,}")
    print("==========================================================================\n")

    # TEST FILTER SETUPS:
    filters = {
        "1. Baseline (No Filter - All MC Focus Trades)": df_cand[df_cand['Is_MC_Focus']],
        
        "2. MC Focus + Range Width Filter (0.4% <= Width <= 1.8%)": 
            df_cand[df_cand['Is_MC_Focus'] & (df_cand['Range_Width_Pct'] >= 0.4) & (df_cand['Range_Width_Pct'] <= 1.8)],
            
        "3. MC Focus + Wick Rejection Filter (Wick >= 0.10%)":
            df_cand[df_cand['Is_MC_Focus'] & (df_cand['Wick_Rejection'] >= 0.10)],
            
        "4. MC Focus + Max Top 2 Trades Per Day (Highest Range Volume)":
            df_cand[df_cand['Is_MC_Focus']].groupby('Date').apply(lambda g: g.nlargest(2, 'Range_Volume')).reset_index(drop=True),
            
        "5. MC Focus + Max Top 3 Trades Per Day (Optimal Range Width 0.4%-1.8%)":
            df_cand[df_cand['Is_MC_Focus'] & (df_cand['Range_Width_Pct'] >= 0.4) & (df_cand['Range_Width_Pct'] <= 1.8)]
                .groupby('Date').apply(lambda g: g.nlargest(3, 'Range_Volume')).reset_index(drop=True),

        "6. General Market Watchlist (Top 3 Volume Stocks Per Day, Width 0.4%-1.8%)":
            df_cand[(df_cand['Range_Width_Pct'] >= 0.4) & (df_cand['Range_Width_Pct'] <= 1.8)]
                .groupby('Date').apply(lambda g: g.nlargest(3, 'Range_Volume')).reset_index(drop=True),
                
        "7. General Market Watchlist (Top 5 Volume Stocks Per Day, Width 0.4%-1.8%)":
            df_cand[(df_cand['Range_Width_Pct'] >= 0.4) & (df_cand['Range_Width_Pct'] <= 1.8)]
                .groupby('Date').apply(lambda g: g.nlargest(5, 'Range_Volume')).reset_index(drop=True)
    }

    for name, f_df in filters.items():
        if f_df.empty: continue
        daily = f_df.groupby('Date')['Net_PnL_Amount'].sum()
        total_days = len(daily)
        win_days = (daily > 0).sum()
        win_day_pct = (win_days / total_days * 100) if total_days > 0 else 0
        net_pnl = daily.sum()
        total_trades = len(f_df)
        avg_trades_per_day = total_trades / total_days if total_days > 0 else 0
        trade_win_rate = (f_df['Is_Winner'].mean() * 100)
        
        print(f"=== {name} ===")
        print(f"  - Total Trades: {total_trades} ({avg_trades_per_day:.1f} trades/day)")
        print(f"  - Trade Win Rate: {trade_win_rate:.1f}%")
        print(f"  - Profitable Trading Days: {win_days} / {total_days} Days ({win_day_pct:.1f}% Win Days)")
        print(f"  - Total Net PnL (After Charges): ${net_pnl:+,.2f}\n")

if __name__ == "__main__":
    main()
