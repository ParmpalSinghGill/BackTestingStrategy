"""
Optimize Real-Time Zero Look-Ahead Engine (No Look-Ahead Bias)
----------------------------------------------------------------
Evaluates:
1. First-Wick Touch with 0.5% SL Buffer
2. First-Wick Touch with 0.8% SL Buffer
3. 15M Confirmation Market Order at Candle Close Price
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

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


def run_test(sl_pct: float, entry_mode: str = 'FIRST_WICK'):
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
        
        if entry_mode == 'FIRST_WICK':
            time_range = pd.date_range(f"{date_str} 09:50:00", f"{date_str} 14:45:00", freq='1min')
            for t in time_range:
                for sname in top5_names:
                    if sname in traded_stocks_today: continue
                    sinfo = daily_stocks[sname]
                    m_bar = sinfo['DF'][sinfo['DF'].index == t]
                    if m_bar.empty: continue
                    m_row = m_bar.iloc[0]
                    r_h, r_l = top5_dict[sname]['Range_High'], top5_dict[sname]['Range_Low']
                    
                    side = None
                    if m_row['Low'] <= r_l * 1.001: side = 'LONG'
                    elif m_row['High'] >= r_h * 0.999: side = 'SHORT'
                    
                    if side:
                        entry_p = r_l if side == 'LONG' else r_h
                        sl_p = entry_p * (1.0 - sl_pct) if side == 'LONG' else entry_p * (1.0 + sl_pct)
                        tp1_p = min(entry_p * 1.010, r_h) if side == 'LONG' else max(entry_p * 0.990, r_l)
                        tp2_p = entry_p * 1.020 if side == 'LONG' else entry_p * 0.980
                        
                        df_rest = sinfo['DF'][sinfo['DF'].index > t]
                        m_records = df_rest.reset_index().to_dict('records')
                        
                        pos_size = 1.0
                        ex1_p, ex1_pnl, ex2_p, ex2_pnl = None, 0.0, None, 0.0
                        for sub_m in m_records:
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
                            last_m = m_records[-1] if m_records else m_row
                            eod_v = last_m['Close']
                            eod_pnl_v = (eod_v - entry_p) / entry_p if side == 'LONG' else (entry_p - eod_v) / entry_p
                            if pos_size == 1.0: ex1_p, ex1_pnl, ex2_p, ex2_pnl = eod_v, eod_pnl_v, eod_v, eod_pnl_v
                            else: ex2_p, ex2_pnl = eod_v, eod_pnl_v

                        qty = max(1, int(NOTIONAL_PER_TRADE / entry_p))
                        gross_pnl_amt = (0.5 * (ex1_p - entry_p) + 0.5 * (ex2_p - entry_p)) * qty if side == 'LONG' else (0.5 * (entry_p - ex1_p) + 0.5 * (entry_p - ex2_p)) * qty
                        fees = fyers_trade_cost(entry_p, qty, "buy" if side == 'LONG' else "sell") + fyers_trade_cost(ex1_p, int(qty/2), "sell" if side == 'LONG' else "buy") + fyers_trade_cost(ex2_p, int(qty/2), "sell" if side == 'LONG' else "buy")
                        net_pnl_amt = gross_pnl_amt - fees
                        executed_trades.append({'Date': date_str, 'Net_PnL': net_pnl_amt, 'Win': net_pnl_amt > 0})
                        traded_stocks_today.add(sname)

    df_res = pd.DataFrame(executed_trades)
    daily = df_res.groupby('Date')['Net_PnL'].sum()
    print(f"=== MODE: {entry_mode} | SL Buffer: {sl_pct*100:.1f}% ===")
    print(f"  - Total Trades: {len(df_res)}")
    print(f"  - Win Rate: {df_res['Win'].mean()*100:.1f}%")
    print(f"  - Win Days: {(daily > 0).sum()} / {len(daily)} Days ({((daily > 0).sum()/len(daily))*100:.1f}%)")
    print(f"  - Net PnL (After Fees): ${df_res['Net_PnL'].sum():+,.2f}\n")


def main():
    print("==========================================================================")
    print("      REAL-TIME NO-LOOKAHEAD OPTIMIZATION TEST                            ")
    print("==========================================================================")
    run_test(0.003, 'FIRST_WICK') # 0.3% SL
    run_test(0.005, 'FIRST_WICK') # 0.5% SL
    run_test(0.008, 'FIRST_WICK') # 0.8% SL
    run_test(0.010, 'FIRST_WICK') # 1.0% SL

if __name__ == "__main__":
    main()
