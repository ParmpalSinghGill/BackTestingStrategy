"""
Incremental 1-Minute Live Simulation Backtest Engine
---------------------------------------------------
Simulates live trading by feeding 1-minute data incrementally minute-by-minute.
At minute 't', the algorithm has ZERO knowledge of minutes > 't'.
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "Reports"

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


def main():
    csv_files = glob.glob(str(DATA_DIR / "*_1m.csv"))
    print(f"Loaded {len(csv_files)} stock data files for Incremental 1-Minute Backtest.\n")
    
    # Load all stock data into memory dictionary
    stocks_data = {}
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        ticker = filename.replace("_1m.csv", "").replace("_BO", ".BO").replace("_NS", ".NS").replace("_F", "=F")
        stock_name = ticker.replace(".NS", "").replace(".BO", "")
        df = load_stock_data(filepath)
        if not df.empty:
            stocks_data[stock_name] = {'Ticker': ticker, 'DF': df}

    # Extract unique trading dates
    all_dates = sorted(list(set(d for s in stocks_data.values() for d in s['DF'].index.date)))
    
    executed_trades = []

    # LOOP DAY BY DAY
    for current_date in all_dates:
        date_str = current_date.strftime("%Y-%m-%d")
        
        # Incremental state for the day
        daily_stocks = {}
        for sname, sdata in stocks_data.items():
            day_df = sdata['DF'][sdata['DF'].index.date == current_date]
            if len(day_df) >= 30: # Ensure full day data available
                daily_stocks[sname] = {'Ticker': sdata['Ticker'], 'DF': day_df}
                
        if not daily_stocks: continue
        
        # 1. AT 09:45 AM: Calculate Range High, Low & Traded Volume
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
        
        # Rank by Volume and Pick Top 5
        top5_today = pd.DataFrame(range_stats).nlargest(5, 'Range_Volume').to_dict('records')
        top5_names = set(item['Stock_Name'] for item in top5_today)
        top5_dict = {item['Stock_Name']: item for item in top5_today}
        
        # Track active/completed trades per stock today (MAX 1 TRADE PER STOCK)
        traded_stocks_today = set()

        # INCREMENTAL 1-MINUTE LOOP FROM 09:50 AM TO 15:30 PM
        time_range = pd.date_range(f"{date_str} 09:50:00", f"{date_str} 15:30:00", freq='1min')
        
        for t in time_range:
            t_str = t.strftime("%H:%M")
            
            # Check 15-minute boundary for trade signals (10:00, 10:15, etc.)
            if t_str in ["10:00", "10:15", "10:30", "10:45", "11:00", "11:15", "11:30", "11:45",
                          "12:00", "12:15", "12:30", "12:45", "13:00", "13:15", "13:30", "13:45",
                          "14:00", "14:15", "14:30", "14:45"]:
                          
                for sname in top5_names:
                    if sname in traded_stocks_today: continue
                    
                    sinfo = daily_stocks[sname]
                    df_up_to_t = sinfo['DF'][sinfo['DF'].index < t] # Incremental Slice: ZERO knowledge of future minutes > t
                    c_15m_start = t - pd.Timedelta(minutes=15)
                    c_15m_df = df_up_to_t[df_up_to_t.index >= c_15m_start]
                    
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
                            # SIMULATE POSITION MANAGEMENT FROM CURRENT MINUTE t
                            entry_p = r_l if side == 'LONG' else r_h
                            entry_t = t
                            sl_p = entry_p * 0.997 if side == 'LONG' else entry_p * 1.003
                            tp1_p = min(entry_p * 1.010, r_h) if side == 'LONG' else max(entry_p * 0.990, r_l)
                            tp2_p = entry_p * 1.020 if side == 'LONG' else entry_p * 0.980
                            
                            df_rest = sinfo['DF'][sinfo['DF'].index >= t]
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
                            
                            executed_trades.append({
                                'Date': date_str, 'Stock_Name': sname, 'Side': side,
                                'Entry_Time': entry_t, 'Net_Trade_PnL_Amount': net_pnl_amt,
                                'Trade_Status': 'WINNER' if net_pnl_amt > 0 else 'LOSER'
                            })
                            traded_stocks_today.add(sname)

    df_inc = pd.DataFrame(executed_trades)
    daily = df_inc.groupby('Date')['Net_Trade_PnL_Amount'].sum()
    total_net = df_inc['Net_Trade_PnL_Amount'].sum()
    roe_pct = (total_net / STARTING_CAPITAL) * 100.0

    print("==========================================================================")
    print("      INCREMENTAL 1-MINUTE MINUTE-BY-MINUTE LIVE BACKTEST RESULTS          ")
    print("==========================================================================")
    print(f"Total Trades Taken:            {len(df_inc):,} (Exactly {len(df_inc)/len(daily):.1f} trades/day)")
    print(f"Trade Win Rate:                {df_inc['Trade_Status'].eq('WINNER').mean()*100:.1f}%")
    print(f"Profitable Trading Days:       {(daily > 0).sum()} / {len(daily)} Days ({((daily > 0).sum()/len(daily))*100:.1f}% Win Days)")
    print(f"Total Net Profit (After Fees): ${total_net:+,.2f}")
    print(f"MONTHLY RETURN ON EQUITY (ROE): +{roe_pct:.2f}% Net Return in 1 Month")
    print("==========================================================================")

if __name__ == "__main__":
    main()
