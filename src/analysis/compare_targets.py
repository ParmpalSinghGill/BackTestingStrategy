"""
Compare Target Preset A (0.8% & 1.5%) vs Target Preset B (1.0% & 2.0%)
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "Reports"

START_CAPITAL = 1000.0
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
    return df.dropna(subset=['Open', 'High', 'Low', 'Close'])


def simulate_trade_with_targets(candles: list, trigger_idx: int, side: str, range_high: float, range_low: float, tp1_pct: float, tp2_pct: float):
    num_candles = len(candles)
    signal_candle = candles[trigger_idx]
    
    if side == 'LONG':
        entry_price = range_low
        sl_price = range_low * 0.997
        tp1_target = min(entry_price * (1.0 + tp1_pct), range_high)
        tp2_target = entry_price * (1.0 + tp2_pct)
    else:
        entry_price = range_high
        sl_price = range_high * 1.003
        tp1_target = max(entry_price * (1.0 - tp1_pct), range_low)
        tp2_target = entry_price * (1.0 - tp2_pct)

    pos_size = 1.0
    exit1_price, exit1_pnl = None, 0.0
    exit2_price, exit2_pnl = None, 0.0
    hit_tp1, hit_tp2 = False, False

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
                pos_size, hit_tp1, sl_price = 0.5, True, entry_price
            if pos_size == 0.5 and c['High'] >= tp2_target:
                exit2_price, exit2_pnl = tp2_target, (tp2_target - entry_price) / entry_price
                pos_size, hit_tp2 = 0.0, True
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
                pos_size, hit_tp1, sl_price = 0.5, True, entry_price
            if pos_size == 0.5 and c['Low'] <= tp2_target:
                exit2_price, exit2_pnl = tp2_target, (entry_price - tp2_target) / entry_price
                pos_size, hit_tp2 = 0.0, True
                break

    if pos_size > 0:
        eod_p = candles[-1]['Close']
        eod_pnl = (eod_p - entry_price) / entry_price if side == 'LONG' else (entry_price - eod_p) / entry_price
        if pos_size == 1.0: exit1_price, exit1_pnl, exit2_price, exit2_pnl = eod_p, eod_pnl, eod_p, eod_pnl
        else: exit2_price, exit2_pnl = eod_p, eod_pnl

    gross_pnl_pct = 0.5 * exit1_pnl + 0.5 * exit2_pnl
    qty = max(1, int(NOTIONAL_PER_TRADE / entry_price))
    gross_pnl_amount = (0.5 * (exit1_price - entry_price) + 0.5 * (exit2_price - entry_price)) * qty if side == 'LONG' else (0.5 * (entry_price - exit1_price) + 0.5 * (entry_price - exit2_price)) * qty
    fees = fyers_trade_cost(entry_price, qty, "buy") + fyers_trade_cost(exit1_price, int(qty/2), "sell") + fyers_trade_cost(exit2_price, int(qty/2), "sell")
    net_pnl_amount = gross_pnl_amount - fees
    net_pnl_pct = (net_pnl_amount / NOTIONAL_PER_TRADE) * 100.0

    return {
        'Hit_TP1': hit_tp1, 'Hit_TP2': hit_tp2, 'Net_Trade_PnL_Pct': net_pnl_pct, 'Net_Trade_PnL_Amount': net_pnl_amount, 'Is_Winner': net_pnl_amount > 0
    }


def main():
    csv_files = glob.glob(str(DATA_DIR / "*_1m.csv"))
    trades_A = []
    trades_B = []

    for filepath in csv_files:
        df = load_stock_data(filepath)
        if df.empty: continue
        
        for day_date, day_df in df.groupby(df.index.date):
            ref_df = day_df.between_time("09:30", "09:45")
            if len(ref_df) < 3: continue
            r_h, r_l = ref_df['High'].max(), ref_df['Low'].min()
            if r_h <= r_l: continue

            eval_df = resample_ohlc(day_df.between_time("09:50", "14:50"), '15min')
            if eval_df.empty: continue
            candles = eval_df.reset_index().to_dict('records')

            long_done, short_done = False, False
            for i, c in enumerate(candles):
                if not long_done and c['Low'] <= r_l * 1.001 and c['Close'] > r_l:
                    trades_A.append(simulate_trade_with_targets(candles, i, 'LONG', r_h, r_l, tp1_pct=0.008, tp2_pct=0.015))
                    trades_B.append(simulate_trade_with_targets(candles, i, 'LONG', r_h, r_l, tp1_pct=0.010, tp2_pct=0.020))
                    long_done = True
                if not short_done and c['High'] >= r_h * 0.999 and c['Close'] < r_h:
                    trades_A.append(simulate_trade_with_targets(candles, i, 'SHORT', r_h, r_l, tp1_pct=0.008, tp2_pct=0.015))
                    trades_B.append(simulate_trade_with_targets(candles, i, 'SHORT', r_h, r_l, tp1_pct=0.010, tp2_pct=0.020))
                    short_done = True

    dfA = pd.DataFrame(trades_A)
    dfB = pd.DataFrame(trades_B)

    print("==========================================================================")
    print("      TARGET COMPARISON: (0.8% & 1.5%) vs (1.0% & 2.0%) ON 15M 1ST ATTEMPT ")
    print("==========================================================================")
    print(f"PRESET A (0.8% & 1.5% Targets):")
    print(f"  - Total Trades: {len(dfA):,}")
    print(f"  - Win Rate: {dfA['Is_Winner'].mean()*100:.1f}%")
    print(f"  - Hit TP1 Reach %: {dfA['Hit_TP1'].mean()*100:.1f}%")
    print(f"  - Hit TP2 Reach %: {dfA['Hit_TP2'].mean()*100:.1f}%")
    print(f"  - Avg Net PnL per Trade: {dfA['Net_Trade_PnL_Pct'].mean():+.2f}%")
    print(f"  - Cumulative Net PnL ($): ${dfA['Net_Trade_PnL_Amount'].sum():+,.2f}\n")

    print(f"PRESET B (1.0% & 2.0% Targets):")
    print(f"  - Total Trades: {len(dfB):,}")
    print(f"  - Win Rate: {dfB['Is_Winner'].mean()*100:.1f}%")
    print(f"  - Hit TP1 Reach %: {dfB['Hit_TP1'].mean()*100:.1f}%")
    print(f"  - Hit TP2 Reach %: {dfB['Hit_TP2'].mean()*100:.1f}%")
    print(f"  - Avg Net PnL per Trade: {dfB['Net_Trade_PnL_Pct'].mean():+.2f}%")
    print(f"  - Cumulative Net PnL ($): ${dfB['Net_Trade_PnL_Amount'].sum():+,.2f}")
    print("==========================================================================")

if __name__ == "__main__":
    main()
