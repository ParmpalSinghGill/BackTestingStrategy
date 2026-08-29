"""
Timeframe Performance Comparison Module (Yearly vs Monthly vs Weekly)

Compares Support Sweep Strategy performance across timeframes:
- Yearly (1Y) Support Sweeps
- Monthly (1M) Support Sweeps
- Weekly (1W) Support Sweeps
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = BASE_DIR / "Reports"
MASTER_TRADES_CSV = REPORTS_DIR / "Support_Liquidity_Strategy_Trades.csv"

def run_timeframe_comparison(starting_capital: float = 100000.0) -> pd.DataFrame:
    df_master = pd.read_csv(MASTER_TRADES_CSV)
    df_sc1_m3 = df_master[(df_master['Scenario'] == 'Scenario 1 (Green & Close > C1 High)') & (df_master['Entry_Mode'] == 'Mode B (C1 High + 0.1%)')].copy()

    tfs = ['Yearly', 'Monthly', 'Weekly']
    res_list = []

    for tf in tfs:
        # Unconstrained overall possible signals for this timeframe
        df_tf_possible = df_sc1_m3[df_sc1_m3['Liquidity_Type'] == tf].copy()
        tot_possible = len(df_tf_possible)
        possible_wins = (df_tf_possible['Outcome'] == 'Success').sum()
        overall_win_rate = (possible_wins / tot_possible * 100.0) if tot_possible > 0 else 0.0

        # Portfolio simulation for trades originating ONLY from this timeframe
        df_tf_possible = df_tf_possible.sort_values('C2_Date').reset_index(drop=True)
        trades_by_date = {}
        for idx, row in df_tf_possible.iterrows():
            trades_by_date.setdefault(pd.to_datetime(row['C2_Date']), []).append(row.to_dict())

        if df_tf_possible.empty:
            continue

        min_dt = pd.to_datetime(df_tf_possible['C2_Date'].min())
        max_dt = max(pd.to_datetime(df_tf_possible['C2_Date'].max()), pd.to_datetime(df_tf_possible['Exit_Date'].max()))
        all_days = pd.date_range(min_dt, max_dt, freq='D')

        equity = starting_capital
        peak_eq = starting_capital
        max_dd = 0.0
        open_trades = []
        accepted = []

        for curr_dt in all_days:
            closed = []
            for i, ot in enumerate(open_trades):
                if ot['exit_date'] <= curr_dt:
                    equity += ot['trade']['Net_PnL']
                    closed.append(i)
            for i in sorted(closed, reverse=True):
                open_trades.pop(i)

            if equity > peak_eq: peak_eq = equity
            dd = ((peak_eq - equity) / peak_eq) * 100.0 if peak_eq > 0 else 0.0
            if dd > max_dd: max_dd = dd

            allocated = sum(ot['cap'] for ot in open_trades)
            avail = max(0.0, equity - allocated)

            if curr_dt in trades_by_date:
                candidates = trades_by_date[curr_dt]
                nifty_rank = {'Nifty 50': 4, 'Nifty 100': 3, 'Nifty 250': 2, 'Other': 1}
                candidates.sort(key=lambda x: -nifty_rank.get(x['Index_Membership'], 0))

                for cand in candidates:
                    pos_val = cand['Entry_Price'] * cand['Position_Size']
                    if pos_val <= avail:
                        open_trades.append({'trade': cand, 'cap': pos_val, 'exit_date': pd.to_datetime(cand['Exit_Date'])})
                        allocated += pos_val
                        avail -= pos_val
                        accepted.append(cand)

        tot_exec = len(accepted)
        exec_wins = sum(1 for t in accepted if t['Outcome'] == 'Success')
        exec_win_rate = (exec_wins / tot_exec * 100.0) if tot_exec > 0 else 0.0

        dur_years = (max_dt - min_dt).days / 365.25
        cagr = ((equity / starting_capital) ** (1.0 / dur_years) - 1.0) * 100.0 if starting_capital > 0 else 0.0
        tot_ret = ((equity - starting_capital) / starting_capital) * 100.0

        gross_w = sum(t['Net_PnL'] for t in accepted if t['Net_PnL'] > 0)
        gross_l = abs(sum(t['Net_PnL'] for t in accepted if t['Net_PnL'] < 0))
        pf = (gross_w / gross_l) if gross_l > 0 else float('inf')

        res_list.append({
            'Liquidity Timeframe': tf,
            'Overall Possible Win Rate (%)': f'{overall_win_rate:.2f}%',
            'Overall Possible Signals': tot_possible,
            'Executed Portfolio Win Rate (%)': f'{exec_win_rate:.2f}%',
            'Executed Trades Count': tot_exec,
            'Final Equity (INR)': f'{equity:,.0f}',
            'Total Portfolio Return (%)': f'{tot_ret:,.2f}%',
            'CAGR (%)': f'{cagr:.2f}%',
            'Max DD (%)': f'{max_dd:.2f}%',
            'Profit Factor': f'{pf:.2f}'
        })

    res_df = pd.DataFrame(res_list)
    csv_out = REPORTS_DIR / "Timeframe_Comparison_1to3_RR_Results.csv"
    res_df.to_csv(csv_out, index=False)
    print("=== TIMEFRAME COMPARISON (1:3 RR - YEARLY vs MONTHLY vs WEEKLY) ===", flush=True)
    print(res_df.to_string(index=False), flush=True)
    print(f"\nReport exported to: {csv_out.resolve()}", flush=True)
    return res_df

if __name__ == "__main__":
    run_timeframe_comparison()
