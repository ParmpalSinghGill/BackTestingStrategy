"""
Parallel Experiment Runner: Single 5-Class Model & Single 6-Class Model

Evaluates under 100k Capital & 1k Risk Cap (Following Guide/Realistic_guide.md):
1. Single 5-Class Model: Predicts Skip, 1:2, 1:3, 1:4, 1:5.
2. Single 6-Class Model: Predicts Skip, 1:2, 1:3, 1:4, 1:5, 1:6.

Executes both models simultaneously in parallel processes for ultra-fast performance comparison.
Updates Master_4Class_Comparison.xlsx with full multi-class performance comparison.
"""

import os
import sys
from pathlib import Path
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

# Force UTF-8 encoding for stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

COMPARISON_DIR = BASE_DIR / "Reports" / "Model_Comparison_4Class_vs_2Stage"
PLOTS_DIR = BASE_DIR / "Plots" / "Model_Comparison_4Class_vs_2Stage"

os.makedirs(COMPARISON_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

from swing_strategy.strategy_multiclass_engine import (
    prepare_multiclass_dataset,
    run_single_5class_model,
    run_single_6class_model
)
from swing_strategy.generate_statement import generate_swing_strategy_statement


def run_single_5class_task(df_ext):
    exp_name = "Single_5Class_Model"
    rep_dir = COMPARISON_DIR / exp_name
    plt_dir = PLOTS_DIR / exp_name

    df_acc_5class = run_single_5class_model(df_ext, probability_threshold=0.38)

    summary = generate_swing_strategy_statement(
        initial_deposit=100000.0,
        max_charts_to_generate=25,
        max_risk_per_trade=1000.0,
        df_acc=df_acc_5class,
        custom_reports_dir=rep_dir,
        custom_plots_dir=plt_dir,
        exp_name=exp_name
    )
    return summary


def run_single_6class_task(df_ext):
    exp_name = "Single_6Class_Model"
    rep_dir = COMPARISON_DIR / exp_name
    plt_dir = PLOTS_DIR / exp_name

    df_acc_6class = run_single_6class_model(df_ext, probability_threshold=0.36)

    summary = generate_swing_strategy_statement(
        initial_deposit=100000.0,
        max_charts_to_generate=25,
        max_risk_per_trade=1000.0,
        df_acc=df_acc_6class,
        custom_reports_dir=rep_dir,
        custom_plots_dir=plt_dir,
        exp_name=exp_name
    )
    return summary


def run_extended_comparison():
    print("==========================================================================", flush=True)
    print("     SINGLE 5-CLASS VS SINGLE 6-CLASS ML MODEL PARALLEL EXECUTION SUITE    ", flush=True)
    print("==========================================================================", flush=True)

    df_ext = prepare_multiclass_dataset()

    print("\n--- Running 5-Class and 6-Class Models SIMULTANEOUSLY in Parallel Processes ---\n", flush=True)

    with ProcessPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(run_single_5class_task, df_ext)
        f2 = executor.submit(run_single_6class_task, df_ext)

        s5 = f1.result()
        s6 = f2.result()

    summary_list = [s5, s6]

    for s in summary_list:
        print(f"[{s['Exp_Name']}] Final Balance: Rs {s['Final_Equity']:,.2f} | CAGR: {s['CAGR_Pct']:.2f}% | Win Rate: {s['Win_Rate_Pct']:.2f}% ({s['Executed_Trades']:,} Trades)", flush=True)

    return s5, s6


if __name__ == "__main__":
    run_extended_comparison()
