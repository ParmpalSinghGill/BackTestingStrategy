"""
Main Execution Script: Intraday Trading Strategy (15-Minute Key Level Reversal Strategy)

Runs:
1. 15-Minute Intraday Resampling & Key Level Reversal Strategy Engine
2. Portfolio Simulation ($1,000 Capital, 5x Leverage, Max 2 Open Positions)
3. Excel Account Statement & Intraday Trade Plot Chart Generation

Usage:
    python intraday_strategy/run_strategy.py
"""

import os
import sys
from pathlib import Path

# Force UTF-8 encoding for stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from intraday_strategy.generate_statement import generate_intraday_strategy_statement


def main():
    print("==========================================================================", flush=True)
    print("      RUNNING INTRADAY STRATEGY: 15m KEY LEVEL REVERSAL STRATEGY          ", flush=True)
    print("==========================================================================", flush=True)

    # Step 1: Run Intraday Statement Simulation & Backtest Evaluation
    excel_path, df_stmt = generate_intraday_strategy_statement(initial_capital=1000.0)

    print("==========================================================================", flush=True)
    print("                    INTRADAY STRATEGY EXECUTION COMPLETE                  ", flush=True)
    print("==========================================================================", flush=True)


if __name__ == "__main__":
    main()
