"""
Main Entry Point: Support Liquidity Sweep Backtest Strategy Engine (2010 to 2026)

Executes src.backtest_engine.backtest_support_liquidity_strategy
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.backtest_engine.backtest_support_liquidity_strategy import run_backtest_all_stocks

if __name__ == "__main__":
    run_backtest_all_stocks()
