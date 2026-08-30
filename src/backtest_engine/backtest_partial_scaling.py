"""
Main Entry Point: Partial Scaling Exit Strategy (50% @ 1:2 RR, 25% @ 1:3 RR, 25% @ 1:4 RR)

Executes src.analysis.backtest_partial_scaling_strategy
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.analysis.backtest_partial_scaling_strategy import run_partial_scaling_portfolio_analysis

if __name__ == "__main__":
    run_partial_scaling_portfolio_analysis()
