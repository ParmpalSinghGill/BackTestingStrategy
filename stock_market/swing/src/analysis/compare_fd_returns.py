"""
Main Entry Point: Bank Fixed Deposit (FD) vs Trading Strategy Comparison

Executes src.analysis.run_fd_comparison
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.analysis.run_fd_comparison import run_full_fd_strategy_comparison

if __name__ == "__main__":
    run_full_fd_strategy_comparison()
