"""
Main Entry Point: Timeframe Comparison Analytics (Yearly vs Monthly vs Weekly)

Executes src.analysis.compare_timeframes_metrics
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.analysis.compare_timeframes_metrics import run_timeframe_comparison

if __name__ == "__main__":
    run_timeframe_comparison()
