"""
Main Entry Point: ML-Filtered 1:2 RR Trade Candlestick Chart Plotter

Executes src.plotting.plot_ml_trade_charts
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.plotting.plot_ml_trade_charts import plot_all_executed_ml_trades

if __name__ == "__main__":
    plot_all_executed_ml_trades(max_trades_to_plot=10)
