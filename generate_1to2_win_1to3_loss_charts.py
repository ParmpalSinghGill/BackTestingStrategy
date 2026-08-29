"""
Main Entry Point: Candlestick Chart Plotter for Trades Reaching 1:2 Target but Failing at 1:3 Target

Executes src.plotting.plot_1to2_win_1to3_loss_trade_charts
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.plotting.plot_1to2_win_1to3_loss_trade_charts import plot_all_1to2_win_1to3_fail_trades

if __name__ == "__main__":
    plot_all_1to2_win_1to3_fail_trades(max_trades_to_plot=10)
