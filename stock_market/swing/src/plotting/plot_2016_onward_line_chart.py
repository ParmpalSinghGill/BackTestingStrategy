"""
Main Entry Point: 5-Strategy Comparison Line Chart Plotter (2016 to 2026)

Executes src.plotting.plot_2016_onward_comparison_line_chart
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.plotting.plot_2016_onward_comparison_line_chart import plot_5_strategy_comparison_line_graph

if __name__ == "__main__":
    plot_5_strategy_comparison_line_graph()
