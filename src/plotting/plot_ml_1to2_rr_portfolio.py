"""
Main Entry Point: ML-Filtered 1:2 RR Portfolio Equity Curve, Yearly Bar Chart & Monthly Heatmap Plotter

Executes src.plotting.plot_ml_1to2_rr_portfolio
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.plotting.plot_ml_1to2_rr_portfolio import run_and_plot_ml_1to2_rr_portfolio

if __name__ == "__main__":
    run_and_plot_ml_1to2_rr_portfolio()
