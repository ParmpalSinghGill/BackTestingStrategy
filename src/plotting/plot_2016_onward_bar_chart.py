"""
Main Entry Point: 2016–2026 Strategy Performance Bar Chart Plotter

Executes src.plotting.plot_2016_onward_bar_chart
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.plotting.plot_2016_onward_bar_chart import generate_2016_onward_bar_chart

if __name__ == "__main__":
    generate_2016_onward_bar_chart()
