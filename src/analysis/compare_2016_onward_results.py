"""
Main Entry Point: 2016 Onward Strategy Comparison (2016 to 2026)

Executes src.analysis.run_2016_onward_analysis
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.analysis.run_2016_onward_analysis import run_2016_onward_comparison

if __name__ == "__main__":
    run_2016_onward_comparison()
