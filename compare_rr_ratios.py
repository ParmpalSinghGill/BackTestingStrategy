"""
Main Entry Point: Risk-to-Reward (RR) Sensitivity Analysis (1:2 to 1:15)

Executes src.analysis.compare_rr_ratios
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.analysis.compare_rr_ratios import run_rr_sensitivity_analysis

if __name__ == "__main__":
    run_rr_sensitivity_analysis()
