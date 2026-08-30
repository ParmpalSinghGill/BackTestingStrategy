"""
Main Entry Point: Indian Brokerage & Statutory Taxes Impact Analytics

Executes src.analysis.run_brokerage_impact_analysis
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.analysis.run_brokerage_impact_analysis import run_full_brokerage_impact_analysis

if __name__ == "__main__":
    run_full_brokerage_impact_analysis()
