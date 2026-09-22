"""
Main Entry Point: Master ML Models, RR (1:2 vs 1:3), and Tax Impact Comparison

Executes src.analysis.compare_all_models_rr_tax_impact
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.analysis.compare_all_models_rr_tax_impact import run_master_tax_and_rr_comparison

if __name__ == "__main__":
    run_master_tax_and_rr_comparison()
