"""
Main Entry Point: Multi-Class Dynamic Risk-Reward ML Selector Strategy

Executes src.analysis.ml_dynamic_rr_selector_model
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.analysis.ml_dynamic_rr_selector_model import run_full_dynamic_rr_benchmark

if __name__ == "__main__":
    run_full_dynamic_rr_benchmark()
