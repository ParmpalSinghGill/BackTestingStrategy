"""
Main Entry Point: Stop Loss Buffer Benchmark (0.0%, 0.1%, 0.2%, 0.5% below C1 Low) & ₹1,000 Fixed Risk

Executes src.analysis.run_sl_buffer_fixed_risk_comparison
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.analysis.run_sl_buffer_fixed_risk_comparison import run_full_sl_buffer_comparison

if __name__ == "__main__":
    run_full_sl_buffer_comparison()
