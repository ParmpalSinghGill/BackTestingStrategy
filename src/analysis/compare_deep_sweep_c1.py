"""
Main Entry Point: Deep Sweep C1 Filter Strategy Comparison (70%+ Submerged / Fully Below Support)

Executes src.analysis.run_deep_sweep_c1_analysis
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.analysis.run_deep_sweep_c1_analysis import run_deep_sweep_c1_comparison

if __name__ == "__main__":
    run_deep_sweep_c1_comparison()
