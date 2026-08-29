"""
Main Entry Point: Feature Selection & Pruning Comparison for 1:2 Random Forest

Executes src.analysis.feature_pruning_analysis
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.analysis.feature_pruning_analysis import run_feature_pruning_comparison

if __name__ == "__main__":
    run_feature_pruning_comparison()
