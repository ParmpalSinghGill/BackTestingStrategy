"""
Main Entry Point: ML Model Benchmark Comparison (Random Forest vs XGBoost vs MLP)

Executes src.analysis.compare_ml_models
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.analysis.compare_ml_models import run_full_ml_model_comparison

if __name__ == "__main__":
    run_full_ml_model_comparison()
