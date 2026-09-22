"""
Main Entry Point: Streamlined 6-Class Dynamic Risk-Reward ML Selector Strategy (Skip, 1:2, 1:3, 1:5, 1:10, 1:15)

Executes src.analysis.run_streamlined_multi_class_benchmark
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.analysis.run_streamlined_multi_class_benchmark import run_streamlined_6class_benchmark

if __name__ == "__main__":
    run_streamlined_6class_benchmark()
