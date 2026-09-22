"""
Main Entry Point: 2016 to TODAY (2026) Streamlined 6-Class ML Selector Benchmark across SL Buffer Levels & Rs 1,000 Fixed Risk

Executes src.analysis.run_2016_to_today_ml_benchmark
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.analysis.run_2016_to_today_ml_benchmark import run_2016_to_today_benchmark

if __name__ == "__main__":
    run_2016_to_today_benchmark()
