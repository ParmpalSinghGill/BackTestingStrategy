"""
Main Entry Point: Average Yearly Return & Winning Percentages Analytics

Executes src.analysis.calculate_yearly_metrics
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.analysis.calculate_yearly_metrics import main

if __name__ == "__main__":
    main()
