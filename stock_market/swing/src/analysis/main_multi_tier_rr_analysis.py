"""
Main Entry Point: Multi-Tier Risk-Reward ML Selector Strategy (SkipTrade, 1:2 to 1:15 RR)

Executes src.analysis.run_multi_tier_rr_benchmark
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.analysis.run_multi_tier_rr_benchmark import run_master_multi_tier_benchmark

if __name__ == "__main__":
    run_master_multi_tier_benchmark()
