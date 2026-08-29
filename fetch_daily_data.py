"""
Main Entry Point: Daily (1d) Stock Data Downloader

Executes src.data_fetchers.fetch_daily_data
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.data_fetchers.fetch_daily_data import main

if __name__ == "__main__":
    main()
