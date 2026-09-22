"""
Plot CURRENT LOWER weekly liquidity on a single ticker (default: RELIANCE.NS).

A pool is kept only if it is a swing-low AND
  (no next lower exists, OR next lower is more than 5% away).
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.liquidity_engine.weekly_liquidity import generate_weekly_liquidity_plots


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    print(f"=== Current LOWER weekly liquidity (none or |next lower| > 5%) : {symbol} ===")
    result = generate_weekly_liquidity_plots(symbol, min_lower_abs_pct=5.0)
    print(f"\nDone. {result['count']} / {result['raw_count']} PNGs written to {result['out_dir']}")


if __name__ == "__main__":
    main()
