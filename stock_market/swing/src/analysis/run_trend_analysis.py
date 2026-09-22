"""
Run Trend Analysis Script

Scans historical stock data CSV files (e.g. from data_daily/ or data/)
and outputs quantitative trend identification and regime diagnostics using TrendFinder.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

# Force UTF-8 encoding for stdout on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.analysis.trend_finder import TrendFinder, calculate_pivot_structure


def analyze_single_file(file_path: Path) -> None:
    print(f"\n==================================================", flush=True)
    print(f"ANALYZING TREND FOR: {file_path.name}", flush=True)
    print(f"==================================================", flush=True)

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return

    req_cols = {"Open", "High", "Low", "Close"}
    if not req_cols.issubset(df.columns):
        print(f"Missing required columns {req_cols}. Columns present: {list(df.columns)}")
        return

    finder = TrendFinder()
    summary = finder.get_latest_summary(df)

    print(f"Latest Date        : {summary['Date']}")
    print(f"Latest Close       : {summary['Close']:.2f}")
    print(f"Market Regime      : {summary['Market_Regime']}")
    print(f"Regime Score       : {summary['Regime_Score']} / 5")
    print(f"Trend Confidence   : {summary['Trend_Confidence_Pct']:.1f}%")
    print(f"--------------------------------------------------")
    print(f"Supertrend Signal  : {summary['Supertrend_Signal']} (Val: {summary['Supertrend_Value']:.2f})")
    print(f"ADX (Trend Strength): {summary['ADX']:.2f} ({summary['Trend_Strength']})")
    print(f"Plus DI / Minus DI : +DI={summary['Plus_DI']:.2f} | -DI={summary['Minus_DI']:.2f}")
    print(f"EMA Slope Angle    : {summary['EMA_Fast_Slope_Angle']:.2f} deg")
    print(f"EMA Stack Alignment: {summary['EMA_Alignment']}")
    print(f"Linear Reg Slope   : {summary['LinReg_Slope_Pct']:.4f}%/bar (R2={summary['LinReg_R2']:.3f})")
    print(f"Pivot Price Action : {summary['Pivot_Structure']}")
    print(f"   - Higher Highs: {summary['Higher_Highs']} | Higher Lows: {summary['Higher_Lows']}")
    print(f"   - Lower Highs : {summary['Lower_Highs']} | Lower Lows : {summary['Lower_Lows']}")
    print(f"==================================================\n")


def scan_directory(data_dir: Path, max_files: int = 10) -> None:
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return

    print(f"\nScanning {min(len(csv_files), max_files)} stocks in '{data_dir}' for trend regimes...\n")

    finder = TrendFinder()
    results = []

    for csv_file in csv_files[:max_files]:
        try:
            df = pd.read_csv(csv_file)
            if not {"Open", "High", "Low", "Close"}.issubset(df.columns) or len(df) < 30:
                continue

            summary = finder.get_latest_summary(df)
            symbol = csv_file.stem.replace("_1d", "").replace("_1m", "")

            results.append({
                "Symbol": symbol,
                "Close": summary["Close"],
                "Market_Regime": summary["Market_Regime"],
                "Regime_Score": summary["Regime_Score"],
                "Confidence_%": summary["Trend_Confidence_Pct"],
                "Supertrend": summary["Supertrend_Signal"],
                "ADX": round(summary["ADX"], 2),
                "Pivot_Structure": summary["Pivot_Structure"],
            })
        except Exception as e:
            continue

    if results:
        res_df = pd.DataFrame(results)
        print(res_df.to_string(index=False))
        print(f"\nTotal Analyzed: {len(results)} stocks.\n")


def main():
    parser = argparse.ArgumentParser(description="Trend Finder CLI Diagnostics")
    parser.add_argument("--file", type=str, help="Path to single stock CSV file")
    parser.add_argument("--dir", type=str, default=str(BASE_DIR / "data_daily"), help="Directory of stock CSVs to scan")
    parser.add_argument("--limit", type=int, default=10, help="Max files to scan")

    args = parser.parse_args()

    if args.file:
        analyze_single_file(Path(args.file))
    else:
        scan_dir = Path(args.dir)
        if scan_dir.exists():
            scan_directory(scan_dir, max_files=args.limit)
        else:
            print(f"Directory {scan_dir} does not exist.")


if __name__ == "__main__":
    main()
