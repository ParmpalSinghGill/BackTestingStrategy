"""
Main Entry Point: Support-Only Multi-Timeframe Event Finder & Plotter Test Runner

Executes src.liquidity_engine and src.plotting workflows for RELIANCE.NS.
"""
import sys
import os
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.liquidity_engine.multi_tf_event_finder import (
    load_stock_daily,
    extract_all_support_events,
    plot_timeframe_support_overview,
    plot_support_event_source_to_break,
    PLOT_DIR,
)

def main():
    symbol = "RELIANCE.NS"
    clean_sym = symbol.replace(".NS", "")
    print(f"=== Support-Only Multi-Timeframe Event Finder Test: {symbol} ===")

    daily_df = load_stock_daily(symbol)
    print(f"Loaded daily candles for {symbol}: {len(daily_df)} rows ({daily_df.index.min().date()} to {daily_df.index.max().date()})")

    tf_data = extract_all_support_events(daily_df)

    print("\n--- Generating Timeframe Support Overview Plots in plot/ ---")
    for tf_name in ["Yearly", "Monthly", "Weekly"]:
        df_tf, labels = tf_data[tf_name]
        out_path = plot_timeframe_support_overview(symbol, tf_name, df_tf, labels, out_dir=PLOT_DIR)
        active_cnt = sum(1 for l in labels if not l["canceled"])
        print(f"[{tf_name}] Total Support Events: {len(labels)} (Active: {active_cnt}) -> Overview Plot: {out_path}")

    events_dir = PLOT_DIR / f"{clean_sym}_events"
    print(f"\n--- Generating Single-Image Source-to-Break Event Plots in {events_dir} ---")

    end_date = daily_df.index.max()
    start_date = end_date - pd.DateOffset(years=2)

    total_zoomed = 0
    for tf_name in ["Yearly", "Monthly", "Weekly"]:
        df_tf, labels = tf_data[tf_name]
        recent_labels = [l for l in labels if l["canceled"] and l["formed_date"] >= start_date]

        for lb in recent_labels:
            zoomed_path = plot_support_event_source_to_break(
                symbol=symbol,
                tf_name=tf_name,
                df_tf=df_tf,
                event=lb,
                out_dir=events_dir,
                buffer_candles=10,
            )
            total_zoomed += 1
            status_str = "Swept (" + lb["cancel_date"].strftime("%Y-%m-%d") + ")" if lb["canceled"] else "ACTIVE"
            print(f"  [{tf_name}] Support Price: INR {lb['price']:<8.2f} | Formed: {lb['formed_date'].strftime('%Y-%m-%d')} | Status: {status_str:<18} -> {Path(zoomed_path).name}")

    print(f"\nSuccessfully generated {total_zoomed} single-image Source-to-Break event charts!")
    print(f"All plots saved under directory: {PLOT_DIR.resolve()}")

if __name__ == "__main__":
    main()
