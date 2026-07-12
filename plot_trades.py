import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Fast headless plotting
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRADES_CSV = BASE_DIR / "backtest_trades.csv"
PLOTS_DIR = BASE_DIR / "plot" / "trades"

def load_stock_df(ticker: str) -> pd.DataFrame:
    path = DATA_DIR / f"{ticker}_1m.csv"
    if not path.exists():
        # Handle filesystem safe names (e.g. CL=F -> CL_F)
        safe_name = ticker.replace("=", "_").replace("/", "_").replace("^", "_")
        path = DATA_DIR / f"{safe_name}_1m.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    ts_col = "Datetime" if "Datetime" in df.columns else "Date"
    df["dt"] = pd.to_datetime(df[ts_col])
    df["date"] = df["dt"].dt.date
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df.sort_values("dt").reset_index(drop=True)

def resample_to_timeframe(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    if tf == "1m":
        return df.copy()
    minutes = int(tf[:-1])
    base = df[["dt", "Open", "High", "Low", "Close", "Volume"]].copy()
    base = base.set_index("dt").sort_index()
    agg = base.resample(f"{minutes}min", label="left", closed="left").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna()
    agg = agg.reset_index().rename(columns={"index": "dt"})
    agg["date"] = agg["dt"].dt.date
    return agg.sort_values("dt").reset_index(drop=True)

def main():
    if not TRADES_CSV.exists():
        print(f"Error: {TRADES_CSV} not found.")
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)
    trades_df = pd.read_csv(TRADES_CSV)
    
    print(f"Loaded {len(trades_df)} trades. Generating plots...")
    
    # Group by stock to load data files only once
    grouped = trades_df.groupby("stock")
    total_plots = 0
    
    for ticker, trades in grouped:
        print(f"Processing stock: {ticker}")
        raw_df = load_stock_df(ticker)
        if raw_df is None:
            print(f"  ! Data file not found for {ticker}")
            continue
            
        for _, trade in trades.iterrows():
            tf = trade["timeframe"]
            trade_date = pd.to_datetime(trade["trade_date"]).date()
            entry_dt = pd.to_datetime(trade["entry_dt"])
            exit_dt = pd.to_datetime(trade["exit_dt"])
            
            # Resample raw data for this timeframe
            df = resample_to_timeframe(raw_df, tf)
            df["dt"] = pd.to_datetime(df["dt"])
            
            # Filter for trade day + last 2 days of history
            unique_dates = sorted(df["date"].unique())
            if trade_date not in unique_dates:
                continue
            idx = unique_dates.index(trade_date)
            start_idx = max(0, idx - 2)
            selected_dates = unique_dates[start_idx : idx + 1]
            df_plot = df[df["date"].isin(selected_dates)].copy().reset_index(drop=True)
            
            if df_plot.empty:
                continue
                
            # Create Plot
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # Plot Candlesticks
            x = np.arange(len(df_plot))
            colors = np.where(df_plot["Close"] >= df_plot["Open"], "green", "red")
            ax.vlines(x, df_plot["Low"], df_plot["High"], color=colors, linewidth=1)
            
            top = np.maximum(df_plot["Open"], df_plot["Close"])
            bottom = np.minimum(df_plot["Open"], df_plot["Close"])
            height = np.maximum(top - bottom, 0.0001)
            ax.bar(x, height, bottom=bottom, color=colors, width=0.6, align="center")
            
            # Plot reference level
            level_val = trade["level_value"]
            ax.axhline(y=level_val, color="purple", linestyle="--", alpha=0.6, label=f"Level: {trade['level']} ({level_val})")
            
            # Find entry/exit positions on x axis
            entry_matches = df_plot[df_plot["dt"] == entry_dt]
            exit_matches = df_plot[df_plot["dt"] == exit_dt]
            
            if not entry_matches.empty and not exit_matches.empty:
                entry_x = entry_matches.index[0]
                exit_x = exit_matches.index[0]
                entry_price = trade["entry"]
                exit_price = trade["exit"]
                
                # Plot SL line
                ax.hlines(y=trade["sl"], xmin=entry_x, xmax=exit_x, color="red", linestyle=":", alpha=0.8, label="Initial SL")
                
                price_range = df_plot["High"].max() - df_plot["Low"].min()
                offset = price_range * 0.03
                
                # Annotate entry
                if trade["side"] == "long":
                    ax.annotate("BUY", xy=(entry_x, entry_price), xytext=(entry_x, entry_price - offset),
                                arrowprops=dict(facecolor='blue', shrink=0.08, width=1.5, headwidth=5),
                                color='blue', fontweight='bold', ha='center')
                    ax.annotate("EXIT", xy=(exit_x, exit_price), xytext=(exit_x, exit_price + offset),
                                arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=5),
                                color='black', fontweight='bold', ha='center')
                else:
                    ax.annotate("SELL", xy=(entry_x, entry_price), xytext=(entry_x, entry_price + offset),
                                arrowprops=dict(facecolor='orange', shrink=0.08, width=1.5, headwidth=5),
                                color='orange', fontweight='bold', ha='center')
                    ax.annotate("EXIT", xy=(exit_x, exit_price), xytext=(exit_x, exit_price - offset),
                                arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=5),
                                color='black', fontweight='bold', ha='center')
            
            # Configure axes
            # Show dates on x axis at the start of each day
            day_starts = df_plot.drop_duplicates(subset=["date"])
            ax.set_xticks(day_starts.index)
            ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in day_starts["date"]], rotation=25, ha='right')
            ax.grid(True, linestyle=":", alpha=0.5)
            
            # Title
            pnl_val = trade["net_pnl"]
            pnl_pct = trade["return_pct"]
            status_str = f"Profit: ${pnl_val:.2f} ({pnl_pct:+.2f}%)" if pnl_val >= 0 else f"Loss: -${abs(pnl_val):.2f} ({pnl_pct:+.2f}%)"
            
            ax.set_title(f"{ticker} | {trade_date} | {tf} | {trade['side'].upper()} | {trade['exit_reason'].upper()}\n{status_str}",
                         fontsize=12, fontweight='bold', color='green' if pnl_val >= 0 else 'red')
            
            plt.legend(loc="upper left")
            plt.tight_layout()
            
            # Save plot
            file_name = f"{ticker}_{trade_date}_{tf}_pass{trade['pass_no']}.png"
            fig.savefig(PLOTS_DIR / file_name, dpi=120)
            plt.close(fig)
            total_plots += 1
            
    print(f"\nDone! Generated {total_plots} plots under: {PLOTS_DIR}")

if __name__ == "__main__":
    main()
