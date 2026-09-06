"""
Swing Strategy Candlestick Trade Chart Plotter

Institutional-grade 200 DPI candlestick charts overlaying:
- Support Liquidity level (#2563EB dashed)
- Liquidity-source candle marker above that bar (#2563EB)
- Dark Green horizontal line at Entry Price (#006400 solid)
- Planned Entry horizontal line (#047857 dashed) if fill differs
- Dark Red horizontal line at Exit Price (#8B0000 solid)
- Dotted Stop-Loss (#DC2626) & Target (#059669) lines
- Wick-preserving green/red entry & exit marker arrows
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DATA_DAILY_DIR = BASE_DIR / "data_daily"
PLOTS_DIR = BASE_DIR / "Plots" / "swing_statement_trades"

plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"


def find_liquidity_source_candle(df: pd.DataFrame, support_price: float, before_dt: pd.Timestamp):
    """Return the bar that printed the HTF support low (Yearly/Monthly/Weekly source)."""
    prior = df[df["Date"] < before_dt]
    if prior.empty or support_price is None or not np.isfinite(support_price):
        return None
    tol = max(0.05, abs(float(support_price)) * 0.001)
    hits = prior[(prior["Low"] - float(support_price)).abs() <= tol]
    if not hits.empty:
        return hits.iloc[-1]
    prior = prior.copy()
    prior["_d"] = (prior["Low"] - float(support_price)).abs()
    best = prior.loc[prior["_d"].idxmin()]
    if float(best["_d"]) <= max(0.50, abs(float(support_price)) * 0.005):
        return best
    return None


def plot_swing_trade_chart(trade_record: dict, output_dir: Path = None) -> str:
    if output_dir is None:
        output_dir = PLOTS_DIR

    os.makedirs(output_dir, exist_ok=True)

    ticker = trade_record["Ticker"]
    trade_id = trade_record["Trade_ID"]
    entry_dt = pd.to_datetime(trade_record["C2_Date"])
    exit_dt = pd.to_datetime(trade_record["Exit_Date"])
    net_pnl = trade_record.get("Net_PnL", 0.0)
    outcome_label = "PROFIT" if net_pnl >= 0 else "LOSS"

    filename = f"Trade_{trade_id:04d}_{ticker.replace('.NS','')}_{entry_dt.strftime('%Y-%m-%d')}_{outcome_label}.png"
    file_path = output_dir / filename

    if file_path.exists():
        return file_path.as_uri()

    csv_candidates = [
        DATA_DAILY_DIR / f"{ticker}_1d.csv",
        DATA_DAILY_DIR / f"{ticker}.csv",
        DATA_DAILY_DIR / f"{ticker.replace('.NS', '')}_1d.csv",
        DATA_DAILY_DIR / f"{ticker.replace('.NS', '')}.csv"
    ]

    csv_file = None
    for cand in csv_candidates:
        if cand.exists():
            csv_file = cand
            break

    if csv_file is None:
        return "N/A"

    try:
        df = pd.read_csv(csv_file)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        liq_explicit = trade_record.get("Liquidity_Date") or trade_record.get("Sweep_Date") or trade_record.get("Support_Formed_Date")
        liq_row = None
        if liq_explicit:
            liq_dt_hint = pd.to_datetime(liq_explicit)
            near = df[(df["Date"] >= liq_dt_hint - pd.Timedelta(days=3)) & (df["Date"] <= liq_dt_hint + pd.Timedelta(days=3))]
            if not near.empty:
                liq_row = near.loc[(near["Date"] - liq_dt_hint).abs().idxmin()]
        if liq_row is None:
            liq_row = find_liquidity_source_candle(df, trade_record.get("Support_Price"), entry_dt)

        start_window = entry_dt - pd.Timedelta(days=60)
        if liq_row is not None:
            start_window = min(start_window, pd.Timestamp(liq_row["Date"]) - pd.Timedelta(days=15))
        end_window = exit_dt + pd.Timedelta(days=20)
        df_sub = df[(df["Date"] >= start_window) & (df["Date"] <= end_window)].copy()

        if len(df_sub) < 5:
            return "N/A"

        fig, ax = plt.subplots(figsize=(13, 7), dpi=200)

        for _, row in df_sub.iterrows():
            d = mdates.date2num(row["Date"])
            open_p, high_p, low_p, close_p = row["Open"], row["High"], row["Low"], row["Close"]
            color = "#10B981" if close_p >= open_p else "#EF4444"

            ax.plot([d, d], [low_p, high_p], color=color, linewidth=1.2, zorder=2)
            body_bottom = min(open_p, close_p)
            body_top = max(open_p, close_p)
            body_height = max(body_top - body_bottom, 0.05)
            ax.add_patch(plt.Rectangle((d - 0.35, body_bottom), 0.7, body_height, facecolor=color, edgecolor=color, alpha=0.9, zorder=3))

        sup_p = trade_record["Support_Price"]
        entry_p = trade_record["Entry_Price"]
        sl_p = trade_record["SL_Price"]
        tp_p = trade_record["Target_Price"]
        exit_p = trade_record["Exit_Price"]
        liq_type = trade_record.get("Liquidity_Type", "Support Level")
        rr_choice = trade_record.get("ML_RR_Choice", "1:3")

        planned_p = trade_record.get("Planned_Entry_Price", entry_p)

        # EXACT USER COLOR RULE: DARK GREEN FOR ENTRY, DARK RED FOR EXIT
        # Support Level Line (#2563EB)
        ax.axhline(sup_p, color="#2563EB", linestyle="--", linewidth=1.2, label=f"Support ({liq_type}): Rs {sup_p:.2f}", zorder=4)

        if abs(planned_p - entry_p) > 0.01:
            ax.axhline(planned_p, color="#047857", linestyle="--", linewidth=1.4, label=f"Planned Entry: Rs {planned_p:.2f}", zorder=4)
            ax.axhline(entry_p, color="#006400", linestyle="-", linewidth=1.6, label=f"Actual Fill Entry (Dark Green): Rs {entry_p:.2f}", zorder=4)
        else:
            ax.axhline(entry_p, color="#006400", linestyle="-", linewidth=1.6, label=f"Entry Price (Dark Green): Rs {entry_p:.2f}", zorder=4)

        ax.axhline(sl_p, color="#DC2626", linestyle=":", linewidth=1.4, label=f"Stop Loss: Rs {sl_p:.2f}", zorder=4)
        ax.axhline(tp_p, color="#059669", linestyle=":", linewidth=1.4, label=f"Target ({rr_choice} RR): Rs {tp_p:.2f}", zorder=4)

        # Exit Price Line (Dark Red #8B0000)
        ax.axhline(exit_p, color="#8B0000", linestyle="-", linewidth=1.6, label=f"Exit Price (Dark Red): Rs {exit_p:.2f}", zorder=4)

        # Calculate Average Candle Size (Height) for offset calculation
        avg_candle_size = (df_sub["High"] - df_sub["Low"]).mean()
        if pd.isna(avg_candle_size) or avg_candle_size <= 0:
            avg_candle_size = entry_p * 0.02

        # Locate Entry Candle High & Low
        entry_rows = df_sub[df_sub["Date"] >= entry_dt]
        if not entry_rows.empty:
            e_row = entry_rows.iloc[0]
            entry_dt_actual = e_row["Date"]
            entry_high = float(e_row["High"])
            entry_low = float(e_row["Low"])
        else:
            entry_dt_actual = entry_dt
            entry_high = entry_p + avg_candle_size
            entry_low = entry_p - avg_candle_size

        # Locate Exit Candle High & Low
        exit_rows = df_sub[df_sub["Date"] >= exit_dt]
        if not exit_rows.empty:
            ex_row = exit_rows.iloc[0]
            exit_dt_actual = ex_row["Date"]
            exit_high = float(ex_row["High"])
            exit_low = float(ex_row["Low"])
        else:
            exit_dt_actual = exit_dt
            exit_high = exit_p + avg_candle_size
            exit_low = exit_p - avg_candle_size

        # Position marker arrows offset by average candle size to leave wicks 100% visible in empty space
        if net_pnl >= 0:
            # Profit: entry above High, exit below Low (empty space, wicks fully visible)
            entry_arrow_y = entry_high + avg_candle_size
            exit_arrow_y = exit_low - avg_candle_size
            entry_marker = "^"
            exit_marker = "v"
            entry_color = "#10B981"
            exit_color = "#10B981"
            entry_label = f"Entry Point (Win)"
            exit_label = f"Target Exit (Win)"
        else:
            # Loss: entry below Low, exit above High (empty space, wicks fully visible)
            entry_arrow_y = entry_low - avg_candle_size
            exit_arrow_y = exit_high + avg_candle_size
            entry_marker = "^"
            exit_marker = "v"
            entry_color = "#10B981"
            exit_color = "#EF4444"
            entry_label = f"Entry Point (Loss)"
            exit_label = f"SL Exit (Loss)"

        ax.scatter([mdates.date2num(entry_dt_actual)], [entry_arrow_y], color=entry_color, s=140, marker=entry_marker, label=entry_label, zorder=6)
        ax.scatter([mdates.date2num(exit_dt_actual)], [exit_arrow_y], color=exit_color, s=140, marker=exit_marker, label=exit_label, zorder=6)

        liq_date_txt = "N/A"
        if liq_row is not None:
            liq_dt_actual = pd.Timestamp(liq_row["Date"])
            liq_high = float(liq_row["High"])
            liq_arrow_y = liq_high + avg_candle_size
            if abs((liq_dt_actual - entry_dt_actual).days) <= 1 and abs(liq_arrow_y - entry_arrow_y) < avg_candle_size:
                liq_arrow_y = entry_arrow_y + avg_candle_size
            ax.scatter(
                [mdates.date2num(liq_dt_actual)], [liq_arrow_y],
                color="#2563EB", s=170, marker="v", label=f"Liquidity Source ({liq_type})", zorder=7,
            )
            ax.annotate(
                f"Liq {liq_type}",
                xy=(mdates.date2num(liq_dt_actual), liq_arrow_y),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color="#1D4ED8",
            )
            liq_date_txt = liq_dt_actual.strftime("%Y-%m-%d")

        info_text = (
            f"Trade ID: #{trade_id} | Stock: {ticker}\n"
            f"Liquidity Source: {liq_type} ({liq_date_txt})\n"
            f"Entry: {entry_dt.strftime('%Y-%m-%d')} @ Rs {entry_p:.2f}\n"
            f"Exit:  {exit_dt.strftime('%Y-%m-%d')} @ Rs {exit_p:.2f}\n"
            f"Target RR: {rr_choice} | Net PnL: Rs {net_pnl:,.2f}\n"
            f"Account Balance: Rs {trade_record.get('Balance_After_Exit', 0.0):,.2f}"
        )

        box_color = "#ECFDF5" if net_pnl >= 0 else "#FEF2F2"
        border_color = "#006400" if net_pnl >= 0 else "#8B0000"
        ax.text(0.02, 0.95, info_text, transform=ax.transAxes, fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.6", facecolor=box_color, edgecolor=border_color, linewidth=1.5, alpha=0.95))

        # Adjust Y limits so text box and legend never overlap candles
        all_prices = list(df_sub["High"]) + list(df_sub["Low"]) + [sup_p, entry_p, sl_p, tp_p, exit_p]
        if liq_row is not None:
            all_prices.append(float(liq_row["High"]) + avg_candle_size)
        min_p = min(all_prices)
        max_p = max(all_prices)
        p_range = max(max_p - min_p, 1.0)
        ax.set_ylim(min_p - 0.08 * p_range, max_p + 0.15 * p_range)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        fig.autofmt_xdate()

        ax.set_title(f"Swing Trade Statement Chart: {ticker} ({entry_dt.strftime('%Y-%m-%d')} to {exit_dt.strftime('%Y-%m-%d')})", fontsize=13, fontweight="bold", pad=12)
        ax.set_ylabel("Price (INR)", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="gray")

        plt.tight_layout()
        plt.savefig(file_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        return file_path.as_uri()
    except Exception as e:
        plt.close()
        return "N/A"
