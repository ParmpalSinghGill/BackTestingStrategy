"""
Multi-Timeframe Support-Only Event Finder & Source-to-Break Event Plotter

Features:
- Extracts Support Events ONLY (ignoring resistance).
- Timeframe Hierarchy & Weightages:
    - Yearly  (1Y) : Weight 3.0 (High Weightage)
    - Monthly (1M) : Weight 2.0 (Medium Weightage)
    - Weekly  (1W) : Weight 1.0 (Low Weightage)
- Separate Overview Graphs for Year, Month, and Week stored in `plot/`.
- Single-Image Event Creation & First Price Break Plots showing both the Source of Support
  and the First Price Break below support (or present date if active) stored in `plot/<SYMBOL>_events/`.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import mplfinance as mpf

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DAILY_DIR = BASE_DIR / "data_daily"
PLOT_DIR = BASE_DIR / "plot"
AGG_DICT = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}

TIMEFRAME_SPECS = {
    "Yearly": {
        "weight": 3.0,
        "label": "Yearly (1Y)",
        "color": "#800080",  # Purple
        "style": "--",
        "linewidth": 2.2,
    },
    "Monthly": {
        "weight": 2.0,
        "label": "Monthly (1M)",
        "color": "#1f77b4",  # Blue
        "style": "--",
        "linewidth": 1.8,
    },
    "Weekly": {
        "weight": 1.0,
        "label": "Weekly (1W)",
        "color": "#2ca02c",  # Green
        "style": "--",
        "linewidth": 1.4,
    },
}


def load_stock_daily(symbol: str) -> pd.DataFrame:
    """Load daily OHLCV dataset for given stock symbol."""
    safe_sym = symbol.replace("=", "_").replace("/", "_").replace("^", "_")
    path = DATA_DAILY_DIR / f"{safe_sym}_1d.csv"
    if not path.exists():
        raise FileNotFoundError(f"Daily data file not found: {path}")

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def _resample_daily(daily: pd.DataFrame, rules: list) -> pd.DataFrame:
    for rule in rules:
        try:
            return daily.resample(rule).agg(AGG_DICT).dropna()
        except (ValueError, KeyError):
            pass
    raise ValueError(f"Could not resample with rules: {rules}")


def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    return _resample_daily(daily, ["W-FRI", "W"])


def to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    return _resample_daily(daily, ["ME", "M"])


def to_yearly(daily: pd.DataFrame) -> pd.DataFrame:
    return _resample_daily(daily, ["YE", "A", "Y"])


def find_support_pivot_levels(
    df: pd.DataFrame,
    tf_name: str,
    left: int = 3,
    right: int = 2,
    big_mult: float = 2.0,
    avg_window: int = 20,
) -> list:
    """Find SUPPORT pivot levels (swing lows) ONLY on a given timeframe DataFrame."""
    n = len(df)
    if n < 5:
        return []

    lows = df["Low"].to_numpy(float)
    body = (df["Close"] - df["Open"]).abs().to_numpy(float)
    avg_body = (
        pd.Series(body).rolling(avg_window, min_periods=3).mean().bfill().to_numpy(float)
    )
    dates = df.index
    spec = TIMEFRAME_SPECS[tf_name]

    def consec_lows(i, step):
        c, j = 0, i + step
        while 0 <= j < n:
            if lows[j] < lows[i]:
                break
            c += 1
            j += step
        return c

    def big_candle(i, step):
        j = i + step
        if not (0 <= j < n):
            return False
        return lows[j] > lows[i] and body[j] >= big_mult * avg_body[j]

    labels = []
    for i in range(n):
        lc = consec_lows(i, -1)
        rc = consec_lows(i, +1)
        normal = (lc >= left and rc >= right) or (lc >= right and rc >= left)
        big_ok = (lc >= right and big_candle(i, +1)) or (rc >= right and big_candle(i, -1))

        if normal or big_ok:
            labels.append({
                "timeframe": tf_name,
                "weight": spec["weight"],
                "type": "support",
                "price": float(lows[i]),
                "idx": i,
                "formed_date": dates[i],
                "canceled": False,
                "cancel_date": None,
                "cancel_price": None,
            })

    return labels


def track_support_cancellation(
    labels: list, daily_df: pd.DataFrame, break_tol: float = 0.0
) -> list:
    """Track support level cancellation (break/sweep) against daily Low prices."""
    daily_dates = daily_df.index
    daily_lows = daily_df["Low"].to_numpy(float)

    processed = []
    for lb in labels:
        formed = lb["formed_date"]
        price = lb["price"]

        post_mask = daily_dates > formed
        if not post_mask.any():
            processed.append(lb)
            continue

        start_idx = int(np.argmax(post_mask))
        beyond = daily_lows[start_idx:] < price * (1.0 - break_tol)

        if beyond.any():
            hit_rel = int(np.argmax(beyond))
            cancel_idx = start_idx + hit_rel
            lb["canceled"] = True
            lb["cancel_date"] = daily_dates[cancel_idx]
            lb["cancel_price"] = daily_lows[cancel_idx]

        processed.append(lb)

    return processed


def extract_all_support_events(daily_df: pd.DataFrame) -> dict:
    """
    Extract Support events for Yearly, Monthly, and Weekly timeframes.
    Returns dict: {"Yearly": (df_y, labels_y), "Monthly": (df_m, labels_m), "Weekly": (df_w, labels_w)}
    """
    weekly_df = to_weekly(daily_df)
    monthly_df = to_monthly(daily_df)
    yearly_df = to_yearly(daily_df)

    y_labels = track_support_cancellation(find_support_pivot_levels(yearly_df, "Yearly", left=2, right=1), daily_df)
    m_labels = track_support_cancellation(find_support_pivot_levels(monthly_df, "Monthly", left=3, right=2), daily_df)
    w_labels = track_support_cancellation(find_support_pivot_levels(weekly_df, "Weekly", left=3, right=2), daily_df)

    return {
        "Yearly": (yearly_df, y_labels),
        "Monthly": (monthly_df, m_labels),
        "Weekly": (weekly_df, w_labels),
    }


def plot_timeframe_support_overview(
    symbol: str, tf_name: str, df_tf: pd.DataFrame, labels: list, out_dir: Path = PLOT_DIR
) -> str:
    """Plot overview candlestick chart for a specific timeframe with Support levels."""
    os.makedirs(out_dir, exist_ok=True)
    clean_sym = symbol.replace(".NS", "")
    out_file = out_dir / f"{clean_sym}_{tf_name}_Support_Overview.png"

    spec = TIMEFRAME_SPECS[tf_name]
    end_date = df_tf.index.max()

    if tf_name == "Yearly":
        start_date = end_date - pd.DateOffset(years=10)
    elif tf_name == "Monthly":
        start_date = end_date - pd.DateOffset(years=5)
    else:
        start_date = end_date - pd.DateOffset(years=2)

    view_df = df_tf.loc[df_tf.index >= start_date].copy()
    if view_df.empty:
        view_df = df_tf.copy()

    n_candles = len(view_df)
    pad = max(10, int(n_candles * 0.12))
    x_end = n_candles - 1 + pad

    fig, ax = mpf.plot(
        view_df,
        type="candle",
        style="yahoo",
        title=f"{symbol} - {spec['label']} Support Events Overview (Weight {spec['weight']:.1f})",
        volume=False,
        figratio=(16, 8),
        figscale=1.2,
        tight_layout=True,
        datetime_format="%Y-%m" if tf_name != "Yearly" else "%Y",
        warn_too_much_data=5000,
        returnfig=True,
    )
    a0 = ax[0]
    a0.set_xlim(-1, x_end)

    for lb in labels:
        formed = lb["formed_date"]
        if formed > end_date:
            continue

        pos_start = int(view_df.index.searchsorted(formed))
        pos_start = min(max(pos_start, 0), n_candles - 1)

        if lb["canceled"] and lb["cancel_date"] in view_df.index:
            pos_cancel = int(view_df.index.searchsorted(lb["cancel_date"]))
            pos_end = min(pos_cancel, n_candles - 1)
            alpha = 0.35
            lbl_txt = f" S {lb['price']:.1f} (Swept)"
        else:
            pos_end = x_end
            alpha = 0.9
            lbl_txt = f" S {lb['price']:.1f}"

        a0.plot(
            [pos_start, pos_end],
            [lb["price"], lb["price"]],
            color=spec["color"],
            linestyle=spec["style"],
            lw=spec["linewidth"],
            alpha=alpha,
        )
        if not lb["canceled"]:
            a0.text(
                pos_end,
                lb["price"],
                lbl_txt,
                color=spec["color"],
                va="center",
                fontsize=8,
                fontweight="bold",
            )

    handle = mlines.Line2D([], [], color=spec["color"], linestyle=spec["style"], lw=spec["linewidth"])
    a0.legend([handle], [f"{spec['label']} Support (Weight {spec['weight']:.1f})"], loc="upper left", fontsize=9)

    fig.savefig(out_file, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(out_file)


def plot_support_event_source_to_break(
    symbol: str,
    tf_name: str,
    df_tf: pd.DataFrame,
    event: dict,
    out_dir: Path,
    buffer_candles: int = 10,
) -> str:
    """
    Plot a single unified chart for a Support Event spanning from its Source of Formation
    up to the First Price Break below support (or present date if active).
    """
    os.makedirs(out_dir, exist_ok=True)
    formed_dt = event["formed_date"]
    price = event["price"]
    clean_sym = symbol.replace(".NS", "")
    date_str = formed_dt.strftime("%Y-%m-%d")

    filename = f"Support_{tf_name}_{date_str}_P{price:.2f}.png"
    out_file = out_dir / filename

    # 1. Determine Start index (Source of Event - buffer_candles)
    formed_idx = int(df_tf.index.searchsorted(formed_dt))
    formed_idx = min(max(formed_idx, 0), len(df_tf) - 1)
    start_idx = max(0, formed_idx - buffer_candles)

    # 2. Determine End index (First Break + buffer_candles OR Present Date)
    if event["canceled"] and event["cancel_date"] in df_tf.index:
        cancel_dt = event["cancel_date"]
        cancel_idx = int(df_tf.index.searchsorted(cancel_dt))
        cancel_idx = min(max(cancel_idx, 0), len(df_tf) - 1)
        end_idx = min(len(df_tf), cancel_idx + buffer_candles + 1)
        status_title = f"SWEPT on {cancel_dt.strftime('%Y-%m-%d')}"
    else:
        cancel_dt = None
        cancel_idx = len(df_tf) - 1
        end_idx = len(df_tf)
        status_title = "ACTIVE to Present Date"

    span_df = df_tf.iloc[start_idx:end_idx].copy()
    n_span = len(span_df)

    rel_formed = formed_idx - start_idx
    rel_formed = min(max(rel_formed, 0), n_span - 1)

    if cancel_dt:
        rel_cancel = cancel_idx - start_idx
        rel_cancel = min(max(rel_cancel, 0), n_span - 1)
    else:
        rel_cancel = n_span - 1

    spec = TIMEFRAME_SPECS[tf_name]

    dt_fmt = "%b %Y" if tf_name == "Monthly" else "%Y"
    if tf_name == "Weekly":
        dt_fmt = "%b %d, %Y"

    title_txt = (
        f"{symbol} - {spec['label']} Support Event: Source to Price Break\n"
        f"Support Level: INR {price:.2f} | Formed: {date_str} | Status: {status_title}"
    )

    fig, ax = mpf.plot(
        span_df,
        type="candle",
        style="yahoo",
        title=title_txt,
        volume=False,
        figratio=(16, 8),
        figscale=1.2,
        tight_layout=True,
        datetime_format=dt_fmt,
        returnfig=True,
    )
    a0 = ax[0]
    a0.set_xlim(-1, n_span + 6)

    # Draw horizontal support line from formed position to cancel position
    a0.plot(
        [rel_formed, rel_cancel],
        [price, price],
        color=spec["color"],
        linestyle=spec["style"],
        lw=spec["linewidth"] + 0.5,
        alpha=0.9,
    )

    # Marker A: Source of Support Creation
    a0.axvline(rel_formed, color="#8c564b", linestyle=":", lw=1.5, alpha=0.8)
    a0.annotate(
        f"SOURCE of Support\nPrice: {price:.2f}\nDate: {date_str}",
        xy=(rel_formed, price),
        xytext=(rel_formed, price * 1.03),
        arrowprops=dict(facecolor="#8c564b", shrink=0.08, width=1.5, headwidth=6),
        fontsize=8,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="#ffe6e6", alpha=0.85),
    )

    # Marker B: First Price Break / Sweep (if cancelled) OR Present Date (if active)
    if cancel_dt:
        break_price = float(span_df["Low"].iloc[rel_cancel])
        a0.axvline(rel_cancel, color="#d62728", linestyle=":", lw=1.5, alpha=0.8)
        a0.annotate(
            f"FIRST PRICE BREAK\nLow: {break_price:.2f}\nDate: {cancel_dt.strftime('%Y-%m-%d')}",
            xy=(rel_cancel, break_price),
            xytext=(rel_cancel, break_price * 0.97),
            arrowprops=dict(facecolor="#d62728", shrink=0.08, width=1.5, headwidth=6),
            fontsize=8,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#ffcccc", alpha=0.85),
        )
    else:
        a0.axvline(rel_cancel, color="#2ca02c", linestyle=":", lw=1.5, alpha=0.8)
        a0.annotate(
            f"ACTIVE to Present\nLatest Date: {span_df.index[-1].strftime('%Y-%m-%d')}",
            xy=(rel_cancel, float(span_df["Close"].iloc[-1])),
            xytext=(max(0, rel_cancel - 5), float(span_df["Close"].iloc[-1]) * 1.02),
            fontsize=8,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#e6ffe6", alpha=0.85),
        )

    fig.savefig(out_file, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(out_file)
