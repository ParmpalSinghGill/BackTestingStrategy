"""
Weekly Liquidity Engine

Finds weekly swing-low (CURRENT LOWER) liquidity pools on a single ticker.
A pool is plotted only when there is no next lower, or the next lower is
more than 5% away. Each chart draws a vertical check line just after the
current lower candle, then keeps a few weekly candles after that line.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd

from src.liquidity_engine.multi_tf_event_finder import (
    load_stock_daily,
    to_weekly,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PLOTS_DIR = BASE_DIR / "Plots" / "weekly_liquidity"
REPORTS_DIR = BASE_DIR / "Reports"

CURRENT_COLOR = "#F59E0B"   # gold  – the pool this PNG is about
UPPER_COLOR = "#DC2626"     # red   – next upper (buy-side / swing high)
LOWER_COLOR = "#16A34A"     # green – next lower (sell-side / swing low)
PIVOT_MARKER_COLOR = "#1D4ED8"

LEFT_BARS = 3
RIGHT_BARS = 2
BIG_MULT = 2.0
AVG_WINDOW = 20
BUFFER_BEFORE = 24
CANDLES_AFTER_LINE = 8   # few weekly bars after the check line
CHECK_LINE_COLOR = "#7C3AED"


def find_weekly_pivot_liquidity(
    weekly_df: pd.DataFrame,
    left: int = LEFT_BARS,
    right: int = RIGHT_BARS,
    big_mult: float = BIG_MULT,
    avg_window: int = AVG_WINDOW,
) -> list[dict]:
    """Detect weekly swing highs (upper) and swing lows (lower)."""
    n = len(weekly_df)
    if n < 5:
        return []

    highs = weekly_df["High"].to_numpy(float)
    lows = weekly_df["Low"].to_numpy(float)
    body = (weekly_df["Close"] - weekly_df["Open"]).abs().to_numpy(float)
    avg_body = (
        pd.Series(body).rolling(avg_window, min_periods=3).mean().bfill().to_numpy(float)
    )
    dates = weekly_df.index

    def consec_ext(values: np.ndarray, i: int, step: int, is_high: bool) -> int:
        count, j = 0, i + step
        while 0 <= j < n:
            if is_high and values[j] > values[i]:
                break
            if not is_high and values[j] < values[i]:
                break
            count += 1
            j += step
        return count

    def big_follow(values: np.ndarray, i: int, step: int, is_high: bool) -> bool:
        j = i + step
        if not (0 <= j < n):
            return False
        extreme_holds = values[j] < values[i] if is_high else values[j] > values[i]
        return extreme_holds and body[j] >= big_mult * avg_body[j]

    def is_pivot(values: np.ndarray, i: int, is_high: bool) -> bool:
        lc = consec_ext(values, i, -1, is_high)
        rc = consec_ext(values, i, +1, is_high)
        normal = (lc >= left and rc >= right) or (lc >= right and rc >= left)
        big_ok = (lc >= right and big_follow(values, i, +1, is_high)) or (
            rc >= right and big_follow(values, i, -1, is_high)
        )
        return normal or big_ok

    pools: list[dict] = []
    for i in range(n):
        if is_pivot(highs, i, is_high=True):
            pools.append({
                "side": "upper",
                "price": float(highs[i]),
                "idx": i,
                "formed_date": dates[i],
            })
        if is_pivot(lows, i, is_high=False):
            pools.append({
                "side": "lower",
                "price": float(lows[i]),
                "idx": i,
                "formed_date": dates[i],
            })
    return pools


def pct_from_current(current: float, other: float) -> float:
    """Signed % distance of `other` from `current`. Upper is +, lower is −."""
    if current == 0:
        return 0.0
    return ((other - current) / current) * 100.0


def fmt_pct(pct: float | None) -> str:
    if pct is None:
        return "N/A"
    return f"{pct:+.2f}%"


def attach_neighbor_liquidity(pools: list[dict]) -> list[dict]:
    """
    For every pool, attach the next upper swing-high above it and the next
    lower swing-low below it (closest in price, excluding itself), plus the
    signed % gap from the current pool.
    """
    uppers = [p for p in pools if p["side"] == "upper"]
    lowers = [p for p in pools if p["side"] == "lower"]

    for pool in pools:
        above = [u for u in uppers if u["price"] > pool["price"] and u is not pool]
        below = [lo for lo in lowers if lo["price"] < pool["price"] and lo is not pool]
        nxt_up = min(above, key=lambda x: x["price"]) if above else None
        nxt_lo = max(below, key=lambda x: x["price"]) if below else None
        cur = pool["price"]

        pool["next_upper"] = {
            "price": nxt_up["price"],
            "formed_date": nxt_up["formed_date"],
            "idx": nxt_up["idx"],
            "pct": pct_from_current(cur, nxt_up["price"]),
        } if nxt_up is not None else None

        pool["next_lower"] = {
            "price": nxt_lo["price"],
            "formed_date": nxt_lo["formed_date"],
            "idx": nxt_lo["idx"],
            "pct": pct_from_current(cur, nxt_lo["price"]),
        } if nxt_lo is not None else None

        pool["dist_upper_pct"] = pool["next_upper"]["pct"] if pool["next_upper"] else None
        pool["dist_lower_pct"] = pool["next_lower"]["pct"] if pool["next_lower"] else None

    return pools


def _chart_window(weekly_df: pd.DataFrame, pool: dict) -> tuple[int, int]:
    """History before the current lower, then a few candles after the check line."""
    n = len(weekly_df)
    formed_idx = int(pool["idx"])
    start = max(0, formed_idx - BUFFER_BEFORE)
    end = min(n, formed_idx + 1 + CANDLES_AFTER_LINE)
    return start, end


def _hline(ax, x0: float, x1: float, price: float, color: str, lw: float, ls: str, label: str) -> None:
    ax.plot([x0, x1], [price, price], color=color, linestyle=ls, lw=lw, alpha=0.95, zorder=5)
    ax.text(
        x1 + 0.4,
        price,
        label,
        color=color,
        va="center",
        ha="left",
        fontsize=8,
        fontweight="bold",
        zorder=6,
    )


def plot_one_weekly_liquidity(
    symbol: str,
    weekly_df: pd.DataFrame,
    pool: dict,
    seq: int,
    total: int,
    out_dir: Path,
) -> str:
    """Render one PNG: current weekly liquidity + next upper + next lower."""
    os.makedirs(out_dir, exist_ok=True)

    start, end = _chart_window(weekly_df, pool)
    span = weekly_df.iloc[start:end].copy()
    n_span = len(span)
    if n_span < 3:
        return ""

    formed_rel = int(pool["idx"]) - start
    formed_rel = min(max(formed_rel, 0), n_span - 1)
    price = pool["price"]
    date_str = pd.Timestamp(pool["formed_date"]).strftime("%Y-%m-%d")

    nxt_up = pool.get("next_upper")
    nxt_lo = pool.get("next_lower")
    up_pct = nxt_up["pct"] if nxt_up else None
    lo_pct = nxt_lo["pct"] if nxt_lo else None
    up_txt = f"INR {nxt_up['price']:.2f}  ({fmt_pct(up_pct)})" if nxt_up else "N/A"
    lo_txt = f"INR {nxt_lo['price']:.2f}  ({fmt_pct(lo_pct)})" if nxt_lo else "none"

    title = (
        f"{symbol}  Weekly LOWER Liquidity  {seq:03d}/{total:03d}   |   "
        f"CURRENT LOWER  INR {price:.2f}  ({date_str})\n"
        f"Next UPPER: {up_txt}     Next LOWER: {lo_txt}"
    )

    filename = f"Liq_{seq:04d}_LOWER_{date_str}_P{price:.2f}.png"
    out_file = out_dir / filename

    fig, axes = mpf.plot(
        span,
        type="candle",
        style="yahoo",
        title=title,
        volume=False,
        figratio=(16, 8),
        figscale=1.25,
        tight_layout=True,
        datetime_format="%d %b %Y",
        returnfig=True,
        warn_too_much_data=80,
    )
    ax = axes[0]
    x_right = n_span + 6
    ax.set_xlim(-1, x_right)

    vis_low = float(span["Low"].min())
    vis_high = float(span["High"].max())
    pad = max((vis_high - vis_low) * 0.12, vis_high * 0.01)
    y_lo, y_hi = vis_low - pad, vis_high + pad
    ax.set_ylim(y_lo, y_hi)

    def in_view(p: float) -> bool:
        return y_lo <= p <= y_hi

    # Vertical check line JUST AFTER the current lower candle (does not cover the wick)
    vline_x = formed_rel + 0.5
    ax.plot(
        [vline_x, vline_x], [y_lo, y_hi],
        color=CHECK_LINE_COLOR, linestyle=":", lw=1.6, alpha=0.95, zorder=3,
    )
    ax.text(
        vline_x + 0.15, y_hi, "after current",
        color=CHECK_LINE_COLOR, fontsize=7, fontweight="bold",
        va="top", ha="left", rotation=90,
    )

    _hline(
        ax, formed_rel, x_right - 1, price, CURRENT_COLOR, 2.4, "-",
        f"  CURRENT LOWER  {price:.2f}",
    )

    off_notes = []
    if nxt_up is not None:
        if in_view(nxt_up["price"]):
            _hline(
                ax, 0, x_right - 1, nxt_up["price"], UPPER_COLOR, 1.7, "--",
                f"  NEXT UPPER  {nxt_up['price']:.2f}  ({fmt_pct(up_pct)})",
            )
            mid_y = (price + nxt_up["price"]) / 2.0
            ax.text(
                n_span * 0.72, mid_y, f"Δ {fmt_pct(up_pct)}",
                color=UPPER_COLOR, fontsize=8, fontweight="bold",
                va="center", ha="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=UPPER_COLOR, alpha=0.9, lw=0.8),
            )
            up_rel = int(nxt_up["idx"]) - start
            if 0 <= up_rel < n_span:
                ax.scatter(
                    [up_rel], [nxt_up["price"]],
                    marker="v", s=90, color=UPPER_COLOR, zorder=7, edgecolors="white", linewidths=0.6,
                )
        else:
            off_notes.append(f"Next UPPER {nxt_up['price']:.2f} ({fmt_pct(up_pct)}) off chart")

    if nxt_lo is not None:
        if in_view(nxt_lo["price"]):
            _hline(
                ax, 0, x_right - 1, nxt_lo["price"], LOWER_COLOR, 1.7, "--",
                f"  NEXT LOWER  {nxt_lo['price']:.2f}  ({fmt_pct(lo_pct)})",
            )
            mid_y = (price + nxt_lo["price"]) / 2.0
            ax.text(
                n_span * 0.72, mid_y, f"Δ {fmt_pct(lo_pct)}",
                color=LOWER_COLOR, fontsize=8, fontweight="bold",
                va="center", ha="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=LOWER_COLOR, alpha=0.9, lw=0.8),
            )
            lo_rel = int(nxt_lo["idx"]) - start
            if 0 <= lo_rel < n_span:
                ax.scatter(
                    [lo_rel], [nxt_lo["price"]],
                    marker="^", s=90, color=LOWER_COLOR, zorder=7, edgecolors="white", linewidths=0.6,
                )
        else:
            off_notes.append(f"Next LOWER {nxt_lo['price']:.2f} ({fmt_pct(lo_pct)}) off chart")

    if off_notes:
        ax.text(
            0.99, 0.02, "  |  ".join(off_notes),
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="#6B7280", fontweight="bold",
        )

    # Current pivot marker on the lower candle
    ax.scatter(
        [formed_rel], [price],
        marker="^", s=140, color=PIVOT_MARKER_COLOR,
        zorder=8, edgecolors="white", linewidths=0.8,
    )

    handles = [
        mlines.Line2D([], [], color=CURRENT_COLOR, lw=2.4, label=f"Current LOWER  {price:.2f}"),
        mlines.Line2D([], [], color=UPPER_COLOR, lw=1.7, ls="--", label=f"Next Upper  {up_txt}"),
        mlines.Line2D([], [], color=LOWER_COLOR, lw=1.7, ls="--", label=f"Next Lower  {lo_txt}"),
        mlines.Line2D([], [], color=CHECK_LINE_COLOR, lw=1.6, ls=":", label="After current candle"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.92)

    fig.savefig(out_file, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(out_file)


def plot_weekly_liquidity_overview(
    symbol: str,
    weekly_df: pd.DataFrame,
    pools: list[dict],
    out_dir: Path,
    years: int = 5,
    filter_note: str = "",
) -> str:
    """Overview chart of every weekly upper/lower pool on the recent window."""
    os.makedirs(out_dir, exist_ok=True)
    end_date = weekly_df.index.max()
    start_date = end_date - pd.DateOffset(years=years)
    view = weekly_df.loc[weekly_df.index >= start_date].copy()
    if view.empty:
        view = weekly_df.copy()

    n = len(view)
    fig, axes = mpf.plot(
        view,
        type="candle",
        style="yahoo",
        title=f"{symbol}  Weekly Liquidity Overview  (last {years}y)  |  {len(pools)} pools{filter_note}",
        volume=False,
        figratio=(16, 8),
        figscale=1.2,
        tight_layout=True,
        datetime_format="%b %Y",
        returnfig=True,
        warn_too_much_data=5000,
    )
    ax = axes[0]
    ax.set_xlim(-1, n + 6)

    for pool in pools:
        formed = pool["formed_date"]
        if formed < view.index.min() or formed > view.index.max():
            continue
        pos = int(view.index.searchsorted(formed))
        pos = min(max(pos, 0), n - 1)
        color = UPPER_COLOR if pool["side"] == "upper" else LOWER_COLOR
        ax.plot(
            [pos, n + 4],
            [pool["price"], pool["price"]],
            color=color, ls="--", lw=1.0, alpha=0.55,
        )

    handles = [
        mlines.Line2D([], [], color=UPPER_COLOR, ls="--", lw=1.4, label="Upper liquidity (swing high)"),
        mlines.Line2D([], [], color=LOWER_COLOR, ls="--", lw=1.4, label="Lower liquidity (swing low)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8)

    out_file = out_dir / f"{symbol.replace('.NS', '')}_Weekly_Liquidity_Overview.png"
    fig.savefig(out_file, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(out_file)


def pools_to_frame(symbol: str, pools: list[dict]) -> pd.DataFrame:
    rows = []
    for i, p in enumerate(pools, start=1):
        nxt_up = p.get("next_upper") or {}
        nxt_lo = p.get("next_lower") or {}
        rows.append({
            "Seq": i,
            "Ticker": symbol,
            "Side": p["side"],
            "Price": round(p["price"], 4),
            "Formed_Date": pd.Timestamp(p["formed_date"]).strftime("%Y-%m-%d"),
            "Next_Upper_Price": round(nxt_up["price"], 4) if nxt_up else None,
            "Next_Upper_Date": pd.Timestamp(nxt_up["formed_date"]).strftime("%Y-%m-%d") if nxt_up else None,
            "Dist_Upper_Pct": round(p.get("dist_upper_pct"), 4) if p.get("dist_upper_pct") is not None else None,
            "Next_Lower_Price": round(nxt_lo["price"], 4) if nxt_lo else None,
            "Next_Lower_Date": pd.Timestamp(nxt_lo["formed_date"]).strftime("%Y-%m-%d") if nxt_lo else None,
            "Dist_Lower_Pct": round(p.get("dist_lower_pct"), 4) if p.get("dist_lower_pct") is not None else None,
        })
    return pd.DataFrame(rows)


def filter_current_lower(pools: list[dict], min_lower_abs_pct: float) -> list[dict]:
    """
    Keep CURRENT LOWER pools only, and only when:
      - there is no next lower liquidity, or
      - the next lower liquidity is more than `min_lower_abs_pct` away.
    """
    kept = []
    for p in pools:
        if p.get("side") != "lower":
            continue
        pct = p.get("dist_lower_pct")
        if pct is None or abs(float(pct)) > min_lower_abs_pct:
            kept.append(p)
    return kept


def _clear_plot_dir(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)


def build_weekly_liquidity(symbol: str) -> tuple[pd.DataFrame, list[dict]]:
    daily = load_stock_daily(symbol)
    weekly = to_weekly(daily)
    pools = attach_neighbor_liquidity(find_weekly_pivot_liquidity(weekly))
    pools.sort(key=lambda p: (pd.Timestamp(p["formed_date"]), p["side"]))
    return weekly, pools


def generate_weekly_liquidity_plots(
    symbol: str = "RELIANCE.NS",
    min_lower_abs_pct: float = 5.0,
) -> dict:
    """Find weekly liquidity pools and write 1 PNG per pool that passes the lower-gap filter."""
    weekly, pools = build_weekly_liquidity(symbol)
    clean = symbol.replace(".NS", "")
    out_dir = PLOTS_DIR / clean
    os.makedirs(REPORTS_DIR, exist_ok=True)
    _clear_plot_dir(out_dir)

    raw_total = len(pools)
    raw_lower = sum(1 for p in pools if p["side"] == "lower")
    pools = filter_current_lower(pools, min_lower_abs_pct)
    total = len(pools)
    no_lower = sum(1 for p in pools if p.get("dist_lower_pct") is None)
    print(
        f"{symbol}: {raw_total} weekly pools scanned ({raw_lower} lower), "
        f"{total} CURRENT LOWER kept "
        f"(no next lower: {no_lower}, |next lower| > {min_lower_abs_pct:.1f}%: {total - no_lower})"
    )

    overview = plot_weekly_liquidity_overview(
        symbol, weekly, pools, out_dir,
        filter_note=f"  (current LOWER only; none or |next lower| > {min_lower_abs_pct:.1f}%)",
    )
    print(f"Overview: {overview}")

    png_paths = []
    for seq, pool in enumerate(pools, start=1):
        path = plot_one_weekly_liquidity(symbol, weekly, pool, seq, total, out_dir)
        png_paths.append(path)
        if seq == 1 or seq == total or seq % 25 == 0:
            print(f"  [{seq}/{total}] {Path(path).name}")

    table = pools_to_frame(symbol, pools)
    table["PNG"] = [Path(p).name if p else "" for p in png_paths]
    csv_path = REPORTS_DIR / f"{clean}_Weekly_Liquidity.csv"
    table.to_csv(csv_path, index=False)
    print(f"Table: {csv_path}")
    print(f"PNGs : {out_dir}  ({len(png_paths)} files)")

    return {
        "symbol": symbol,
        "count": total,
        "raw_count": raw_total,
        "min_lower_abs_pct": min_lower_abs_pct,
        "out_dir": str(out_dir),
        "csv": str(csv_path),
        "overview": overview,
        "pngs": png_paths,
    }
