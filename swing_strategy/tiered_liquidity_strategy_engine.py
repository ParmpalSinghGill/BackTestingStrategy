"""
Tiered Liquidity Strategy Engine (No ML)

Liquidity:
- On sweep, if next support below is within 5% of current → use lower level only.
- C1: green candle, close below liquidity.
- C2: any candle after C1 (before C1 low breaks) where high > C1 high; entry at C1_High×1.001 (TouchPlannedEntry).
- C1 invalidated if any bar low < C1 low → search next C1.

Exits (partial):
- TP1 1:2 → 33% | TP2 1:3 → 33% | TP3 1:4 → remaining 34%
- SL → full remaining | Gap fills at open × 0.999
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DATA_DAILY_DIR = BASE_DIR / "data_daily"
REPORTS_DIR = BASE_DIR / "Reports" / "Tiered_Liquidity_Strategy"

from src.backtest_engine.backtest_support_liquidity_strategy import (
    INDEX_CLASSIFIER,
    get_all_stock_supports,
)

ENTRY_BUFFER = 0.001
GAP_EXIT_BUFFER = 0.001
LIQUIDITY_SKIP_PCT = 0.05
MAX_C2_WAIT = 40
MAX_POST_SWEEP = 90
LEG1_PCT = 0.33
LEG2_PCT = 0.33
LEG3_PCT = 0.34
TF_RANK = {"Yearly": 3, "Monthly": 2, "Weekly": 1}
NIFTY_RANK = {"Nifty 50": 4, "Nifty 100": 3, "Nifty 250": 2, "Other": 1}


def _load_daily(symbol: str) -> Optional[pd.DataFrame]:
    for name in (
        f"{symbol}_1d.csv",
        f"{symbol}.csv",
        f"{symbol.replace('.NS', '')}_1d.csv",
        f"{symbol.replace('.NS', '')}.csv",
    ):
        path = DATA_DAILY_DIR / name
        if path.exists():
            try:
                df = pd.read_csv(path)
                df["Date"] = pd.to_datetime(df["Date"])
                return df.sort_values("Date").reset_index(drop=True)
            except Exception:
                pass
    return None


def _resolve_effective_liquidity(
    swept_price: float,
    swept_tf: str,
    active: list[dict],
) -> tuple[float, str]:
    """If nearest *intact* support below is within 5%, trade the lower level instead.

    Already-swept levels are ignored. Using a taken low as the 'next support'
    silently substitutes old highs/doji prints (upper liquidity) for the real sweep.
    """
    below = [
        s for s in active
        if float(s["price"]) < swept_price and not s.get("swept", False)
    ]
    if not below:
        return swept_price, swept_tf
    nearest = max(below, key=lambda s: float(s["price"]))
    nb_price = float(nearest["price"])
    if swept_price > 0 and (swept_price - nb_price) / swept_price <= LIQUIDITY_SKIP_PCT:
        return nb_price, str(nearest["timeframe"])
    return swept_price, swept_tf


def _upper_liquidity_metrics(
    active: list[dict],
    eff_price: float,
    entry_price: float,
) -> dict:
    """Nearest support above traded liquidity — farther = more room before overhead level."""
    above = [s for s in active if float(s["price"]) > eff_price]
    if not above:
        return {
            "Upper_Liquidity_Price": 0.0,
            "Upper_Liquidity_TF": "None",
            "Upper_Liquidity_TF_Rank": 0,
            "Upper_Liquidity_Dist_Pct": 9999.0,
        }
    nearest = min(above, key=lambda s: float(s["price"]))
    up_price = float(nearest["price"])
    dist_pct = ((up_price - entry_price) / entry_price * 100.0) if entry_price > 0 else 0.0
    up_tf = str(nearest["timeframe"])
    return {
        "Upper_Liquidity_Price": round(up_price, 2),
        "Upper_Liquidity_TF": up_tf,
        "Upper_Liquidity_TF_Rank": TF_RANK.get(up_tf, 0),
        "Upper_Liquidity_Dist_Pct": round(dist_pct, 4),
    }


def _touch_entry(planned: float, o: float, h: float, l: float) -> Optional[float]:
    if o > planned:
        if l <= planned:
            return planned
        return None
    if h >= planned:
        return planned
    return None


def _gap_exit(o: float) -> float:
    return round(o * (1.0 - GAP_EXIT_BUFFER), 2)


def _simulate_tiered_exits(
    entry_idx: int,
    entry: float,
    sl: float,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    dates: np.ndarray,
    n: int,
    rr1: float = 2.0,
    rr2: float = 3.0,
    rr3: float = 4.0,
) -> list[dict]:
    risk = entry - sl
    if risk <= 0.05:
        return []
    tp1 = round(entry + rr1 * risk, 2)
    tp2 = round(entry + rr2 * risk, 2)
    tp3 = round(entry + rr3 * risk, 2)

    rem = 1.0
    leg1 = leg2 = leg3 = False
    legs: list[dict] = []

    for m in range(entry_idx, n):
        o, h, l = float(opens[m]), float(highs[m]), float(lows[m])
        dt = pd.Timestamp(dates[m]).strftime("%Y-%m-%d")

        if rem > 0 and o < sl:
            legs.append({"date": dt, "price": _gap_exit(o), "qty_pct": rem, "leg": "SL_GAP"})
            return legs

        if rem > 0 and o > tp3:
            legs.append({"date": dt, "price": _gap_exit(o), "qty_pct": rem, "leg": "TP3_GAP"})
            return legs

        if rem > 0 and o > tp2 and not leg2:
            pct = min(0.66 if not leg1 else LEG2_PCT, rem)
            legs.append({"date": dt, "price": _gap_exit(o), "qty_pct": pct, "leg": "TP2_GAP"})
            rem = round(rem - pct, 4)
            leg1 = leg2 = True
            if rem <= 0:
                return legs

        if rem > 0 and o > tp1 and not leg1:
            pct = min(LEG1_PCT, rem)
            legs.append({"date": dt, "price": _gap_exit(o), "qty_pct": pct, "leg": "TP1_GAP"})
            rem = round(rem - pct, 4)
            leg1 = True
            if rem <= 0:
                return legs

        if rem > 0 and l <= sl:
            legs.append({"date": dt, "price": sl, "qty_pct": rem, "leg": "SL"})
            return legs

        if not leg1 and rem > 0 and h >= tp1:
            pct = min(LEG1_PCT, rem)
            legs.append({"date": dt, "price": tp1, "qty_pct": pct, "leg": "TP1"})
            rem = round(rem - pct, 4)
            leg1 = True

        if not leg2 and rem > 0 and h >= tp2:
            pct = min(LEG2_PCT, rem)
            legs.append({"date": dt, "price": tp2, "qty_pct": pct, "leg": "TP2"})
            rem = round(rem - pct, 4)
            leg2 = True

        if rem > 0 and h >= tp3:
            legs.append({"date": dt, "price": tp3, "qty_pct": rem, "leg": "TP3"})
            return legs

    if rem > 0:
        last = n - 1
        legs.append({
            "date": pd.Timestamp(dates[last]).strftime("%Y-%m-%d"),
            "price": round(float(lows[last]), 2),
            "qty_pct": rem,
            "leg": "EOD",
        })
    return legs


def _resolve_setup(
    sweep_idx: int,
    liq_price: float,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    dates: np.ndarray,
    n: int,
    rr1: float = 2.0,
    rr2: float = 3.0,
    rr3: float = 4.0,
) -> Optional[tuple[int, int, float, float, float, float, list]]:
    """Returns c1_idx, entry_idx, entry, c1_high, c1_low, sl, exit_legs."""
    end = min(n, sweep_idx + MAX_POST_SWEEP)
    pos = sweep_idx

    while pos < end:
        if not (float(closes[pos]) > float(opens[pos]) and float(closes[pos]) < liq_price):
            pos += 1
            continue

        c1_idx = pos
        c1_high = float(highs[c1_idx])
        c1_low = float(lows[c1_idx])
        planned = round(c1_high * (1.0 + ENTRY_BUFFER), 2)
        sl = round(c1_low, 2)
        search_end = min(n, c1_idx + 1 + MAX_C2_WAIT)

        invalidated = False
        entry_idx = None
        entry_price = None

        for k in range(c1_idx + 1, search_end):
            if float(lows[k]) < c1_low:
                pos = k + 1
                invalidated = True
                break
            if float(highs[k]) > c1_high:
                fill = _touch_entry(planned, float(opens[k]), float(highs[k]), float(lows[k]))
                if fill is not None:
                    entry_idx = k
                    entry_price = fill
                    break

        if invalidated:
            continue
        if entry_idx is None:
            pos = search_end
            continue

        exit_legs = _simulate_tiered_exits(
            entry_idx, entry_price, sl, opens, highs, lows, dates, n, rr1, rr2, rr3
        )
        if not exit_legs:
            pos = entry_idx + 1
            continue

        return c1_idx, entry_idx, entry_price, c1_high, c1_low, sl, exit_legs

    return None


def scan_ticker(
    symbol: str,
    start_date: str = "2010-01-01",
    rr1: float = 2.0,
    rr2: float = 3.0,
    rr3: float = 4.0,
) -> list[dict]:
    df = _load_daily(symbol)
    if df is None or len(df) < 120:
        return []

    idx_tag = INDEX_CLASSIFIER.classify(symbol)
    all_supports = get_all_stock_supports(df.set_index("Date"))

    dates = df["Date"].to_numpy()
    opens = df["Open"].to_numpy(float)
    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    closes = df["Close"].to_numpy(float)
    n = len(df)
    start_dt = pd.to_datetime(start_date)

    sup_by_date: dict = {}
    for s in all_supports:
        sup_by_date.setdefault(s["formed_date"], []).append(s)

    active: list[dict] = []
    trades: list[dict] = []

    for i in range(n):
        curr_dt = pd.Timestamp(dates[i])

        if curr_dt in sup_by_date:
            for s in sup_by_date[curr_dt]:
                active.append({
                    "price": float(s["price"]),
                    "timeframe": s["timeframe"],
                    "formed_date": s["formed_date"],
                    "swept": False,
                })

        if curr_dt < start_dt:
            for sup in active:
                if not sup["swept"] and lows[i] < sup["price"]:
                    sup["swept"] = True
            continue

        for sup in list(active):
            if sup["swept"] or lows[i] >= sup["price"]:
                continue

            sup["swept"] = True
            eff_price, eff_tf = _resolve_effective_liquidity(
                float(sup["price"]), str(sup["timeframe"]), active
            )

            resolved = _resolve_setup(
                i, eff_price, opens, highs, lows, closes, dates, n, rr1, rr2, rr3
            )
            if resolved is None:
                continue

            c1_idx, entry_idx, entry_price, c1_high, c1_low, sl, exit_legs = resolved
            risk = entry_price - sl
            tp1 = round(entry_price + rr1 * risk, 2)
            tp2 = round(entry_price + rr2 * risk, 2)
            tp3 = round(entry_price + rr3 * risk, 2)
            rr_mode = f"{LEG1_PCT:.0%} 1:{rr1:g} / {LEG2_PCT:.0%} 1:{rr2:g} / {LEG3_PCT:.0%} 1:{rr3:g}"
            last_exit = exit_legs[-1]["date"]
            gross_pnl_per_share = sum(
                (leg["price"] - entry_price) * leg["qty_pct"] for leg in exit_legs
            )
            outcome = "Success" if gross_pnl_per_share > 0 else "Failure"
            upper = _upper_liquidity_metrics(active, eff_price, entry_price)

            trades.append({
                "Ticker": symbol,
                "Index_Membership": idx_tag,
                "Nifty_Rank": NIFTY_RANK.get(idx_tag, 1),
                "Liquidity_Type": eff_tf,
                "TF_Rank": TF_RANK.get(eff_tf, 1),
                "Support_Price": round(eff_price, 2),
                "Swept_Support_Price": round(float(sup["price"]), 2),
                "Upper_Liquidity_Price": upper["Upper_Liquidity_Price"],
                "Upper_Liquidity_TF": upper["Upper_Liquidity_TF"],
                "Upper_Liquidity_TF_Rank": upper["Upper_Liquidity_TF_Rank"],
                "Upper_Liquidity_Dist_Pct": upper["Upper_Liquidity_Dist_Pct"],
                "C1_Date": pd.Timestamp(dates[c1_idx]).strftime("%Y-%m-%d"),
                "C2_Date": pd.Timestamp(dates[entry_idx]).strftime("%Y-%m-%d"),
                "Entry_Date": pd.Timestamp(dates[entry_idx]).strftime("%Y-%m-%d"),
                "Exit_Date": last_exit,
                "C1_High": round(c1_high, 2),
                "C1_Low": round(c1_low, 2),
                "Planned_Entry_Price": round(c1_high * (1.0 + ENTRY_BUFFER), 2),
                "Entry_Price": round(entry_price, 2),
                "SL_Price": sl,
                "TP1_Price": tp1,
                "TP2_Price": tp2,
                "TP3_Price": tp3,
                "Outcome": outcome,
                "Target_RR_Mode": rr_mode,
                "Exit_Legs_JSON": json.dumps(exit_legs),
                "Exit_Leg_Count": len(exit_legs),
            })

    return trades


def _scan_worker(payload: tuple) -> list[dict]:
    symbol, start_date, rr1, rr2, rr3 = payload
    try:
        return scan_ticker(symbol, start_date, rr1, rr2, rr3)
    except Exception:
        return []


def scan_all_tickers(
    start_date: str = "2010-01-01",
    max_workers: int = 8,
    rr1: float = 2.0,
    rr2: float = 3.0,
    rr3: float = 4.0,
) -> pd.DataFrame:
    tickers = sorted({p.name.split("_1d.csv")[0] for p in DATA_DAILY_DIR.glob("*_1d.csv")})
    print(
        f"Scanning {len(tickers):,} tickers | 5% lower-liq | C1 green close-below | "
        f"exits {LEG1_PCT:.0%} 1:{rr1:g} / {LEG2_PCT:.0%} 1:{rr2:g} / {LEG3_PCT:.0%} 1:{rr3:g}...",
        flush=True,
    )
    all_trades: list[dict] = []

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_scan_worker, (t, start_date, rr1, rr2, rr3)): t for t in tickers
        }
        done = 0
        for fut in as_completed(futures):
            all_trades.extend(fut.result())
            done += 1
            if done % 200 == 0 or done == len(tickers):
                print(f"  • {done:,}/{len(tickers):,} | setups: {len(all_trades):,}", flush=True)

    if not all_trades:
        return pd.DataFrame()

    df = pd.DataFrame(all_trades)
    df["Entry_Date"] = pd.to_datetime(df["Entry_Date"])
    df = df.sort_values(["Entry_Date", "Ticker", "TF_Rank"], ascending=[True, True, False]).drop_duplicates(
        subset=["Entry_Date", "Ticker"], keep="first"
    )
    df = df.sort_values("Entry_Date").reset_index(drop=True)
    print(f"Deduplicated: {len(df):,} setups (Yearly > Monthly > Weekly).", flush=True)
    return df


def build_trades_dataset(
    start_date: str = "2010-01-01",
    force_rescan: bool = False,
    rr1: float = 2.0,
    rr2: float = 3.0,
    rr3: float = 4.0,
    cache_csv: Optional[Path] = None,
) -> pd.DataFrame:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = cache_csv or (REPORTS_DIR / "Tiered_Liquidity_Trades.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists() and not force_rescan:
        print(f"Loading cached: {out_csv}", flush=True)
        df = pd.read_csv(out_csv)
        df["Entry_Date"] = pd.to_datetime(df["Entry_Date"])
        return df
    df = scan_all_tickers(start_date=start_date, rr1=rr1, rr2=rr2, rr3=rr3)
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv} ({len(df):,})", flush=True)
    return df
