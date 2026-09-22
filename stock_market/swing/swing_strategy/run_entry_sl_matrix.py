"""
C1 / C2 entry / SL matrix — exclusive R buckets (no look-ahead mixing).

C1 (green candle) variants:
  close  — Close < liquidity
  high   — High  < liquidity
  open   — Open  < liquidity

Entry variants (C1 low must not break first):
  a1 — C2 Close > C1 High; fill next bar (C3) at C3 Open
  a2 — same C2; fill C3 Open only if C3 Open > C1 High
  a3 — same C2; fill C1_High * 1.01 on C3 if that price trades (no chase)
  b  — C2 Open inside C1 range, High > C1 High; fill C1_High * 1.01 on C2 (no gap-up)

SL variants:
  c1       — C1 Low
  c1x99    — C1 Low * 0.99
  sweep    — lowest low from sweep through C1 (bars that traded below liquidity)
  sweepx99 — that low * 0.99

Exclusive buckets: 1:10, 1:5, 1:4, 1:3, 1:2, 1:1, SL, Open_EOD
(e.g. Count_1to4 = reached 1:4 but not 1:5)

Capital sims: Rs 50k / Rs 500 and Rs 100k / Rs 500.
Qty = min(floor(risk_cap / (entry-SL)), floor(cash/entry));
if that is 0 but 1 share fits cash and 1-share risk <= risk_cap, buy 1.
Entries first; exit cash usable next day. Full exit at SL or 10R.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DATA_DAILY_DIR = BASE_DIR / "data_daily"
OUT_DIR = BASE_DIR / "Reports" / "Entry_SL_Matrix"
CACHE_TRADES = OUT_DIR / "All_Variant_Trades.csv"
OUT_SIGNAL = OUT_DIR / "Master_Signal_RR_Buckets.csv"
OUT_PORT = OUT_DIR / "Master_Portfolio_50k_100k.csv"

from src.analysis.indian_brokerage_calculator import calculate_indian_trade_charges
from src.backtest_engine.backtest_support_liquidity_strategy import (
    INDEX_CLASSIFIER,
    get_all_stock_supports,
)
from swing_strategy.tiered_liquidity_strategy_engine import (
    DATA_DAILY_DIR as _DD,
    _load_daily,
    _resolve_effective_liquidity,
)

assert _DD == DATA_DAILY_DIR

START = pd.Timestamp("2010-01-01")
MAX_C2_WAIT = 40
MAX_POST_SWEEP = 90
PLANNED_MULT = 1.01
GAP_EXIT = 0.999
TF_RANK = {"Yearly": 3, "Monthly": 2, "Weekly": 1}
NIFTY_RANK = {"Nifty 50": 4, "Nifty 100": 3, "Nifty 250": 2, "Other": 1}

C1_NAMES = ("close", "high", "open")
ENTRY_NAMES = ("a1", "a2", "a3", "b")
SL_NAMES = ("c1", "c1x99", "sweep", "sweepx99")
BUCKETS = ("1:10", "1:5", "1:4", "1:3", "1:2", "1:1", "SL", "Open_EOD")
CAPITALS = (
    {"exp": "Exp_50k_0.5k", "capital": 50_000.0, "risk": 500.0},
    {"exp": "Exp_100k_0.5k", "capital": 100_000.0, "risk": 500.0},
)


def _is_c1(c1: str, o: float, h: float, c: float, liq: float) -> bool:
    if c <= o:
        return False
    if c1 == "close":
        return c < liq
    if c1 == "high":
        return h < liq
    return o < liq


def _sweep_low(sweep_idx: int, c1_idx: int, lows: np.ndarray, liq: float) -> float:
    vals = [
        float(lows[j])
        for j in range(sweep_idx, c1_idx + 1)
        if float(lows[j]) < liq
    ]
    return min(vals) if vals else float(lows[c1_idx])


def _sl_price(kind: str, c1_low: float, sweep_lo: float) -> float:
    if kind == "c1":
        return round(c1_low, 2)
    if kind == "c1x99":
        return round(c1_low * 0.99, 2)
    if kind == "sweep":
        return round(sweep_lo, 2)
    return round(sweep_lo * 0.99, 2)


def _touch_planned(planned: float, o: float, h: float, l: float) -> Optional[float]:
    if o > planned:
        return planned if l <= planned else None
    if h >= planned:
        return planned
    return None


def _bucket(mfe: float, hit_sl: bool) -> str:
    if mfe >= 10.0 - 1e-12:
        return "1:10"
    if mfe >= 5.0 - 1e-12:
        return "1:5"
    if mfe >= 4.0 - 1e-12:
        return "1:4"
    if mfe >= 3.0 - 1e-12:
        return "1:3"
    if mfe >= 2.0 - 1e-12:
        return "1:2"
    if mfe >= 1.0 - 1e-12:
        return "1:1"
    if hit_sl:
        return "SL"
    return "Open_EOD"


def _mfe_and_exit(
    entry_idx: int,
    entry: float,
    sl: float,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    n: int,
) -> tuple[float, bool, int, float]:
    risk = entry - sl
    if risk <= 0.05:
        return 0.0, True, entry_idx, sl
    tp10 = entry + 10.0 * risk
    mfe = max(0.0, (float(highs[entry_idx]) - entry) / risk)
    if float(lows[entry_idx]) <= sl:
        return mfe, True, entry_idx, sl
    if float(highs[entry_idx]) >= tp10:
        return max(mfe, 10.0), False, entry_idx, round(tp10, 2)

    for m in range(entry_idx + 1, n):
        o = float(opens[m])
        if o < sl:
            return mfe, True, m, round(o * GAP_EXIT, 2)
        if o > tp10:
            return max(mfe, 10.0), False, m, round(o * GAP_EXIT, 2)
        mfe = max(mfe, (float(highs[m]) - entry) / risk)
        if float(lows[m]) <= sl:
            return mfe, True, m, sl
        if float(highs[m]) >= tp10:
            return max(mfe, 10.0), False, m, round(tp10, 2)
    last = n - 1
    return mfe, False, last, round(float(closes[last]), 2)


def _try_a_entries(
    c2: int,
    c1_high: float,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    n: int,
) -> dict[str, tuple[int, float]]:
    out: dict[str, tuple[int, float]] = {}
    c3 = c2 + 1
    if c3 >= n:
        return out
    o, h, l = float(opens[c3]), float(highs[c3]), float(lows[c3])
    out["a1"] = (c3, round(o, 2))
    if o > c1_high:
        out["a2"] = (c3, round(o, 2))
    planned = round(c1_high * PLANNED_MULT, 2)
    fill = _touch_planned(planned, o, h, l)
    if fill is not None:
        out["a3"] = (c3, round(fill, 2))
    return out


def _try_b_entry(
    c2: int,
    c1_high: float,
    c1_low: float,
    opens: np.ndarray,
    highs: np.ndarray,
) -> Optional[tuple[int, float]]:
    o, h = float(opens[c2]), float(highs[c2])
    if not (c1_low <= o <= c1_high and h > c1_high):
        return None
    planned = round(c1_high * PLANNED_MULT, 2)
    if o > planned:
        return None
    if h < planned:
        return None
    return c2, planned


def scan_ticker(symbol: str) -> list[dict]:
    df = _load_daily(symbol)
    if df is None or len(df) < 120:
        return []

    idx_tag = INDEX_CLASSIFIER.classify(symbol)
    nifty_rank = NIFTY_RANK.get(idx_tag, 1)
    all_supports = get_all_stock_supports(df.set_index("Date"))

    dates = df["Date"].to_numpy()
    opens = df["Open"].to_numpy(float)
    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    closes = df["Close"].to_numpy(float)
    n = len(df)

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
                    "swept": False,
                })

        if curr_dt < START:
            for sup in active:
                if not sup["swept"] and lows[i] < sup["price"]:
                    sup["swept"] = True
            continue

        for sup in list(active):
            if sup["swept"] or lows[i] >= sup["price"]:
                continue
            sup["swept"] = True
            liq, tf = _resolve_effective_liquidity(
                float(sup["price"]), str(sup["timeframe"]), active
            )
            tf_rank = TF_RANK.get(tf, 1)
            end = min(n, i + MAX_POST_SWEEP)

            for c1_name in C1_NAMES:
                pos = i
                got_a = got_b = False
                while pos < end and not (got_a and got_b):
                    o, h, c = float(opens[pos]), float(highs[pos]), float(closes[pos])
                    if not _is_c1(c1_name, o, h, c, liq):
                        pos += 1
                        continue

                    c1_idx = pos
                    c1_high = float(highs[c1_idx])
                    c1_low = float(lows[c1_idx])
                    search_end = min(n, c1_idx + 1 + MAX_C2_WAIT)
                    c2_a = None
                    c2_b = None
                    invalidated = False

                    for k in range(c1_idx + 1, search_end):
                        if float(lows[k]) < c1_low:
                            pos = k + 1
                            invalidated = True
                            break
                        if c2_a is None and float(closes[k]) > c1_high:
                            c2_a = k
                        if (
                            c2_b is None
                            and c1_low <= float(opens[k]) <= c1_high
                            and float(highs[k]) > c1_high
                        ):
                            c2_b = k
                        if c2_a is not None and c2_b is not None:
                            break

                    if invalidated:
                        continue
                    if c2_a is None and c2_b is None:
                        pos = search_end
                        continue

                    sweep_lo = _sweep_low(i, c1_idx, lows, liq)
                    fills: dict[str, tuple[int, float]] = {}
                    if c2_a is not None and not got_a:
                        fills.update(_try_a_entries(c2_a, c1_high, opens, highs, lows, n))
                        if fills:
                            got_a = True
                    if c2_b is not None and not got_b:
                        bfill = _try_b_entry(c2_b, c1_high, c1_low, opens, highs)
                        if bfill is not None:
                            fills["b"] = bfill
                            got_b = True

                    if not fills:
                        pos = c1_idx + 1
                        continue

                    c1_date = pd.Timestamp(dates[c1_idx]).strftime("%Y-%m-%d")
                    for em, (eidx, entry) in fills.items():
                        for sln in SL_NAMES:
                            sl = _sl_price(sln, c1_low, sweep_lo)
                            if entry - sl <= 0.05:
                                continue
                            mfe, hit_sl, xidx, xpx = _mfe_and_exit(
                                eidx, entry, sl, opens, highs, lows, closes, n
                            )
                            trades.append({
                                "Variant": f"{c1_name}_{em}_{sln}",
                                "C1_Filter": c1_name,
                                "Entry_Method": em,
                                "SL_Method": sln,
                                "Ticker": symbol,
                                "Index_Membership": idx_tag,
                                "Nifty_Rank": nifty_rank,
                                "Liquidity_Type": tf,
                                "TF_Rank": tf_rank,
                                "C1_Date": c1_date,
                                "Entry_Date": pd.Timestamp(dates[eidx]).strftime("%Y-%m-%d"),
                                "Exit_Date": pd.Timestamp(dates[xidx]).strftime("%Y-%m-%d"),
                                "Entry_Price": entry,
                                "SL_Price": sl,
                                "Exit_Price": xpx,
                                "MFE_R": round(mfe, 4),
                                "Hit_SL": int(hit_sl),
                                "Best_Bucket": _bucket(mfe, hit_sl),
                            })
                    pos = c1_idx + 1
    return trades


def _worker(symbol: str) -> list[dict]:
    try:
        return scan_ticker(symbol)
    except Exception:
        return []


def _signal_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (c1, em, sln), g in df.groupby(["C1_Filter", "Entry_Method", "SL_Method"], sort=False):
        n = len(g)
        vc = g["Best_Bucket"].value_counts()
        row = {
            "C1_Filter": c1,
            "Entry_Method": em,
            "SL_Method": sln,
            "Variant": f"{c1}_{em}_{sln}",
            "Total_Possible": n,
        }
        for b in BUCKETS:
            row[f"Count_{b.replace(':', 'to')}"] = int(vc.get(b, 0))
        for b in BUCKETS:
            key = f"Pct_{b.replace(':', 'to')}"
            row[key] = round(100.0 * row[f"Count_{b.replace(':', 'to')}"] / n, 2) if n else 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    out = out.sort_values("Pct_1to10", ascending=False).reset_index(drop=True)
    return out


def _selection_key(row: dict) -> tuple:
    return (-int(row["TF_Rank"]), -int(row["Nifty_Rank"]))


def _run_portfolio(df_var: pd.DataFrame, capital: float, risk_cap: float) -> dict:
    if df_var.empty:
        return {
            "Executed": 0, "Final_Equity": capital, "Net_Return_Pct": 0.0,
            "CAGR_Pct": 0.0, "Win_Rate_Pct": 0.0, "Max_DD_Pct": 0.0,
            **{f"Exec_{b.replace(':', 'to')}": 0 for b in BUCKETS},
        }

    df = df_var.copy()
    df["Entry_Date"] = pd.to_datetime(df["Entry_Date"])
    df["Exit_Date"] = pd.to_datetime(df["Exit_Date"])
    by_entry: dict = defaultdict(list)
    for rec in df.to_dict("records"):
        by_entry[rec["Entry_Date"]].append(rec)

    min_dt = START
    max_dt = max(df["Entry_Date"].max(), df["Exit_Date"].max())
    all_days = pd.date_range(min_dt, max_dt, freq="D")

    cash = capital
    pending = 0.0
    peak = capital
    max_dd = 0.0
    open_pos: dict[int, dict] = {}
    pid = 0
    executed = 0
    wins = 0
    bucket_counts = {b: 0 for b in BUCKETS}
    by_exit: dict = defaultdict(list)

    for day in all_days:
        cash += pending
        pending = 0.0

        cands = by_entry.get(day, [])
        cands.sort(key=_selection_key)
        available = cash
        for cand in cands:
            entry = float(cand["Entry_Price"])
            sl = float(cand["SL_Price"])
            risk_ps = entry - sl
            if risk_ps <= 0.05:
                continue
            qty_risk = int(risk_cap / risk_ps)
            qty_cash = int(available / entry) if entry > 0 else 0
            qty = min(qty_risk, qty_cash)
            if qty < 1:
                if available >= entry and risk_ps <= risk_cap:
                    qty = 1
                else:
                    continue
            spend = qty * entry
            if spend > available or qty < 1:
                continue
            available -= spend
            cash -= spend
            pid += 1
            executed += 1
            bucket_counts[cand["Best_Bucket"]] += 1
            pos = {
                "qty": qty,
                "entry": entry,
                "exit_px": float(cand["Exit_Price"]),
                "exit_dt": cand["Exit_Date"],
                "bucket": cand["Best_Bucket"],
            }
            open_pos[pid] = pos
            by_exit[cand["Exit_Date"]].append(pid)

        holding = sum(p["qty"] * p["entry"] for p in open_pos.values())
        equity = cash + holding
        if equity > peak:
            peak = equity
        if peak > 0:
            dd = (peak - equity) / peak * 100.0
            if dd > max_dd:
                max_dd = dd

        for tid in by_exit.get(day, []):
            pos = open_pos.pop(tid, None)
            if pos is None:
                continue
            ch = calculate_indian_trade_charges(pos["entry"], pos["exit_px"], pos["qty"], 0.0)
            proceeds = pos["qty"] * pos["exit_px"]
            pending += proceeds - ch["total_charges"]
            if ch["net_pnl"] > 0:
                wins += 1

    cash += pending
    holding = sum(p["qty"] * p["entry"] for p in open_pos.values())
    final_eq = cash + holding
    years = max((max_dt - min_dt).days / 365.25, 1e-9)
    ret = (final_eq - capital) / capital * 100.0
    if final_eq <= 0:
        cagr = -100.0
    else:
        cagr = ((final_eq / capital) ** (1.0 / years) - 1.0) * 100.0
    wr = (100.0 * wins / executed) if executed else 0.0
    out = {
        "Executed": executed,
        "Final_Equity": round(final_eq, 2),
        "Net_Return_Pct": round(ret, 2),
        "CAGR_Pct": round(cagr, 2),
        "Win_Rate_Pct": round(wr, 2),
        "Max_DD_Pct": round(max_dd, 2),
    }
    for b in BUCKETS:
        out[f"Exec_{b.replace(':', 'to')}"] = bucket_counts[b]
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tickers = sorted({p.name.split("_1d.csv")[0] for p in DATA_DAILY_DIR.glob("*_1d.csv")})
    print(f"Scanning {len(tickers):,} tickers | 3 C1 x 4 entry x 4 SL = 48 variants", flush=True)

    all_trades: list[dict] = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_worker, t): t for t in tickers}
        done = 0
        for fut in as_completed(futs):
            all_trades.extend(fut.result())
            done += 1
            if done % 200 == 0 or done == len(tickers):
                print(f"  • {done:,}/{len(tickers):,} | rows {len(all_trades):,}", flush=True)

    if not all_trades:
        print("No trades found.", flush=True)
        return

    df = pd.DataFrame(all_trades)
    df = df.sort_values(["Variant", "Entry_Date", "Ticker"]).reset_index(drop=True)
    df.to_csv(CACHE_TRADES, index=False)
    print(f"Saved trades: {CACHE_TRADES} ({len(df):,})", flush=True)

    sig = _signal_table(df)
    sig.to_csv(OUT_SIGNAL, index=False)
    print(f"Saved signal buckets: {OUT_SIGNAL}", flush=True)
    print(sig.to_string(index=False), flush=True)

    port_rows = []
    variants = list(df.groupby(["C1_Filter", "Entry_Method", "SL_Method"], sort=False))
    print(f"\nPortfolio sims: {len(variants)} variants x {len(CAPITALS)} capitals...", flush=True)
    for i, ((c1, em, sln), g) in enumerate(variants, 1):
        g = g.drop_duplicates(subset=["Entry_Date", "Ticker"], keep="first")
        row = {"C1_Filter": c1, "Entry_Method": em, "SL_Method": sln, "Variant": f"{c1}_{em}_{sln}"}
        for cap in CAPITALS:
            m = _run_portfolio(g, cap["capital"], cap["risk"])
            prefix = cap["exp"].replace("Exp_", "")
            for k, v in m.items():
                row[f"{prefix}_{k}"] = v
        port_rows.append(row)
        if i % 8 == 0 or i == len(variants):
            print(f"  • portfolio {i}/{len(variants)}", flush=True)

    pdf = pd.DataFrame(port_rows)
    pdf = pdf.sort_values("50k_0.5k_CAGR_Pct", ascending=False).reset_index(drop=True)
    pdf.to_csv(OUT_PORT, index=False)
    print(f"Saved portfolio: {OUT_PORT}", flush=True)
    cols = [
        "Variant", "50k_0.5k_Executed", "50k_0.5k_Final_Equity", "50k_0.5k_Net_Return_Pct",
        "50k_0.5k_CAGR_Pct", "100k_0.5k_Executed", "100k_0.5k_Final_Equity",
        "100k_0.5k_Net_Return_Pct", "100k_0.5k_CAGR_Pct",
    ]
    print(pdf[cols].to_string(index=False), flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
