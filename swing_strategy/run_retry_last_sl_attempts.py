"""
Retry chains: if attempt 1 hits SL before 1:2, last SL becomes next liquidity.
When that level is swept and C1/C2 form again, enter attempt 2, then 3.

Reports which RR (1:2 / 1:3 / 1:4 / 1:5 / 1:10) first appeared on attempt 1, 2, 3, or never.
Uses the same 48 C1/entry/SL variants as the MFE matrix.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.backtest_engine.backtest_support_liquidity_strategy import (
    get_all_stock_supports,
)
from swing_strategy.run_c1_entry_sl_matrix import (
    C1_KINDS,
    ENT_KINDS,
    MAX_POST_SWEEP,
    SL_KINDS,
    START,
    _c1_match,
    _entries_for_c1,
    _mfe,
    _sl_price,
)
from swing_strategy.tiered_liquidity_strategy_engine import (
    DATA_DAILY_DIR,
    TF_RANK,
    _load_daily,
    _resolve_effective_liquidity,
)

OUT_DIR = BASE_DIR / "Reports" / "C1_Entry_SL_Matrix"
OUT_CSV = OUT_DIR / "Retry_SL_then_LastSL_Attempts.csv"
MAX_RETRY_WAIT = 252
RRS = (2, 3, 4, 5, 10)


def _empty() -> dict[str, int]:
    s = {
        "First_Attempts": 0,
        "Fail_SL_before_1to2": 0,
        "Attempt2_Entries": 0,
        "Attempt3_Entries": 0,
        "No_2nd_Setup": 0,
        "No_3rd_Setup": 0,
    }
    for r in RRS:
        s[f"R1to{r}_Att1"] = 0
        s[f"R1to{r}_Att2"] = 0
        s[f"R1to{r}_Att3"] = 0
        s[f"R1to{r}_Never"] = 0
    return s


def _try_setup(
    sweep_idx: int,
    support: float,
    ck: str,
    ek: str,
    sk: str,
    opens,
    highs,
    lows,
    closes,
    n: int,
) -> tuple[float, float, float, bool, int, int] | None:
    end = min(n - 1, sweep_idx + MAX_POST_SWEEP)
    for c1 in range(sweep_idx, end):
        if not _c1_match(ck, float(opens[c1]), float(highs[c1]), float(closes[c1]), support):
            continue
        ents = _entries_for_c1(c1, opens, highs, lows, closes, n)
        if ek not in ents:
            continue
        eidx, entry = ents[ek]
        if entry <= 0:
            continue
        c1_low = float(lows[c1])
        sweep_low = float(min(lows[sweep_idx : c1 + 1]))
        sl = _sl_price(sk, c1_low, sweep_low)
        if entry - sl <= 0.05:
            continue
        mfe, hit_sl, xidx = _mfe(eidx, entry, sl, opens, highs, lows, n)
        return float(entry), float(sl), float(mfe), bool(hit_sl), int(xidx), int(eidx)
    return None


def _next_sweep(start: int, level: float, lows, n: int) -> int | None:
    end = min(n, start + MAX_RETRY_WAIT)
    for j in range(start, end):
        if float(lows[j]) < level:
            return j
    return None


def _follow_retries(
    sl: float,
    xidx: int,
    ck: str,
    ek: str,
    sk: str,
    opens,
    highs,
    lows,
    closes,
    n: int,
) -> list[float]:
    """Return MFE list for attempt 2 and 3 (may be length 0, 1, or 2). Uses last SL as liquidity."""
    mfes: list[float] = []
    search_from = xidx + 1
    last_sl = sl
    for _ in range(2):
        sw = _next_sweep(search_from, last_sl, lows, n)
        if sw is None:
            break
        got = _try_setup(sw, last_sl, ck, ek, sk, opens, highs, lows, closes, n)
        if got is None:
            break
        _entry, new_sl, mfe, hit_sl, nx, _eidx = got
        mfes.append(mfe)
        if not (hit_sl and mfe < 2.0):
            break
        last_sl = new_sl
        search_from = nx + 1
    return mfes


def scan_ticker(symbol: str) -> dict[str, dict[str, int]]:
    df = _load_daily(symbol)
    if df is None or len(df) < 120:
        return {}

    all_supports = get_all_stock_supports(df.set_index("Date"))
    dates_np = df["Date"].to_numpy()
    date_s = pd.to_datetime(df["Date"])
    opens = df["Open"].to_numpy(float)
    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    closes = df["Close"].to_numpy(float)
    n = len(df)

    sup_by_date: dict = {}
    for s in all_supports:
        sup_by_date.setdefault(s["formed_date"], []).append(s)

    active: list[dict] = []
    firsts: list[tuple] = []

    for i in range(n):
        curr_dt = pd.Timestamp(dates_np[i])
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
            support, tf = _resolve_effective_liquidity(
                float(sup["price"]), str(sup["timeframe"]), active
            )
            tf_rank = TF_RANK.get(tf, 1)
            for ck in C1_KINDS:
                for ek in ENT_KINDS:
                    for sk in SL_KINDS:
                        got = _try_setup(i, support, ck, ek, sk, opens, highs, lows, closes, n)
                        if got is None:
                            continue
                        entry, sl, mfe, hit_sl, xidx, eidx = got
                        firsts.append((
                            f"{ck}|{ek}|{sk}",
                            date_s.iloc[eidx].strftime("%Y-%m-%d"),
                            tf_rank,
                            sl,
                            mfe,
                            hit_sl,
                            xidx,
                            ck,
                            ek,
                            sk,
                        ))

    firsts.sort(key=lambda t: (t[0], t[1], -t[2]))
    seen: set[tuple] = set()
    stats: dict[str, dict[str, int]] = defaultdict(_empty)

    for vk, edate, _tf, sl, mfe1, hit_sl, xidx, ck, ek, sk in firsts:
        key = (vk, edate)
        if key in seen:
            continue
        seen.add(key)
        st = stats[vk]
        st["First_Attempts"] += 1

        failed_12 = bool(hit_sl and mfe1 < 2.0)
        if failed_12:
            st["Fail_SL_before_1to2"] += 1

        mfe2 = mfe3 = None
        if failed_12:
            extra = _follow_retries(sl, xidx, ck, ek, sk, opens, highs, lows, closes, n)
            if not extra:
                st["No_2nd_Setup"] += 1
            else:
                mfe2 = extra[0]
                st["Attempt2_Entries"] += 1
                if mfe2 < 2.0:
                    if len(extra) >= 2:
                        mfe3 = extra[1]
                        st["Attempt3_Entries"] += 1
                    else:
                        st["No_3rd_Setup"] += 1

        for r in RRS:
            if mfe1 >= r:
                st[f"R1to{r}_Att1"] += 1
            elif mfe2 is not None and mfe2 >= r:
                st[f"R1to{r}_Att2"] += 1
            elif mfe3 is not None and mfe3 >= r:
                st[f"R1to{r}_Att3"] += 1
            else:
                st[f"R1to{r}_Never"] += 1

    return dict(stats)


def _scan_worker(symbol: str) -> dict[str, dict[str, int]]:
    try:
        return scan_ticker(symbol)
    except Exception:
        return {}


def _merge(dst: dict[str, dict[str, int]], src: dict[str, dict[str, int]]) -> None:
    for vk, st in src.items():
        if vk not in dst:
            dst[vk] = _empty()
        for k, v in st.items():
            dst[vk][k] += v


def main() -> None:
    tickers = sorted({p.name.split("_1d.csv")[0] for p in DATA_DAILY_DIR.glob("*_1d.csv")})
    print(
        f"Retry chains on {len(tickers):,} tickers | SL before 1:2 → last SL is next liquidity",
        flush=True,
    )
    merged: dict[str, dict[str, int]] = {}
    done = 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_scan_worker, t): t for t in tickers}
        for fut in as_completed(futs):
            _merge(merged, fut.result())
            done += 1
            if done % 200 == 0 or done == len(tickers):
                tot = sum(v["First_Attempts"] for v in merged.values())
                print(f"  • {done:,}/{len(tickers):,} | first attempts {tot:,}", flush=True)

    rows = []
    for ck in C1_KINDS:
        for ek in ENT_KINDS:
            for sk in SL_KINDS:
                vk = f"{ck}|{ek}|{sk}"
                st = merged.get(vk, _empty())
                n = st["First_Attempts"]
                row = {"C1": ck, "Entry": ek, "SL": sk, **st}
                if n:
                    row["Pct_Fail_SL_before_1to2"] = round(100 * st["Fail_SL_before_1to2"] / n, 2)
                    row["Pct_Got_Attempt2"] = round(100 * st["Attempt2_Entries"] / n, 2)
                    for r in RRS:
                        row[f"Pct_R1to{r}_Att1"] = round(100 * st[f"R1to{r}_Att1"] / n, 2)
                        row[f"Pct_R1to{r}_Never"] = round(100 * st[f"R1to{r}_Never"] / n, 2)
                else:
                    row["Pct_Fail_SL_before_1to2"] = 0.0
                    row["Pct_Got_Attempt2"] = 0.0
                    for r in RRS:
                        row[f"Pct_R1to{r}_Att1"] = 0.0
                        row[f"Pct_R1to{r}_Never"] = 0.0
                rows.append(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}", flush=True)

    show_cols = [
        "C1", "Entry", "SL", "First_Attempts", "Fail_SL_before_1to2",
        "Attempt2_Entries", "Attempt3_Entries",
        "R1to2_Att1", "R1to2_Att2", "R1to2_Att3", "R1to2_Never",
        "R1to5_Att1", "R1to5_Att2", "R1to5_Att3", "R1to5_Never",
        "R1to10_Att1", "R1to10_Att2", "R1to10_Att3", "R1to10_Never",
    ]
    top = df.sort_values("First_Attempts", ascending=False)
    print("\nRETRY CHAINS (sorted by first attempts)\n", flush=True)
    print(top[show_cols].to_string(index=False), flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
