"""
Rescan setups with swing-low HTF liquidity (strategy/LIQUIDITY.md),
then walk-forward meta + vol-managed eval on the three live books.

N=2/N2=2, or 3 on one side, with candles on both sides. At least two HTF
bars after the low; the next bar is never a trade bar. Optimize M2 in
{0, 1, 2}. C1 / C2 / C3 / SL / 1:2 / ML unchanged.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.backtest_engine.backtest_support_liquidity_strategy import (
    INDEX_CLASSIFIER,
    SWING_N2_DEFAULT,
    SWING_N_DEFAULT,
    SWING_SKIP_AFTER,
    find_htf_swing_lows,
    get_swing_low_supports,
)
from swing_strategy.run_c1_entry_sl_matrix import START
from swing_strategy.run_ml_next_search import META_FEAT, select_meta, walk_meta
from swing_strategy.run_ml_target_books import BOOKS, CORE_COLS, feat_worker, walk
from swing_strategy.run_ml_wave3 import run_vol_managed
from swing_strategy.run_retry2_rr_returns import _rr_exit, _try_setup
from swing_strategy.tiered_liquidity_strategy_engine import (
    DATA_DAILY_DIR,
    LIQUIDITY_SKIP_PCT,
    NIFTY_RANK,
    TF_RANK,
    _load_daily,
)

OUT = BASE_DIR / "Reports" / "SwingLowLiquidity_v3"
SUMMARY = OUT / "M2_Comparison.json"
M2_VALUES = (0, 1, 2)


def _first_idx_after(dates: pd.DatetimeIndex, ts) -> int | None:
    idx = int(dates.searchsorted(pd.Timestamp(ts).normalize(), side="right"))
    if idx >= len(dates):
        return None
    return idx


def _resolve_valid_below(
    swept_price: float,
    swept_tf: str,
    active: list[dict],
    m2: int,
) -> tuple[float, str]:
    below = [
        s
        for s in active
        if float(s["price"]) < swept_price
        and not s.get("gone", False)
        and int(s.get("sweep_count", 0)) <= m2
    ]
    if not below:
        return swept_price, swept_tf
    nearest = max(below, key=lambda s: float(s["price"]))
    nb_price = float(nearest["price"])
    if swept_price > 0 and (swept_price - nb_price) / swept_price <= LIQUIDITY_SKIP_PCT:
        return nb_price, str(nearest["timeframe"])
    return swept_price, swept_tf


def _precount_state(lows, closes, start: int | None, end: int, price: float) -> tuple[int, bool]:
    if start is None or start > end:
        return 0, False
    sweeps = 0
    for j in range(start, end + 1):
        if float(closes[j]) < price:
            return sweeps, True
        if float(lows[j]) < price:
            sweeps += 1
    return sweeps, False


def _dedupe_day(rows: list[dict]) -> list[dict]:
    rows.sort(key=lambda t: (t["Entry_Date"], -t["TF_Rank"]))
    kept, seen = [], set()
    for t in rows:
        k = t["Entry_Date"]
        if k in seen:
            continue
        seen.add(k)
        kept.append(t)
    return kept


def scan_ticker_swing(symbol: str) -> dict[int, list[dict]]:
    empty = {m2: [] for m2 in M2_VALUES}
    df = _load_daily(symbol)
    if df is None or len(df) < 120:
        return empty
    nifty = NIFTY_RANK.get(INDEX_CLASSIFIER.classify(symbol), 1)
    indexed = df.set_index("Date")
    swings = get_swing_low_supports(indexed, n=SWING_N_DEFAULT, n2=SWING_N2_DEFAULT)
    if not swings:
        return empty

    date_s = pd.to_datetime(df["Date"]).dt.normalize()
    dates = pd.DatetimeIndex(date_s)
    opens = df["Open"].to_numpy(float)
    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    closes = df["Close"].to_numpy(float)
    n = len(df)

    activate_at: dict[int, list[dict]] = {}
    for s in swings:
        aidx = _first_idx_after(dates, s["confirm_date"])
        if aidx is None:
            continue
        start = _first_idx_after(dates, s["liquidity_date"])
        sweeps, gone = _precount_state(lows, closes, start, aidx - 1, float(s["price"]))
        src = pd.Timestamp(s.get("source_date", s["liquidity_date"])).normalize()
        src_i = int(dates.searchsorted(src, side="left"))
        if src_i < n and dates[src_i] == src:
            trade_from = src_i + 1 + SWING_SKIP_AFTER
        else:
            trade_from = (start if start is not None else aidx) + SWING_SKIP_AFTER
        activate_at.setdefault(aidx, []).append({
            "price": float(s["price"]),
            "timeframe": s["timeframe"],
            "liquidity_date": pd.Timestamp(s["liquidity_date"]).normalize(),
            "source_date": src,
            "sweep_count": sweeps,
            "gone": gone,
            "trade_from": int(trade_from),
        })

    active: list[dict] = []
    raw: dict[int, list[dict]] = {m2: [] for m2 in M2_VALUES}

    for i in range(n):
        if i in activate_at:
            active.extend(activate_at[i])
        low_i = float(lows[i])
        close_i = float(closes[i])
        tradeable = date_s.iloc[i] >= START
        for sup in active:
            if sup["gone"]:
                continue
            price = float(sup["price"])
            if close_i < price:
                sup["gone"] = True
                continue
            if low_i >= price:
                continue
            prior = int(sup["sweep_count"])
            if tradeable and i >= int(sup.get("trade_from", 0)):
                for m2 in M2_VALUES:
                    if prior > m2:
                        continue
                    support, tf = _resolve_valid_below(price, str(sup["timeframe"]), active, m2)
                    got = _try_setup(i, support, opens, highs, lows, closes, n)
                    if got is None:
                        continue
                    eidx, entry, sl, _c1 = got
                    risk = entry - sl
                    if risk <= 0.05:
                        continue
                    exit_p, xidx, mfe, _hit_sl, reason = _rr_exit(
                        eidx, entry, sl, 2.0, opens, highs, lows, n
                    )
                    raw[m2].append({
                        "Ticker": symbol,
                        "Liquidity_Type": tf,
                        "TF_Rank": TF_RANK.get(tf, 1),
                        "Nifty_Rank": nifty,
                        "Support_Price": round(support, 2),
                        "Liquidity_Date": sup["liquidity_date"].strftime("%Y-%m-%d"),
                        "Sweep_Count": prior,
                        "Entry_Date": date_s.iloc[eidx].strftime("%Y-%m-%d"),
                        "Exit_Date": date_s.iloc[xidx].strftime("%Y-%m-%d"),
                        "Entry_Price": round(entry, 2),
                        "SL_Price": round(sl, 2),
                        "Exit_Price": round(exit_p, 2),
                        "Realized_R": round(float((exit_p - entry) / risk), 4),
                        "MFE_R": round(float(mfe), 4),
                        "Outcome": "Success" if reason == "TP" else "Failure",
                        "Target_RR_Mode": "1:2",
                    })
            sup["sweep_count"] = prior + 1

    return {m2: _dedupe_day(raw[m2]) for m2 in M2_VALUES}


def _worker(symbol: str) -> dict[int, list[dict]]:
    try:
        return scan_ticker_swing(symbol)
    except Exception:
        return {m2: [] for m2 in M2_VALUES}


def scan_setups() -> dict[int, pd.DataFrame]:
    existing = {}
    all_present = True
    for m2 in M2_VALUES:
        path = OUT / f"All_Setups_RR2_M2{m2}.csv"
        if path.exists():
            existing[m2] = pd.read_csv(path)
        else:
            all_present = False
    if all_present:
        for m2, df in existing.items():
            print(f"Loading {OUT / f'All_Setups_RR2_M2{m2}.csv'} n={len(df):,}", flush=True)
        return existing

    tickers = sorted({p.name.split("_1d.csv")[0] for p in DATA_DAILY_DIR.glob("*_1d.csv")})
    print(
        f"Scanning {len(tickers):,} tickers "
        f"(swing-low 2/2 or 3-on-one-side, skip {SWING_SKIP_AFTER} dailies after source, "
        f"M2={list(M2_VALUES)}) ...",
        flush=True,
    )
    buckets: dict[int, list[dict]] = {m2: [] for m2 in M2_VALUES}
    done = 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_worker, t): t for t in tickers}
        for fut in as_completed(futs):
            got = fut.result()
            for m2 in M2_VALUES:
                buckets[m2].extend(got.get(m2, []))
            done += 1
            if done % 200 == 0 or done == len(tickers):
                counts = " ".join(f"M2{m2}={len(buckets[m2]):,}" for m2 in M2_VALUES)
                print(f"  scan {done:,}/{len(tickers):,} {counts}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    for m2 in M2_VALUES:
        df = pd.DataFrame(buckets[m2])
        path = OUT / f"All_Setups_RR2_M2{m2}.csv"
        df.to_csv(path, index=False)
        print(f"Saved {path} n={len(df):,}", flush=True)
        out[m2] = df
    return out


def build_features(raw: pd.DataFrame, feat_path: Path) -> pd.DataFrame:
    if feat_path.exists():
        print(f"Loading {feat_path}", flush=True)
        return pd.read_parquet(feat_path)
    if raw.empty:
        return raw
    by = defaultdict(list)
    for rec in raw.to_dict("records"):
        by[rec["Ticker"]].append(rec)
    jobs = list(by.items())
    rows: list[dict] = []
    print(f"Building features {len(jobs):,} tickers ...", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(feat_worker, j): j[0] for j in jobs}
        for fut in as_completed(futs):
            rows.extend(fut.result())
            done += 1
            if done % 300 == 0 or done == len(jobs):
                print(f"  feat {done:,}/{len(jobs):,} rows={len(rows):,}", flush=True)
    df = pd.DataFrame(rows)
    df["Entry_Date"] = pd.to_datetime(df["Entry_Date"])
    df["Exit_Date"] = pd.to_datetime(df["Exit_Date"])
    df["year"] = df["Entry_Date"].dt.year
    df["y_win"] = (df["Outcome"] == "Success").astype(int)
    df["y_r"] = df["Realized_R"].clip(-2.0, 4.0)
    g = df.groupby("Entry_Date")
    df["n_cands"] = g["Ticker"].transform("size")
    df["idio_ret20"] = df["ret_20"] - g["ret_20"].transform("median")
    df["idio_vol"] = df["vol20"] - g["vol20"].transform("median")
    df["rel_ret20"] = g["ret_20"].rank(pct=True)
    df["rel_dolvol"] = g["dolvol_log"].rank(pct=True)
    df["rel_risk_atr"] = g["risk_atr"].rank(pct=True)
    df["rel_c2_thru"] = g["c2_thru_atr"].rank(pct=True)
    df["rel_r_to_hh60"] = g["r_to_hh60"].rank(pct=True)
    for c in CORE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    feat_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(feat_path, index=False)
    print(f"Saved {feat_path} n={len(df):,} win={df.y_win.mean():.3f}", flush=True)
    return df


def eval_m2(m2: int, raw: pd.DataFrame) -> dict:
    tag = f"M2{m2}"
    feat_path = OUT / f"Features_v6_{tag}.parquet"
    scored_path = OUT / f"Scored_v6_meta_{tag}.parquet"
    print(f"\n===== {tag}  setups={len(raw):,} =====", flush=True)
    if raw.empty:
        return {"m2": m2, "setups": 0, "error": "no setups"}
    feat = build_features(raw, feat_path)
    if feat.empty:
        return {"m2": m2, "setups": 0, "error": "no features"}
    print("Primary walk-forward XGB ...", flush=True)
    scored = walk(feat, CORE_COLS, "xgb")
    print("Meta-label walk-forward ...", flush=True)
    scored = walk_meta(scored, META_FEAT)
    scored.to_parquet(scored_path, index=False)
    picked = select_meta(scored, 32, 0.38, "Meta_P")
    accept = 100.0 * len(picked) / max(len(scored), 1)
    print(f"ML accepted {len(picked):,} / {len(scored):,} ({accept:.1f}%)", flush=True)
    books = {}
    print("Vol-managed books ...", flush=True)
    for cap, risk, name in BOOKS:
        res = run_vol_managed(picked, cap, risk, 0.04)
        books[name] = {
            "CAGR": round(float(res["CAGR"]), 2),
            "N": int(res["N"]),
            "DD": float(res["DD"]),
        }
        print(
            f"  {name}: CAGR {res['CAGR']:+.2f}%  n={res['N']:,}  DD {res['DD']:.2f}%",
            flush=True,
        )
    return {
        "m2": m2,
        "setups": int(len(raw)),
        "featured": int(len(feat)),
        "ml_accepted": int(len(picked)),
        "ml_accept_pct": round(accept, 1),
        "win_rate_raw": round(float(feat["y_win"].mean() * 100.0), 2),
        "books": books,
    }


def _synthetic_swing_test() -> None:
    """2 on both sides AND 3 on one side; a neighbour touch/undercut moves the candidate."""
    idx = pd.date_range("2015-01-02", periods=10, freq="W-FRI")
    base = {
        "Open": 11.0,
        "High": 13.0,
        "Close": 11.5,
        "Volume": 1000.0,
    }
    both3 = pd.DataFrame({**base, "Low": [12, 12, 12, 8, 12, 12, 12, 12, 12, 12]}, index=idx)
    swings = find_htf_swing_lows(both3)
    assert len(swings) == 1 and abs(swings[0]["price"] - 8.0) < 1e-9, swings
    assert swings[0]["need_right"] == 2, swings[0]

    left2_right3 = pd.DataFrame({**base, "Low": [12, 12, 8, 12, 12, 12, 12, 12, 12, 12]}, index=idx)
    s23 = find_htf_swing_lows(left2_right3)
    assert len(s23) == 1 and abs(s23[0]["price"] - 8.0) < 1e-9, s23
    assert s23[0]["need_right"] == 3, s23[0]

    left3_right2 = pd.DataFrame({**base, "Low": [12, 12, 12, 8, 12, 12, 7.5, 12, 12, 12]}, index=idx)
    s32 = find_htf_swing_lows(left3_right2)
    assert any(abs(s["price"] - 8.0) < 1e-9 and s["need_right"] == 2 for s in s32), s32

    only22 = pd.DataFrame({**base, "Low": [12, 12, 8, 12, 12, 7.5, 12, 12, 12, 12]}, index=idx)
    prices22 = {round(s["price"], 4) for s in find_htf_swing_lows(only22)}
    assert 8.0 not in prices22, prices22

    left1 = pd.DataFrame({**base, "Low": [12, 8, 12, 12, 12, 12, 12, 12, 12, 12]}, index=idx)
    assert all(abs(s["price"] - 8.0) > 1e-9 for s in find_htf_swing_lows(left1))

    touched = pd.DataFrame({**base, "Low": [12, 12, 12, 8, 12, 8, 12, 12, 12, 12]}, index=idx)
    assert find_htf_swing_lows(touched) == []

    under = pd.DataFrame({**base, "Low": [12, 12, 12, 8, 12, 7, 12, 12, 12, 12]}, index=idx)
    swings2 = find_htf_swing_lows(under)
    assert len(swings2) == 1 and abs(swings2[0]["price"] - 7.0) < 1e-9, swings2

    flat = both3.copy()
    flat.loc[idx[3], "High"] = 8.0
    flat.loc[idx[3], "Volume"] = 0.0
    assert find_htf_swing_lows(flat) == []
    print("synthetic swing-low tests ok", flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if "--sanity" in sys.argv:
        _synthetic_swing_test()
        return
    raws = scan_setups()
    summary = []
    for m2 in M2_VALUES:
        summary.append(eval_m2(m2, raws[m2]))
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n===== M2 comparison (net Zerodha, live ML) =====", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved {SUMMARY}", flush=True)


if __name__ == "__main__":
    main()
