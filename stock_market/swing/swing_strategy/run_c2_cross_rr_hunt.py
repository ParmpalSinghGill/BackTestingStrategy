"""Live-valid C2-cross entry + tighter SL + condition RR on Swing_PP.

Enter / exit are CONDITIONS, not assumed level fills:

ENTER (after C1 green open-below closes)
  Next session C2: buy-stop at C1 high.
  Gap above C1 high: fill at open x 1.002, OR wait for a pullback to C1 high
  (else skip). Same-bar low through SL while open is still below the trigger
  is skipped (unknown order). C2 close is NOT required before the fill.

EXIT
  Default 1:2, fair BE (SL first until +1R, then stop at entry).
  Same-bar close back below C1 high after a fill -> scratch at that close.
  After a +1R close, if 1:2 is not already tagged: next bar may raise TP to
  1:3 when the 1R bar closed in its upper 40% and high stayed < 1.8R.

SL: C1 low x 0.99 (tighter) vs sweep-to-C1 x 0.99 (control).

C3 limit variants keep the existing Meta_P list (score uses C2 close, so
entry cannot be during C2). C2-cross ranks at C1 close only.

Do not write Swing_low / Swing_Live. Not a frozen Rs 500 book.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from src.backtest_engine.backtest_support_liquidity_strategy import (
    INDEX_CLASSIFIER,
    SWING_N2_DEFAULT,
    SWING_N_DEFAULT,
    SWING_SKIP_AFTER,
    get_swing_low_supports,
)
from swing_strategy.run_2pct_50_25_hunt import age_ok, sim
from swing_strategy.run_2pct_valid_hunt import be_fair, be_fair_after_1r
from swing_strategy.run_c1_entry_sl_matrix import MAX_POST_SWEEP, START, _c1_match
from swing_strategy.run_ml_next_search import META_FEAT, walk_meta
from swing_strategy.run_ml_target_books import CORE_COLS
from swing_strategy.run_ml_top5_selector import _rsi, _sma
from swing_strategy.run_pred_paper_gate import M46 as GATE_M46
from swing_strategy.run_pred_paper_gate import load_pred_list, sim_pred_gate
from swing_strategy.run_swing_low_cagr_hunt import _walk_kind
from swing_strategy.run_swing_low_liquidity_eval import (
    _first_idx_after,
    _precount_state,
    _resolve_valid_below,
)
from swing_strategy.tiered_liquidity_strategy_engine import (
    DATA_DAILY_DIR,
    NIFTY_RANK,
    TF_RANK,
    _load_daily,
)

OUT = BASE / "Reports" / "SwingLow_OldLiquidity"
LOG = OUT / "c2_cross_rr_hunt.json"
CACHE_C1 = OUT / "C2Cross_C1_pending.parquet"
CACHE_FEAT = OUT / "C2Cross_C1_features.parquet"
CACHE_SCORED = OUT / "C2Cross_C1_scored.parquet"
PRED = OUT / "Pred_fairBE_age1m_t048_top16.parquet"

BASE_CAGR = 55.39
BASE_DD = 24.5
GAP_SLIP = 1.002
M2_LOCK = 2

C1_FEAT = [
    c
    for c in CORE_COLS
    if c
    not in (
        "gap_pct",
        "open_vs_brk_atr",
        "c2_thru_atr",
        "rel_c2_thru",
        "TF_Rank",
        "Nifty_Rank",
        "sweep_age_bars",
        "vol20",
    )
]


def _sl(kind: str, c1_low: float, sweep_low: float) -> float:
    if kind == "C1_x99":
        return round(c1_low * 0.99, 2)
    if kind == "C1_LOW":
        return round(c1_low, 2)
    return round(sweep_low * 0.99, 2)


def fill_stop(c1: int, opens, highs, lows, sl: float, gap: str) -> tuple[int, float, str] | None:
    """Buy-stop at C1 high on the next bar. gap=open|wait."""
    c2 = c1 + 1
    n = len(opens)
    if c2 >= n:
        return None
    trig = float(highs[c1])
    o, h, l = float(opens[c2]), float(highs[c2]), float(lows[c2])
    if o > trig:
        if gap == "open":
            return c2, round(o * GAP_SLIP, 2), "gap_open"
        if l <= trig:
            return c2, round(trig, 2), "gap_wait"
        return None
    if h < trig:
        return None
    if l <= sl:
        return None
    return c2, round(trig, 2), "cross"


def fill_c3_limit(c1: int, opens, highs, lows, closes, sl: float, gap: str) -> tuple[int, float, str] | None:
    """A1 confirmed; C3 limit at C1 high (no chase unless gap=open)."""
    c2 = c1 + 1
    c3 = c2 + 1
    n = len(opens)
    if c3 >= n:
        return None
    if float(closes[c2]) <= float(highs[c1]):
        return None
    trig = float(highs[c1])
    o, h, l = float(opens[c3]), float(highs[c3]), float(lows[c3])
    if o > trig:
        if gap == "open":
            return c3, round(o * GAP_SLIP, 2), "gap_open"
        if l <= trig:
            return c3, round(trig, 2), "gap_wait"
        return None
    if h < trig:
        return None
    if l <= sl:
        return None
    return c3, round(trig, 2), "limit"


def exit_path(
    entry_idx: int,
    entry: float,
    sl: float,
    trigger: float,
    opens,
    highs,
    lows,
    closes,
    n: int,
    rr: float = 2.0,
    scratch_fail: bool = True,
) -> tuple[float, int, float]:
    risk = entry - sl
    last = n - 1
    if risk <= 0.05:
        px = round(float(opens[last]), 2)
        return px, last, (px - entry) / max(risk, 1e-6)
    tp = entry + rr * risk
    o0, h0, l0, c0 = float(opens[entry_idx]), float(highs[entry_idx]), float(lows[entry_idx]), float(closes[entry_idx])
    if l0 <= sl:
        return round(sl, 2), entry_idx, -1.0
    if h0 >= tp:
        return round(tp, 2), entry_idx, float(rr)
    if scratch_fail and c0 < trigger:
        return round(c0, 2), entry_idx, (c0 - entry) / risk
    px, xidx, rv = be_fair(entry_idx, entry, sl, opens, highs, lows, n, rr=rr)
    return float(px), int(xidx), float(rv)


def can_upgrade_bar(entry_idx, entry, sl, opens, highs, lows, n) -> tuple[bool, int]:
    risk = entry - sl
    r1 = entry + risk
    r2 = entry + 2.0 * risk
    hit1 = None
    for m in range(entry_idx, n):
        h = float(highs[m])
        if hit1 is None and h >= r1:
            hit1 = m
        if h >= r2:
            return False, int(hit1 if hit1 is not None else m)
        if hit1 is None and m > entry_idx and float(opens[m]) < sl:
            return False, -1
        if hit1 is None and float(lows[m]) <= sl:
            return False, -1
        if hit1 is not None and float(lows[m]) <= entry:
            return False, int(hit1)
        if hit1 is not None:
            break
    if hit1 is None or hit1 >= n - 1:
        return False, -1
    if float(lows[hit1]) <= sl or float(highs[hit1]) >= r2:
        return False, int(hit1)
    return True, int(hit1)


def cond_raise_3(opens, highs, lows, closes, entry, sl, hit1: int) -> bool:
    risk = entry - sl
    h = float(highs[hit1])
    l = float(lows[hit1])
    c = float(closes[hit1])
    rng = max(h - l, 1e-9)
    loc = (c - l) / rng
    high_r = (h - entry) / risk
    return loc >= 0.60 and high_r < 1.80


def exit_rr_cond(entry_idx, entry, sl, trigger, opens, highs, lows, closes, n, scratch_fail=True):
    ok, hit1 = can_upgrade_bar(entry_idx, entry, sl, opens, highs, lows, n)
    px2, x2, r2 = exit_path(entry_idx, entry, sl, trigger, opens, highs, lows, closes, n, 2.0, scratch_fail)
    if not ok or not cond_raise_3(opens, highs, lows, closes, entry, sl, hit1):
        return px2, x2, r2, 2
    px, xidx, rv = be_fair_after_1r(entry_idx, hit1 + 1, entry, sl, opens, highs, lows, n, 3.0)
    return float(px), int(xidx), float(rv), 3


def locate_c1(entry_idx: int, opens, highs, lows, closes, support: float) -> int | None:
    """C3 A1 entry: C2 is entry-1, C1 is the green open-below before C2."""
    p = entry_idx - 1
    if p < 1:
        return None
    c1 = None
    for j in range(p - 1, max(-1, p - 90), -1):
        if float(closes[j]) > float(opens[j]) and float(opens[j]) < support:
            if any(float(lows[k]) < float(lows[j]) for k in range(j + 1, p)):
                continue
            c1 = j
            break
    if c1 is None:
        return None
    if float(closes[p]) <= float(highs[c1]):
        return None
    return int(c1)


KINDS = [
    ("c3open_c1sl", "c3_open", "open", "C1_x99", False, True),
    ("c3open_c1sl_rr3", "c3_open", "open", "C1_x99", True, True),
    ("c3open_c1sl_hold", "c3_open", "open", "C1_x99", False, False),
    ("c3lim_open_c1sl", "c3_lim", "open", "C1_x99", False, True),
    ("c3lim_wait_c1sl", "c3_lim", "wait", "C1_x99", False, True),
    ("c3lim_wait_c1sl_rr3", "c3_lim", "wait", "C1_x99", True, True),
    ("c3lim_open_c1sl_rr3", "c3_lim", "open", "C1_x99", True, True),
    ("c2cross_open_c1sl", "c2", "open", "C1_x99", False, True),
    ("c2cross_wait_c1sl", "c2", "wait", "C1_x99", False, True),
    ("c2cross_open_c1sl_rr3", "c2", "open", "C1_x99", True, True),
    ("c2cross_wait_c1sl_rr3", "c2", "wait", "C1_x99", True, True),
]


def _apply_kind(rec, dates, opens, highs, lows, closes, n, c1, kind, mode, gap, sl_kind, use_rr, scratch):
    trig = float(highs[c1])
    sl = _sl(sl_kind, float(lows[c1]), float(min(lows[max(0, c1 - 90) : c1 + 1])))
    if mode == "c3_open":
        eidx = int(rec["_c3"])
        entry = float(opens[eidx])
        how = "c3_open"
        if entry > trig:
            entry = round(entry * GAP_SLIP, 2)
    elif mode == "c3_lim":
        got = fill_c3_limit(c1, opens, highs, lows, closes, sl, gap)
        if got is None:
            return None
        eidx, entry, how = got
    else:
        got = fill_stop(c1, opens, highs, lows, sl, gap)
        if got is None:
            return None
        eidx, entry, how = got
    if entry - sl <= 0.05:
        return None
    if use_rr:
        px, xidx, rv, rr = exit_rr_cond(eidx, entry, sl, trig, opens, highs, lows, closes, n, scratch)
    else:
        px, xidx, rv = exit_path(eidx, entry, sl, trig, opens, highs, lows, closes, n, 2.0, scratch)
        rr = 2
    out = {k: v for k, v in rec.items() if not str(k).startswith("_")}
    out["Entry_Date"] = pd.Timestamp(dates.iloc[eidx])
    out["Exit_Date"] = pd.Timestamp(dates.iloc[min(int(xidx), n - 1)])
    out["Entry_Price"] = float(entry)
    out["SL_Price"] = float(sl)
    out["Exit_Price"] = float(px)
    out["Realized_R"] = float(rv)
    out["Outcome"] = "Success" if rv >= 1.5 else ("BE" if rv >= 0 else "Failure")
    out["Target_RR_Mode"] = f"1:{int(rr)}"
    out["fill_how"] = how
    out["sl_kind"] = sl_kind
    return out


def rewrite_all(base: pd.DataFrame) -> dict[str, pd.DataFrame]:
    cache: dict = {}
    buckets: dict[str, list] = {k[0]: [] for k in KINDS}
    miss = 0
    recs = base.to_dict("records")
    print(f"[rewrite] {len(KINDS)} kinds on {len(recs):,}", flush=True)
    for i, rec in enumerate(recs):
        if i and i % 800 == 0:
            print(f"  {i:,}/{len(recs):,}", flush=True)
        sym = rec["Ticker"]
        if sym not in cache:
            df = _load_daily(sym)
            if df is None or "Date" not in getattr(df, "columns", []):
                cache[sym] = None
            else:
                d = df.copy()
                d["Date"] = pd.to_datetime(d["Date"])
                cache[sym] = d.sort_values("Date").reset_index(drop=True)
        df = cache[sym]
        if df is None:
            miss += 1
            continue
        edt = pd.Timestamp(rec["Entry_Date"]).normalize()
        dates = pd.to_datetime(df["Date"]).dt.normalize()
        idx = int(dates.searchsorted(edt))
        if idx >= len(df) or dates.iloc[idx] != edt:
            miss += 1
            continue
        opens = df["Open"].to_numpy(float)
        highs = df["High"].to_numpy(float)
        lows = df["Low"].to_numpy(float)
        closes = df["Close"].to_numpy(float)
        n = len(df)
        c1 = locate_c1(idx, opens, highs, lows, closes, float(rec["Support_Price"]))
        if c1 is None:
            miss += 1
            continue
        rec = dict(rec)
        rec["_c3"] = idx
        for kind, mode, gap, sl_kind, use_rr, scratch in KINDS:
            got = _apply_kind(rec, dates, opens, highs, lows, closes, n, c1, kind, mode, gap, sl_kind, use_rr, scratch)
            if got is not None:
                buckets[kind].append(got)
    print(f"  miss {miss}", flush=True)
    return {k: pd.DataFrame(v) for k, v in buckets.items()}


def _eval(tag: str, df: pd.DataFrame, gate: bool) -> dict:
    if df is None or df.empty:
        return {"tag": tag, "empty": True}
    wr = float((df["Realized_R"] >= 1.5).mean() * 100.0)
    mean_r = float(df["Realized_R"].mean())
    cfg = {**GATE_M46, "roll_n": 50, "fail_max": 0.74, "tag": tag}
    if gate:
        r = sim_pred_gate(df, cfg)
    else:
        r = sim(df, {"scale_hi": 1.3, "rank_hi": 1.2, "rank_lo": 0.6, "scale_lo": 0.40})
    row = {
        "tag": tag,
        "gate": gate,
        "n_list": int(len(df)),
        "wr": round(wr, 2),
        "mean_R": round(mean_r, 3),
        "CAGR": r["CAGR"],
        "DD": r["DD"],
        "N": r["N"],
        "Final": r.get("Final"),
        "ok_cagr": r["CAGR"] >= BASE_CAGR - 0.05,
        "ok_dd": r["DD"] <= BASE_DD + 0.15,
    }
    flag = ""
    if row["ok_cagr"] and row["ok_dd"] and (r["CAGR"] > BASE_CAGR + 0.2 or r["DD"] < BASE_DD - 0.3):
        flag = " IMPROVE"
    print(
        f"  {tag:36s} {r['CAGR']:+6.2f}% DD {r['DD']:5.1f} n={r['N']:4d}  "
        f"WR {wr:5.1f}% meanR {mean_r:+.3f} list={len(df):,}{flag}",
        flush=True,
    )
    return row


def scan_ticker_c1(symbol: str) -> list[dict]:
    df = _load_daily(symbol)
    if df is None or len(df) < 120:
        return []
    nifty = NIFTY_RANK.get(INDEX_CLASSIFIER.classify(symbol), 1)
    swings = get_swing_low_supports(df.set_index("Date"), n=SWING_N_DEFAULT, n2=SWING_N2_DEFAULT)
    if not swings:
        return []
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
            "sweep_count": sweeps,
            "gone": gone,
            "trade_from": int(trade_from),
        })
    active: list[dict] = []
    out: list[dict] = []
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
            if (not tradeable) or i < int(sup.get("trade_from", 0)) or prior > M2_LOCK:
                sup["sweep_count"] = prior + 1
                continue
            support, tf = _resolve_valid_below(price, str(sup["timeframe"]), active, M2_LOCK)
            filled = False
            end = min(n - 1, i + MAX_POST_SWEEP)
            for c1 in range(i, end):
                if filled:
                    break
                if not _c1_match("OPEN_BELOW", float(opens[c1]), float(highs[c1]), float(closes[c1]), support):
                    continue
                if c1 + 1 >= n:
                    continue
                sw = float(min(lows[i : c1 + 1]))
                rec = {
                    "Ticker": symbol,
                    "Liquidity_Type": tf,
                    "TF_Rank": TF_RANK.get(tf, 1),
                    "Nifty_Rank": nifty,
                    "Support_Price": round(support, 2),
                    "Liquidity_Date": sup["liquidity_date"].strftime("%Y-%m-%d"),
                    "Sweep_Count": prior,
                    "Decision_Date": date_s.iloc[c1].strftime("%Y-%m-%d"),
                    "C1_idx": int(c1),
                    "C1_High": round(float(highs[c1]), 2),
                    "C1_Low": round(float(lows[c1]), 2),
                    "Sweep_Low": round(sw, 2),
                    "c1_close": float(closes[c1]),
                    "c1_open": float(opens[c1]),
                }
                out.append(rec)
                if fill_stop(c1, opens, highs, lows, _sl("C1_x99", float(lows[c1]), sw), "open"):
                    filled = True
            sup["sweep_count"] = prior + 1
    return out


def _scan_worker(symbol: str) -> list[dict]:
    try:
        return scan_ticker_c1(symbol)
    except Exception:
        return []


def feat_c1_worker(payload: tuple) -> list[dict]:
    symbol, recs = payload
    path = DATA_DAILY_DIR / f"{symbol}_1d.csv"
    if not path.exists() or not recs:
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    if "Date" not in df.columns or len(df) < 80:
        return []
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    opens = df["Open"].to_numpy(float)
    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    closes = df["Close"].to_numpy(float)
    vols = df["Volume"].to_numpy(float) if "Volume" in df.columns else np.ones(len(df))
    date_to_i = {pd.Timestamp(d).normalize(): i for i, d in enumerate(df["Date"])}
    logc = np.log(np.clip(closes, 1e-6, None))
    rets = np.diff(logc, prepend=logc[0])
    sma50 = _sma(closes, 50)
    sma200 = _sma(closes, 200)
    tr = np.maximum(
        highs - lows,
        np.maximum(np.abs(highs - np.roll(closes, 1)), np.abs(lows - np.roll(closes, 1))),
    )
    tr[0] = highs[0] - lows[0]
    atr14 = _sma(tr, 14)
    vol20 = pd.Series(rets).rolling(20, min_periods=10).std().to_numpy()
    vol60 = pd.Series(rets).rolling(60, min_periods=20).std().to_numpy()
    rsi = _rsi(closes, 14)
    vsma = _sma(vols, 20)
    hh20 = pd.Series(highs).rolling(20, min_periods=5).max().to_numpy()
    hh60 = pd.Series(highs).rolling(60, min_periods=10).max().to_numpy()
    maxret20 = pd.Series(rets).rolling(20, min_periods=5).max().to_numpy()
    out = []
    for rec in recs:
        p = int(rec["C1_idx"])
        if p < 70 or p >= len(df):
            cdt = pd.Timestamp(rec["Decision_Date"]).normalize()
            p = date_to_i.get(cdt, -1)
        if p is None or p < 70 or p >= len(df):
            continue
        c = float(closes[p])
        if c <= 0:
            continue
        trig = float(rec["C1_High"])
        sl = _sl("C1_x99", float(rec["C1_Low"]), float(rec["Sweep_Low"]))
        entry_est = trig
        risk = max(entry_est - sl, 1e-6)
        atr = float(atr14[p]) if atr14[p] == atr14[p] and atr14[p] > 0 else max(c * 0.02, 1e-6)
        sup = float(rec["Support_Price"])

        def ret_n(k):
            j = p - k
            if j < 0 or closes[j] <= 0:
                return 0.0
            return float(closes[p] / closes[j] - 1.0)

        c1_o, c1_h, c1_l, c1_c = float(opens[p]), float(highs[p]), float(lows[p]), float(closes[p])
        c1_rng = max(c1_h - c1_l, 1e-6)
        row = dict(rec)
        row.update({
            "risk_pct": risk / entry_est,
            "dist_support_pct": (entry_est - sup) / sup if sup else 0.0,
            "ret_5": ret_n(5),
            "ret_20": ret_n(20),
            "ret_60": ret_n(60),
            "ret_120": ret_n(120),
            "sma50_dist": (c / sma50[p] - 1.0) if sma50[p] == sma50[p] and sma50[p] else 0.0,
            "sma200_dist": (c / sma200[p] - 1.0) if sma200[p] == sma200[p] and sma200[p] else 0.0,
            "atr14_pct": atr / c,
            "vol20": float(vol20[p]) if vol20[p] == vol20[p] else 0.0,
            "vol60": float(vol60[p]) if vol60[p] == vol60[p] else 0.0,
            "rsi14": float(rsi[p]) if rsi[p] == rsi[p] else 50.0,
            "vol_ratio20": float(vols[p] / vsma[p]) if vsma[p] == vsma[p] and vsma[p] else 1.0,
            "dolvol_log": float(np.log(max(c * vols[p], 1.0))),
            "maxret20": float(maxret20[p]) if maxret20[p] == maxret20[p] else 0.0,
            "dist_high20": (c / hh20[p] - 1.0) if hh20[p] == hh20[p] and hh20[p] else 0.0,
            "dist_high60": (c / hh60[p] - 1.0) if hh60[p] == hh60[p] and hh60[p] else 0.0,
            "risk_atr": risk / atr,
            "r_to_hh60": (float(hh60[p]) - entry_est) / risk if hh60[p] == hh60[p] else 0.0,
            "sweep_depth_pct": (sup - float(rec["Sweep_Low"])) / sup if sup else 0.0,
            "c1_reclaim_pct": (c1_c - sup) / sup if sup else 0.0,
            "c1_close_loc": (c1_c - c1_l) / c1_rng,
            "sweep_age_bars": 0.0,
        })
        out.append(row)
    return out


def apply_c2_fills(pending: pd.DataFrame, gap: str, sl_kind: str, use_rr: bool) -> pd.DataFrame:
    cache: dict = {}
    rows = []
    recs = pending.to_dict("records")
    print(f"[fill] gap={gap} sl={sl_kind} rr={use_rr} n={len(recs):,}", flush=True)
    for i, rec in enumerate(recs):
        if i and i % 4000 == 0:
            print(f"  {i:,}/{len(recs):,}", flush=True)
        sym = rec["Ticker"]
        if sym not in cache:
            df = _load_daily(sym)
            if df is None:
                cache[sym] = None
            else:
                d = df.copy()
                d["Date"] = pd.to_datetime(d["Date"])
                cache[sym] = d.sort_values("Date").reset_index(drop=True)
        df = cache[sym]
        if df is None:
            continue
        dates = pd.to_datetime(df["Date"]).dt.normalize()
        c1 = int(rec["C1_idx"])
        if c1 < 0 or c1 + 1 >= len(df):
            continue
        opens = df["Open"].to_numpy(float)
        highs = df["High"].to_numpy(float)
        lows = df["Low"].to_numpy(float)
        closes = df["Close"].to_numpy(float)
        n = len(df)
        sl = _sl(sl_kind, float(rec["C1_Low"]), float(rec["Sweep_Low"]))
        got = fill_stop(c1, opens, highs, lows, sl, gap)
        out = dict(rec)
        out["filled"] = int(got is not None)
        if got is None:
            rows.append(out)
            continue
        eidx, entry, how = got
        if entry - sl <= 0.05:
            out["filled"] = 0
            rows.append(out)
            continue
        trig = float(rec["C1_High"])
        if use_rr:
            px, xidx, rv, rr = exit_rr_cond(eidx, entry, sl, trig, opens, highs, lows, closes, n, True)
        else:
            px, xidx, rv = exit_path(eidx, entry, sl, trig, opens, highs, lows, closes, n, 2.0, True)
            rr = 2
        out["Entry_Date"] = pd.Timestamp(dates.iloc[eidx])
        out["Exit_Date"] = pd.Timestamp(dates.iloc[min(int(xidx), n - 1)])
        out["Entry_Price"] = float(entry)
        out["SL_Price"] = float(sl)
        out["Exit_Price"] = float(px)
        out["Realized_R"] = float(rv)
        out["Outcome"] = "Success" if rv >= 1.5 else ("BE" if rv >= 0 else "Failure")
        out["Target_RR_Mode"] = f"1:{int(rr)}"
        out["fill_how"] = how
        rows.append(out)
    return pd.DataFrame(rows)


def select_decision(scored: pd.DataFrame, top_n: int, thresh: float) -> pd.DataFrame:
    d = scored[pd.to_numeric(scored["Meta_P"], errors="coerce") >= thresh].copy()
    d["Decision_Date"] = pd.to_datetime(d["Decision_Date"])
    d = d.sort_values(["Decision_Date", "Meta_P"], ascending=[True, False])
    d["rk"] = d.groupby("Decision_Date").cumcount()
    return d[d["rk"] < top_n]


def phase_pred() -> list[dict]:
    base = load_pred_list()
    print(f"[pred] Swing_PP list {len(base):,}", flush=True)
    rows = [_eval("baseline_list", base, True)]
    books = rewrite_all(base)
    for kind, df in books.items():
        if df is None or df.empty:
            rows.append({"tag": kind, "empty": True})
            continue
        gate = not kind.startswith("c2cross")
        row = _eval(kind, df, gate)
        if kind.startswith("c2cross"):
            row["note"] = "lookahead: Meta_P uses C2 close; C2 fill is diagnostic only"
            print("    (C2-cross on this list uses C2-close Meta_P — diagnostic, not live)", flush=True)
        rows.append(row)
    return rows


def phase_c2() -> list[dict]:
    rows = []
    if CACHE_C1.exists():
        pending = pd.read_parquet(CACHE_C1)
        print(f"[c1] cache {len(pending):,}", flush=True)
    else:
        tickers = sorted({p.name.split("_1d.csv")[0] for p in DATA_DAILY_DIR.glob("*_1d.csv")})
        print(f"[c1] scan {len(tickers):,} tickers", flush=True)
        raw: list[dict] = []
        done = 0
        with ProcessPoolExecutor(max_workers=8) as ex:
            futs = [ex.submit(_scan_worker, t) for t in tickers]
            for fut in as_completed(futs):
                raw.extend(fut.result())
                done += 1
                if done % 200 == 0 or done == len(tickers):
                    print(f"  scan {done:,}/{len(tickers):,} c1={len(raw):,}", flush=True)
        pending = pd.DataFrame(raw)
        OUT.mkdir(parents=True, exist_ok=True)
        pending.to_parquet(CACHE_C1, index=False)
        print(f"[c1] wrote {CACHE_C1} n={len(pending):,}", flush=True)
    pending["Decision_Date"] = pd.to_datetime(pending["Decision_Date"])
    pending["Liquidity_Date"] = pd.to_datetime(pending["Liquidity_Date"])
    pending = pending[age_ok(pending["Liquidity_Date"], pending["Decision_Date"] + pd.Timedelta(days=1), 1)].copy()
    print(f"[c1] age>=1m {len(pending):,}", flush=True)

    if CACHE_FEAT.exists() and len(pd.read_parquet(CACHE_FEAT)) == len(pending):
        feat = pd.read_parquet(CACHE_FEAT)
        print(f"[feat] cache {len(feat):,}", flush=True)
    else:
        by = defaultdict(list)
        for rec in pending.to_dict("records"):
            by[rec["Ticker"]].append(rec)
        jobs = list(by.items())
        feat_rows: list[dict] = []
        print(f"[feat] {len(jobs):,} tickers", flush=True)
        with ProcessPoolExecutor(max_workers=8) as ex:
            futs = [ex.submit(feat_c1_worker, j) for j in jobs]
            done = 0
            for fut in as_completed(futs):
                feat_rows.extend(fut.result())
                done += 1
                if done % 400 == 0 or done == len(jobs):
                    print(f"  feat {done:,}/{len(jobs):,}", flush=True)
        feat = pd.DataFrame(feat_rows)
        feat.to_parquet(CACHE_FEAT, index=False)
        print(f"[feat] {len(feat):,}", flush=True)

    feat["Decision_Date"] = pd.to_datetime(feat["Decision_Date"])
    feat["year"] = feat["Decision_Date"].dt.year
    g = feat.groupby("Decision_Date")
    feat["n_cands"] = g["Ticker"].transform("size")
    feat["idio_ret20"] = feat["ret_20"] - g["ret_20"].transform("median")
    feat["idio_vol"] = feat["vol20"] - g["vol20"].transform("median")
    feat["rel_ret20"] = g["ret_20"].rank(pct=True)
    feat["rel_dolvol"] = g["dolvol_log"].rank(pct=True)
    feat["rel_risk_atr"] = g["risk_atr"].rank(pct=True)
    feat["rel_r_to_hh60"] = g["r_to_hh60"].rank(pct=True)
    for c in C1_FEAT + ["n_cands", "idio_ret20", "idio_vol", "rel_ret20", "rel_dolvol", "rel_risk_atr", "rel_r_to_hh60"]:
        if c in feat.columns:
            feat[c] = pd.to_numeric(feat[c], errors="coerce").fillna(0.0)

    variants = [
        ("c2_open_c1sl", "open", "C1_x99", False),
        ("c2_wait_c1sl", "wait", "C1_x99", False),
        ("c2_open_sweep", "open", "SWEEP_x99", False),
        ("c2_open_c1sl_rr3", "open", "C1_x99", True),
        ("c2_wait_c1sl_rr3", "wait", "C1_x99", True),
    ]
    for tag, gap, sl_kind, use_rr in variants:
        all_c1 = apply_c2_fills(feat, gap, sl_kind, use_rr)
        all_c1["Decision_Date"] = pd.to_datetime(all_c1["Decision_Date"])
        filled_n = int(all_c1["filled"].sum()) if "filled" in all_c1.columns else 0
        if filled_n < 200:
            rows.append({"tag": tag, "empty": True, "n_fills": filled_n})
            continue
        all_c1["Entry_Date"] = (
            pd.to_datetime(all_c1["Entry_Date"], errors="coerce")
            if "Entry_Date" in all_c1.columns
            else pd.NaT
        )
        all_c1["Entry_Date"] = all_c1["Entry_Date"].fillna(all_c1["Decision_Date"])
        all_c1["Exit_Date"] = (
            pd.to_datetime(all_c1["Exit_Date"], errors="coerce")
            if "Exit_Date" in all_c1.columns
            else pd.NaT
        )
        all_c1["Exit_Date"] = all_c1["Exit_Date"].fillna(all_c1["Decision_Date"])
        all_c1["Realized_R"] = pd.to_numeric(all_c1.get("Realized_R"), errors="coerce").fillna(0.0)
        all_c1["y_win"] = ((all_c1["filled"] == 1) & (all_c1["Realized_R"] >= 1.5)).astype(int)
        all_c1["year"] = all_c1["Decision_Date"].dt.year
        use = [
            c
            for c in C1_FEAT
            + ["n_cands", "idio_ret20", "idio_vol", "rel_ret20", "rel_dolvol", "rel_risk_atr", "rel_r_to_hh60"]
            if c in all_c1.columns
        ]
        print(f"[ml] {tag} fills {filled_n:,} / pending {len(all_c1):,}", flush=True)
        scored = _walk_kind(all_c1, use, "hgb")
        meta_cols = [c for c in META_FEAT if c in scored.columns]
        scored = walk_meta(scored, meta_cols)
        for thresh, top in ((0.42, 16), (0.48, 16), (0.42, 24)):
            picked = select_decision(scored, top, thresh)
            trades = picked[picked["filled"] == 1].copy()
            if trades.empty:
                continue
            r = _eval(f"{tag}/t{thresh}/top{top}", trades, False)
            r["fill_rate"] = round(100.0 * filled_n / max(len(all_c1), 1), 2)
            r["n_fills"] = filled_n
            r["n_pending"] = int(len(all_c1))
            r["n_picked"] = int(len(picked))
            rows.append(r)
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(
        f"Beat Swing_PP 1:2  CAGR>={BASE_CAGR:.2f}%  DD<={BASE_DD:.1f}%  "
        f"2% equity, net Zerodha",
        flush=True,
    )
    all_rows = []
    print("\n===== A  existing Swing_PP list (C3 fills live-valid; C2 diagnostic) =====", flush=True)
    all_rows.extend(phase_pred())
    LOG.write_text(json.dumps(all_rows, indent=2, default=str), encoding="utf-8")
    print("\n===== B  C2-cross universe, rank at C1 close (live) =====", flush=True)
    all_rows.extend(phase_c2())
    LOG.write_text(json.dumps(all_rows, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {LOG}", flush=True)
    keep = [
        r
        for r in all_rows
        if r.get("ok_cagr") and r.get("ok_dd") and not str(r.get("tag", "")).startswith("c2cross")
    ]
    print("\nLive-valid rows that hold CAGR and DD:", flush=True)
    if not keep:
        print("  none", flush=True)
    for r in keep:
        print(f"  {r}", flush=True)


if __name__ == "__main__":
    main()
