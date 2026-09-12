"""Valid 2%-only search. No fake BE. Fair BE = SL first until 1R, then 2R before scratch."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from swing_strategy.run_2pct_50_25_hunt import (
    CACHE_1M,
    FEAT,
    LOG,
    OUT,
    SCORED,
    age_ok,
    attach_nifty,
    ensure_1m,
    is_hit,
    load_nifty,
    nifty_filter,
    pick,
    sim,
)
from swing_strategy.run_conservative_be_2pct import be_exit
from swing_strategy.tiered_liquidity_strategy_engine import _load_daily

LOG2 = OUT / "hunt_2pct_valid.json"


def be_fair(entry_idx, entry, sl, opens, highs, lows, n, rr=2.0):
    risk = entry - sl
    tp = entry + rr * risk
    be = entry
    armed = False
    for m in range(entry_idx, n):
        o, h, l = float(opens[m]), float(highs[m]), float(lows[m])
        stop = be if armed else sl
        if m > entry_idx and o < stop:
            px = round(o * 0.998, 2)
            return px, m, (px - entry) / risk
        if m > entry_idx and o > tp:
            px = round(o * 0.998, 2)
            return px, m, (px - entry) / risk
        if not armed:
            if l <= sl:
                return sl, m, -1.0
            if h >= entry + risk:
                armed = True
                if h >= tp:
                    return round(tp, 2), m, float(rr)
            continue
        if h >= tp:
            return round(tp, 2), m, float(rr)
        if l <= be:
            return round(be, 2), m, 0.0
    last = n - 1
    px = round(float(opens[last]), 2)
    return px, last, (px - entry) / risk


def rewrite(picked: pd.DataFrame, fn) -> pd.DataFrame:
    cache: dict = {}
    rows = []
    for rec in picked.to_dict("records"):
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
            rows.append(rec)
            continue
        edt = pd.Timestamp(rec["Entry_Date"]).normalize()
        dates = pd.to_datetime(df["Date"]).dt.normalize()
        idx = int(dates.searchsorted(edt))
        if idx >= len(df) or dates.iloc[idx] != edt:
            rows.append(rec)
            continue
        px, xidx, rr = fn(
            idx,
            float(rec["Entry_Price"]),
            float(rec["SL_Price"]),
            df["Open"].to_numpy(),
            df["High"].to_numpy(),
            df["Low"].to_numpy(),
            len(df),
        )
        rec = dict(rec)
        rec["Exit_Price"] = float(px)
        rec["Exit_Date"] = pd.Timestamp(dates.iloc[min(int(xidx), len(dates) - 1)])
        rec["Realized_R"] = float(rr)
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    nifty = load_nifty()
    raw = pd.read_parquet(SCORED)
    raw["Entry_Date"] = pd.to_datetime(raw["Entry_Date"])
    raw["Exit_Date"] = pd.to_datetime(raw["Exit_Date"])
    raw["Liquidity_Date"] = pd.to_datetime(raw["Liquidity_Date"])
    raw = attach_nifty(raw, nifty)
    age1 = raw[age_ok(raw["Liquidity_Date"], raw["Entry_Date"], 1)].copy()
    age2 = raw[age_ok(raw["Liquidity_Date"], raw["Entry_Date"], 2)].copy()
    re1 = ensure_1m(nifty)
    books = {
        "full": raw,
        "age1": age1,
        "age2": age2,
        "re1": re1,
    }
    pauses = [
        ("base", {}),
        ("p6_15", {"k": 6, "pause_days": 15}),
        ("p5_20", {"k": 5, "pause_days": 20}),
        ("p6_12", {"k": 6, "pause_days": 12}),
        ("p7_15", {"k": 7, "pause_days": 15}),
        ("p8_20", {"k": 8, "pause_days": 20}),
        ("dl2_10", {"day_loss_n": 2, "day_loss_pause": 10}),
        ("p6_h3", {"k": 6, "pause_days": 15, "cut_after": 3, "size_cut": 0.5}),
        ("max6", {"max_day": 6}),
        ("max8_p6", {"max_day": 8, "k": 6, "pause_days": 15}),
        ("h3", {"cut_after": 3, "size_cut": 0.5}),
        ("h2", {"cut_after": 2, "size_cut": 0.5}),
    ]
    nfty = ["all", "sma50", "sma200", "ret20", "sma50_ret20"]
    knobs = [(32, 0.42), (24, 0.42), (16, 0.38), (16, 0.48), (12, 0.42), (20, 0.40)]
    hits = []
    rows = []

    def go(tag, picked, cfg):
        if picked is None or picked.empty:
            return
        r = sim(picked, cfg)
        r["tag"] = tag
        rows.append(r)
        if r["CAGR"] >= 40 or r["DD"] <= 28 or is_hit(r):
            print(f"  {tag:72s} {r['CAGR']:+6.2f}% DD {r['DD']:5.1f} n={r['N']}{' HIT' if is_hit(r) else ''}", flush=True)
        if is_hit(r):
            hits.append(r)
            print("FOUND", r, flush=True)

    print("===== fair BE on age2/re1 (bar walk) =====", flush=True)
    for bname in ("age2", "re1"):
        for top, th in ((32, 0.42), (16, 0.48)):
            picked = pick(books[bname], top, th)
            print(f"rewrite fair {bname} t{th} top{top} n={len(picked)}", flush=True)
            rw = rewrite(picked, be_fair)
            go(f"fairBE/{bname}/t{th}/top{top}", rw, {})
            if hits:
                OUT.joinpath("hunt_2pct_valid.json").write_text(json.dumps(hits + rows[-20:], indent=2, default=str), encoding="utf-8")
                return
            go(f"fairBE/{bname}/t{th}/top{top}/p6", rw, {"k": 6, "pause_days": 15})
            if hits:
                return

    print("===== raw 2% overlays (no BE) =====", flush=True)
    for bname, u0 in books.items():
        for nf in nfty:
            u = nifty_filter(u0, nf)
            if len(u) < 800:
                continue
            for top, th in knobs:
                picked = pick(u, top, th)
                for pname, cfg in pauses:
                    go(f"{bname}/{nf}/t{th}/top{top}/{pname}", picked, cfg)
                    if hits:
                        LOG2.write_text(json.dumps({"hits": hits, "n": len(rows)}, indent=2), encoding="utf-8")
                        return
        print(f"  done {bname} rows={len(rows)}", flush=True)

    LOG2.write_text(json.dumps({"hits": hits, "best": sorted(rows, key=lambda x: -(x['CAGR']-0.35*x['DD']))[:30]}, indent=2), encoding="utf-8")
    print("no valid hit", flush=True)
    for r in sorted(rows, key=lambda x: -(x["CAGR"] - 0.35 * x["DD"]))[:15]:
        print(f"  {r['CAGR']:+6.2f}% DD {r['DD']:5.1f} {r['tag']}", flush=True)


if __name__ == "__main__":
    main()
