"""Among 1:2 profit trades: how far price went after 2R, and how it came back."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from swing_strategy.run_pred_paper_gate import CACHE_BE, load_pred_list
from swing_strategy.tiered_liquidity_strategy_engine import _load_daily

OUT = BASE / "Reports" / "SwingLow_OldLiquidity"
LOG = OUT / "swing_pp_winner_extension.json"
STMT = BASE / "Reports" / "SwingPP" / "Swing_Strategy_Account_Statement.csv"

# exclusive MFE buckets after 2R, until first return to entry
BUCKETS = [
    ("1:2 only", 2.0, 3.0),
    ("1:3", 3.0, 4.0),
    ("1:4", 4.0, 5.0),
    ("1:5", 5.0, 6.0),
    ("1:6", 6.0, 10.0),
    ("1:10", 10.0, 11.0),
    ("1:10+", 11.0, 1e9),
]
FUNNEL = [3.0, 4.0, 5.0, 6.0, 10.0]


def bucket(mfe: float) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= mfe < hi:
            return name
    return "1:10+"


def walk(entry_idx, entry, sl, opens, highs, lows, n) -> dict | None:
    risk = entry - sl
    if risk <= 0.05:
        return None
    tp2 = entry + 2.0 * risk
    hit2 = None
    for m in range(entry_idx, n):
        if float(highs[m]) >= tp2:
            hit2 = m
            break
        if m > entry_idx and float(opens[m]) < sl:
            return None
        if float(lows[m]) <= sl:
            return None
    if hit2 is None:
        return None

    def extend(start: int, include_start_high: bool) -> dict:
        peak = 2.0
        how = "still_up_at_data_end"
        last = start
        for m in range(start, n):
            o, h, l = float(opens[m]), float(highs[m]), float(lows[m])
            if include_start_high or m > start:
                peak = max(peak, (h - entry) / risk)
            if m > start and o < entry:
                how = "gap_below_entry"
                last = m
                break
            if l <= sl:
                how = "came_back_to_SL"
                last = m
                break
            if l <= entry:
                how = "came_back_to_entry"
                last = m
                break
            last = m
        reached = []
        for lv in FUNNEL:
            reached.append(peak + 1e-9 >= lv)
        return {
            "peak_R": round(float(peak), 3),
            "bucket": bucket(peak),
            "how": how,
            "bars_after_2R": int(last - start),
            "reached_3": reached[0],
            "reached_4": reached[1],
            "reached_5": reached[2],
            "reached_6": reached[3],
            "reached_10": reached[4],
        }

    # same-bar extra high counts (daily high after 2R on that candle)
    same = extend(hit2, include_start_high=True)
    # runner after we would have exited that day
    nxt = hit2 + 1
    after = extend(nxt, include_start_high=True) if nxt < n else {
        "peak_R": 2.0, "bucket": "1:2 only", "how": "no_bar_after_2R",
        "bars_after_2R": 0, "reached_3": False, "reached_4": False,
        "reached_5": False, "reached_6": False, "reached_10": False,
    }
    return {"same_bar": same, "after_exit_bar": after}


def summarize(rows: list[dict], key: str) -> dict:
    parts = [r[key] for r in rows]
    n = len(parts)
    bc = Counter(p["bucket"] for p in parts)
    hc = Counter(p["how"] for p in parts)
    funnel = {"took_1to2": n}
    labels = [("1:3", "reached_3"), ("1:4", "reached_4"), ("1:5", "reached_5"),
              ("1:6", "reached_6"), ("1:10", "reached_10")]
    for lab, k in labels:
        funnel[lab] = int(sum(1 for p in parts if p[k]))
    stopped = {
        "only_1:2_then_back": n - funnel["1:3"],
        "to_1:3_then_back": funnel["1:3"] - funnel["1:4"],
        "to_1:4_then_back": funnel["1:4"] - funnel["1:5"],
        "to_1:5_then_back": funnel["1:5"] - funnel["1:6"],
        "to_1:6_then_back_before_1:10": funnel["1:6"] - funnel["1:10"],
        "to_1:10_or_more": funnel["1:10"],
    }
    return {
        "n": n,
        "median_peak_R": round(float(pd.Series([p["peak_R"] for p in parts]).median()), 2),
        "buckets": {name: int(bc.get(name, 0)) for name, _, _ in BUCKETS},
        "how_returned": dict(hc),
        "funnel_reached": funnel,
        "stopped_after": stopped,
    }


def main() -> None:
    ml = load_pred_list()
    ml["Entry_Date"] = pd.to_datetime(ml["Entry_Date"])
    wins = ml[pd.to_numeric(ml["Realized_R"], errors="coerce") >= 1.5].copy()
    print(f"paper 1:2 profits {len(wins):,}", flush=True)

    filled_keys = set()
    if STMT.exists():
        st = pd.read_csv(STMT)
        buys = st[st["Type"] == "BUY (ENTRY)"]
        sells = st[st["Type"] == "SELL (EXIT)"]
        filled_keys = set(zip(buys["Ticker"], pd.to_datetime(buys["Date"]).dt.strftime("%Y-%m-%d")))
        print(f"executed fills {len(filled_keys):,}", flush=True)

    cache: dict = {}
    paper_rows = []
    filled_rows = []
    miss = 0
    for rec in wins.to_dict("records"):
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
        got = walk(
            idx,
            float(rec["Entry_Price"]),
            float(rec["SL_Price"]),
            df["Open"].to_numpy(),
            df["High"].to_numpy(),
            df["Low"].to_numpy(),
            len(df),
        )
        if not got:
            miss += 1
            continue
        paper_rows.append(got)
        key = (sym, edt.strftime("%Y-%m-%d"))
        if key in filled_keys:
            filled_rows.append(got)

    print(f"walked {len(paper_rows):,}  filled-winners {len(filled_rows):,}  miss {miss}", flush=True)
    out = {
        "paper_1to2_profits": summarize(paper_rows, "same_bar"),
        "paper_after_exit_bar": summarize(paper_rows, "after_exit_bar"),
        "filled_1to2_profits": summarize(filled_rows, "same_bar") if filled_rows else None,
        "filled_after_exit_bar": summarize(filled_rows, "after_exit_bar") if filled_rows else None,
        "note": (
            "same_bar: MFE on/after the 2R daily bar until first return to entry. "
            "after_exit_bar: only bars after the 2R day (runner left after a 1:2 fill that day)."
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for title, block in (
        ("FILLED 1:2 profits — MFE including 2R bar", out["filled_1to2_profits"]),
        ("FILLED — after the 2R day only", out["filled_after_exit_bar"]),
        ("ALL paper 1:2 profits — MFE including 2R bar", out["paper_1to2_profits"]),
        ("ALL paper — after the 2R day only", out["paper_after_exit_bar"]),
    ):
        if not block:
            continue
        print(f"\n===== {title}  n={block['n']}  median peak {block['median_peak_R']}R =====", flush=True)
        print("  exclusive buckets:", block["buckets"], flush=True)
        print("  reached at least:", block["funnel_reached"], flush=True)
        print("  stopped (returned before next level):", block["stopped_after"], flush=True)
        print("  how came back:", block["how_returned"], flush=True)
    print(f"wrote {LOG}", flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
