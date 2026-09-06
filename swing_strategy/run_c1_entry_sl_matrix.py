"""
C1 / entry / SL experiment matrix — exclusive MFE buckets.

C1 (green candle):
  CLOSE_BELOW | HIGH_BELOW | OPEN_BELOW
Entry:
  A1 C3 at C3 open after C2 close > C1 high
  A2 same, only if C3 open > C1 high
  A3 C3 at C1_high*1.01 if touched (no chase)
  B  C2 open inside C1 range, C2 high > C1 high, fill C1_high*1.01, no gap-up chase
SL:
  C1_LOW | C1_LOW*0.99 | sweep-low | sweep-low*0.99

Universes: All possible | 50k/500 executed | 100k/500 executed
Exclusive buckets: 1:10, 1:5, 1:4, 1:3, 1:2, 1:1, SL, Open_EOD
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.backtest_engine.backtest_support_liquidity_strategy import (
    INDEX_CLASSIFIER,
    get_all_stock_supports,
)
from swing_strategy.tiered_liquidity_strategy_engine import (
    DATA_DAILY_DIR,
    NIFTY_RANK,
    TF_RANK,
    _load_daily,
    _resolve_effective_liquidity,
)

OUT_DIR = BASE_DIR / "Reports" / "C1_Entry_SL_Matrix"
OUT_CSV = OUT_DIR / "MFE_Exclusive_Buckets_Matrix.csv"
START = pd.Timestamp("2010-01-01")
MAX_POST_SWEEP = 90
ENTRY_MULT = 1.01
RISK_CAP = 500.0
C1_KINDS = ("CLOSE_BELOW", "HIGH_BELOW", "OPEN_BELOW")
ENT_KINDS = ("A1", "A2", "A3", "B")
SL_KINDS = ("C1_LOW", "C1_x99", "SWEEP_LOW", "SWEEP_x99")
BUCKETS = ("1:10", "1:5", "1:4", "1:3", "1:2", "1:1", "SL", "Open_EOD")


def _touch(planned: float, o: float, h: float, l: float) -> float | None:
    if o > planned:
        return planned if l <= planned else None
    if h >= planned:
        return planned
    return None


def _c1_match(kind: str, o: float, h: float, c: float, support: float) -> bool:
    if c <= o:
        return False
    if kind == "CLOSE_BELOW":
        return c < support
    if kind == "HIGH_BELOW":
        return h < support
    return o < support


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
    return "SL" if hit_sl else "Open_EOD"


def _mfe(entry_idx: int, entry: float, sl: float, opens, highs, lows, n: int) -> tuple[float, bool, int]:
    risk = entry - sl
    if risk <= 0.05:
        return 0.0, True, entry_idx
    mfe = max(0.0, (float(highs[entry_idx]) - entry) / risk)
    if float(lows[entry_idx]) <= sl:
        return mfe, True, entry_idx
    for m in range(entry_idx + 1, n):
        if float(opens[m]) < sl:
            return mfe, True, m
        mfe = max(mfe, (float(highs[m]) - entry) / risk)
        if float(lows[m]) <= sl:
            return mfe, True, m
    return mfe, False, n - 1


def _entries_for_c1(c1: int, opens, highs, lows, closes, n: int) -> dict[str, tuple[int, float]]:
    c2 = c1 + 1
    if c2 >= n:
        return {}
    c1_high = float(highs[c1])
    c1_low = float(lows[c1])
    planned = round(c1_high * ENTRY_MULT, 2)
    out: dict[str, tuple[int, float]] = {}

    if float(closes[c2]) > c1_high and c2 + 1 < n:
        c3 = c2 + 1
        o3, h3, l3 = float(opens[c3]), float(highs[c3]), float(lows[c3])
        out["A1"] = (c3, o3)
        if o3 > c1_high:
            out["A2"] = (c3, o3)
        fill = _touch(planned, o3, h3, l3)
        if fill is not None:
            out["A3"] = (c3, fill)

    o2, h2, l2 = float(opens[c2]), float(highs[c2]), float(lows[c2])
    if c1_low <= o2 <= c1_high and h2 > c1_high:
        fill = _touch(planned, o2, h2, l2)
        if fill is not None:
            out["B"] = (c2, fill)
    return out


def _sl_price(kind: str, c1_low: float, sweep_low: float) -> float:
    if kind == "C1_LOW":
        return round(c1_low, 2)
    if kind == "C1_x99":
        return round(c1_low * 0.99, 2)
    if kind == "SWEEP_LOW":
        return round(sweep_low, 2)
    return round(sweep_low * 0.99, 2)


def scan_ticker(symbol: str) -> list[tuple]:
    df = _load_daily(symbol)
    if df is None or len(df) < 120:
        return []

    idx_tag = INDEX_CLASSIFIER.classify(symbol)
    nifty = NIFTY_RANK.get(idx_tag, 1)
    all_supports = get_all_stock_supports(df.set_index("Date"))
    date_s = pd.to_datetime(df["Date"])
    dates_np = df["Date"].to_numpy()
    opens = df["Open"].to_numpy(float)
    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    closes = df["Close"].to_numpy(float)
    n = len(df)

    sup_by_date: dict = {}
    for s in all_supports:
        sup_by_date.setdefault(s["formed_date"], []).append(s)

    active: list[dict] = []
    raw: list[tuple] = []

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
            end = min(n - 1, i + MAX_POST_SWEEP)
            found = set()
            for c1 in range(i, end):
                if len(found) == len(C1_KINDS):
                    break
                o, h, c = float(opens[c1]), float(highs[c1]), float(closes[c1])
                c1_low = float(lows[c1])
                sweep_low = float(min(lows[i : c1 + 1]))
                ents = None
                for ck in C1_KINDS:
                    if ck in found:
                        continue
                    if not _c1_match(ck, o, h, c, support):
                        continue
                    if ents is None:
                        ents = _entries_for_c1(c1, opens, highs, lows, closes, n)
                    if not ents:
                        continue
                    found.add(ck)
                    for ek, (eidx, entry) in ents.items():
                        if entry <= 0:
                            continue
                        for sk in SL_KINDS:
                            sl = _sl_price(sk, c1_low, sweep_low)
                            if entry - sl <= 0.05:
                                continue
                            mfe, hit_sl, xidx = _mfe(eidx, entry, sl, opens, highs, lows, n)
                            raw.append((
                                f"{ck}|{ek}|{sk}",
                                date_s.iloc[eidx].strftime("%Y-%m-%d"),
                                date_s.iloc[xidx].strftime("%Y-%m-%d"),
                                tf_rank,
                                nifty,
                                float(entry),
                                float(sl),
                                _bucket(mfe, hit_sl),
                            ))

    # one setup per variant + entry date (higher TF wins)
    raw.sort(key=lambda t: (t[0], t[1], -t[3]))
    kept: list[tuple] = []
    seen: set[tuple] = set()
    for row in raw:
        key = (row[0], row[1])
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
    return kept


def _scan_worker(symbol: str) -> list[tuple]:
    try:
        return scan_ticker(symbol)
    except Exception:
        return []


def _empty_counts() -> dict[str, int]:
    return {b: 0 for b in BUCKETS} | {"Total": 0}


def _add(dst: dict[str, int], bucket: str, n: int = 1) -> None:
    dst[bucket] += n
    dst["Total"] += n


def run_book(trades: list[tuple], capital: float, risk_cap: float = RISK_CAP) -> dict[str, int]:
    """trades fields: variant, entry_d, exit_d, tf, nifty, entry, sl, bucket"""
    by_entry: dict[str, list] = defaultdict(list)
    for t in trades:
        by_entry[t[1]].append(t)

    min_d = START
    max_d = START
    for t in trades:
        ed = pd.Timestamp(t[2])
        if ed > max_d:
            max_d = ed
        ie = pd.Timestamp(t[1])
        if ie > max_d:
            max_d = ie

    cash = capital
    open_pos: list[tuple[pd.Timestamp, float]] = []
    counts = _empty_counts()

    for day in pd.date_range(min_d, max_d, freq="D"):
        ds = day.strftime("%Y-%m-%d")
        avail = cash
        cands = by_entry.get(ds, [])
        cands.sort(key=lambda t: (-t[3], -t[4]))
        for t in cands:
            entry, sl, bucket = t[5], t[6], t[7]
            risk = entry - sl
            if risk <= 0.05 or risk > risk_cap:
                continue
            qty = min(int(risk_cap // risk), int(avail // entry))
            if qty < 1:
                continue
            spend = entry * qty
            if spend > avail + 1e-9:
                continue
            cash -= spend
            avail -= spend
            open_pos.append((pd.Timestamp(t[2]), spend))
            _add(counts, bucket)

        still = []
        for xdt, spend in open_pos:
            if xdt <= day:
                cash += spend
            else:
                still.append((xdt, spend))
        open_pos = still

    return counts


def _row(c1: str, ent: str, sl: str, universe: str, c: dict[str, int]) -> dict:
    tot = c["Total"]
    r = {
        "C1": c1,
        "Entry": ent,
        "SL": sl,
        "Universe": universe,
        "Total": tot,
    }
    for b in BUCKETS:
        r[f"Count_{b.replace(':', 'to').replace('_', '')}"] = c[b]
    for b in BUCKETS:
        key = f"Pct_{b.replace(':', 'to').replace('_', '')}"
        r[key] = round(100.0 * c[b] / tot, 2) if tot else 0.0
    return r


def main() -> None:
    tickers = sorted({p.name.split("_1d.csv")[0] for p in DATA_DAILY_DIR.glob("*_1d.csv")})
    print(f"Scanning {len(tickers):,} tickers | 3 C1 x 4 entry x 4 SL = 48 variants", flush=True)

    by_var: dict[str, list[tuple]] = defaultdict(list)
    done = 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_scan_worker, t): t for t in tickers}
        for fut in as_completed(futs):
            for row in fut.result():
                by_var[row[0]].append(row)
            done += 1
            if done % 200 == 0 or done == len(tickers):
                ntr = sum(len(v) for v in by_var.values())
                print(f"  • {done:,}/{len(tickers):,} | raw variant-rows {ntr:,}", flush=True)

    all_counts: dict[str, dict[str, int]] = {}
    for vk, rows in by_var.items():
        c = _empty_counts()
        for t in rows:
            _add(c, t[7])
        all_counts[vk] = c

    print("Running 50k/500 and 100k/500 cash books (entries-first, hold to SL)...", flush=True)
    exec50: dict[str, dict[str, int]] = {}
    exec100: dict[str, dict[str, int]] = {}
    keys = [f"{ck}|{ek}|{sk}" for ck in C1_KINDS for ek in ENT_KINDS for sk in SL_KINDS]
    for i, vk in enumerate(keys, 1):
        rows = by_var.get(vk, [])
        exec50[vk] = run_book(rows, 50_000.0) if rows else _empty_counts()
        exec100[vk] = run_book(rows, 100_000.0) if rows else _empty_counts()
        if i % 12 == 0 or i == len(keys):
            print(f"  • books {i}/{len(keys)}", flush=True)

    out_rows = []
    for ck in C1_KINDS:
        for ek in ENT_KINDS:
            for sk in SL_KINDS:
                vk = f"{ck}|{ek}|{sk}"
                c_all = all_counts.get(vk, _empty_counts())
                out_rows.append(_row(ck, ek, sk, "All_Possible", c_all))
                out_rows.append(_row(ck, ek, sk, "Exec_50k_0.5k", exec50.get(vk, _empty_counts())))
                out_rows.append(_row(ck, ek, sk, "Exec_100k_0.5k", exec100.get(vk, _empty_counts())))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(out_rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}", flush=True)

    show = df[df["Universe"] == "All_Possible"].sort_values("Count_1to10", ascending=False)
    cols = ["C1", "Entry", "SL", "Total", "Count_1to10", "Count_1to5", "Count_1to4",
            "Count_1to3", "Count_1to2", "Count_1to1", "Count_SL", "Count_OpenEOD",
            "Pct_1to10", "Pct_1to5", "Pct_SL"]
    print("\nALL POSSIBLE — exclusive buckets (sorted by Count 1:10)\n", flush=True)
    print(show[cols].to_string(index=False), flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
