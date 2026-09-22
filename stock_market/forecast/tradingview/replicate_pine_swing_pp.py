"""Python twin of tradingview/Swing_PP.pine — per-stock, single-symbol, NO ML.

Goal: reproduce, symbol by symbol, exactly what the Pine strategy does on a
TradingView chart, so you can compare TradingView's Strategy Tester "Net Profit"
for each ticker against this engine.

What this matches from the Pine script (defaults):
  - HTF Yearly/Monthly/Weekly swing-low liquidity (same get_swing_low_supports),
    M2=2 wick sweeps, close-below kills, skip 2 dailies after source low.
  - >= 1 month age from liquidity to entry.
  - On a sweep, if the nearest intact support below is within 5%, trade that lower level.
  - C1 = green, open below liquidity. C2 close > C1 high, C1 low intact ->
    ENTER C3 at C3 OPEN.
  - SL = min(low from sweep..C1) x 0.99. Default target 1:2.
  - Entry bar only: SL, else 1:2, else if close < C1 high -> SELL AT CLOSE (scratch).
  - After first +1R (high >= entry+risk): SL -> breakeven, target STAYS 1:2
    (Pine default "Stay 1:2"; the live k comes from Swing_PP_RR.txt, not Pine).
  - Single position, no pyramiding. New sweeps ignored while hunting / in a trade.
  - Sizing: qty = max(1, floor(equity * 2% / (entry - SL))). No cash cap (like TV).
  - Commission 0.03% of notional per side. Gap fills at the bar open.

What is deliberately NOT here (also NOT in Pine): Meta_P>=0.48, daily top-16,
paper-gate, portfolio sizing, walk-forward XGB RR. This is the raw single-symbol
path only.

CAVEAT: data_daily/*.csv are RAW (unadjusted) OHLC. TradingView charts are split/
dividend ADJUSTED by default. Post-split symbols will differ in price and therefore
in Net Profit even though the logic is identical. Toggle "Adjust data for..." off on
the TV chart to get closer.

Usage:
  python tradingview/replicate_pine_swing_pp.py                 # all stocks
  python tradingview/replicate_pine_swing_pp.py --start 2010-01-01
  python tradingview/replicate_pine_swing_pp.py --tickers RELIANCE.NS TCS.NS
"""
from __future__ import annotations

import argparse
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.backtest_engine.backtest_support_liquidity_strategy import (  # noqa: E402
    SWING_N2_DEFAULT,
    SWING_N_DEFAULT,
    SWING_SKIP_AFTER,
    get_swing_low_supports,
)

DATA_DIR = BASE_DIR / "data_daily"
OUT_DIR = BASE_DIR / "Reports" / "TV_Pine_Replica"

# --- Pine defaults ---------------------------------------------------------
INIT_CAPITAL = 50_000.0
RISK_PCT = 0.02
COMMISSION = 0.0003          # 0.03% per side
M2_LOCK = 2
MIN_AGE_MONTHS = 1
HUNT_BARS = 90
NEAR_PCT = 0.05
SL_MULT = 0.99
MIN_RISK = 0.05
MIN_RANGE_PCT = 0.003        # skip flat HTF (matches get_all; scanner uses real range)
TF_RANK = {"Yearly": 3, "Monthly": 2, "Weekly": 1}
START_DEFAULT = pd.Timestamp("1990-01-01")


def _first_idx_after(dates: pd.DatetimeIndex, ts) -> int | None:
    idx = int(dates.searchsorted(pd.Timestamp(ts).normalize(), side="right"))
    return None if idx >= len(dates) else idx


def _precount_state(lows, closes, start, end, price):
    if start is None or start > end:
        return 0, False
    sweeps = 0
    for j in range(start, end + 1):
        if float(closes[j]) < price:
            return sweeps, True
        if float(lows[j]) < price:
            sweeps += 1
    return sweeps, False


def _age_ok(liq_ts, entry_ts) -> bool:
    if MIN_AGE_MONTHS <= 0:
        return True
    return pd.Timestamp(liq_ts) + pd.DateOffset(months=MIN_AGE_MONTHS) <= pd.Timestamp(entry_ts)


def _resolve_below(swept_price, swept_tf, active, m2):
    below = [
        s for s in active
        if float(s["price"]) < swept_price and not s.get("gone", False)
        and int(s.get("sweep_count", 0)) <= m2
    ]
    if not below:
        return swept_price, swept_tf
    nearest = max(below, key=lambda s: float(s["price"]))
    nb = float(nearest["price"])
    if swept_price > 0 and (swept_price - nb) / swept_price <= NEAR_PCT:
        return nb, str(nearest["timeframe"])
    return swept_price, swept_tf


def _norm_ohlc(df: pd.DataFrame) -> pd.DataFrame | None:
    """Map any OHLC csv (data_daily, TradingView export, tvdatafeed) to a
    canonical Date/Open/High/Low/Close/Volume frame."""
    cols = {c.lower().strip(): c for c in df.columns}
    date_c = cols.get("date") or cols.get("time") or cols.get("datetime")
    if date_c is None:
        return None
    o = cols.get("open"); h = cols.get("high"); l = cols.get("low"); c = cols.get("close")
    if not all([o, h, l, c]):
        return None
    v = cols.get("volume")
    t = df[date_c]
    # Keep the EXCHANGE-LOCAL calendar day (TradingView shows the session date).
    if pd.api.types.is_numeric_dtype(t) and t.max() > 1e8:
        # unix seconds -> assume NSE/BSE local (Asia/Kolkata)
        dt = pd.to_datetime(t, unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    else:
        dt = pd.to_datetime(t, errors="coerce")
        try:
            if getattr(dt.dt, "tz", None) is not None:
                dt = dt.dt.tz_localize(None)   # drop tz, keep wall-clock local time
        except (AttributeError, TypeError):
            pass
    out = pd.DataFrame({
        "Date": pd.to_datetime(dt).dt.normalize(),
        "Open": pd.to_numeric(df[o], errors="coerce"),
        "High": pd.to_numeric(df[h], errors="coerce"),
        "Low": pd.to_numeric(df[l], errors="coerce"),
        "Close": pd.to_numeric(df[c], errors="coerce"),
    })
    out["Volume"] = pd.to_numeric(df[v], errors="coerce") if v else 1.0
    out = out.dropna(subset=["Date", "Open", "High", "Low", "Close"])
    return out.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)


def simulate(symbol: str, start: pd.Timestamp, clip_gap: bool = False,
             drop_zero_vol: bool = True, src_path: Path | None = None) -> dict | None:
    path = src_path if src_path is not None else (DATA_DIR / f"{symbol}_1d.csv")
    if not path.exists():
        return None
    try:
        raw = pd.read_csv(path)
    except Exception:
        return None
    df = _norm_ohlc(raw)
    if df is None:
        return None
    if drop_zero_vol and "Volume" in df.columns:
        # A zero-volume day is a non-trading day: no fill/sweep can occur on it.
        # This removes corrupt single-print spike bars (O=H=L=C, vol 0).
        nz = df[df["Volume"].fillna(0) > 0]
        if len(nz) >= 120:            # only drop if we still have enough bars
            df = nz.reset_index(drop=True)
    if len(df) < 120:
        return None

    swings = get_swing_low_supports(df.set_index("Date"), n=SWING_N_DEFAULT, n2=SWING_N2_DEFAULT)
    if not swings:
        return None

    dates = pd.DatetimeIndex(pd.to_datetime(df["Date"]).dt.normalize())
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
        st = _first_idx_after(dates, s["liquidity_date"])
        sweeps, gone = _precount_state(lows, closes, st, aidx - 1, float(s["price"]))
        src = pd.Timestamp(s.get("source_date", s["liquidity_date"])).normalize()
        src_i = int(dates.searchsorted(src, side="left"))
        trade_from = (src_i + 1 + SWING_SKIP_AFTER) if (src_i < n and dates[src_i] == src) \
            else ((st if st is not None else aidx) + SWING_SKIP_AFTER)
        activate_at.setdefault(aidx, []).append({
            "price": float(s["price"]),
            "timeframe": s["timeframe"],
            "liquidity_date": pd.Timestamp(s["liquidity_date"]).normalize(),
            "sweep_count": int(sweeps),
            "gone": bool(gone),
            "trade_from": int(trade_from),
        })

    equity = INIT_CAPITAL
    peak = INIT_CAPITAL
    max_dd = 0.0
    trades: list[dict] = []
    active: list[dict] = []

    # single-symbol state machine
    hunting = False
    hunt = None            # dict: support, tf, liq_ts, until, sweep_low
    c1_bar = None
    c1_high = c1_low = np.nan
    in_pos = False
    entry_i = None
    entry_px = sl_px = tp2 = r1 = np.nan
    qty = 0
    be_armed = False
    first_entry_ts = None
    last_exit_ts = None

    def close_trade(exit_i, exit_px, reason):
        nonlocal equity, peak, max_dd, in_pos, hunting, hunt, c1_bar, entry_i
        nonlocal entry_px, sl_px, tp2, r1, qty, be_armed, last_exit_ts
        gross = (exit_px - entry_px) * qty
        fees = COMMISSION * (entry_px + exit_px) * qty
        equity += gross - fees
        peak = max(peak, equity)
        if peak > 0:
            max_dd = max(max_dd, (peak - equity) / peak)
        risk_sh = entry_px - sl_px
        trades.append({
            "Ticker": symbol,
            "Entry_Date": dates[entry_i].strftime("%Y-%m-%d"),
            "Exit_Date": dates[exit_i].strftime("%Y-%m-%d"),
            "TF": hunt["tf"] if hunt else "",
            "Entry": round(entry_px, 4),
            "SL": round(sl_px, 4),
            "Target_1to2": round(tp2, 4),
            "Exit": round(exit_px, 4),
            "Qty": int(qty),
            "R": round((exit_px - entry_px) / risk_sh, 4) if risk_sh > 0 else 0.0,
            "PnL": round(gross - fees, 2),
            "Reason": reason,
            "Equity_After": round(equity, 2),
        })
        last_exit_ts = dates[exit_i]
        # reset
        in_pos = False
        hunting = False
        hunt = None
        c1_bar = None
        entry_i = None
        qty = 0
        be_armed = False

    for i in range(n):
        if i in activate_at:
            active.extend(activate_at[i])
        o, h, l, c = float(opens[i]), float(highs[i]), float(lows[i]), float(closes[i])

        # ---- manage an open position first (this bar) ----
        if in_pos:
            is_entry_bar = (i == entry_i)
            if is_entry_bar:
                if l <= sl_px:
                    close_trade(i, sl_px, "SL")
                elif h >= tp2:
                    close_trade(i, tp2, "1:2")
                elif c < c1_high:
                    close_trade(i, c, "SCRATCH")
                # else hold to next bar
            else:
                stop_lv = entry_px if be_armed else sl_px
                if o < stop_lv:                      # gap through stop
                    fill = stop_lv if clip_gap else o
                    close_trade(i, fill, "BE" if be_armed else "SL_gap")
                elif o > tp2:                        # gap through target
                    fill = tp2 if clip_gap else o
                    close_trade(i, fill, "1:2_gap")
                elif l <= stop_lv:
                    close_trade(i, stop_lv, "BE" if be_armed else "SL")
                elif h >= tp2:
                    close_trade(i, tp2, "1:2")
                elif (not be_armed) and h >= r1:
                    be_armed = True                  # +1R -> SL to breakeven, stay 1:2
            # if still in position after management, no new hunt logic this bar
            if in_pos:
                # keep updating sweep counts / kills below, then continue
                pass

        # ---- update liquidity levels: kills + sweep counts ----
        for sup in active:
            if sup["gone"]:
                continue
            price = float(sup["price"])
            if c < price:
                sup["gone"] = True
                continue
            if l < price:
                prior = int(sup["sweep_count"])
                # start a hunt only if flat, not hunting, not in position
                if (not hunting) and (not in_pos) and i >= int(sup["trade_from"]) and prior <= M2_LOCK:
                    support, tf = _resolve_below(price, str(sup["timeframe"]), active, M2_LOCK)
                    if (hunt is None) or (TF_RANK.get(tf, 1) >= TF_RANK.get(hunt["tf"], 1)):
                        hunt = {
                            "support": support, "tf": tf,
                            "liq_ts": sup["liquidity_date"],
                            "until": i + HUNT_BARS, "sweep_low": l,
                        }
                        hunting = True
                        c1_bar = None
                        c1_high = c1_low = np.nan
                sup["sweep_count"] = prior + 1

        # ---- hunt for C1 / C2 -> enter C3 next open ----
        if hunting and (not in_pos):
            if c1_bar is None or i <= c1_bar:
                hunt["sweep_low"] = min(hunt["sweep_low"], l)
            support = hunt["support"]
            if i > hunt["until"]:
                hunting = False
                hunt = None
                c1_bar = None
            elif c1_bar is None:
                if c > o and o < support:            # C1 green open-below
                    c1_bar = i
                    c1_high = h
                    c1_low = l
            elif i == c1_bar + 1:                    # C2 candidate
                c2_ok = (c > c1_high) and (l >= c1_low)
                entered = False
                if c2_ok and i + 1 < n:              # need C3 bar to exist
                    entry_ts = dates[i + 1]
                    if _age_ok(hunt["liq_ts"], entry_ts):
                        sl_try = min(hunt["sweep_low"], c1_low) * SL_MULT
                        est = float(opens[i + 1])    # C3 open
                        if est - sl_try > MIN_RISK:
                            # open position at next bar open (C3)
                            in_pos = True
                            entry_i = i + 1
                            entry_px = est
                            sl_px = sl_try
                            risk = entry_px - sl_px
                            tp2 = entry_px + 2.0 * risk
                            r1 = entry_px + risk
                            be_armed = False
                            risk_cash = equity * RISK_PCT
                            qty = max(1, int(math.floor(risk_cash / risk)))
                            if first_entry_ts is None:
                                first_entry_ts = entry_ts
                            entered = True
                if not entered:
                    # C2 failed: this bar may itself be a fresh C1
                    if c > o and o < support:
                        c1_bar = i
                        c1_high = h
                        c1_low = l
                    else:
                        c1_bar = None
            else:  # bar_index > c1_bar + 1 (shouldn't linger, but re-seek C1)
                if c > o and o < support:
                    c1_bar = i
                    c1_high = h
                    c1_low = l
                else:
                    c1_bar = None

    if not trades:
        return {
            "Ticker": symbol, "Trades": 0, "Wins_1to2": 0, "Scratch": 0, "SL": 0,
            "BE": 0, "Net_Return_%": 0.0, "CAGR_%": 0.0, "Max_DD_%": 0.0,
            "Final_Equity": round(equity, 2), "First_Entry": "", "Last_Exit": "",
            "Bars": n, "Data_Start": dates[0].strftime("%Y-%m-%d"),
            "Data_End": dates[-1].strftime("%Y-%m-%d"),
        }

    net = equity / INIT_CAPITAL - 1.0
    yrs = max((last_exit_ts - first_entry_ts).days / 365.25, 1e-9) if first_entry_ts is not None else 0.0
    try:
        ratio = max(equity / INIT_CAPITAL, 1e-9)
        cagr = math.pow(ratio, 1.0 / yrs) - 1.0 if yrs > 0 and math.isfinite(ratio) else 0.0
        if not math.isfinite(cagr):
            cagr = 0.0
    except (OverflowError, ValueError):
        cagr = float("inf")
    reasons = [t["Reason"] for t in trades]
    wins = sum(1 for r in reasons if r.startswith("1:2"))
    scr = sum(1 for r in reasons if r == "SCRATCH")
    sls = sum(1 for r in reasons if r.startswith("SL"))
    bes = sum(1 for r in reasons if r == "BE")
    return {
        "Ticker": symbol,
        "Trades": len(trades),
        "Wins_1to2": wins,
        "Scratch": scr,
        "SL": sls,
        "BE": bes,
        "Net_Return_%": round(net * 100, 2),
        "CAGR_%": round(cagr * 100, 2),
        "Max_DD_%": round(max_dd * 100, 2),
        "Final_Equity": round(equity, 2),
        "First_Entry": first_entry_ts.strftime("%Y-%m-%d") if first_entry_ts is not None else "",
        "Last_Exit": last_exit_ts.strftime("%Y-%m-%d") if last_exit_ts is not None else "",
        "Bars": n,
        "Data_Start": dates[0].strftime("%Y-%m-%d"),
        "Data_End": dates[-1].strftime("%Y-%m-%d"),
        "_trades": trades,
    }


def _worker(args):
    symbol, start, clip_gap, keep_zero_vol, src = args
    try:
        return simulate(symbol, start, clip_gap, drop_zero_vol=not keep_zero_vol,
                        src_path=Path(src) if src else None)
    except Exception as exc:
        return {"Ticker": symbol, "error": str(exc)}


def fetch_tv(tickers: list[str], exchange: str, out_dir: Path, n_bars: int = 5000) -> None:
    """Pull TradingView's own daily OHLC via tvdatafeed into out_dir/<TICKER>.csv.

    Requires: pip install --upgrade tvdatafeed   (and a TradingView login for
    more history; anonymous works with fewer bars). This is the free-plan way to
    get the SAME series TradingView charts show, since native CSV export is paid.
    """
    try:
        from tvDatafeed import Interval, TvDatafeed
    except Exception:
        try:
            from tvdatafeed import Interval, TvDatafeed  # some installs
        except Exception as exc:
            raise SystemExit(
                "tvdatafeed not installed. Run:  pip install --upgrade "
                "git+https://github.com/rongardF/tvdatafeed.git\n" + str(exc)
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    tv = TvDatafeed()  # anonymous; pass username/password for full history
    for t in tickers:
        sym = t.replace(".NS", "").replace(".BO", "")
        try:
            df = tv.get_hist(symbol=sym, exchange=exchange, interval=Interval.in_daily, n_bars=n_bars)
            if df is None or df.empty:
                print(f"  ! {sym}: no data", flush=True)
                continue
            df = df.reset_index().rename(columns={"datetime": "time"})
            df.to_csv(out_dir / f"{sym}.csv", index=False)
            print(f"  fetched {sym}: {len(df):,} bars -> {out_dir / f'{sym}.csv'}", flush=True)
        except Exception as exc:
            print(f"  ! {sym}: {exc}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None, help="ignore setups before this date (default: all history)")
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--clip-gap", action="store_true",
                    help="cap gap-through fills at target/stop (kills dirty-data windfalls; closer to TV adjusted)")
    ap.add_argument("--keep-zero-vol", action="store_true",
                    help="keep volume=0 rows (default drops them; they are corrupt/non-trading bars)")
    ap.add_argument("--tv-dir", default=None,
                    help="folder of TradingView-exported / tvdatafeed CSVs; ticker = filename. "
                         "Run on TV's OWN data for an apples-to-apples match.")
    ap.add_argument("--fetch-tv", action="store_true",
                    help="first pull TradingView data via tvdatafeed into --tv-dir (needs --tickers)")
    ap.add_argument("--exchange", default="NSE", help="exchange for --fetch-tv (NSE/BSE)")
    ap.add_argument("--n-bars", type=int, default=5000, help="bars to fetch with --fetch-tv")
    args = ap.parse_args()

    start = pd.Timestamp(args.start) if args.start else START_DEFAULT

    tv_dir = Path(args.tv_dir) if args.tv_dir else None
    if args.fetch_tv:
        if not args.tickers:
            raise SystemExit("--fetch-tv needs --tickers RELIANCE.NS TCS.NS ...")
        if tv_dir is None:
            tv_dir = BASE_DIR / "data_tradingview"
        print(f"Fetching TradingView data ({args.exchange}) -> {tv_dir}", flush=True)
        fetch_tv(args.tickers, args.exchange, tv_dir, args.n_bars)

    src_map: dict[str, str] = {}
    if tv_dir is not None:
        files = sorted(tv_dir.glob("*.csv"))
        for f in files:
            src_map[f.stem] = str(f)
        if args.tickers:
            want = {t.replace(".NS", "").replace(".BO", "").replace("_1d", "") for t in args.tickers}
            src_map = {k: v for k, v in src_map.items() if k in want}
        syms = sorted(src_map)
        print(f"Replicating Pine Swing_PP on {len(syms):,} TradingView CSVs from {tv_dir}", flush=True)
    else:
        if args.tickers:
            syms = [t.replace("_1d.csv", "") for t in args.tickers]
        else:
            syms = sorted(p.name[:-len("_1d.csv")] for p in DATA_DIR.glob("*_1d.csv"))
        print(f"Replicating Pine Swing_PP on {len(syms):,} symbols (data_daily)  start>={start.date()}", flush=True)

    rows: list[dict] = []
    all_trades: list[dict] = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_worker, (s, start, args.clip_gap, args.keep_zero_vol, src_map.get(s))): s
                for s in syms}
        for fut in as_completed(futs):
            r = fut.result()
            done += 1
            if r is None or "error" in r:
                if r and "error" in r:
                    print(f"  ! {r['Ticker']}: {r['error']}", flush=True)
            else:
                tr = r.pop("_trades", [])
                all_trades.extend(tr)
                rows.append(r)
            if done % 200 == 0 or done == len(syms):
                print(f"  {done:,}/{len(syms):,}  usable {len(rows):,}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows).sort_values("Net_Return_%", ascending=False)
    per_stock = OUT_DIR / "Per_Stock_Returns.csv"
    res.to_csv(per_stock, index=False)
    pd.DataFrame(all_trades).to_csv(OUT_DIR / "All_Trades.csv", index=False)

    traded = res[res["Trades"] > 0]
    print("\n===== Pine Swing_PP replica (single-symbol, no ML) =====", flush=True)
    print(f"symbols with >=1 trade : {len(traded):,} / {len(res):,}", flush=True)
    if len(traded):
        print(f"total trades           : {int(traded['Trades'].sum()):,}", flush=True)
        print(f"avg net return / stock : {traded['Net_Return_%'].mean():+.2f}%", flush=True)
        print(f"median net return      : {traded['Net_Return_%'].median():+.2f}%", flush=True)
        print(f"stocks net positive    : {(traded['Net_Return_%'] > 0).sum():,} "
              f"({(traded['Net_Return_%'] > 0).mean() * 100:.1f}%)", flush=True)
        w = int(traded["Wins_1to2"].sum())
        s = int(traded["Scratch"].sum())
        sl = int(traded["SL"].sum())
        be = int(traded["BE"].sum())
        print(f"exits  1:2={w:,}  scratch={s:,}  SL={sl:,}  BE={be:,}", flush=True)
    print(f"\nper-stock -> {per_stock}", flush=True)
    print(f"all trades -> {OUT_DIR / 'All_Trades.csv'}", flush=True)


if __name__ == "__main__":
    main()
