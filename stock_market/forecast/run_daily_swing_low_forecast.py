"""
Daily swing-low entries: OPEN_BELOW + A1 + HGB meta (M41 / SWING_LOW_BOOKS.md).

Liquidity: HTF Yearly/Monthly/Weekly swing low (2 both sides AND 3 on one),
M2=2 wick sweeps, close-below kills, skip 2 dailies after source low.
Does NOT use calendar min-low. Does NOT write Swing_Live*.txt.

Clock (local):
  before 16:00 — last complete bar = previous trading day
  at/after 16:00 — last complete bar = today if weekday
  after 15:30, if today's bar is already on disk, use it
  Daily schedule: run via run_daily_all_forecasts.py at 16:00 (fetch once).

Writes (default daily job):
  forecast_stocks/Swing_low_<DD_Mon_YYYY>.txt
  forecast_stocks/Swing_low.txt
  Watchlist/Swing_low.txt

--min-age N  (N>=1) writes only Swing_low_{N}m*.txt — never Swing_low or Swing_Live.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.backtest_engine.backtest_support_liquidity_strategy import (
    INDEX_CLASSIFIER,
    SWING_N2_DEFAULT,
    SWING_N_DEFAULT,
    SWING_SKIP_AFTER,
    get_swing_low_supports,
)
from swing_strategy.run_c1_entry_sl_matrix import MAX_POST_SWEEP, _c1_match
from swing_strategy.run_daily_swing_forecast import (
    _feats_at_c2,
    add_cross_section,
    fetch_latest,
    latest_bar_on_disk,
    next_session,
    to_fyers,
)
from swing_strategy.run_ml_next_search import META_FEAT
from swing_strategy.run_ml_target_books import CORE_COLS
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

FORECAST_DIR = BASE_DIR / "forecast_stocks"
WATCHLIST_DIR = Path(r"C:\Users\parmp\Downloads\Watchlist")
FEAT_PATH = BASE_DIR / "Reports" / "SwingLowLiquidity_v2" / "Features_v6_M22.parquet"
SCORED_PATH = BASE_DIR / "Reports" / "SwingLowCagrHunt" / "Scored_hgb.parquet"
CUT = 16, 0
META_MIN = 0.42
TOP_N = 32
M2_LOCK = 2
MIN_AGE_MONTHS = 0
KEEP = list(CORE_COLS)


def _liq_date_for(support: float, tf: str, active: list[dict], fallback) -> pd.Timestamp:
    for s in active:
        if abs(float(s["price"]) - float(support)) < 1e-6 and str(s["timeframe"]) == tf:
            return pd.Timestamp(s["liquidity_date"]).normalize()
    return pd.Timestamp(fallback).normalize()


def _age_ok(liq, entry, months: int) -> bool:
    if months <= 0:
        return True
    return pd.Timestamp(liq) + pd.DateOffset(months=int(months)) <= pd.Timestamp(entry)


def last_complete_session(now: datetime) -> datetime.date:
    today = now.date()
    if now.time() < time(*CUT):
        d = today - timedelta(days=1)
        print(f"[clock] {now:%H:%M} before 16:00 - use previous day unless today's bar is on disk", flush=True)
    else:
        d = today
        print(f"[clock] {now:%H:%M} at/after 16:00 - include today's bar if it exists", flush=True)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _swing_low_pending_worker(payload: tuple) -> dict | None:
    path, asof_s = payload
    symbol = path.name.replace("_1d.csv", "").replace("_NS", ".NS").replace("_BO", ".BO")
    if "ticker_cache" in symbol:
        return None
    asof = pd.Timestamp(asof_s)
    df = _load_daily(symbol)
    if df is None or len(df) < 120:
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Date"] <= asof].sort_values("Date").reset_index(drop=True)
    if len(df) < 120:
        return None
    last = pd.Timestamp(df["Date"].iloc[-1]).normalize()
    if last != asof.normalize():
        return None

    try:
        swings = get_swing_low_supports(df.set_index("Date"), n=SWING_N_DEFAULT, n2=SWING_N2_DEFAULT)
    except Exception:
        return None
    if not swings:
        return None

    date_s = pd.to_datetime(df["Date"]).dt.normalize()
    dates = pd.DatetimeIndex(date_s)
    opens = df["Open"].to_numpy(float)
    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    closes = df["Close"].to_numpy(float)
    vols = df["Volume"].to_numpy(float) if "Volume" in df.columns else np.ones(len(df))
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
    best = None
    for i in range(n):
        if i in activate_at:
            active.extend(activate_at[i])
        low_i = float(lows[i])
        close_i = float(closes[i])
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
            if i >= int(sup.get("trade_from", 0)) and prior <= M2_LOCK:
                support, tf = _resolve_valid_below(price, str(sup["timeframe"]), active, M2_LOCK)
                end = min(n - 1, i + MAX_POST_SWEEP)
                for c1 in range(i, end):
                    if not _c1_match("OPEN_BELOW", float(opens[c1]), float(highs[c1]), float(closes[c1]), support):
                        continue
                    c2 = c1 + 1
                    if c2 != n - 1:
                        continue
                    if float(closes[c2]) <= float(highs[c1]):
                        continue
                    if float(lows[c2]) < float(lows[c1]):
                        continue
                    sweep_low = float(min(lows[i : c1 + 1]))
                    sl = round(sweep_low * 0.99, 2)
                    entry_est = round(float(closes[c2]), 2)
                    if entry_est - sl <= 0.05:
                        continue
                    liq_d = _liq_date_for(support, tf, active, sup["liquidity_date"])
                    rec = {
                        "Ticker": symbol,
                        "Liquidity_Type": tf,
                        "TF_Rank": TF_RANK.get(tf, 1),
                        "Nifty_Rank": NIFTY_RANK.get(INDEX_CLASSIFIER.classify(symbol), 1),
                        "Support_Price": round(float(support), 2),
                        "Liquidity_Date": liq_d.strftime("%Y-%m-%d"),
                        "C1_Date": pd.Timestamp(dates[c1]).strftime("%Y-%m-%d"),
                        "C2_Date": pd.Timestamp(dates[c2]).strftime("%Y-%m-%d"),
                        "Entry_Price": entry_est,
                        "SL_Price": sl,
                        "c1": c1,
                        "sweep": i,
                        "sweep_age_bars": float(c1 - i),
                    }
                    feats = _feats_at_c2(opens, highs, lows, closes, vols, rec)
                    if feats:
                        rec.update(feats)
                        if best is None or rec["TF_Rank"] > best["TF_Rank"]:
                            best = rec
                    break
            sup["sweep_count"] = prior + 1
    return best


def train_models(asof: pd.Timestamp, min_age_months: int = 0):
    from xgboost import XGBClassifier

    if not FEAT_PATH.exists():
        raise SystemExit(f"missing {FEAT_PATH} — train swing-low features first")
    feat = pd.read_parquet(FEAT_PATH)
    feat["Exit_Date"] = pd.to_datetime(feat["Exit_Date"])
    feat["Entry_Date"] = pd.to_datetime(feat["Entry_Date"])
    if min_age_months > 0 and "Liquidity_Date" in feat.columns:
        feat["Liquidity_Date"] = pd.to_datetime(feat["Liquidity_Date"])
        feat = feat[
            feat["Liquidity_Date"] + pd.DateOffset(months=min_age_months) <= feat["Entry_Date"]
        ].copy()
        print(
            f"[train] keep Liquidity_Date + {min_age_months}m <= Entry_Date  "
            f"({len(feat):,} rows)",
            flush=True,
        )
    train = feat[feat["Exit_Date"] < asof]
    cols = [c for c in KEEP if c in train.columns]
    if len(train) < 800:
        raise SystemExit("not enough purged history to train")
    print(f"[train] HGB primary on {len(train):,} rows (exit < {asof.date()})", flush=True)
    primary = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.06, max_iter=220)
    primary.fit(train[cols].to_numpy(np.float32), train["y_win"].to_numpy())

    if SCORED_PATH.exists():
        sc = pd.read_parquet(SCORED_PATH)
        sc["Entry_Date"] = pd.to_datetime(sc["Entry_Date"])
        sc["Exit_Date"] = pd.to_datetime(sc["Exit_Date"])
        m = train.drop(columns=["ML_Score"], errors="ignore").merge(
            sc[["Ticker", "Entry_Date", "ML_Score"]], on=["Ticker", "Entry_Date"], how="left"
        )
        m["ML_Score"] = m["ML_Score"].fillna(m["y_win"].astype(float) * 0.4 + 0.3)
    else:
        m = train.copy()
        m["ML_Score"] = primary.predict_proba(train[cols].to_numpy(np.float32))[:, 1]
    mtr = m[m["Exit_Date"] < asof]
    meta_cols = [c for c in META_FEAT if c in mtr.columns]
    print(f"[train] XGB meta on {len(mtr):,}", flush=True)
    meta = XGBClassifier(
        n_estimators=280, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.8, n_jobs=4,
        eval_metric="logloss", tree_method="hist",
        reg_lambda=1.5,
    )
    meta.fit(mtr[meta_cols].to_numpy(np.float32), mtr["y_win"].to_numpy())
    return primary, meta, cols, meta_cols


def write_txt(symbols: list[str], entry_day: datetime.date, min_age_months: int = 0) -> None:
    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    WATCHLIST_DIR.mkdir(parents=True, exist_ok=True)
    head = ["NSE:NIFTY50-INDEX", "BSE:SENSEX-INDEX"]
    line = ",".join(head + symbols)
    stamp = entry_day.strftime("%d_%b_%Y")
    if min_age_months > 0:
        tag = f"Swing_low_{min_age_months}m"
        paths = [
            FORECAST_DIR / f"{tag}_{stamp}.txt",
            FORECAST_DIR / f"{tag}.txt",
            WATCHLIST_DIR / f"{tag}.txt",
        ]
    else:
        paths = [
            FORECAST_DIR / f"Swing_low_{stamp}.txt",
            FORECAST_DIR / "Swing_low.txt",
            WATCHLIST_DIR / "Swing_low.txt",
        ]
    for path in paths:
        path.write_text(line, encoding="utf-8")
        print(f"[write] {path}  ({len(symbols)} names)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--asof", default="", help="YYYY-MM-DD override for last complete bar")
    ap.add_argument(
        "--min-age",
        type=int,
        default=0,
        help="keep liquidity at least N months old; writes Swing_low_Nm.txt, not Swing_low.txt",
    )
    args = ap.parse_args()
    min_age = int(args.min_age)

    now = datetime.now()
    if now.weekday() >= 5 and not args.asof:
        print(f"[clock] {now:%A %Y-%m-%d} weekend - skip (no Sat/Sun run)", flush=True)
        return
    asof_d = datetime.strptime(args.asof, "%Y-%m-%d").date() if args.asof else last_complete_session(now)

    if not args.skip_fetch:
        fetch_latest()

    disk = latest_bar_on_disk()
    if args.asof:
        if disk is not None and disk < asof_d:
            print(f"[asof] data on disk only through {disk} - clamping", flush=True)
            asof_d = disk
    else:
        # After the cash close, use today's bar if the fetch actually wrote it.
        if (
            disk is not None
            and now.weekday() < 5
            and now.time() >= time(15, 30)
            and disk >= now.date()
        ):
            asof_d = now.date()
            print(f"[asof] today's close is on disk ({disk}) - using {asof_d}", flush=True)
        elif disk is not None and disk < asof_d:
            print(f"[asof] data on disk only through {disk} - clamping", flush=True)
            asof_d = disk

    asof = pd.Timestamp(asof_d)
    entry_d = next_session(asof_d)
    print(f"[asof] last complete bar {asof_d}  ->  enter next session {entry_d}", flush=True)

    files = sorted(DATA_DAILY_DIR.glob("*_1d.csv"))
    print(f"[scan] {len(files):,} tickers for pending swing-low A1 on {asof_d}", flush=True)
    rows: list[dict] = []
    jobs = [(p, asof_d.isoformat()) for p in files]
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(_swing_low_pending_worker, j) for j in jobs]
        done = 0
        for fut in as_completed(futs):
            rec = fut.result()
            if rec:
                rows.append(rec)
            done += 1
            if done % 400 == 0 or done == len(jobs):
                print(f"  scan {done}/{len(jobs)}  pending {len(rows)}", flush=True)

    if not rows:
        print("[ml] no pending A1 setups", flush=True)
        write_txt([], entry_d, min_age)
        return

    if min_age > 0:
        before = len(rows)
        rows = [r for r in rows if _age_ok(r.get("Liquidity_Date"), entry_d, min_age)]
        print(f"[age] >= {min_age} month at {entry_d}: kept {len(rows)} / {before} pending A1", flush=True)
        if not rows:
            print("[ml] no pending A1 after min-age filter", flush=True)
            write_txt([], entry_d, min_age)
            return

    raw = add_cross_section(rows)
    for c in KEEP:
        if c not in raw.columns:
            raw[c] = 0.0
        else:
            raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0.0)
    print(f"[ml] {len(raw)} pending A1  -> score", flush=True)
    primary, meta, cols, meta_cols = train_models(asof, min_age)
    raw["ML_Score"] = primary.predict_proba(raw[cols].to_numpy(np.float32))[:, 1]
    use = [c for c in meta_cols if c in raw.columns]
    raw["Meta_P"] = meta.predict_proba(raw[use].to_numpy(np.float32))[:, 1]
    picked = raw[raw["Meta_P"] >= META_MIN].sort_values("Meta_P", ascending=False).head(TOP_N)
    print(f"[ml] Meta_P>={META_MIN}: {len(raw[raw['Meta_P']>=META_MIN])}  kept {len(picked)}", flush=True)
    if min_age > 0:
        print("[ml] all pending after age filter, by Meta_P:", flush=True)
        for _, r in raw.sort_values("Meta_P", ascending=False).iterrows():
            flag = "KEEP" if float(r["Meta_P"]) >= META_MIN else "drop"
            print(
                f"  {flag:4s} {r['Ticker']:16s}  Meta_P {r['Meta_P']:.3f}  "
                f"liq {r.get('Liquidity_Date','')}  {r['Liquidity_Type']}",
                flush=True,
            )
    for _, r in picked.iterrows():
        print(
            f"  {r['Ticker']:16s}  Meta_P {r['Meta_P']:.3f}  SL {r['SL_Price']:.2f}  "
            f"est_entry {r['Entry_Price']:.2f}  {r['Liquidity_Type']}",
            flush=True,
        )
    write_txt([to_fyers(str(t)) for t in picked["Ticker"]], entry_d, min_age)


if __name__ == "__main__":
    main()
