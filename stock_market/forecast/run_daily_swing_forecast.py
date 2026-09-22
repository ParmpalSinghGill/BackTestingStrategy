"""
Daily live entries: OPEN_BELOW + A1 + meta-label (same rules as ALGORITHM.md).

Liquidity is the live scanner only:
  Year/Month/Week calendar min-Low, skip flat/doji HTF prints,
  first wick through marks the level swept (never reused),
  5% lower-level swap uses intact (not yet swept) supports only.
  That is the KHAITAN correction. Do not call get_swing_low_supports here.

Clock (local):
  before 16:00 - last complete bar = previous trading day (ignore today's partial bar)
  at/after 16:00 - last complete bar = today if it is a weekday, else last weekday

Writes only text files (no Excel). Names say Live so this is not the parked swing-low scanner:
  forecast_stocks/Swing_Live_<DD_Mon_YYYY>.txt   dated list (entry session)
  forecast_stocks/Swing_Live.txt                 overwritten each run

  How the list is built: Year/Month/Week calendar min-Low (skip flat/doji),
  first wick through marks swept, 5% swap only among intact supports,
  C1 open-below + A1, Meta_P >= 0.38, daily top 32.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from swing_strategy.run_c1_entry_sl_matrix import MAX_POST_SWEEP, START, _c1_match
from swing_strategy.run_ml_next_search import META_FEAT
from swing_strategy.run_ml_target_books import CORE_COLS
from swing_strategy.run_ml_top5_selector import _rsi, _sma
from swing_strategy.tiered_liquidity_strategy_engine import (
    DATA_DAILY_DIR,
    TF_RANK,
    _load_daily,
    _resolve_effective_liquidity,
)
from src.backtest_engine.backtest_support_liquidity_strategy import get_all_stock_supports

FORECAST_DIR = BASE_DIR / "forecast_stocks"
WATCHLIST_DIR = Path(r"C:\Users\parmp\Downloads\Watchlist")
LIVE_ML_DIR = BASE_DIR / "Reports" / "LiquidityFix_IntactSupport"
FEAT_PATH = LIVE_ML_DIR / "Features_v6.parquet"
SCORED_PATH = LIVE_ML_DIR / "Scored_v6_meta.parquet"
CUT = 16, 0
META_MIN = 0.38
TOP_N = 32
KEEP = [c for c in CORE_COLS if c not in ("TF_Rank", "Nifty_Rank", "sweep_age_bars")]


def to_fyers(ticker: str) -> str:
    clean = (
        ticker.replace(".NS", "").replace(".BO", "")
        .replace("_NS", "").replace("_BO", "").split("=")[0].upper()
    )
    if ".BO" in ticker or ticker.endswith(".BO"):
        return f"BSE:{clean}-EQ"
    return f"NSE:{clean}-EQ"


def last_complete_session(now: datetime) -> datetime.date:
    today = now.date()
    if now.time() < time(*CUT):
        d = today - timedelta(days=1)
        print(f"[clock] {now:%H:%M} before 16:00 - use previous day (no today bar)", flush=True)
    else:
        d = today
        print(f"[clock] {now:%H:%M} at/after 16:00 - include today's bar if it exists", flush=True)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def latest_bar_on_disk() -> datetime.date | None:
    probe = DATA_DAILY_DIR / "RELIANCE.NS_1d.csv"
    if not probe.exists():
        files = list(DATA_DAILY_DIR.glob("*_1d.csv"))
        probe = files[0] if files else None
    if probe is None or not probe.exists():
        return None
    try:
        tail = pd.read_csv(probe, usecols=["Date"])
        return pd.to_datetime(tail["Date"]).max().date()
    except Exception:
        return None


def next_session(asof: datetime.date) -> datetime.date:
    d = asof + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def fetch_latest() -> None:
    from src.data_fetchers.fetch_daily_data import get_all_tickers, process_symbol

    symbols = [s for s in get_all_tickers() if s and not s.endswith("=F")]
    print(f"[fetch] updating {len(symbols):,} daily files ...", flush=True)
    done = 0
    from concurrent.futures import ThreadPoolExecutor, as_completed as tac

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(process_symbol, s, "1990-01-01"): s for s in symbols}
        for fut in tac(futs):
            done += 1
            if done % 200 == 0 or done == len(symbols):
                print(f"  fetch {done}/{len(symbols)}", flush=True)
            try:
                fut.result()
            except Exception:
                pass
    print("[fetch] done", flush=True)


def _pending_worker(payload: tuple) -> dict | None:
    path, asof_s = payload
    symbol = path.name.replace("_1d.csv", "").replace("_NS", ".NS").replace("_BO", ".BO")
    if "ticker_cache" in symbol:
        return None
    asof = pd.Timestamp(asof_s)
    df = _load_daily(symbol)
    if df is None or len(df) < 80:
        return None
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Date"] <= asof].sort_values("Date").reset_index(drop=True)
    if len(df) < 80:
        return None
    dates_np = df["Date"].to_numpy()
    opens = df["Open"].to_numpy(float)
    highs = df["High"].to_numpy(float)
    lows = df["Low"].to_numpy(float)
    closes = df["Close"].to_numpy(float)
    vols = df["Volume"].to_numpy(float) if "Volume" in df.columns else np.ones(len(df))
    n = len(df)
    last = pd.Timestamp(dates_np[-1]).normalize()
    if last != asof.normalize():
        return None

    try:
        all_supports = get_all_stock_supports(df.set_index("Date"))
    except Exception:
        return None
    sup_by_date: dict = {}
    for s in all_supports:
        sup_by_date.setdefault(pd.Timestamp(s["formed_date"]).normalize(), []).append(s)
    active: list[dict] = []
    best = None
    for i in range(n):
        curr_dt = pd.Timestamp(dates_np[i]).normalize()
        if curr_dt in sup_by_date:
            for s in sup_by_date[curr_dt]:
                active.append({"price": float(s["price"]), "timeframe": s["timeframe"], "swept": False})
        if curr_dt < START:
            for sup in active:
                if not sup["swept"] and lows[i] < sup["price"]:
                    sup["swept"] = True
            continue
        for sup in list(active):
            if sup["swept"] or lows[i] >= sup["price"]:
                continue
            sup["swept"] = True
            support, tf = _resolve_effective_liquidity(float(sup["price"]), str(sup["timeframe"]), active)
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
                rec = {
                    "Ticker": symbol,
                    "Liquidity_Type": tf,
                    "TF_Rank": TF_RANK.get(tf, 1),
                    "Support_Price": round(float(support), 2),
                    "C1_Date": pd.Timestamp(dates_np[c1]).strftime("%Y-%m-%d"),
                    "C2_Date": pd.Timestamp(dates_np[c2]).strftime("%Y-%m-%d"),
                    "Entry_Price": entry_est,
                    "SL_Price": sl,
                    "c1": c1,
                    "sweep": i,
                }
                feats = _feats_at_c2(opens, highs, lows, closes, vols, rec)
                if feats:
                    rec.update(feats)
                    if best is None or rec["TF_Rank"] > best["TF_Rank"]:
                        best = rec
                break
    return best


def _feats_at_c2(opens, highs, lows, closes, vols, rec: dict) -> dict | None:
    p = int(rec["c1"]) + 1
    if p < 70:
        return None
    c = float(closes[p])
    if c <= 0:
        return None
    o_today = c
    entry = float(rec["Entry_Price"])
    sl = float(rec["SL_Price"])
    sup = float(rec["Support_Price"])
    risk = max(entry - sl, 1e-6)
    sma50 = _sma(closes, 50)
    sma200 = _sma(closes, 200)
    tr = np.maximum(
        highs - lows,
        np.maximum(np.abs(highs - np.roll(closes, 1)), np.abs(lows - np.roll(closes, 1))),
    )
    tr[0] = highs[0] - lows[0]
    atr14 = _sma(tr, 14)
    logc = np.log(np.clip(closes, 1e-6, None))
    rets = np.diff(logc, prepend=logc[0])
    vol20 = pd.Series(rets).rolling(20, min_periods=10).std().to_numpy()
    vol60 = pd.Series(rets).rolling(60, min_periods=20).std().to_numpy()
    rsi = _rsi(closes, 14)
    vsma = _sma(vols, 20)
    hh20 = pd.Series(highs).rolling(20, min_periods=5).max().to_numpy()
    hh60 = pd.Series(highs).rolling(60, min_periods=10).max().to_numpy()
    maxret20 = pd.Series(rets).rolling(20, min_periods=5).max().to_numpy()
    atr = float(atr14[p]) if atr14[p] == atr14[p] and atr14[p] > 0 else max(c * 0.02, 1e-6)
    c1 = int(rec["c1"])
    sweep = int(rec["sweep"])
    sweep_low = float(min(lows[sweep : c1 + 1]))
    c1_o, c1_h, c1_l, c1_c = float(opens[c1]), float(highs[c1]), float(lows[c1]), float(closes[c1])
    c1_rng = max(c1_h - c1_l, 1e-6)

    def ret_n(k):
        j = p - k
        if j < 0 or closes[j] <= 0:
            return 0.0
        return float(closes[p] / closes[j] - 1.0)

    return {
        "risk_pct": risk / entry,
        "dist_support_pct": (entry - sup) / sup if sup else 0.0,
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
        "gap_pct": o_today / c - 1.0,
        "risk_atr": risk / atr,
        "r_to_hh60": (float(hh60[p]) - entry) / risk if hh60[p] == hh60[p] else 0.0,
        "sweep_depth_pct": (sup - sweep_low) / sup if sup else 0.0,
        "c1_reclaim_pct": (c1_c - sup) / sup if sup else 0.0,
        "c1_close_loc": (c1_c - c1_l) / c1_rng,
        "c2_thru_atr": (c - c1_h) / atr,
        "open_vs_brk_atr": (o_today - c1_h) / atr,
        "idio_vol": 0.0,
        "n_cands": 0.0,
        "idio_ret20": 0.0,
        "rel_ret20": 0.5,
        "rel_dolvol": 0.5,
        "rel_risk_atr": 0.5,
        "rel_c2_thru": 0.5,
        "rel_r_to_hh60": 0.5,
    }


def add_cross_section(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["n_cands"] = float(len(df))
    med_r = df["ret_20"].median()
    med_v = df["vol20"].median()
    df["idio_ret20"] = df["ret_20"] - med_r
    df["idio_vol"] = df["vol20"] - med_v
    df["rel_ret20"] = df["ret_20"].rank(pct=True)
    df["rel_dolvol"] = df["dolvol_log"].rank(pct=True)
    df["rel_risk_atr"] = df["risk_atr"].rank(pct=True)
    df["rel_c2_thru"] = df["c2_thru_atr"].rank(pct=True)
    df["rel_r_to_hh60"] = df["r_to_hh60"].rank(pct=True)
    for c in KEEP:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def train_models(asof: pd.Timestamp):
    from xgboost import XGBClassifier

    if not FEAT_PATH.exists():
        raise SystemExit(f"missing {FEAT_PATH} — train features first")
    feat = pd.read_parquet(FEAT_PATH)
    feat["Exit_Date"] = pd.to_datetime(feat["Exit_Date"])
    feat["Entry_Date"] = pd.to_datetime(feat["Entry_Date"])
    train = feat[feat["Exit_Date"] < asof]
    cols = [c for c in KEEP if c in train.columns]
    if len(train) < 800:
        raise SystemExit("not enough purged history to train")
    print(f"[train] primary on {len(train):,} rows (exit < {asof.date()})", flush=True)
    primary = XGBClassifier(
        n_estimators=350, max_depth=5, learning_rate=0.04,
        subsample=0.85, colsample_bytree=0.8, n_jobs=4,
        eval_metric="logloss", tree_method="hist",
    )
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
    print(f"[train] meta on {len(mtr):,}", flush=True)
    meta = XGBClassifier(
        n_estimators=280, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.8, n_jobs=4,
        eval_metric="logloss", tree_method="hist",
        reg_lambda=1.5,
    )
    meta.fit(mtr[meta_cols].to_numpy(np.float32), mtr["y_win"].to_numpy())
    return primary, meta, cols, meta_cols


def write_txt(symbols: list[str], entry_day: datetime.date) -> None:
    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    WATCHLIST_DIR.mkdir(parents=True, exist_ok=True)
    head = ["NSE:NIFTY50-INDEX", "BSE:SENSEX-INDEX"]
    line = ",".join(head + symbols)
    stamp = entry_day.strftime("%d_%b_%Y")
    paths = [
        FORECAST_DIR / f"Swing_Live_{stamp}.txt",
        FORECAST_DIR / "Swing_Live.txt",
        WATCHLIST_DIR / "Swing_Live.txt",
    ]
    for path in paths:
        path.write_text(line, encoding="utf-8")
        print(f"[write] {path}  ({len(symbols)} names)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--asof", default="", help="YYYY-MM-DD override for last complete bar")
    args = ap.parse_args()

    now = datetime.now()
    if now.weekday() >= 5 and not args.asof:
        print(f"[clock] {now:%A %Y-%m-%d} weekend - skip (no Sat/Sun run)", flush=True)
        return
    asof_d = datetime.strptime(args.asof, "%Y-%m-%d").date() if args.asof else last_complete_session(now)
    asof = pd.Timestamp(asof_d)
    entry_d = next_session(asof_d)
    print(f"[asof] last complete bar {asof_d}  ->  enter next session {entry_d}", flush=True)

    if not args.skip_fetch:
        fetch_latest()

    disk = latest_bar_on_disk()
    if disk is not None and disk < asof_d:
        print(f"[asof] data on disk only through {disk} - clamping (run without --skip-fetch to download)", flush=True)
        asof_d = disk
        asof = pd.Timestamp(asof_d)
        entry_d = next_session(asof_d)

    files = sorted(DATA_DAILY_DIR.glob("*_1d.csv"))
    print(f"[scan] {len(files):,} tickers for pending A1 on {asof_d}", flush=True)
    rows: list[dict] = []
    jobs = [(p, asof_d.isoformat()) for p in files]
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(_pending_worker, j) for j in jobs]
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
        write_txt([], entry_d)
        return

    raw = add_cross_section(rows)
    print(f"[ml] {len(raw)} pending A1  -> score", flush=True)
    primary, meta, cols, meta_cols = train_models(asof)
    raw["ML_Score"] = primary.predict_proba(raw[cols].to_numpy(np.float32))[:, 1]
    use = [c for c in meta_cols if c in raw.columns]
    raw["Meta_P"] = meta.predict_proba(raw[use].to_numpy(np.float32))[:, 1]
    picked = raw[raw["Meta_P"] >= META_MIN].sort_values("Meta_P", ascending=False).head(TOP_N)
    print(f"[ml] Meta_P>={META_MIN}: {len(raw[raw['Meta_P']>=META_MIN])}  kept {len(picked)}", flush=True)
    for _, r in picked.iterrows():
        print(
            f"  {r['Ticker']:16s}  Meta_P {r['Meta_P']:.3f}  SL {r['SL_Price']:.2f}  "
            f"est_entry {r['Entry_Price']:.2f}  {r['Liquidity_Type']}",
            flush=True,
        )
    write_txt([to_fyers(str(t)) for t in picked["Ticker"]], entry_d)


if __name__ == "__main__":
    main()
