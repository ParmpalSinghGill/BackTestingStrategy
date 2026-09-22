"""
Hunt 45%+ net Zerodha CAGR on locked swing-low liquidity.

Liquidity method is fixed: HTF swing low, >=2 candles both sides AND >=3 on
one side, never trade the next HTF bar, skip 2 dailies after the source low.
No future bars in features or model scores. Oracle rows are ceiling-only.
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
    SWING_MIN_ONE_SIDE,
    SWING_N2_DEFAULT,
    SWING_N_DEFAULT,
    SWING_SKIP_AFTER,
    get_swing_low_supports,
)
from swing_strategy.run_c1_entry_sl_matrix import (
    MAX_POST_SWEEP,
    START,
    _c1_match,
    _entries_for_c1,
    _sl_price,
)
from swing_strategy.run_ml_next_search import META_FEAT, select_meta, walk_meta
from swing_strategy.run_ml_target_books import CORE_COLS, feat_worker, walk
from swing_strategy.run_ml_wave3 import run_vol_managed
from swing_strategy.run_retry2_rr_returns import _rr_exit
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

V2 = BASE_DIR / "Reports" / "SwingLowLiquidity_v2"
OUT = BASE_DIR / "Reports" / "SwingLowCagrHunt"
SCORED_M22 = V2 / "Scored_v6_meta_M22.parquet"
RAW_M22 = V2 / "All_Setups_RR2_M22.csv"
FEAT_M22 = V2 / "Features_v6_M22.parquet"
TARGET = 45.0
M2_LOCK = 2

C1_KINDS = ("OPEN_BELOW", "CLOSE_BELOW", "HIGH_BELOW")
ENT_KINDS = ("A1", "A2", "A3", "B", "C2C")
RR_VALUES = (2.0, 3.0)
KNOB_BOOKS = [
    (50_000.0, 500.0, "50k_0.5k"),
    (50_000.0, 1_000.0, "50k_1k"),
    (50_000.0, 2_000.0, "50k_2k"),
    (50_000.0, 2_800.0, "50k_2.8k"),
    (50_000.0, 4_000.0, "50k_4k"),
    (100_000.0, 1_000.0, "100k_1k"),
    (100_000.0, 2_000.0, "100k_2k"),
]
ORACLE_TOP = (5, 10, 16, 32)
THRESHS = (None, 0.32, 0.36, 0.38, 0.42)
TOPNS = (12, 16, 24, 32)
TVS = (0.03, 0.04, 0.05)


def _log(row: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "hunt_log.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print("  " + json.dumps(row), flush=True)


def _best_so_far(rows: list[dict]) -> dict | None:
    ok = [r for r in rows if r.get("CAGR") is not None and not r.get("oracle")]
    if not ok:
        return None
    return max(ok, key=lambda r: float(r["CAGR"]))


def _entries_plus(c1: int, opens, highs, lows, closes, n: int) -> dict[str, tuple[int, float]]:
    out = _entries_for_c1(c1, opens, highs, lows, closes, n)
    c2 = c1 + 1
    if c2 + 1 < n and float(closes[c2]) > float(highs[c1]):
        out["C2C"] = (c2 + 1, float(closes[c2]))
    return out


def _try_variant(sweep_idx, support, opens, highs, lows, closes, n, c1_kind, ent_kind, sl_kind):
    end = min(n - 1, sweep_idx + MAX_POST_SWEEP)
    for c1 in range(sweep_idx, end):
        if not _c1_match(c1_kind, float(opens[c1]), float(highs[c1]), float(closes[c1]), support):
            continue
        ents = _entries_plus(c1, opens, highs, lows, closes, n)
        if ent_kind not in ents:
            continue
        eidx, entry = ents[ent_kind]
        if entry <= 0 or eidx >= n:
            continue
        sl = _sl_price(sl_kind, float(lows[c1]), float(min(lows[sweep_idx : c1 + 1])))
        if entry - sl <= 0.05:
            continue
        return int(eidx), float(entry), float(sl), int(c1)
    return None


def pick_oracle(raw: pd.DataFrame, top_n: int) -> pd.DataFrame:
    raw = raw.copy()
    raw["Entry_Date"] = pd.to_datetime(raw["Entry_Date"])
    keep = []
    for _, g in raw.groupby("Entry_Date", sort=False):
        keep.append(g.sort_values("Realized_R", ascending=False).head(top_n))
    return pd.concat(keep, ignore_index=True) if keep else raw.iloc[0:0]


def phase_oracle() -> list[dict]:
    print("===== Oracle ceiling (lookahead rank; not tradable) =====", flush=True)
    raw = pd.read_csv(RAW_M22)
    raw["Entry_Date"] = pd.to_datetime(raw["Entry_Date"])
    feat = pd.read_parquet(FEAT_M22)
    feat["Entry_Date"] = pd.to_datetime(feat["Entry_Date"])
    keys = ["Ticker", "Entry_Date"]
    merged = raw.merge(feat[keys + ["idio_vol"]], on=keys, how="left")
    rows = []
    for top_n in ORACLE_TOP:
        picked = pick_oracle(merged, top_n)
        for cap, risk, name in KNOB_BOOKS:
            res = run_vol_managed(picked, cap, risk, 0.04)
            row = {
                "phase": "oracle",
                "oracle": True,
                "top_n": top_n,
                "book": name,
                "CAGR": round(float(res["CAGR"]), 2),
                "N": int(res["N"]),
                "DD": float(res["DD"]),
            }
            _log(row)
            rows.append(row)
    path = OUT / "oracle.json"
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    best = max(rows, key=lambda r: r["CAGR"])
    print(f"Oracle peak {best['CAGR']:+.2f}%  {best['book']} top{best['top_n']}", flush=True)
    return rows


def phase_knobs() -> list[dict]:
    print("===== ML knobs on locked A1 M2=2 scored book =====", flush=True)
    scored = pd.read_parquet(SCORED_M22)
    scored["Entry_Date"] = pd.to_datetime(scored["Entry_Date"])
    rows = []
    for thresh in THRESHS:
        for top_n in TOPNS:
            picked = select_meta(scored, top_n, thresh, "Meta_P")
            if picked.empty:
                continue
            accept = 100.0 * len(picked) / max(len(scored), 1)
            for tv in TVS:
                for cap, risk, name in KNOB_BOOKS:
                    res = run_vol_managed(picked, cap, risk, tv)
                    row = {
                        "phase": "knobs",
                        "oracle": False,
                        "thresh": thresh,
                        "top_n": top_n,
                        "tv": tv,
                        "book": name,
                        "accept_pct": round(accept, 1),
                        "ml_n": int(len(picked)),
                        "CAGR": round(float(res["CAGR"]), 2),
                        "N": int(res["N"]),
                        "DD": float(res["DD"]),
                    }
                    rows.append(row)
                    if row["CAGR"] >= 35.0:
                        _log(row)
    rows.sort(key=lambda r: -r["CAGR"])
    (OUT / "knobs.json").write_text(json.dumps(rows[:80], indent=2), encoding="utf-8")
    print(f"Knob peak {rows[0]['CAGR']:+.2f}%  {rows[0]}", flush=True)
    for r in rows[:12]:
        print(
            f"  {r['CAGR']:+.2f}% {r['book']} t={r['thresh']} top{r['top_n']} tv{r['tv']} "
            f"n={r['N']} DD {r['DD']}",
            flush=True,
        )
    return rows


def _walk_kind(df: pd.DataFrame, cols: list[str], kind: str) -> pd.DataFrame:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    parts = []
    use = [c for c in cols if c in df.columns]
    for y in sorted(df["year"].unique()):
        cutoff = pd.Timestamp(f"{y}-01-01")
        train = df[(df["year"] < y) & (df["Exit_Date"] < cutoff)]
        test = df[df["year"] == y].copy()
        if test.empty:
            continue
        if len(train) < 800:
            test["ML_Score"] = test.get("rel_c2_thru", test.get("gap_pct", 0))
            parts.append(test)
            continue
        Xtr = train[use].to_numpy(np.float32)
        Xte = test[use].to_numpy(np.float32)
        ytr = train["y_win"].to_numpy()
        if kind == "hgb":
            clf = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.06, max_iter=220)
            clf.fit(Xtr, ytr)
            test["ML_Score"] = clf.predict_proba(Xte)[:, 1]
        elif kind == "mlp":
            sc = StandardScaler()
            clf = MLPClassifier(
                hidden_layer_sizes=(96, 48),
                activation="relu",
                max_iter=80,
                random_state=42,
                early_stopping=True,
                n_iter_no_change=6,
            )
            clf.fit(sc.fit_transform(Xtr), ytr)
            test["ML_Score"] = clf.predict_proba(sc.transform(Xte))[:, 1]
        elif kind == "cat":
            from catboost import CatBoostClassifier

            clf = CatBoostClassifier(
                iterations=280,
                depth=6,
                learning_rate=0.05,
                verbose=False,
                random_seed=42,
            )
            clf.fit(Xtr, ytr)
            test["ML_Score"] = clf.predict_proba(Xte)[:, 1]
        else:
            clf = XGBClassifier(
                n_estimators=350,
                max_depth=5,
                learning_rate=0.04,
                subsample=0.85,
                colsample_bytree=0.8,
                n_jobs=4,
                eval_metric="logloss",
                tree_method="hist",
            )
            clf.fit(Xtr, ytr)
            test["ML_Score"] = clf.predict_proba(Xte)[:, 1]
        parts.append(test)
        print(f"    {kind} {y} train={len(train):,}", flush=True)
    return pd.concat(parts, ignore_index=True)


def phase_models() -> list[dict]:
    print("===== Alternate models (purged yearly walk-forward) =====", flush=True)
    feat = pd.read_parquet(FEAT_M22)
    feat["Entry_Date"] = pd.to_datetime(feat["Entry_Date"])
    feat["Exit_Date"] = pd.to_datetime(feat["Exit_Date"])
    if "year" not in feat.columns:
        feat["year"] = feat["Entry_Date"].dt.year
    kinds = ["hgb", "mlp"]
    try:
        import catboost  # noqa: F401

        kinds.append("cat")
    except Exception:
        print("catboost not installed; skipping", flush=True)
    rows = []
    for kind in kinds:
        print(f"Primary {kind} ...", flush=True)
        scored = _walk_kind(feat, CORE_COLS, kind)
        print(f"Meta on {kind} ...", flush=True)
        scored = walk_meta(scored, META_FEAT)
        scored.to_parquet(OUT / f"Scored_{kind}.parquet", index=False)
        for thresh in (0.36, 0.38, 0.42):
            for top_n in (16, 24, 32):
                picked = select_meta(scored, top_n, thresh, "Meta_P")
                if picked.empty:
                    continue
                for cap, risk, name in (
                    (50_000.0, 1_000.0, "50k_1k"),
                    (50_000.0, 2_800.0, "50k_2.8k"),
                    (50_000.0, 4_000.0, "50k_4k"),
                    (100_000.0, 1_000.0, "100k_1k"),
                ):
                    res = run_vol_managed(picked, cap, risk, 0.04)
                    row = {
                        "phase": "models",
                        "oracle": False,
                        "model": kind,
                        "thresh": thresh,
                        "top_n": top_n,
                        "book": name,
                        "CAGR": round(float(res["CAGR"]), 2),
                        "N": int(res["N"]),
                        "DD": float(res["DD"]),
                    }
                    rows.append(row)
                    if row["CAGR"] >= 30.0:
                        _log(row)
    rows.sort(key=lambda r: -r["CAGR"])
    (OUT / "models.json").write_text(json.dumps(rows[:60], indent=2), encoding="utf-8")
    if rows:
        print(f"Model peak {rows[0]['CAGR']:+.2f}%  {rows[0]}", flush=True)
    return rows


def _scan_ticker_entries(symbol: str) -> list[dict]:
    df = _load_daily(symbol)
    if df is None or len(df) < 120:
        return []
    nifty = NIFTY_RANK.get(INDEX_CLASSIFIER.classify(symbol), 1)
    indexed = df.set_index("Date")
    swings = get_swing_low_supports(indexed, n=SWING_N_DEFAULT, n2=SWING_N2_DEFAULT)
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
            "sweep_count": sweeps,
            "gone": gone,
            "trade_from": int(trade_from),
        })
    active: list[dict] = []
    out: list[dict] = []
    seen: dict[tuple, set] = defaultdict(set)
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
            for c1_kind in C1_KINDS:
                for ent_kind in ENT_KINDS:
                    got = _try_variant(
                        i, support, opens, highs, lows, closes, n, c1_kind, ent_kind, "SWEEP_x99"
                    )
                    if got is None:
                        continue
                    eidx, entry, sl, _c1 = got
                    risk = entry - sl
                    if risk <= 0.05:
                        continue
                    for rr in RR_VALUES:
                        key = (c1_kind, ent_kind, rr, date_s.iloc[eidx].strftime("%Y-%m-%d"))
                        if key in seen[symbol]:
                            continue
                        seen[symbol].add(key)
                        exit_p, xidx, mfe, _hit, reason = _rr_exit(
                            eidx, entry, sl, rr, opens, highs, lows, n
                        )
                        out.append({
                            "Ticker": symbol,
                            "C1": c1_kind,
                            "EntryKind": ent_kind,
                            "RR": rr,
                            "Liquidity_Type": tf,
                            "TF_Rank": TF_RANK.get(tf, 1),
                            "Nifty_Rank": nifty,
                            "Support_Price": round(support, 2),
                            "Sweep_Count": prior,
                            "Entry_Date": date_s.iloc[eidx].strftime("%Y-%m-%d"),
                            "Exit_Date": date_s.iloc[xidx].strftime("%Y-%m-%d"),
                            "Entry_Price": round(entry, 2),
                            "SL_Price": round(sl, 2),
                            "Exit_Price": round(exit_p, 2),
                            "Realized_R": round(float((exit_p - entry) / risk), 4),
                            "MFE_R": round(float(mfe), 4),
                            "Outcome": "Success" if reason == "TP" else "Failure",
                        })
            sup["sweep_count"] = prior + 1
    return out


def phase_entries() -> list[dict]:
    print("===== C1 x entry x RR scan on locked swing-low =====", flush=True)
    cache = OUT / "All_Entry_Variants.csv"
    if cache.exists():
        print(f"Loading {cache}", flush=True)
        df = pd.read_csv(cache)
    else:
        tickers = sorted({p.name.split("_1d.csv")[0] for p in DATA_DAILY_DIR.glob("*_1d.csv")})
        rows: list[dict] = []
        done = 0
        with ProcessPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(_scan_ticker_entries, t): t for t in tickers}
            for fut in as_completed(futs):
                rows.extend(fut.result())
                done += 1
                if done % 200 == 0 or done == len(tickers):
                    print(f"  scan {done:,}/{len(tickers):,} rows={len(rows):,}", flush=True)
        df = pd.DataFrame(rows)
        OUT.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
        print(f"Saved {cache} n={len(df):,}", flush=True)
    summary = []
    for (c1, ent, rr), g in df.groupby(["C1", "EntryKind", "RR"]):
        wr = float((g["Outcome"] == "Success").mean() * 100.0)
        er = float(g["Realized_R"].mean())
        summary.append({
            "C1": c1,
            "EntryKind": ent,
            "RR": float(rr),
            "n": int(len(g)),
            "win_rate": round(wr, 2),
            "mean_R": round(er, 4),
            "edge": round(er * len(g), 1),
        })
    summary.sort(key=lambda r: -r["edge"])
    (OUT / "entry_raw.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Raw entry ranking (mean R * n):", flush=True)
    for r in summary[:15]:
        print(
            f"  {r['C1']:12} {r['EntryKind']:4} 1:{int(r['RR'])}  n={r['n']:,}  "
            f"WR {r['win_rate']:.1f}%  meanR {r['mean_R']:+.3f}",
            flush=True,
        )
    return summary


def _eval_raw_ml(raw: pd.DataFrame, tag: str) -> list[dict]:
    feat_path = OUT / f"Features_{tag}.parquet"
    scored_path = OUT / f"Scored_{tag}.parquet"
    if feat_path.exists():
        feat = pd.read_parquet(feat_path)
    else:
        by = defaultdict(list)
        for rec in raw.to_dict("records"):
            by[rec["Ticker"]].append(rec)
        rows: list[dict] = []
        jobs = list(by.items())
        print(f"Features {tag} {len(jobs):,} tickers ...", flush=True)
        with ProcessPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(feat_worker, j): j[0] for j in jobs}
            done = 0
            for fut in as_completed(futs):
                rows.extend(fut.result())
                done += 1
                if done % 400 == 0 or done == len(jobs):
                    print(f"  feat {done:,}/{len(jobs):,}", flush=True)
        feat = pd.DataFrame(rows)
        if feat.empty:
            return []
        feat["Entry_Date"] = pd.to_datetime(feat["Entry_Date"])
        feat["Exit_Date"] = pd.to_datetime(feat["Exit_Date"])
        feat["year"] = feat["Entry_Date"].dt.year
        feat["y_win"] = (feat["Outcome"] == "Success").astype(int)
        feat["y_r"] = feat["Realized_R"].clip(-2.0, 4.0)
        g = feat.groupby("Entry_Date")
        feat["n_cands"] = g["Ticker"].transform("size")
        feat["idio_ret20"] = feat["ret_20"] - g["ret_20"].transform("median")
        feat["idio_vol"] = feat["vol20"] - g["vol20"].transform("median")
        feat["rel_ret20"] = g["ret_20"].rank(pct=True)
        feat["rel_dolvol"] = g["dolvol_log"].rank(pct=True)
        feat["rel_risk_atr"] = g["risk_atr"].rank(pct=True)
        feat["rel_c2_thru"] = g["c2_thru_atr"].rank(pct=True)
        feat["rel_r_to_hh60"] = g["r_to_hh60"].rank(pct=True)
        for c in CORE_COLS:
            if c in feat.columns:
                feat[c] = pd.to_numeric(feat[c], errors="coerce").fillna(0.0)
        feat.to_parquet(feat_path, index=False)
    print(f"Walk {tag} ...", flush=True)
    scored = walk(feat, CORE_COLS, "xgb")
    scored = walk_meta(scored, META_FEAT)
    scored.to_parquet(scored_path, index=False)
    rows = []
    for thresh in (0.36, 0.38):
        for top_n in (16, 24, 32):
            picked = select_meta(scored, top_n, thresh, "Meta_P")
            if picked.empty:
                continue
            for cap, risk, name in (
                (50_000.0, 1_000.0, "50k_1k"),
                (50_000.0, 2_800.0, "50k_2.8k"),
                (50_000.0, 4_000.0, "50k_4k"),
                (100_000.0, 1_000.0, "100k_1k"),
            ):
                res = run_vol_managed(picked, cap, risk, 0.04)
                row = {
                    "phase": "entry_ml",
                    "oracle": False,
                    "tag": tag,
                    "thresh": thresh,
                    "top_n": top_n,
                    "book": name,
                    "CAGR": round(float(res["CAGR"]), 2),
                    "N": int(res["N"]),
                    "DD": float(res["DD"]),
                }
                rows.append(row)
                _log(row)
    return rows


def phase_entry_ml(summary: list[dict]) -> list[dict]:
    cache = OUT / "All_Entry_Variants.csv"
    df = pd.read_csv(cache)
    top = [s for s in summary if s["n"] >= 4000][:6]
    rows = []
    for s in top:
        tag = f"{s['C1']}_{s['EntryKind']}_RR{int(s['RR'])}"
        print(f"===== ML {tag} n={s['n']:,} meanR={s['mean_R']:+.3f} =====", flush=True)
        sub = df[
            (df["C1"] == s["C1"])
            & (df["EntryKind"] == s["EntryKind"])
            & (df["RR"] == s["RR"])
        ].copy()
        rows.extend(_eval_raw_ml(sub, tag))
    rows.sort(key=lambda r: -r["CAGR"])
    (OUT / "entry_ml.json").write_text(json.dumps(rows[:80], indent=2), encoding="utf-8")
    return rows


def phase_liq_params() -> list[dict]:
    """N=2/N2=2/S=3 is the locked method; M36 already tested 3/3."""
    print("===== Liquidity N/N2/S =====", flush=True)
    print("Locked (2,2,3). N=N2=3 already lost (M36). Not re-gridded.", flush=True)
    return []


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    all_rows.extend(phase_oracle())
    knobs = phase_knobs()
    all_rows.extend(knobs)
    all_rows.extend(phase_models())
    summary = phase_entries()
    all_rows.extend(phase_entry_ml(summary))
    phase_liq_params()
    tradable = [r for r in all_rows if r.get("CAGR") is not None and not r.get("oracle")]
    tradable.sort(key=lambda r: -float(r["CAGR"]))
    (OUT / "best.json").write_text(json.dumps(tradable[:40], indent=2), encoding="utf-8")
    if tradable:
        best = tradable[0]
        print("\n===== BEST TRADABLE =====", flush=True)
        print(json.dumps(best, indent=2), flush=True)
        print(f"Target {TARGET}%  hit={best['CAGR'] >= TARGET}", flush=True)


if __name__ == "__main__":
    main()
