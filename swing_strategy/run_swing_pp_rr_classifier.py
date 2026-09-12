"""Walk-forward RR picker for Swing_PP. Default stays 1:2.

Why 65% tagging 3R is not a 65% money edge
------------------------------------------
Banking 2R vs holding for 3R: a miss scratches at entry (0R), so you give back
2R. Break-even is P(3R | would have hit 2R) > 2/3. Unconditional ~65% is
slightly below that, and holding longer locks cash. This file only raises the
target when a purged walk-forward model is precise enough to clear that bar.

Live-valid decision points
--------------------------
- at_entry: choose 1:2 vs 1:3/1:4/1:5 from features known at C3 open.
- at_1R: enter with 1:2; after +1R (BE armed) optionally replace TP with 1:3+.
  Path-to-1R is known then; 2R order is not filled yet.
- scale50: half qty keeps 1:2, half uses the chosen higher target (two tickets).

Paper-gate fail% always uses the original 1:2 paper book so skip-days match M46.
Do not write Swing_low / Swing_Live.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from src.analysis.indian_brokerage_calculator import calculate_indian_trade_charges
from swing_strategy.run_2pct_50_25_hunt import CAPITAL, PCT, TV
from swing_strategy.run_2pct_valid_hunt import be_fair
from swing_strategy.run_ml_target_books import CORE_COLS
from swing_strategy.run_ml_top5_selector import _metrics
from swing_strategy.run_pred_paper_gate import M46, exits_by_day, load_pred_list
from swing_strategy.tiered_liquidity_strategy_engine import _load_daily

OUT = BASE / "Reports" / "SwingLow_OldLiquidity"
CACHE = OUT / "Pred_rr_path_cache.parquet"
LOG = OUT / "swing_pp_rr_classifier.json"
BASELINE_CAGR = 55.41
BASELINE_DD = 24.5

ENTRY_FEAT = [
    c
    for c in CORE_COLS
    if c
    not in (
        "TF_Rank",
        "Nifty_Rank",
        "sweep_age_bars",
        "vol20",
    )
] + [
    "ML_Score",
    "Meta_P",
    "sweep_age_bars",
    "Sweep_Count",
    "vol20_x",
    "vol20_y",
    "ret20",
    "sma50_dist",
]
PATH1 = [
    "bars_to_1R",
    "arm_open_R",
    "arm_high_R",
    "arm_close_R",
    "arm_gap_thru",
    "arm_close_thru",
    "arm_range_R",
    "arm_vol_spike",
    "arm_close_loc",
]


def _be(entry_idx, entry, sl, opens, highs, lows, n, rr: float):
    return be_fair(entry_idx, entry, sl, opens, highs, lows, n, rr=rr)


def scan_one(entry_idx, entry, sl, opens, highs, lows, vols, n) -> dict | None:
    risk = entry - sl
    if risk <= 0.05:
        return None
    r1 = entry + risk
    r2 = entry + 2.0 * risk
    hit1 = hit2 = None
    for m in range(entry_idx, n):
        h = float(highs[m])
        if hit1 is None and h >= r1:
            hit1 = m
        if h >= r2:
            hit2 = m
            break
        if hit1 is None and float(lows[m]) <= sl:
            break
        if hit1 is not None and float(lows[m]) <= entry:
            break
    out = {"armed": hit1 is not None, "hit2": hit2 is not None}
    if hit1 is not None:
        m = hit1
        o, h, l, c = float(opens[m]), float(highs[m]), float(lows[m]), float(highs[m])
        # close is not in this signature — caller passes closes separately via highs? No.
        out["bars_to_1R"] = int(m - entry_idx)
        out["arm_idx"] = int(m)
    for rr in (2.0, 3.0, 4.0, 5.0):
        px, xidx, rv = _be(entry_idx, entry, sl, opens, highs, lows, n, rr)
        k = int(rr)
        out[f"px_{k}"] = float(px)
        out[f"xidx_{k}"] = int(xidx)
        out[f"r_{k}"] = float(rv)
        out[f"win_{k}"] = int(rv >= rr - 0.5)
    return out


def scan_one_full(entry_idx, entry, sl, opens, highs, lows, closes, vols, n) -> dict | None:
    risk = entry - sl
    if risk <= 0.05:
        return None
    r1 = entry + risk
    r2 = entry + 2.0 * risk
    hit1 = hit2 = None
    for m in range(entry_idx, n):
        h = float(highs[m])
        if hit1 is None and h >= r1:
            hit1 = m
        if h >= r2:
            hit2 = m
            break
        if hit1 is None and m > entry_idx and float(opens[m]) < sl:
            break
        if hit1 is None and float(lows[m]) <= sl:
            break
        if hit1 is not None and float(lows[m]) <= entry:
            break

    def arm_feats(idx: int, prefix: str) -> dict:
        o, h, l, c = float(opens[idx]), float(highs[idx]), float(lows[idx]), float(closes[idx])
        mean_v = float(np.nanmean(vols[max(0, idx - 20) : idx])) if idx > 0 else 0.0
        v = float(vols[idx]) if np.isfinite(vols[idx]) else 0.0
        spike = (v / mean_v) if mean_v > 1e-9 else 1.0
        rng = max(h - l, 1e-9)
        return {
            f"{prefix}bars": int(idx - entry_idx),
            f"{prefix}open_R": (o - entry) / risk,
            f"{prefix}high_R": (h - entry) / risk,
            f"{prefix}close_R": (c - entry) / risk,
            f"{prefix}gap_thru": int(o >= (r1 if prefix == "arm_" else r2)),
            f"{prefix}close_thru": int(c >= (r1 if prefix == "arm_" else r2)),
            f"{prefix}range_R": (h - l) / risk,
            f"{prefix}vol_spike": float(spike),
            f"{prefix}close_loc": (c - l) / rng,
        }

    out: dict = {
        "armed": int(hit1 is not None),
        "hit2": int(hit2 is not None),
        "arm_idx": int(hit1) if hit1 is not None else -1,
        "hit2_idx": int(hit2) if hit2 is not None else -1,
    }
    if hit1 is not None:
        f = arm_feats(hit1, "arm_")
        out["bars_to_1R"] = f["arm_bars"]
        out["arm_open_R"] = f["arm_open_R"]
        out["arm_high_R"] = f["arm_high_R"]
        out["arm_close_R"] = f["arm_close_R"]
        out["arm_gap_thru"] = f["arm_gap_thru"]
        out["arm_close_thru"] = f["arm_close_thru"]
        out["arm_range_R"] = f["arm_range_R"]
        out["arm_vol_spike"] = f["arm_vol_spike"]
        out["arm_close_loc"] = f["arm_close_loc"]
    else:
        for k in PATH1:
            out[k] = 0.0
    if hit2 is not None:
        f = arm_feats(hit2, "h2_")
        out["bars_to_2R"] = f["h2_bars"]
        out["h2_close_R"] = f["h2_close_R"]
        out["h2_high_R"] = f["h2_high_R"]
        out["h2_gap_thru"] = f["h2_gap_thru"]
        out["h2_close_thru"] = f["h2_close_thru"]
    for rr in (2.0, 3.0, 4.0, 5.0, 6.0):
        px, xidx, rv = _be(entry_idx, entry, sl, opens, highs, lows, n, rr)
        k = int(rr)
        out[f"px_{k}"] = float(px)
        out[f"xidx_{k}"] = int(xidx)
        out[f"r_{k}"] = float(rv)
        out[f"win_{k}"] = int(float(rv) >= rr - 0.5)
    return out


def build_cache(base: pd.DataFrame) -> pd.DataFrame:
    if CACHE.exists():
        c = pd.read_parquet(CACHE)
        if len(c) == len(base) and "win_6" in c.columns and "arm_close_R" in c.columns:
            print(f"[cache] {CACHE.name} n={len(c):,}", flush=True)
            return c
    cache: dict = {}
    rows = []
    miss = 0
    recs = base.to_dict("records")
    print(f"[scan] path + RR exits on {len(recs):,} names ...", flush=True)
    for i, rec in enumerate(recs):
        if i and i % 500 == 0:
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
        rec_out = dict(rec)
        if df is None:
            miss += 1
            rows.append(rec_out)
            continue
        edt = pd.Timestamp(rec["Entry_Date"]).normalize()
        dates = pd.to_datetime(df["Date"]).dt.normalize()
        idx = int(dates.searchsorted(edt))
        if idx >= len(df) or dates.iloc[idx] != edt:
            miss += 1
            rows.append(rec_out)
            continue
        vols = (
            df["Volume"].to_numpy(float)
            if "Volume" in df.columns
            else np.ones(len(df), dtype=float)
        )
        got = scan_one_full(
            idx,
            float(rec["Entry_Price"]),
            float(rec["SL_Price"]),
            df["Open"].to_numpy(float),
            df["High"].to_numpy(float),
            df["Low"].to_numpy(float),
            df["Close"].to_numpy(float),
            vols,
            len(df),
        )
        if not got:
            miss += 1
            rows.append(rec_out)
            continue
        rec_out.update(got)
        for k in (2, 3, 4, 5, 6):
            xidx = int(got[f"xidx_{k}"])
            rec_out[f"xdt_{k}"] = pd.Timestamp(dates.iloc[min(xidx, len(dates) - 1)])
        rec_out["label_date"] = rec_out["xdt_6"]
        rows.append(rec_out)
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CACHE, index=False)
    print(f"[scan] wrote {CACHE} miss={miss}", flush=True)
    return out


def _feat_cols(df: pd.DataFrame, extra: list[str]) -> list[str]:
    cols = []
    for c in ENTRY_FEAT + extra:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def walk_proba(df: pd.DataFrame, ycol: str, feats: list[str], mask: pd.Series, min_train: int = 180) -> pd.Series:
    from xgboost import XGBClassifier

    df = df.copy()
    df["Entry_Date"] = pd.to_datetime(df["Entry_Date"])
    df["year"] = df["Entry_Date"].dt.year
    lab = pd.to_datetime(df["label_date"])
    parts = []
    for y in sorted(df["year"].unique()):
        cutoff = pd.Timestamp(f"{y}-01-01")
        test = df[df["year"] == y].copy()
        if test.empty:
            continue
        train_idx = (df["year"] < y) & (lab < cutoff) & mask
        train = df.loc[train_idx]
        use = [c for c in feats if c in df.columns]
        if len(train) < min_train or train[ycol].nunique() < 2:
            test["p"] = np.nan
            parts.append(test)
            continue
        pos = float(train[ycol].mean())
        spw = (1.0 - pos) / max(pos, 1e-6)
        clf = XGBClassifier(
            n_estimators=140,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.75,
            min_child_weight=12,
            reg_lambda=2.0,
            n_jobs=4,
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=min(spw, 4.0),
        )
        xtr = train[use].to_numpy(np.float32)
        xtr = np.nan_to_num(xtr, nan=0.0, posinf=0.0, neginf=0.0)
        clf.fit(xtr, train[ycol].to_numpy(np.int32))
        xte = test[use].to_numpy(np.float32)
        xte = np.nan_to_num(xte, nan=0.0, posinf=0.0, neginf=0.0)
        test["p"] = clf.predict_proba(xte)[:, 1]
        parts.append(test)
        print(f"    {ycol} {y} train={len(train):,} pos={pos:.3f}", flush=True)
    if not parts:
        return pd.Series(np.nan, index=df.index)
    scored = pd.concat(parts, ignore_index=False)
    return scored.reindex(df.index)["p"]


def oos_table(y: np.ndarray, p: np.ndarray, thresholds: list[float]) -> list[dict]:
    rows = []
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) < 20:
        return rows
    for t in thresholds:
        pred = p >= t
        n = int(pred.sum())
        if n < 5:
            continue
        tp = int((y[pred] == 1).sum())
        fp = n - tp
        prec = tp / n
        rec = tp / max(int(y.sum()), 1)
        rows.append(
            {
                "t": t,
                "n_flag": n,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "tp": tp,
                "fp": fp,
            }
        )
    return rows


def apply_rr(df: pd.DataFrame, chosen: pd.Series) -> pd.DataFrame:
    out = df.copy()
    px = out["px_2"].to_numpy(float)
    r = out["r_2"].to_numpy(float)
    xdt = pd.to_datetime(out["xdt_2"])
    for k in (3, 4, 5, 6):
        m = chosen.to_numpy() == k
        if not m.any():
            continue
        px = np.where(m, out[f"px_{k}"].to_numpy(float), px)
        r = np.where(m, out[f"r_{k}"].to_numpy(float), r)
        xdt = pd.Series(xdt).where(~m, pd.to_datetime(out[f"xdt_{k}"]))
    out["Exit_Price"] = px
    out["Realized_R"] = r
    out["Exit_Date"] = pd.to_datetime(xdt)
    miss = out["Exit_Date"].isna()
    if miss.any():
        out.loc[miss, "Exit_Date"] = pd.to_datetime(out.loc[miss, "Entry_Date"])
        out.loc[miss, "Exit_Price"] = pd.to_numeric(out.loc[miss, "Entry_Price"], errors="coerce")
        out.loc[miss, "Realized_R"] = 0.0
    out["Target_RR_Mode"] = chosen.map({2: "1:2", 3: "1:3", 4: "1:4", 5: "1:5", 6: "1:6"}).fillna("1:2")
    out["Outcome"] = np.where(out["Realized_R"] >= 1.5, "Success", np.where(out["Realized_R"] >= -0.25, "BE", "Failure"))
    return out


def sim_lots(fill: pd.DataFrame, paper_ml: pd.DataFrame, cfg: dict) -> dict:
    """Same 2% + paper-gate as M46, but each candidate may carry two lots."""
    roll_n = int(cfg.get("roll_n", 50))
    fail_max = float(cfg.get("fail_max", 0.74))
    scale_hi = float(cfg.get("scale_hi", 1.3))
    rank_hi = float(cfg.get("rank_hi", 1.2))
    rank_lo = float(cfg.get("rank_lo", 0.6))
    scale_lo = float(cfg.get("scale_lo", 0.40))
    paper = exits_by_day(paper_ml, 1.5)
    by_entry = defaultdict(list)
    for rec in fill.to_dict("records"):
        by_entry[pd.Timestamp(rec["Entry_Date"]).normalize()].append(rec)
    for day, cands in by_entry.items():
        cands.sort(key=lambda x: float(x.get("Meta_P", 0.0)), reverse=True)

    min_dt = pd.Timestamp("2010-01-01")
    max_dt = max(fill["Entry_Date"].max(), pd.to_datetime(fill["Exit_Date"]).max())
    cash = CAPITAL
    peak = CAPITAL
    max_dd = 0.0
    open_pos: dict[int, dict] = {}
    executed = 0
    skipped_days = 0
    trade_id = 1
    n_hi = 0
    rr_used = defaultdict(int)
    recent: deque[int] = deque(maxlen=max(roll_n, 1))
    extra_x = []
    for recs in by_entry.values():
        for r in recs:
            xd = pd.Timestamp(r["Exit_Date"])
            if pd.notna(xd):
                extra_x.append(xd.normalize())
    event_days = sorted(set(by_entry.keys()) | set(paper.keys()) | set(extra_x))
    for day in event_days:
        fail_block = False
        if roll_n and len(recent) >= roll_n:
            fail_block = (1.0 - (sum(recent) / len(recent))) >= fail_max
        if fail_block and day in by_entry:
            skipped_days += 1
        port = cash + sum(p["spend"] for p in open_pos.values())
        risk = max(port * PCT, 100.0)
        avail = cash
        cands = by_entry.get(day, [])
        n = max(len(cands), 1)
        vols = [float(c.get("idio_vol", np.nan)) for c in cands]
        vols = [v for v in vols if np.isfinite(v) and v > 0]
        med = float(np.median(vols)) if vols else TV
        scale = float(np.clip(TV / max(med, 1e-6), scale_lo, scale_hi))
        for i, cand in enumerate(cands):
            if fail_block:
                continue
            if pd.isna(cand.get("Exit_Date")):
                continue
            entry_p = float(cand["Entry_Price"])
            sl_p = float(cand["SL_Price"])
            rsk = entry_p - sl_p
            if rsk <= 0.05 or rsk > risk * 1.8:
                continue
            strength = (rank_hi - (rank_hi - rank_lo) * (i / n)) * scale
            qty = min(int((risk * strength) // rsk), int(avail // entry_p))
            spend = round(entry_p * qty, 2)
            while qty >= 1 and spend > avail:
                qty -= 1
                spend = round(entry_p * qty, 2)
            if qty < 1:
                continue
            lots = cand.get("lots")
            if not lots:
                lots = [
                    {
                        "frac": 1.0,
                        "exit": float(cand["Exit_Price"]),
                        "xdt": pd.Timestamp(cand["Exit_Date"]).normalize(),
                    }
                ]
            q_left = qty
            spends = []
            for j, lot in enumerate(lots):
                if j == len(lots) - 1:
                    qj = q_left
                else:
                    qj = int(qty * float(lot["frac"]))
                    q_left -= qj
                if qj < 1:
                    continue
                sp = round(entry_p * qj, 2)
                spends.append((qj, sp, float(lot["exit"]), pd.Timestamp(lot["xdt"]).normalize()))
            if not spends:
                continue
            total_sp = round(sum(s[1] for s in spends), 2)
            if total_sp > avail + 0.01:
                continue
            cash = round(cash - total_sp, 2)
            avail = round(avail - total_sp, 2)
            executed += 1
            mode = str(cand.get("Target_RR_Mode", "1:2"))
            rr_used[mode] += 1
            if mode != "1:2":
                n_hi += 1
            for qj, sp, ex, xdt in spends:
                open_pos[trade_id] = {"entry": entry_p, "exit": ex, "qty": qj, "spend": sp, "xdt": xdt}
                trade_id += 1

        for tid, pos in list(open_pos.items()):
            if pos["xdt"] > day:
                continue
            qty, entry_p, exit_p, spend = pos["qty"], pos["entry"], pos["exit"], pos["spend"]
            gross = round((exit_p - entry_p) * qty, 2)
            tax = round(calculate_indian_trade_charges(entry_p, exit_p, qty, 0.0)["total_charges"], 2)
            cash = round(cash + spend + gross - tax, 2)
            del open_pos[tid]

        for won in paper.get(day, []):
            recent.append(1 if won else 0)

        port = cash + sum(p["spend"] for p in open_pos.values())
        peak = max(peak, port)
        max_dd = max(max_dd, ((peak - port) / peak * 100) if peak else 0)

    years = max((max_dt - min_dt).days / 365.25, 0.01)
    _, cagr = _metrics(CAPITAL, cash, years)
    return {
        **{k: v for k, v in cfg.items() if k in ("tag", "roll_n", "fail_max")},
        "CAGR": round(float(cagr), 2),
        "N": executed,
        "DD": round(max_dd, 2),
        "skipped_days": skipped_days,
        "Final": round(cash, 2),
        "n_higher_rr": n_hi,
        "rr_used": dict(rr_used),
        "delta_cagr": round(float(cagr) - BASELINE_CAGR, 2),
        "delta_dd": round(max_dd - BASELINE_DD, 2),
        "keep": bool(cagr >= BASELINE_CAGR - 0.05 and max_dd <= BASELINE_DD + 0.3),
        "better": bool(cagr >= BASELINE_CAGR and max_dd <= BASELINE_DD),
    }


def attach_lots(df: pd.DataFrame, chosen: pd.Series, scale50: bool) -> pd.DataFrame:
    out = apply_rr(df, chosen)
    lots = []
    ch = chosen.to_numpy()
    for i, rec in enumerate(out.to_dict("records")):
        k = int(ch[i])
        if scale50 and k > 2:
            lots.append(
                [
                    {"frac": 0.5, "exit": float(rec["px_2"]), "xdt": rec["xdt_2"]},
                    {"frac": 0.5, "exit": float(rec[f"px_{k}"]), "xdt": rec[f"xdt_{k}"]},
                ]
            )
        else:
            lots.append(None)
    out["lots"] = lots
    if scale50:
        out.loc[chosen > 2, "Target_RR_Mode"] = chosen[chosen > 2].map(
            {3: "50/50 1:2+1:3", 4: "50/50 1:2+1:4", 5: "50/50 1:2+1:5", 6: "50/50 1:2+1:6"}
        )
    return out


def choose_ladder(p3, p4, p5, t3, t4, t5, need_armed: np.ndarray | None, p6=None, t6=None) -> pd.Series:
    p3 = np.nan_to_num(np.asarray(p3, float), nan=-1.0)
    p4 = np.nan_to_num(np.asarray(p4, float), nan=-1.0)
    p5 = np.nan_to_num(np.asarray(p5, float), nan=-1.0)
    p6 = np.nan_to_num(np.asarray(p6 if p6 is not None else p5, float), nan=-1.0)
    ch = np.full(len(p3), 2, dtype=int)
    if t6 is not None:
        ch = np.where(p6 >= t6, 6, ch)
    if t5 is not None:
        ch = np.where((ch == 2) & (p5 >= t5), 5, ch)
    if t4 is not None:
        ch = np.where((ch == 2) & (p4 >= t4), 4, ch)
    if t3 is not None:
        ch = np.where((ch == 2) & (p3 >= t3), 3, ch)
    if need_armed is not None:
        ch = np.where(need_armed.astype(bool), ch, 2)
    return pd.Series(ch)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    paper = load_pred_list()
    paper["Entry_Date"] = pd.to_datetime(paper["Entry_Date"])
    paper["Exit_Date"] = pd.to_datetime(paper["Exit_Date"])
    df = build_cache(paper)
    df["Entry_Date"] = pd.to_datetime(df["Entry_Date"])
    for c in ("xdt_2", "xdt_3", "xdt_4", "xdt_5", "label_date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    df["armed"] = pd.to_numeric(df.get("armed", 0), errors="coerce").fillna(0).astype(int)
    df["hit2"] = pd.to_numeric(df.get("hit2", 0), errors="coerce").fillna(0).astype(int)
    for k in (2, 3, 4, 5):
        df[f"win_{k}"] = pd.to_numeric(df.get(f"win_{k}", 0), errors="coerce").fillna(0).astype(int)

    n = len(df)
    print(
        f"universe {n:,}  armed {int(df.armed.sum()):,}  hit2 {int(df.hit2.sum()):,}  "
        f"win3 {int(df.win_3.sum()):,} win4 {int(df.win_4.sum()):,} win5 {int(df.win_5.sum()):,}",
        flush=True,
    )

    entry_cols = _feat_cols(df, [])
    arm_cols = _feat_cols(df, PATH1)
    print(f"entry feats {len(entry_cols)}  1R feats {len(arm_cols)}", flush=True)

    print("[wf] at-entry P(win 1:3/4/5) ...", flush=True)
    all_mask = pd.Series(True, index=df.index)
    df["p3_entry"] = walk_proba(df, "win_3", entry_cols, all_mask)
    df["p4_entry"] = walk_proba(df, "win_4", entry_cols, all_mask)
    df["p5_entry"] = walk_proba(df, "win_5", entry_cols, all_mask)

    print("[wf] at-1R P(win 1:3/4/5 | armed) ...", flush=True)
    armed_mask = df["armed"] == 1
    df["p3_1r"] = walk_proba(df, "win_3", arm_cols, armed_mask)
    df["p4_1r"] = walk_proba(df, "win_4", arm_cols, armed_mask)
    df["p5_1r"] = walk_proba(df, "win_5", arm_cols, armed_mask)

    th = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    acc = {
        "entry_win3": oos_table(df["win_3"].to_numpy(), df["p3_entry"].to_numpy(), th),
        "entry_win4": oos_table(df["win_4"].to_numpy(), df["p4_entry"].to_numpy(), th),
        "arm_win3": oos_table(df.loc[armed_mask, "win_3"].to_numpy(), df.loc[armed_mask, "p3_1r"].to_numpy(), th),
        "arm_win4": oos_table(df.loc[armed_mask, "win_4"].to_numpy(), df.loc[armed_mask, "p4_1r"].to_numpy(), th),
        "arm_win5": oos_table(df.loc[armed_mask, "win_5"].to_numpy(), df.loc[armed_mask, "p5_1r"].to_numpy(), th),
        "hit2_win3": oos_table(
            df.loc[df.hit2 == 1, "win_3"].to_numpy(),
            df.loc[df.hit2 == 1, "p3_1r"].to_numpy(),
            th,
        ),
    }
    print("OOS precision (at-1R, win_3 among armed):", flush=True)
    for row in acc["arm_win3"]:
        print(f"  t={row['t']:.2f}  n={row['n_flag']:4d}  prec={row['precision']:.1%}  rec={row['recall']:.1%}", flush=True)

    gate = {"roll_n": 50, "fail_max": 0.74, **M46}
    rows = []

    def run_policy(tag, chosen, scale50=False):
        n_hi = int((chosen > 2).sum())
        fill = attach_lots(df, chosen, scale50)
        r = sim_lots(fill, paper, {**gate, "tag": tag})
        r["n_flagged_names"] = n_hi
        rows.append(r)
        mark = " KEEP" if r["keep"] else ""
        if r["better"]:
            mark = " BETTER"
        if r["CAGR"] >= 50 or r["keep"] or n_hi == 0:
            print(
                f"  {tag:42s} CAGR {r['CAGR']:+6.2f}% DD {r['DD']:5.1f} n={r['N']:4d} "
                f"hi={r['n_higher_rr']:4d} names={n_hi:4d} {r.get('rr_used')}{mark}",
                flush=True,
            )
        return r

    print("\n===== baseline 1:2 =====", flush=True)
    run_policy("baseline_1to2", pd.Series(2, index=df.index))

    print("\n===== oracle ceiling (lookahead, not tradable) =====", flush=True)
    ora3 = pd.Series(np.where(df["win_3"] == 1, 3, 2), index=df.index)
    run_policy("ORACLE_1to3_if_win3", ora3)
    ora4 = pd.Series(np.where(df["win_4"] == 1, 4, 2), index=df.index)
    run_policy("ORACLE_1to4_if_win4", ora4)

    print("\n===== always higher (sanity) =====", flush=True)
    for k in (3, 4, 5):
        run_policy(f"always_1to{k}", pd.Series(k, index=df.index))

    print("\n===== at-entry XGB =====", flush=True)
    for t3 in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
        ch = choose_ladder(df["p3_entry"], df["p4_entry"], df["p5_entry"], t3, None, None, None)
        run_policy(f"entry_1to3_t{t3:.2f}", ch)
    for t4 in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        ch = choose_ladder(df["p3_entry"], df["p4_entry"], df["p5_entry"], None, t4, None, None)
        run_policy(f"entry_1to4_t{t4:.2f}", ch)
    for t3, t4, t5 in (
        (0.70, 0.55, 0.50),
        (0.75, 0.60, 0.55),
        (0.80, 0.65, 0.55),
        (0.65, 0.50, 0.45),
    ):
        ch = choose_ladder(df["p3_entry"], df["p4_entry"], df["p5_entry"], t3, t4, t5, None)
        run_policy(f"entry_ladder_{t3:.2f}_{t4:.2f}_{t5:.2f}", ch)

    print("\n===== at-1R XGB (raise TP after BE arm) =====", flush=True)
    armed = df["armed"].to_numpy() == 1
    for t3 in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
        ch = choose_ladder(df["p3_1r"], df["p4_1r"], df["p5_1r"], t3, None, None, armed)
        run_policy(f"arm_1to3_t{t3:.2f}", ch)
    for t4 in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75):
        ch = choose_ladder(df["p3_1r"], df["p4_1r"], df["p5_1r"], None, t4, None, armed)
        run_policy(f"arm_1to4_t{t4:.2f}", ch)
    for t3, t4, t5 in (
        (0.70, 0.55, 0.50),
        (0.75, 0.60, 0.55),
        (0.80, 0.65, 0.55),
        (0.65, 0.50, 0.45),
        (0.85, 0.70, 0.60),
    ):
        ch = choose_ladder(df["p3_1r"], df["p4_1r"], df["p5_1r"], t3, t4, t5, armed)
        run_policy(f"arm_ladder_{t3:.2f}_{t4:.2f}_{t5:.2f}", ch)

    print("\n===== scale-out 50/50 (half banks 1:2) =====", flush=True)
    for t3 in (0.55, 0.65, 0.75, 0.85):
        ch = choose_ladder(df["p3_1r"], df["p4_1r"], df["p5_1r"], t3, None, None, armed)
        run_policy(f"scale50_arm3_t{t3:.2f}", ch, scale50=True)
    for t4 in (0.45, 0.55, 0.65):
        ch = choose_ladder(df["p3_1r"], df["p4_1r"], df["p5_1r"], None, t4, None, armed)
        run_policy(f"scale50_arm4_t{t4:.2f}", ch, scale50=True)
    ch = choose_ladder(df["p3_entry"], df["p4_entry"], df["p5_entry"], 0.70, None, None, None)
    run_policy("scale50_entry3_t0.70", ch, scale50=True)
    # always split 2+4 on every name (no ML)
    ch = pd.Series(4, index=df.index)
    run_policy("scale50_always_2and4", ch, scale50=True)
    ch = pd.Series(3, index=df.index)
    run_policy("scale50_always_2and3", ch, scale50=True)

    print("\n===== rule at 1R (no ML) =====", flush=True)
    close_r = pd.to_numeric(df.get("arm_close_R"), errors="coerce").fillna(0)
    bars = pd.to_numeric(df.get("bars_to_1R"), errors="coerce").fillna(99)
    meta = pd.to_numeric(df["Meta_P"], errors="coerce").fillna(0)
    thru = pd.to_numeric(df.get("arm_close_thru"), errors="coerce").fillna(0)
    for close_min, bar_max, meta_min, k in (
        (1.05, 3, 0.55, 3),
        (1.20, 5, 0.55, 3),
        (1.40, 5, 0.60, 3),
        (1.20, 3, 0.65, 4),
        (1.50, 5, 0.55, 4),
        (0.80, 2, 0.70, 3),
    ):
        m = armed & (close_r >= close_min) & (bars <= bar_max) & (meta >= meta_min)
        ch = pd.Series(np.where(m, k, 2), index=df.index)
        run_policy(f"rule_c{close_min:.2f}_b{bar_max}_p{meta_min:.2f}_1to{k}", ch)
    m = armed & (thru == 1) & (meta >= 0.60)
    ch = pd.Series(np.where(m, 3, 2), index=df.index)
    run_policy("rule_closeThru1R_p0.60_1to3", ch)

    keep = [r for r in rows if r.get("keep")]
    better = [r for r in rows if r.get("better")]
    ranked = sorted(rows, key=lambda x: -(x["CAGR"] - 0.35 * x["DD"]))
    print("\n===== policies that do not worsen vs 1:2 (+55.41% / 24.5% DD) =====", flush=True)
    if not keep:
        print("  none", flush=True)
    for r in keep[:15]:
        print(f"  {r['tag']:42s} {r['CAGR']:+6.2f}% DD {r['DD']:5.1f} hi={r['n_higher_rr']}", flush=True)
    print("strictly better (CAGR>= and DD<=):", flush=True)
    if not better:
        print("  none", flush=True)
    for r in better:
        print(f"  {r['tag']:42s} {r['CAGR']:+6.2f}% DD {r['DD']:5.1f} hi={r['n_higher_rr']}", flush=True)
    print("top score CAGR-0.35*DD:", flush=True)
    for r in ranked[:12]:
        print(
            f"  {r['CAGR']:+6.2f}% DD {r['DD']:5.1f} hi={r['n_higher_rr']:4d} {r['tag']}",
            flush=True,
        )

    payload = {
        "baseline": {"CAGR": BASELINE_CAGR, "DD": BASELINE_DD, "note": "Swing_PP 1:2 fair BE paper-gate n50 fail74"},
        "oos_accuracy": acc,
        "keep": keep,
        "better": better,
        "rows": rows,
        "note": (
            "65% of 1:2 winners later tag 3R is not a 65% edge: miss gives back 2R "
            "(need >67% precision) plus capital lock. Oracle rows are lookahead."
        ),
    }
    LOG.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"wrote {LOG}", flush=True)


if __name__ == "__main__":
    main()
