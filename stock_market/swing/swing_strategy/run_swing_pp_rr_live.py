"""Live Swing_PP: C3-open fill, entry-bar scratch, next-bar RR after +1R close.

Enter tomorrow at that session's open. Same bar: SL first, then 1:2, else if
the session closes below C1 high sell at that close (scratch). After the first
+1R close (and not already 2R/SL/scratch), maybe raise TP from the next session.

Does not write Swing_low / Swing_Live.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from swing_strategy.run_pred_paper_gate import load_pred_list
from swing_strategy.run_swing_pp_rr_classifier import (
    PATH1,
    build_cache,
    choose_ladder,
    score_at_1r,
    train_at_1r_models,
)
from swing_strategy.run_swing_pp_rr_statement import T3, T5, T6
from swing_strategy.tiered_liquidity_strategy_engine import _load_daily

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "Reports" / "SwingLow_OldLiquidity"
OPEN_BOOK = OUT / "Swing_PP_open.json"


def load_open_book() -> list[dict]:
    if not OPEN_BOOK.exists():
        return []
    try:
        raw = json.loads(OPEN_BOOK.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(raw, dict):
        raw = raw.get("positions", [])
    return list(raw) if isinstance(raw, list) else []


def save_open_book(rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    OPEN_BOOK.write_text(
        json.dumps({"positions": rows}, indent=2, default=str),
        encoding="utf-8",
    )


def _feat_snapshot(rec: dict | pd.Series) -> dict:
    if isinstance(rec, pd.Series):
        rec = rec.to_dict()
    out = {}
    for k, v in rec.items():
        if k in ("c1", "sweep"):
            continue
        if isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(float(v)):
            out[str(k)] = float(v)
    return out


def add_pending_entries(book: list[dict], picked: pd.DataFrame, entry_d) -> list[dict]:
    entry_s = pd.Timestamp(entry_d).strftime("%Y-%m-%d")
    have = {(str(r.get("Ticker")), str(r.get("Entry_Date"))) for r in book}
    for rec in picked.to_dict("records"):
        key = (str(rec["Ticker"]), entry_s)
        if key in have:
            continue
        book.append(
            {
                "Ticker": str(rec["Ticker"]),
                "Entry_Date": entry_s,
                "Entry_Price": None,
                "SL_Price": float(rec["SL_Price"]),
                "Support_Price": float(rec.get("Support_Price") or 0.0),
                "C1_High": float(rec["C1_High"]) if rec.get("C1_High") not in (None, "") else None,
                "C2_Close": float(rec["C2_Close"]) if rec.get("C2_Close") not in (None, "") else None,
                "Liquidity_Type": rec.get("Liquidity_Type", "Weekly"),
                "Liquidity_Date": rec.get("Liquidity_Date"),
                "Meta_P": float(rec.get("Meta_P") or 0.0),
                "ML_Score": float(rec.get("ML_Score") or 0.0),
                "features": _feat_snapshot(rec),
                "chosen_rr": 2,
                "Target_Price_2": None,
                "Target_Price": None,
                "upgrade_done": False,
                "status": "pending_entry",
            }
        )
        have.add(key)
    return book


def _path1(entry_idx, idx, entry, sl, opens, highs, lows, closes, vols) -> dict:
    risk = max(entry - sl, 1e-9)
    r1 = entry + risk
    o, h, l, c = float(opens[idx]), float(highs[idx]), float(lows[idx]), float(closes[idx])
    mean_v = float(np.nanmean(vols[max(0, idx - 20) : idx])) if idx > 0 else 0.0
    v = float(vols[idx]) if np.isfinite(vols[idx]) else 0.0
    spike = (v / mean_v) if mean_v > 1e-9 else 1.0
    rng = max(h - l, 1e-9)
    return {
        "bars_to_1R": int(idx - entry_idx),
        "arm_open_R": (o - entry) / risk,
        "arm_high_R": (h - entry) / risk,
        "arm_close_R": (c - entry) / risk,
        "arm_gap_thru": int(o >= r1),
        "arm_close_thru": int(c >= r1),
        "arm_range_R": (h - l) / risk,
        "arm_vol_spike": float(spike),
        "arm_close_loc": (c - l) / rng,
    }


def inspect_asof(pos: dict, asof: pd.Timestamp) -> dict:
    """Where this published name is as of the last complete bar."""
    tick = str(pos["Ticker"])
    edt = pd.Timestamp(pos["Entry_Date"]).normalize()
    asof = pd.Timestamp(asof).normalize()
    df = _load_daily(tick)
    if df is None or "Date" not in getattr(df, "columns", []):
        return {"status": pos.get("status", "pending_entry"), "note": "no bars"}
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"]).dt.normalize()
    d = d[d["Date"] <= asof].sort_values("Date").reset_index(drop=True)
    if d.empty:
        return {"status": "pending_entry", "note": "no bars yet"}
    dates = d["Date"]
    idx = int(dates.searchsorted(edt))
    if idx >= len(d) or dates.iloc[idx] != edt:
        if asof < edt:
            return {"status": "pending_entry", "note": "entry session not closed"}
        return {"status": "pending_entry", "note": "entry bar missing"}
    opens = d["Open"].to_numpy(float)
    highs = d["High"].to_numpy(float)
    lows = d["Low"].to_numpy(float)
    closes = d["Close"].to_numpy(float)
    vols = (
        d["Volume"].to_numpy(float)
        if "Volume" in d.columns
        else np.ones(len(d), dtype=float)
    )
    entry = float(opens[idx])
    sl = float(pos["SL_Price"])
    risk = entry - sl
    if risk <= 0.05:
        return {"status": "closed", "note": "bad risk", "Entry_Price": entry}
    r1 = entry + risk
    r2 = entry + 2.0 * risk
    c1_high = pos.get("C1_High")
    try:
        c1_high = float(c1_high) if c1_high not in (None, "") else None
    except (TypeError, ValueError):
        c1_high = None
    n = len(d)
    o0, h0, l0, c0 = float(opens[idx]), float(highs[idx]), float(lows[idx]), float(closes[idx])
    if l0 <= sl:
        return {"status": "closed", "note": "SL", "Entry_Price": entry}
    if h0 >= r2:
        return {
            "status": "closed",
            "note": "1:2 filled on entry bar",
            "Entry_Price": entry,
        }
    if c1_high is not None and c0 < c1_high:
        return {
            "status": "closed",
            "note": "scratch: entry-bar close below C1 high — sell at close",
            "Entry_Price": entry,
            "Exit_Price": round(c0, 2),
            "C1_High": c1_high,
            "Close": round(c0, 2),
        }
    armed_i = None
    for m in range(idx, n):
        o, h, l = float(opens[m]), float(highs[m]), float(lows[m])
        if armed_i is None:
            if m > idx and o < sl:
                return {"status": "closed", "note": "SL gap", "Entry_Price": entry}
            if l <= sl:
                return {"status": "closed", "note": "SL", "Entry_Price": entry}
            if h >= r2:
                return {"status": "closed", "note": "1:2 filled", "Entry_Price": entry}
            if h >= r1:
                armed_i = m
            continue
        if h >= r2:
            return {"status": "closed", "note": "1:2 filled", "Entry_Price": entry}
        if l <= entry:
            return {"status": "closed", "note": "BE", "Entry_Price": entry}
    if armed_i is None:
        return {"status": "open", "Entry_Price": entry, "note": "not yet +1R"}
    if pd.Timestamp(dates.iloc[armed_i]).normalize() != asof:
        return {
            "status": "open",
            "Entry_Price": entry,
            "note": "1R already passed (upgrade window was that close)",
            "arm_idx": int(armed_i),
        }
    path = _path1(idx, armed_i, entry, sl, opens, highs, lows, closes, vols)
    return {
        "status": "open",
        "Entry_Price": entry,
        "can_upgrade": True,
        "note": "1R close today",
        "path1": path,
        "r1": r1,
        "risk": risk,
    }


def _choose_from_p(p3, p5, p6) -> int:
    ch = choose_ladder(
        [p3], [0.0], [p5], T3, None, T5, np.array([True]), [p6], T6,
    )
    return int(ch.iloc[0])


def scan_upgrades(book: list[dict], asof: pd.Timestamp, models: dict | None) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Returns (updated book, raise rows, 1R-keep rows, entry-bar scratches)."""
    raises: list[dict] = []
    keeps: list[dict] = []
    scratches: list[dict] = []
    out: list[dict] = []
    for pos in book:
        if pos.get("status") == "closed":
            continue
        info = inspect_asof(pos, asof)
        pos = dict(pos)
        if info.get("Entry_Price") is not None:
            pos["Entry_Price"] = float(info["Entry_Price"])
        pos["status"] = info.get("status", pos.get("status"))
        pos["last_note"] = info.get("note", "")
        if pos["status"] == "closed":
            if "scratch" in str(info.get("note", "")).lower():
                scratches.append({
                    "Ticker": pos["Ticker"],
                    "Entry_Date": pos["Entry_Date"],
                    "Entry_Price": pos.get("Entry_Price"),
                    "C1_High": pos.get("C1_High") or info.get("C1_High"),
                    "Close": info.get("Close") or info.get("Exit_Price"),
                    "note": info.get("note", ""),
                })
            out.append(pos)
            continue
        if pos.get("Entry_Price") is not None and pos.get("SL_Price") is not None:
            entry = float(pos["Entry_Price"])
            sl = float(pos["SL_Price"])
            risk = entry - sl
            if risk > 0.05:
                pos.setdefault("Target_Price_2", round(entry + 2.0 * risk, 2))
                if not pos.get("Target_Price"):
                    pos["Target_Price"] = pos["Target_Price_2"]
        if not info.get("can_upgrade") or pos.get("upgrade_done"):
            out.append(pos)
            continue
        feats = dict(pos.get("features") or {})
        feats["ML_Score"] = float(pos.get("ML_Score") or 0.0)
        feats["Meta_P"] = float(pos.get("Meta_P") or 0.0)
        feats.update(info.get("path1") or {})
        for k in PATH1:
            feats.setdefault(k, 0.0)
        if not models:
            pos["last_note"] = "1R close but RR models unavailable"
            out.append(pos)
            continue
        ps = score_at_1r(models, feats)
        k = _choose_from_p(ps["p3_1r"], ps["p5_1r"], ps["p6_1r"])
        entry = float(pos["Entry_Price"])
        sl = float(pos["SL_Price"])
        risk = entry - sl
        tp2 = round(entry + 2.0 * risk, 2)
        tpk = round(entry + k * risk, 2)
        rec = {
            "Ticker": pos["Ticker"],
            "Entry_Date": pos["Entry_Date"],
            "Entry_Price": round(entry, 2),
            "SL_Price": round(sl, 2),
            "BE": round(entry, 2),
            "chosen_rr": k,
            "Target_Price_2": tp2,
            "Target_Price": tpk,
            **ps,
            "note": info.get("note", ""),
        }
        pos["upgrade_done"] = True
        pos["chosen_rr"] = k
        pos["Target_Price_2"] = tp2
        pos["Target_Price"] = tpk
        pos["p3_1r"] = ps["p3_1r"]
        pos["p5_1r"] = ps["p5_1r"]
        pos["p6_1r"] = ps["p6_1r"]
        if k > 2:
            raises.append(rec)
        else:
            keeps.append(rec)
        out.append(pos)
    return out, raises, keeps, scratches


def load_rr_models(asof: pd.Timestamp) -> dict | None:
    print("[rr] walk-forward 1R models ...", flush=True)
    paper = load_pred_list()
    paper["Entry_Date"] = pd.to_datetime(paper["Entry_Date"])
    df = build_cache(paper)
    df["Entry_Date"] = pd.to_datetime(df["Entry_Date"])
    for c in ("label_date", "xdt_6"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    models = train_at_1r_models(df, asof)
    if not any(models.values()):
        print("[rr] no 1R models (not enough purged train)", flush=True)
        return None
    return models
