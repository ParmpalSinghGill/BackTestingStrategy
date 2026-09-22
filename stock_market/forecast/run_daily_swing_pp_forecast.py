"""
Daily Swing_PP entries: swing-low OPEN_BELOW + A1, M46 paper-gate book,
C3-close scratch, plus next-bar RR after a +1R close.

Same liquidity scan as Swing_low. Then:
  liquidity age >= 1 month
  HGB + XGB meta trained on >=1m history
  Meta_P >= 0.48, daily top 16
  skip NEW entries if last 50 predicted paper results failed >= 74%
  ENTER next session at that session's OPEN
  same bar: SL, else 1:2, else if close < C1 high SELL AT CLOSE (scratch)
  after first +1R close, maybe SHIFT TP from 1:2 to 1:3/5/6 next session
  (P6>=0.85->1:6 else P5>=0.85->1:5 else P3>=0.80->1:3)

Writes (never touches Swing_low / Swing_Live):
  forecast_stocks/Swing_PP.txt
  forecast_stocks/Swing_PP_<DD_Mon_YYYY>.txt
  Watchlist/Swing_PP.txt
  forecast_stocks/Swing_PP_RR.txt  (SHIFT target FROM x TO y)
  Watchlist/Swing_PP_RR.txt
  Swing_PP_ins.txt always (how to enter, scratch, SHIFT) — not deleted when names publish
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, time
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent / "swing"
sys.path.insert(0, str(BASE_DIR))

from swing_strategy.run_daily_swing_forecast import (
    add_cross_section,
    fetch_latest,
    latest_bar_on_disk,
    next_session,
    to_fyers,
)
from swing_strategy.run_daily_swing_low_forecast import (
    DATA_DAILY_DIR,
    FORECAST_DIR,
    KEEP,
    WATCHLIST_DIR,
    _age_ok,
    _swing_low_pending_worker,
    last_complete_session,
    train_models,
)
from swing_strategy.run_pred_paper_gate import gate_asof, load_pred_list
from swing_strategy.run_swing_pp_rr_live import (
    add_pending_entries,
    load_open_book,
    load_rr_models,
    save_open_book,
    scan_upgrades,
)

TAG = "Swing_PP"
INS_NAME = "Swing_PP_ins.txt"
RR_TAG = "Swing_PP_RR"
META_MIN = 0.48
TOP_N = 16
MIN_AGE = 1
ROLL_N = 50
FAIL_MAX = 0.74
OUT = BASE_DIR / "Reports" / "SwingLow_OldLiquidity"
PICKS_LOG = OUT / "Swing_PP_last.json"


def ins_paths(entry_day: datetime.date | None = None) -> list[Path]:
    paths = [
        FORECAST_DIR / INS_NAME,
        WATCHLIST_DIR / INS_NAME,
        WATCHLIST_DIR / f"RAN_{INS_NAME}",
    ]
    if entry_day is not None:
        stamp = entry_day.strftime("%d_%b_%Y")
        paths.append(FORECAST_DIR / f"Swing_PP_ins_{stamp}.txt")
    return paths


def write_watchlist(symbols: list[str], entry_day: datetime.date) -> None:
    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    WATCHLIST_DIR.mkdir(parents=True, exist_ok=True)
    head = ["NSE:NIFTY50-INDEX", "BSE:SENSEX-INDEX"]
    line = ",".join(head + symbols)
    stamp = entry_day.strftime("%d_%b_%Y")
    paths = [
        FORECAST_DIR / f"{TAG}_{stamp}.txt",
        FORECAST_DIR / f"{TAG}.txt",
        WATCHLIST_DIR / f"{TAG}.txt",
        WATCHLIST_DIR / f"RAN_{TAG}.txt",
    ]
    for path in paths:
        path.write_text(line, encoding="utf-8")
        print(f"[write] {path}  ({len(symbols)} names)", flush=True)


def write_rr(
    raises: list[dict],
    keeps: list[dict],
    next_d: datetime.date,
    asof_d: datetime.date,
) -> None:
    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    WATCHLIST_DIR.mkdir(parents=True, exist_ok=True)
    stamp = next_d.strftime("%d_%b_%Y")
    lines = [
        f"Swing_PP RR  asof {asof_d.isoformat()}  next session {next_d.isoformat()}",
        "After first +1R close: SHIFT the working target from next open. SL to breakeven (entry).",
        "No SHIFT if that 1R bar already hit 2R or SL, or if entry bar scratched (close < C1 high).",
        "Cuts: P6>=0.85 -> 1:6 else P5>=0.85 -> 1:5 else P3>=0.80 -> 1:3 else stay 1:2.",
        "",
    ]
    if raises:
        lines.append("SHIFT target next session (condition matched today: +1R close):")
        for r in raises:
            fy = to_fyers(str(r["Ticker"]))
            k = int(r["chosen_rr"])
            tp2 = float(r.get("Target_Price_2") or r["Target_Price"])
            tpk = float(r["Target_Price"])
            lines.append(
                f"  SHIFT {fy:22s}  target FROM {tp2:.2f} (1:2) TO {tpk:.2f} (1:{k})  "
                f"because +1R closed today and P6={r['p6_1r']:.2f} P5={r['p5_1r']:.2f} P3={r['p3_1r']:.2f}  "
                f"entry {r['Entry_Price']:.2f}  SL->BE {r['BE']:.2f}"
            )
    else:
        lines.append("No TP raises for next session.")
    if keeps:
        lines.append("")
        lines.append("1R close today — keep 1:2 (model below cut):")
        for r in keeps:
            fy = to_fyers(str(r["Ticker"]))
            tp2 = float(r.get("Target_Price_2") or 0.0)
            if tp2 <= 0:
                entry = float(r["Entry_Price"])
                sl = float(r["SL_Price"])
                tp2 = round(entry + 2.0 * (entry - sl), 2)
            lines.append(
                f"  KEEP  {fy:22s}  target stays {tp2:.2f} (1:2)  "
                f"P3={r['p3_1r']:.2f} P5={r['p5_1r']:.2f} P6={r['p6_1r']:.2f}"
            )
    body = "\n".join(lines) + "\n"
    paths = [
        FORECAST_DIR / f"{RR_TAG}_{stamp}.txt",
        FORECAST_DIR / f"{RR_TAG}.txt",
        WATCHLIST_DIR / f"{RR_TAG}.txt",
        WATCHLIST_DIR / f"RAN_{RR_TAG}.txt",
    ]
    for path in paths:
        path.write_text(body, encoding="utf-8")
        print(f"[rr] {path}", flush=True)


def write_ins(
    entry_day: datetime.date,
    asof_d: datetime.date,
    gate: dict,
    why: str,
    scored_lines: list[str],
    picked: pd.DataFrame,
    rr_raises: list[dict] | None = None,
    rr_keeps: list[dict] | None = None,
    scratches: list[dict] | None = None,
) -> None:
    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    WATCHLIST_DIR.mkdir(parents=True, exist_ok=True)
    fail = gate.get("fail_pct")
    fail_s = "n/a" if fail is None else f"{fail:.1f}%"
    gate_line = (
        f"SKIP new entries (last {gate.get('need', ROLL_N)} paper results failed {fail_s}, "
        f"threshold {FAIL_MAX * 100:.0f}%)"
        if gate.get("skip")
        else f"TRADE allowed (last {gate.get('ready')} paper results failed {fail_s}, skip if >= {FAIL_MAX * 100:.0f}%)"
    )
    n_new = 0 if picked is None or picked.empty else len(picked)
    lines = [
        "Swing_PP — session instructions",
        f"next session (ENTER): {entry_day.isoformat()}  ({entry_day.strftime('%d %b %Y')})",
        f"asof (last complete bar): {asof_d.isoformat()}",
        f"gate: {gate_line}",
        f"why: {why}",
        "",
        "HOW TO TRADE",
        "  NEW names: ENTER tomorrow at that session's OPEN.",
        "  Default target 1:2. SL = published sweep-to-C1 low x 0.99.",
        "  Same session (entry bar) only, in this order:",
        "    1) If SL trades -> out.",
        "    2) If 1:2 trades -> out.",
        "    3) Else if that session CLOSES below C1 high -> SELL AT CLOSE (scratch).",
        "       Do not hold overnight for the sweep stop.",
        "  RUNNING names: after first +1R close (and not already 2R/SL/scratch),",
        "    SHIFT the working target FROM the 1:2 price TO the 1:k price from next open.",
        "    Move SL to breakeven (entry). If the model is below the cut, KEEP 1:2.",
        "",
    ]
    if gate.get("skip") or n_new == 0:
        lines.append("NEW ENTRIES: NONE for tomorrow.")
        if gate.get("skip"):
            lines.append("  Paper-gate skip — do not enter new Swing_PP names this session.")
        else:
            lines.append("  No Meta_P names to publish.")
        lines.append("")
    else:
        lines.append(f"NEW ENTRIES: ENTER AT TOMORROW OPEN  ({n_new} names)")
        for rec in picked.to_dict("records"):
            fy = to_fyers(str(rec["Ticker"]))
            sl = float(rec.get("SL_Price") or 0.0)
            c1 = rec.get("C1_High")
            try:
                c1s = f"{float(c1):.2f}" if c1 not in (None, "") else "?"
            except (TypeError, ValueError):
                c1s = "?"
            lines.append(
                f"  ENTER {fy:22s}  at OPEN  SL {sl:.2f}  "
                f"C1 high {c1s}  -> if tomorrow CLOSES below C1 high, SELL AT CLOSE  "
                f"else hold 1:2 (TP = open + 2*(open-SL))"
            )
        lines.append("")
    if scratches:
        lines.append("ENTRY-BAR SCRATCH (session just closed below C1 high — you should already be out at close):")
        for r in scratches:
            fy = to_fyers(str(r["Ticker"]))
            c1 = r.get("C1_High")
            cl = r.get("Close")
            c1s = "?" if c1 in (None, "") else f"{float(c1):.2f}"
            cls = "?" if cl in (None, "") else f"{float(cl):.2f}"
            lines.append(f"  SCRATCH {fy:22s}  close {cls} < C1 high {c1s}  SELL AT CLOSE")
        lines.append("")
    if rr_raises:
        lines.append("RUNNING — SHIFT target FROM x TO y (condition matched: +1R closed today):")
        for r in rr_raises:
            fy = to_fyers(str(r["Ticker"]))
            k = int(r["chosen_rr"])
            tp2 = float(r.get("Target_Price_2") or r["Target_Price"])
            tpk = float(r["Target_Price"])
            lines.append(
                f"  SHIFT {fy:22s}  target FROM {tp2:.2f} (1:2) TO {tpk:.2f} (1:{k})  "
                f"from next open  SL->BE {float(r['BE']):.2f}  "
                f"P6={r['p6_1r']:.2f} P5={r['p5_1r']:.2f} P3={r['p3_1r']:.2f}"
            )
        lines.append("")
    if rr_keeps:
        lines.append("RUNNING — +1R closed today, KEEP 1:2 (model below cut):")
        for r in rr_keeps:
            fy = to_fyers(str(r["Ticker"]))
            tp2 = float(r.get("Target_Price_2") or (float(r["Entry_Price"]) + 2.0 * (float(r["Entry_Price"]) - float(r["SL_Price"]))))
            lines.append(
                f"  KEEP  {fy:22s}  target stays {tp2:.2f} (1:2)  "
                f"P6={r['p6_1r']:.2f} P5={r['p5_1r']:.2f} P3={r['p3_1r']:.2f}"
            )
        lines.append("")
    if scored_lines:
        lines.append("scored today:")
        lines.extend(f"  {s}" for s in scored_lines)
    body = "\n".join(lines) + "\n"
    for path in ins_paths(entry_day):
        path.write_text(body, encoding="utf-8")
        print(f"[ins] {path}", flush=True)


def delete_ins() -> None:
    """Kept for manual cleanup. Daily job always rewrites ins; it is not deleted when names publish."""
    return


def resolve_asof(now: datetime, asof_arg: str) -> datetime.date:
    asof_d = datetime.strptime(asof_arg, "%Y-%m-%d").date() if asof_arg else last_complete_session(now)
    disk = latest_bar_on_disk()
    if asof_arg:
        if disk is not None and disk < asof_d:
            print(f"[asof] data on disk only through {disk} - clamping", flush=True)
            asof_d = disk
        return asof_d
    if (
        disk is not None
        and now.weekday() < 5
        and now.time() >= time(15, 30)
        and disk >= now.date()
    ):
        print(f"[asof] today's close is on disk ({disk}) - using {disk}", flush=True)
        return disk
    if disk is not None and disk < asof_d:
        print(f"[asof] data on disk only through {disk} - clamping", flush=True)
        return disk
    return asof_d


def scan_pending(asof_d: datetime.date) -> list[dict]:
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
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--asof", default="", help="YYYY-MM-DD override for last complete bar")
    args = ap.parse_args()

    now = datetime.now()
    if now.weekday() >= 5 and not args.asof:
        print(f"[clock] {now:%A %Y-%m-%d} weekend - skip (no Sat/Sun run)", flush=True)
        return

    if not args.skip_fetch:
        fetch_latest()

    asof_d = resolve_asof(now, args.asof)
    asof = pd.Timestamp(asof_d)
    entry_d = next_session(asof_d)
    print(f"[asof] last complete bar {asof_d}  ->  enter next session {entry_d}", flush=True)

    ml = load_pred_list()
    gate = gate_asof(ml, asof, ROLL_N, FAIL_MAX)
    print(
        f"[gate] last {gate.get('ready')} paper results  fail% {gate.get('fail_pct')}  "
        f"{'SKIP' if gate['skip'] else 'TRADE'}",
        flush=True,
    )

    rows = scan_pending(asof_d)
    picked = pd.DataFrame()
    scored_lines: list[str] = []
    n_raw = 0
    n_age = 0
    why = "no pending swing-low A1 setups"
    if not rows:
        print("[ml] no pending A1 setups", flush=True)
    else:
        n_raw = len(rows)
        rows = [r for r in rows if _age_ok(r.get("Liquidity_Date"), entry_d, MIN_AGE)]
        n_age = len(rows)
        print(f"[age] >= {MIN_AGE} month at {entry_d}: kept {n_age} / {n_raw} pending A1", flush=True)
        if not rows:
            why = f"{n_raw} pending A1, 0 after >= {MIN_AGE} month liquidity age"
            print("[ml] no pending A1 after min-age filter", flush=True)
        else:
            raw = add_cross_section(rows)
            for c in KEEP:
                if c not in raw.columns:
                    raw[c] = 0.0
                else:
                    raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0.0)
            print(f"[ml] {len(raw)} pending A1  -> score  Meta_P>={META_MIN} top {TOP_N}", flush=True)
            primary, meta, cols, meta_cols = train_models(asof, MIN_AGE)
            raw["ML_Score"] = primary.predict_proba(raw[cols].to_numpy(np.float32))[:, 1]
            use = [c for c in meta_cols if c in raw.columns]
            raw["Meta_P"] = meta.predict_proba(raw[use].to_numpy(np.float32))[:, 1]
            picked = raw[raw["Meta_P"] >= META_MIN].sort_values("Meta_P", ascending=False).head(TOP_N)
            print(
                f"[ml] Meta_P>={META_MIN}: {len(raw[raw['Meta_P'] >= META_MIN])}  kept {len(picked)}",
                flush=True,
            )
            for _, r in raw.sort_values("Meta_P", ascending=False).iterrows():
                flag = "KEEP" if float(r["Meta_P"]) >= META_MIN else "drop"
                line = (
                    f"{flag:4s} {r['Ticker']:16s}  Meta_P {r['Meta_P']:.3f}  "
                    f"liq {r.get('Liquidity_Date', '')}  {r['Liquidity_Type']}"
                )
                scored_lines.append(line.strip())
                print(f"  {line}", flush=True)
            n_keep = int((raw["Meta_P"] >= META_MIN).sum())
            why = (
                f"{n_age} setups after >= {MIN_AGE} month age; "
                f"{n_keep} with Meta_P >= {META_MIN:.2f} (need top {TOP_N})"
            )

    symbols = [to_fyers(str(t)) for t in picked["Ticker"]] if len(picked) else []

    book = load_open_book()
    if len(picked):
        book = add_pending_entries(book, picked, entry_d)
    rr_raises: list[dict] = []
    rr_keeps: list[dict] = []
    scratches: list[dict] = []
    models = None
    if book:
        try:
            models = load_rr_models(asof)
        except Exception as exc:
            print(f"[rr] models failed: {exc}", flush=True)
        try:
            book, rr_raises, rr_keeps, scratches = scan_upgrades(book, asof, models)
            book = [p for p in book if p.get("status") != "closed"]
        except Exception as exc:
            print(f"[rr] scan failed: {exc}", flush=True)
        save_open_book(book)
        print(
            f"[rr] open-book {len(book)}  SHIFT {len(rr_raises)}  keep-1:2 {len(rr_keeps)}  "
            f"scratch {len(scratches)}",
            flush=True,
        )
    write_rr(rr_raises, rr_keeps, entry_d, asof_d)

    if gate["skip"]:
        why = (
            f"paper-gate skip — last {gate.get('ready')} predicted paper results failed "
            f"{gate.get('fail_pct')}% (threshold {FAIL_MAX * 100:.0f}%)"
        )
        write_watchlist([], entry_d)
        write_ins(entry_d, asof_d, gate, why, scored_lines, pd.DataFrame(), rr_raises, rr_keeps, scratches)
    elif not symbols:
        write_watchlist([], entry_d)
        write_ins(entry_d, asof_d, gate, why, scored_lines, pd.DataFrame(), rr_raises, rr_keeps, scratches)
    else:
        write_watchlist(symbols, entry_d)
        write_ins(entry_d, asof_d, gate, why, scored_lines, picked, rr_raises, rr_keeps, scratches)

    OUT.mkdir(parents=True, exist_ok=True)
    PICKS_LOG.write_text(
        json.dumps(
            {
                "asof": str(asof_d),
                "entry": str(entry_d),
                "skip": bool(gate["skip"]),
                "why": why,
                "gate": gate,
                "n": len(symbols),
                "symbols": symbols,
                "scored": scored_lines,
                "rr_raises": rr_raises,
                "rr_keeps": rr_keeps,
                "scratches": scratches,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"[log] {PICKS_LOG}", flush=True)


if __name__ == "__main__":
    main()
