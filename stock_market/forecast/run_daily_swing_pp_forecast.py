"""
Daily Swing_PP entries: swing-low OPEN_BELOW + A1, M46 paper-gate book,
plus next-bar RR after a +1R close.

Same liquidity scan as Swing_low. Then:
  liquidity age >= 1 month
  HGB + XGB meta trained on >=1m history
  Meta_P >= 0.48, daily top 16
  skip NEW entries if last 50 predicted paper results failed >= 74%
  new names enter 1:2; after first +1R bar closes, maybe raise TP from next session
  (P6>=0.85->1:6 else P5>=0.85->1:5 else P3>=0.80->1:3). No raise if that bar
  already tagged 2R or SL.

Writes (never touches Swing_low / Swing_Live):
  forecast_stocks/Swing_PP.txt
  forecast_stocks/Swing_PP_<DD_Mon_YYYY>.txt
  Watchlist/Swing_PP.txt
  forecast_stocks/Swing_PP_RR.txt  (TP raises for next session)
  Watchlist/Swing_PP_RR.txt

Instruction file Swing_PP_ins.txt (forecast + Watchlist + dated copy):
  Written when there is nothing NEW to enter (paper-gate skip, or no Meta_P names).
  RR raises are still written to Swing_PP_RR.txt. Date and why are inside the ins file.
  Ins is deleted only when new names are published.
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

BASE_DIR = Path(__file__).resolve().parent.parent
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
        "Rule: after first +1R close, new TP from next open. SL to breakeven (entry).",
        "No raise if that 1R bar already hit 2R or SL. Default stays 1:2.",
        f"Cuts: P6>=0.85 -> 1:6 else P5>=0.85 -> 1:5 else P3>=0.80 -> 1:3.",
        "",
    ]
    if raises:
        lines.append("RAISE TP next session:")
        for r in raises:
            fy = to_fyers(str(r["Ticker"]))
            lines.append(
                f"  RAISE {fy:22s}  1:{int(r['chosen_rr'])}  "
                f"P3={r['p3_1r']:.2f} P5={r['p5_1r']:.2f} P6={r['p6_1r']:.2f}  "
                f"entry {r['Entry_Price']:.2f}  SL->BE {r['BE']:.2f}  "
                f"TP {r['Target_Price']:.2f}"
            )
    else:
        lines.append("No TP raises for next session.")
    if keeps:
        lines.append("")
        lines.append("1R close today — keep 1:2 (model below cut):")
        for r in keeps:
            fy = to_fyers(str(r["Ticker"]))
            lines.append(
                f"  KEEP  {fy:22s}  1:2  "
                f"P3={r['p3_1r']:.2f} P5={r['p5_1r']:.2f} P6={r['p6_1r']:.2f}"
            )
    body = "\n".join(lines) + "\n"
    paths = [
        FORECAST_DIR / f"{RR_TAG}_{stamp}.txt",
        FORECAST_DIR / f"{RR_TAG}.txt",
        WATCHLIST_DIR / f"{RR_TAG}.txt",
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
    rr_raises: list[dict] | None = None,
) -> None:
    FORECAST_DIR.mkdir(parents=True, exist_ok=True)
    WATCHLIST_DIR.mkdir(parents=True, exist_ok=True)
    fail = gate.get("fail_pct")
    fail_s = "n/a" if fail is None else f"{fail:.1f}%"
    gate_line = (
        f"SKIP (last {gate.get('need', ROLL_N)} paper results failed {fail_s}, "
        f"threshold {FAIL_MAX * 100:.0f}%)"
        if gate.get("skip")
        else f"TRADE allowed (last {gate.get('ready')} paper results failed {fail_s}, skip if >= {FAIL_MAX * 100:.0f}%)"
    )
    lines = [
        "NO TRADE",
        f"date: {entry_day.isoformat()}  ({entry_day.strftime('%d %b %Y')})",
        f"asof: {asof_d.isoformat()}",
        "code: ran",
        f"gate: {gate_line}",
        f"why: {why}",
    ]
    if scored_lines:
        lines.append("scored:")
        lines.extend(f"  {s}" for s in scored_lines)
    if rr_raises:
        lines.append("RR raise next session (still manage open names):")
        for r in rr_raises:
            lines.append(
                f"  RAISE {to_fyers(str(r['Ticker']))}  1:{int(r['chosen_rr'])}  "
                f"TP {r['Target_Price']:.2f}"
            )
    lines.append("Do not enter NEW Swing_PP names this session.")
    body = "\n".join(lines) + "\n"
    for path in ins_paths(entry_day):
        path.write_text(body, encoding="utf-8")
        print(f"[ins] {path}", flush=True)


def delete_ins() -> None:
    for path in ins_paths():
        if path.exists():
            path.unlink()
            print(f"[ins] names to trade -> deleted {path}", flush=True)
        else:
            print(f"[ins] names to trade -> no {path} to delete", flush=True)


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
    models = None
    if book:
        try:
            models = load_rr_models(asof)
        except Exception as exc:
            print(f"[rr] models failed: {exc}", flush=True)
        try:
            book, rr_raises, rr_keeps = scan_upgrades(book, asof, models)
            book = [p for p in book if p.get("status") != "closed"]
        except Exception as exc:
            print(f"[rr] scan failed: {exc}", flush=True)
        save_open_book(book)
        print(
            f"[rr] open-book {len(book)}  raise {len(rr_raises)}  keep-1:2 {len(rr_keeps)}",
            flush=True,
        )
    write_rr(rr_raises, rr_keeps, entry_d, asof_d)

    if gate["skip"]:
        why = (
            f"paper-gate skip — last {gate.get('ready')} predicted paper results failed "
            f"{gate.get('fail_pct')}% (threshold {FAIL_MAX * 100:.0f}%)"
        )
        write_watchlist([], entry_d)
        write_ins(entry_d, asof_d, gate, why, scored_lines, rr_raises)
    elif not symbols:
        write_watchlist([], entry_d)
        write_ins(entry_d, asof_d, gate, why, scored_lines, rr_raises)
    else:
        write_watchlist(symbols, entry_d)
        delete_ins()

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
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"[log] {PICKS_LOG}", flush=True)


if __name__ == "__main__":
    main()
