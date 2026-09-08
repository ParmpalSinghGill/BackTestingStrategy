"""
Pause from ALL predicted names' paper outcomes — not from fills.

Every name that cleared Meta_P (the daily prediction list) is scored 1:2 / SL / BE
from prices. Rolling fail% uses only those paper exits with Exit_Date < today.
Whether you had cash to buy does not matter. Deterministic on any asof date.
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
from swing_strategy.run_2pct_50_25_hunt import (
    CAPITAL,
    PCT,
    TV,
    ensure_1m,
    is_hit,
    load_nifty,
    pick,
)
from swing_strategy.run_2pct_valid_hunt import be_fair, rewrite
from swing_strategy.run_ml_top5_selector import _metrics

OUT = BASE / "Reports" / "SwingLow_OldLiquidity"
CACHE_BE = OUT / "Pred_fairBE_age1m_t048_top16.parquet"
LOG = OUT / "pred_paper_gate.json"
M46 = {"scale_hi": 1.3, "rank_hi": 1.2, "rank_lo": 0.6, "scale_lo": 0.40}


def paper_win(rec: dict) -> bool | None:
    r = float(rec.get("Realized_R", -9))
    if r >= 1.5:
        return True
    if r < -0.25:
        return False
    return None  # BE / scratch — not a fail, not a win


def exits_by_day(df: pd.DataFrame) -> dict:
    out = defaultdict(list)
    for rec in df.to_dict("records"):
        w = paper_win(rec)
        if w is None:
            continue
        out[pd.Timestamp(rec["Exit_Date"]).normalize()].append(w)
    return out


def sim_pred_gate(ml: pd.DataFrame, cfg: dict) -> dict:
    """2% fills on the prediction list, but skip days from paper fail% of ALL predictions."""
    roll_n = int(cfg.get("roll_n", 0))
    fail_max = float(cfg.get("fail_max", 1.0))
    k = int(cfg.get("k", 99))
    pause_days = int(cfg.get("pause_days", 0))
    scale_hi = float(cfg.get("scale_hi", 1.3))
    rank_hi = float(cfg.get("rank_hi", 1.2))
    rank_lo = float(cfg.get("rank_lo", 0.6))
    scale_lo = float(cfg.get("scale_lo", 0.40))

    paper = exits_by_day(ml)
    by_entry = defaultdict(list)
    for rec in ml.to_dict("records"):
        by_entry[pd.Timestamp(rec["Entry_Date"]).normalize()].append(rec)
    for day, cands in by_entry.items():
        cands.sort(key=lambda x: float(x.get("Meta_P", 0.0)), reverse=True)

    min_dt = pd.Timestamp("2010-01-01")
    max_dt = max(ml["Entry_Date"].max(), pd.to_datetime(ml["Exit_Date"]).max())
    cash = CAPITAL
    peak = CAPITAL
    max_dd = 0.0
    open_pos: dict[int, dict] = {}
    executed = 0
    skipped_days = 0
    trade_id = 1
    consec = 0
    pause_until = pd.Timestamp("1900-01-01")
    recent: deque[int] = deque(maxlen=max(roll_n, 1))
    event_days = sorted(set(by_entry.keys()) | set(paper.keys()) | {pd.Timestamp(r["Exit_Date"]).normalize() for recs in by_entry.values() for r in recs})

    for day in event_days:
        paused = day < pause_until
        fail_block = False
        if roll_n and len(recent) >= roll_n:
            fail_rate = 1.0 - (sum(recent) / len(recent))
            fail_block = fail_rate >= fail_max
        block = paused or fail_block
        if block and day in by_entry:
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
            if block:
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
            cash = round(cash - spend, 2)
            avail = round(avail - spend, 2)
            executed += 1
            open_pos[trade_id] = {
                "entry": entry_p,
                "exit": float(cand["Exit_Price"]),
                "qty": qty,
                "spend": spend,
                "xdt": pd.Timestamp(cand["Exit_Date"]).normalize(),
            }
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
            if won:
                consec = 0
            else:
                consec += 1
                if consec >= k and pause_days > 0:
                    pause_until = max(pause_until, day + pd.Timedelta(days=pause_days))
                    consec = 0

        port = cash + sum(p["spend"] for p in open_pos.values())
        peak = max(peak, port)
        max_dd = max(max_dd, ((peak - port) / peak * 100) if peak else 0)

    years = max((max_dt - min_dt).days / 365.25, 0.01)
    _, cagr = _metrics(CAPITAL, cash, years)
    fail_now = None
    if roll_n and len(recent) >= roll_n:
        fail_now = round(1.0 - sum(recent) / len(recent), 4)
    return {
        **{k: v for k, v in cfg.items() if k in ("roll_n", "fail_max", "k", "pause_days", "tag")},
        "CAGR": round(float(cagr), 2),
        "N": executed,
        "DD": round(max_dd, 2),
        "skipped_days": skipped_days,
        "fail_now": fail_now,
        "Final": round(cash, 2),
    }


def gate_asof(ml: pd.DataFrame, asof, roll_n: int, fail_max: float) -> dict:
    """Deterministic: paper exits with Exit_Date <= asof only."""
    asof = pd.Timestamp(asof).normalize()
    recent: deque[int] = deque(maxlen=roll_n)
    paper = exits_by_day(ml)
    for day in sorted(d for d in paper if d <= asof):
        for won in paper[day]:
            recent.append(1 if won else 0)
    n = len(recent)
    if n < roll_n:
        return {"asof": str(asof.date()), "ready": n, "need": roll_n, "skip": False, "fail_pct": None}
    fail = 1.0 - sum(recent) / n
    return {
        "asof": str(asof.date()),
        "ready": n,
        "need": roll_n,
        "fail_pct": round(fail * 100, 1),
        "skip": fail >= fail_max,
        "rule": f"last {roll_n} predicted paper results, skip if fail% >= {fail_max*100:.0f}%",
    }


def load_pred_list() -> pd.DataFrame:
    if CACHE_BE.exists():
        df = pd.read_parquet(CACHE_BE)
        df["Entry_Date"] = pd.to_datetime(df["Entry_Date"])
        df["Exit_Date"] = pd.to_datetime(df["Exit_Date"])
        return df
    print("[build] >=1m Meta_P>=0.48 top16 + fair BE ...", flush=True)
    nifty = load_nifty()
    re1 = ensure_1m(nifty)
    picked = pick(re1, 16, 0.48)
    rw = rewrite(picked, be_fair)
    OUT.mkdir(parents=True, exist_ok=True)
    rw.to_parquet(CACHE_BE, index=False)
    return rw


def main() -> None:
    ml = load_pred_list()
    print(f"prediction universe {len(ml):,}  (all Meta_P names, not fills)", flush=True)
    rows = []
    base = sim_pred_gate(ml, {"tag": "no_gate", **M46})
    rows.append(base)
    print(f"no_gate {base['CAGR']:+.2f}% DD {base['DD']} n={base['N']}", flush=True)

    cfgs = [{"tag": "no_gate"}]
    for n in (10, 15, 20, 25, 30, 40):
        for fm in (0.60, 0.65, 0.70, 0.75, 0.78, 0.80, 0.85):
            cfgs.append({"tag": f"fail{int(fm*100)}_n{n}", "roll_n": n, "fail_max": fm})
    for k in (5, 6, 8, 10, 12):
        for d in (5, 8, 10, 15, 20):
            cfgs.append({"tag": f"pconsec{k}_{d}d", "k": k, "pause_days": d})
    for n, fm, k, d in (
        (20, 0.75, 6, 10),
        (20, 0.80, 8, 8),
        (15, 0.75, 6, 8),
        (25, 0.75, 8, 10),
        (20, 0.70, 6, 15),
    ):
        cfgs.append({"tag": f"mix_n{n}_f{int(fm*100)}_k{k}_{d}d", "roll_n": n, "fail_max": fm, "k": k, "pause_days": d})

    hits = []
    for i, cfg in enumerate(cfgs, 1):
        if cfg["tag"] == "no_gate":
            continue
        r = sim_pred_gate(ml, {**M46, **cfg})
        rows.append(r)
        flag = " HIT" if is_hit(r) else ""
        if r["CAGR"] >= 45 or r["DD"] <= 26 or flag:
            print(
                f"  {cfg['tag']:28s} {r['CAGR']:+6.2f}% DD {r['DD']:5.1f} n={r['N']} skipd={r['skipped_days']}{flag}",
                flush=True,
            )
        if is_hit(r):
            hits.append(r)

    asof = pd.Timestamp("2026-09-08")
    print("\n===== closest CAGR>50 DD<25 (paper of ALL predictions) =====", flush=True)
    ok50 = [r for r in rows if r["CAGR"] > 50]
    for r in sorted(ok50, key=lambda x: x["DD"])[:12]:
        print(f"  {r['CAGR']:+6.2f}% DD {r['DD']:5.1f} n={r['N']} skipd={r.get('skipped_days')} {r.get('tag')}", flush=True)
    print("top CAGR", flush=True)
    for r in sorted(rows, key=lambda x: -x["CAGR"])[:8]:
        print(f"  {r['CAGR']:+6.2f}% DD {r['DD']:5.1f} {r.get('tag')}", flush=True)

    # pick best dual: CAGR>50 minimize DD, else best score
    dual = [r for r in rows if r["CAGR"] > 50 and r["DD"] < 25]
    if dual:
        best = min(dual, key=lambda x: x["DD"])
    else:
        best = max(ok50, key=lambda x: x["CAGR"] - 0.4 * x["DD"]) if ok50 else base
    print("best dual/score", best, flush=True)
    rn = int(best.get("roll_n") or 20)
    fm = float(best.get("fail_max") or 0.75)
    g = gate_asof(ml, asof, rn, fm)
    print(f"\n[forecast {asof.date()} -> next session] {g}", flush=True)

    LOG.write_text(json.dumps({"best": best, "gate_today": g, "hits": hits, "rows": rows}, indent=2, default=str), encoding="utf-8")
    print(f"wrote {LOG}", flush=True)


if __name__ == "__main__":
    main()
