"""
>=2 month liquidity + min volume scaled to Nifty (no fixed share count).

Volume methods (all Nifty-referenced):
  to_n   — C2 rupee turnover / Nifty close
  sh_n   — C2 shares / Nifty close
  to_n2  — C2 rupee turnover / Nifty^2
  plus optional vol_ratio20 / rel_dolvol floors

Then ML knobs, % equity, paper gates, filled-SL pauses, DD circuit.
Stops early if CAGR >= 50 and DD < 25. Does not write Swing_low / Swing_Live.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from swing_strategy.run_ml_next_search import META_FEAT, select_meta, walk_meta
from swing_strategy.run_ml_target_books import CORE_COLS
from swing_strategy.run_ml_wave3 import run_vol_managed
from swing_strategy.run_paper_gate import _exits_by_day
from swing_strategy.run_paper_gate import simulate as paper_sim
from swing_strategy.run_pause_search_v2 import simulate as pause_sim
from swing_strategy.run_swing_low_cagr_extra import run_pct_equity
from swing_strategy.run_swing_low_cagr_hunt import _walk_kind

SCORED = BASE / "Reports" / "SwingLowCagrHunt" / "Scored_hgb.parquet"
FEAT = BASE / "Reports" / "SwingLowLiquidity_v2" / "Features_v6_M22.parquet"
NIFTY_PATH = BASE / "data_daily" / "_NSEI_1d.csv"
OUT = BASE / "Reports" / "SwingLow_OldLiquidity"
HIT_CAGR = 50.0
HIT_DD = 25.0
MIN_AGE = 2


def age_ok(liq, entry, months: int) -> pd.Series:
    return pd.to_datetime(liq) + pd.DateOffset(months=months) <= pd.to_datetime(entry)


def load_nifty() -> pd.DataFrame:
    if NIFTY_PATH.exists():
        n = pd.read_csv(NIFTY_PATH)
        n["Date"] = pd.to_datetime(n["Date"])
        if pd.to_datetime(n["Date"]).max() >= pd.Timestamp("2026-08-01"):
            return n[["Date", "Close"]].dropna().sort_values("Date")
    import yfinance as yf

    print("[nifty] downloading ^NSEI ...", flush=True)
    raw = yf.download("^NSEI", start="2009-01-01", auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raw = yf.download("NIFTYBEES.NS", start="2009-01-01", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    raw = raw.reset_index()
    date_col = "Date" if "Date" in raw.columns else raw.columns[0]
    raw = raw.rename(columns={date_col: "Date"})
    raw["Date"] = pd.to_datetime(raw["Date"]).dt.tz_localize(None)
    close = raw["Close"] if "Close" in raw.columns else raw.iloc[:, -1]
    n = pd.DataFrame({"Date": raw["Date"], "Close": pd.to_numeric(close, errors="coerce")})
    n = n.dropna().sort_values("Date")
    NIFTY_PATH.parent.mkdir(parents=True, exist_ok=True)
    n.to_csv(NIFTY_PATH, index=False)
    print(f"[nifty] {len(n):,} bars  {n['Date'].min().date()} -> {n['Date'].max().date()}", flush=True)
    return n


def attach(df: pd.DataFrame, nifty: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Entry_Date"] = pd.to_datetime(out["Entry_Date"])
    out["Exit_Date"] = pd.to_datetime(out["Exit_Date"])
    out["Liquidity_Date"] = pd.to_datetime(out["Liquidity_Date"])
    n = nifty.rename(columns={"Date": "Entry_Date", "Close": "Nifty"}).sort_values("Entry_Date")
    out = out.sort_values("Entry_Date")
    out = pd.merge_asof(out, n, on="Entry_Date", direction="backward")
    out["turnover"] = np.exp(pd.to_numeric(out["dolvol_log"], errors="coerce").fillna(0.0).clip(upper=40))
    px = pd.to_numeric(out["Entry_Price"], errors="coerce").replace(0, np.nan)
    out["shares"] = out["turnover"] / px
    nif = pd.to_numeric(out["Nifty"], errors="coerce").replace(0, np.nan)
    out["to_n"] = out["turnover"] / nif
    out["sh_n"] = out["shares"] / nif
    out["to_n2"] = out["turnover"] / (nif ** 2)
    return out


def hit(res: dict) -> bool:
    return float(res.get("CAGR", -999)) >= HIT_CAGR and float(res.get("DD", 999)) < HIT_DD


def eval_pct(picked: pd.DataFrame, pct: float = 0.02) -> dict:
    if picked is None or picked.empty:
        return {"CAGR": None, "N": 0, "DD": None}
    r = run_pct_equity(picked, 50_000.0, pct)
    return {"CAGR": round(float(r["CAGR"]), 2), "N": int(r["N"]), "DD": float(r["DD"]), "Final": float(r["Final"])}


def pick(scored: pd.DataFrame, top_n: int, thresh: float) -> pd.DataFrame:
    return select_meta(scored, top_n, thresh, "Meta_P")


def dump(path: Path, rows: list) -> None:
    path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hits: list[dict] = []
    log: list[dict] = []

    nifty = load_nifty()
    scored = attach(pd.read_parquet(SCORED), nifty)
    scored = scored[age_ok(scored["Liquidity_Date"], scored["Entry_Date"], MIN_AGE)].copy()
    print(f"[base] >=2m setups {len(scored):,}  nifty {scored['Nifty'].min():.0f}-{scored['Nifty'].max():.0f}", flush=True)
    for col in ("to_n", "sh_n", "to_n2"):
        q = scored[col].quantile([0.15, 0.25, 0.35, 0.50, 0.65, 0.75, 0.85])
        print(f"  {col} q { {k: round(float(v), 4) for k, v in q.items()} }", flush=True)

    base_pick = pick(scored, 32, 0.42)
    base = eval_pct(base_pick)
    rec = {"phase": "baseline_2m", "tag": "no_vol_t0.42_top32_2pct", **base}
    log.append(rec)
    print(f"[base] 2% {base['CAGR']:+.2f}% DD {base['DD']} n={base['N']}", flush=True)
    if hit(base):
        hits.append(rec)
        print("HIT on baseline", flush=True)

    methods = [
        ("to_n", "turnover / Nifty"),
        ("sh_n", "shares / Nifty"),
        ("to_n2", "turnover / Nifty^2"),
    ]
    qs = (0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85)
    extras = [
        ("none", lambda d: d),
        ("vr1", lambda d: d[d["vol_ratio20"] >= 1.0]),
        ("vr1.2", lambda d: d[d["vol_ratio20"] >= 1.2]),
        ("rd50", lambda d: d[d["rel_dolvol"] >= 0.50]),
        ("rd70", lambda d: d[d["rel_dolvol"] >= 0.70]),
    ]

    print("===== phase 1: volume × extra × same-scores 2% =====", flush=True)
    for mcol, mname in methods:
        for q in qs:
            thr = float(scored[mcol].quantile(q))
            for extra_tag, extra_fn in extras:
                sub = extra_fn(scored[scored[mcol] >= thr])
                if len(sub) < 800:
                    continue
                picked = pick(sub, 32, 0.42)
                res = eval_pct(picked)
                row = {
                    "phase": "vol_grid",
                    "method": mcol,
                    "method_name": mname,
                    "q": q,
                    "min_ratio": round(thr, 6),
                    "extra": extra_tag,
                    "setups": int(len(sub)),
                    "ml_n": int(len(picked)),
                    "tag": f"{mcol}_q{q}_{extra_tag}",
                    **res,
                }
                log.append(row)
                mark = " **" if hit(res) else ""
                print(
                    f"  {row['tag']:28s} setups {len(sub):5d} ML {len(picked):4d}  "
                    f"{res['CAGR']:+6.2f}%  DD {res['DD']:5.1f}  n={res['N']}{mark}",
                    flush=True,
                )
                if hit(res):
                    hits.append(row)

    dump(OUT / "vol_nifty_2m.json", log)
    if hits:
        print("HIT found in phase 1", hits[-1], flush=True)

    vol_ok = [r for r in log if r.get("phase") == "vol_grid" and r.get("CAGR") is not None]
    vol_ok.sort(key=lambda r: (-(r["CAGR"] >= 45), r["DD"] - 0.4 * r["CAGR"], -r["CAGR"]))
    seeds = []
    seen = set()
    for r in vol_ok:
        key = (r["method"], r["q"], r["extra"])
        if key in seen:
            continue
        seen.add(key)
        seeds.append(r)
        if len(seeds) >= 8:
            break
    print("===== phase 2: knobs on best volume cuts =====", flush=True)
    knob_rows = []
    for seed in seeds:
        thr = float(seed["min_ratio"])
        mcol = seed["method"]
        extra_fn = dict(extras)[seed["extra"]]
        sub = extra_fn(scored[scored[mcol] >= thr])
        for top_n in (16, 24, 32):
            for thresh in (0.36, 0.38, 0.42, 0.48):
                for pct in (0.02, 0.03):
                    picked = pick(sub, top_n, thresh)
                    res = eval_pct(picked, pct)
                    row = {
                        "phase": "knobs",
                        "parent": seed["tag"],
                        "method": mcol,
                        "q": seed["q"],
                        "extra": seed["extra"],
                        "min_ratio": seed["min_ratio"],
                        "top_n": top_n,
                        "thresh": thresh,
                        "pct": pct,
                        "ml_n": int(len(picked)),
                        "tag": f"{seed['tag']}_t{thresh}_top{top_n}_p{int(pct*100)}",
                        **res,
                    }
                    knob_rows.append(row)
                    log.append(row)
                    if hit(res):
                        hits.append(row)
                        print(f"  HIT {row['tag']} {res['CAGR']:+.2f}% DD {res['DD']}", flush=True)
        best = max(
            [r for r in knob_rows if r["parent"] == seed["tag"]],
            key=lambda r: (r["CAGR"] >= 50 and r["DD"] < 25, r["CAGR"] - 0.35 * r["DD"]),
        )
        print(
            f"  seed {seed['tag']:28s} best {best['tag']}  {best['CAGR']:+.2f}% DD {best['DD']}",
            flush=True,
        )
        if hits:
            break

    dump(OUT / "vol_nifty_2m.json", log)
    if hits:
        print("HIT in phase 2 — skip remaining heavy search", flush=True)
        _finish(log, hits)
        return

    print("===== phase 3: retrain HGB/XGB on best volume universes =====", flush=True)
    feat = attach(pd.read_parquet(FEAT), nifty)
    feat = feat[age_ok(feat["Liquidity_Date"], feat["Entry_Date"], MIN_AGE)].copy()
    if "year" not in feat.columns:
        feat["year"] = feat["Entry_Date"].dt.year
    retrain_seeds = seeds[:4]
    retrain_books = []
    for seed in retrain_seeds:
        thr = float(seed["min_ratio"])
        mcol = seed["method"]
        extra_fn = dict(extras)[seed["extra"]]
        sub_ft = extra_fn(feat[feat[mcol] >= thr]).copy()
        if len(sub_ft) < 2000:
            print(f"  skip retrain {seed['tag']} n={len(sub_ft)}", flush=True)
            continue
        for kind in ("hgb", "xgb"):
            print(f"  retrain {kind} {seed['tag']} n={len(sub_ft):,}", flush=True)
            walked = _walk_kind(sub_ft, CORE_COLS, kind)
            walked = walk_meta(walked, META_FEAT)
            for top_n, thresh, pct in (
                (32, 0.42, 0.02),
                (32, 0.38, 0.02),
                (24, 0.42, 0.02),
                (32, 0.42, 0.03),
                (16, 0.42, 0.02),
            ):
                picked = pick(walked, top_n, thresh)
                res = eval_pct(picked, pct)
                row = {
                    "phase": "retrain",
                    "kind": kind,
                    "parent": seed["tag"],
                    "method": mcol,
                    "q": seed["q"],
                    "extra": seed["extra"],
                    "min_ratio": seed["min_ratio"],
                    "top_n": top_n,
                    "thresh": thresh,
                    "pct": pct,
                    "setups": int(len(sub_ft)),
                    "ml_n": int(len(picked)),
                    "tag": f"re_{kind}_{seed['tag']}_t{thresh}_top{top_n}_p{int(pct*100)}",
                    **res,
                }
                log.append(row)
                retrain_books.append((row, picked, walked))
                print(
                    f"    {row['tag']}  {res['CAGR']:+.2f}% DD {res['DD']} n={res['N']}",
                    flush=True,
                )
                if hit(res):
                    hits.append(row)
            if hits:
                break
        if hits:
            break

    dump(OUT / "vol_nifty_2m.json", log)
    if hits:
        print("HIT in retrain", flush=True)
        _finish(log, hits)
        return

    print("===== phase 4: paper / pause / DD circuit on best books =====", flush=True)
    candidates = []
    for r in log:
        if r.get("CAGR") is None:
            continue
        if r.get("phase") in ("vol_grid", "knobs", "retrain", "baseline_2m"):
            candidates.append(r)
    candidates.sort(key=lambda r: (r["CAGR"] - 0.35 * r["DD"]), reverse=True)
    gate_parents = []
    used = set()
    for r in candidates:
        key = r.get("tag")
        if key in used:
            continue
        used.add(key)
        gate_parents.append(r)
        if len(gate_parents) >= 4:
            break

    def rebuild_picked(row: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
        extra_fn = dict(extras)[row.get("extra", "none")]
        if row.get("phase") == "retrain":
            thr = float(row["min_ratio"])
            sub_ft = extra_fn(feat[feat[row["method"]] >= thr]).copy()
            walked = _walk_kind(sub_ft, CORE_COLS, row["kind"])
            walked = walk_meta(walked, META_FEAT)
            picked = pick(walked, int(row.get("top_n", 32)), float(row.get("thresh", 0.42)))
            return picked, walked
        thr = float(row.get("min_ratio", 0.0))
        mcol = row.get("method", "to_n")
        if row.get("phase") == "baseline_2m":
            sub = scored
        else:
            sub = extra_fn(scored[scored[mcol] >= thr])
        picked = pick(sub, int(row.get("top_n", 32)), float(row.get("thresh", 0.42)))
        return picked, sub

    paper_cfgs = [
        {"tag": "base", "k": 99},
        {"tag": "ml_roll20_0.22", "shadow": "ml", "roll_n": 20, "roll_min": 0.22},
        {"tag": "ml_roll20_0.25", "shadow": "ml", "roll_n": 20, "roll_min": 0.25},
        {"tag": "ml_slratio20_0.75", "shadow": "ml", "sl_n": 20, "sl_max": 0.75},
        {"tag": "ml_slratio20_0.80", "shadow": "ml", "sl_n": 20, "sl_max": 0.80},
        {"tag": "ml_slratio30_0.75", "shadow": "ml", "sl_n": 30, "sl_max": 0.75},
        {"tag": "all_slratio20_0.75", "shadow": "all", "sl_n": 20, "sl_max": 0.75},
        {"tag": "ml_k6_d15", "shadow": "ml", "k": 6, "pause_days": 15},
        {"tag": "all_k6_d15", "shadow": "all", "k": 6, "pause_days": 15},
        {"tag": "ml_k5_d20", "shadow": "ml", "k": 5, "pause_days": 20},
    ]
    pause_cfgs = [
        {"tag": "pause6_15d", "k": 6, "pause_days": 15},
        {"tag": "pause5_20d", "k": 5, "pause_days": 20},
        {"tag": "dayloss2_10d", "day_loss_n": 2, "day_loss_pause": 10},
        {"tag": "skip5_k5", "k": 5, "skip_trades": 5},
        {"tag": "dd15_20d", "dd_pause": 0.15, "dd_days": 20, "dd_resume": 0.05},
        {"tag": "dd20_20d", "dd_pause": 0.20, "dd_days": 20, "dd_resume": 0.05},
        {"tag": "dd25_20d", "dd_pause": 0.25, "dd_days": 20, "dd_resume": 0.05},
        {"tag": "half_after3", "cut_after": 3, "size_cut": 0.50},
        {"tag": "roll20_0.20", "roll_n": 20, "roll_min": 0.20},
        {"tag": "both_k6_s5_d15", "k": 6, "skip_trades": 5, "pause_days": 15},
    ]

    for parent in gate_parents:
        print(f"  gates on {parent['tag']}  {parent['CAGR']:+.2f}% DD {parent['DD']}", flush=True)
        picked, universe = rebuild_picked(parent)
        paper_all = _exits_by_day(universe)
        paper_ml = _exits_by_day(picked)
        for cfg in paper_cfgs:
            shadow = cfg.get("shadow", "none")
            paper = {} if shadow == "none" else (paper_ml if shadow == "ml" else paper_all)
            r = paper_sim(picked, paper, cfg)
            row = {
                "phase": "paper",
                "parent": parent["tag"],
                "tag": f"{parent['tag']}__paper_{cfg['tag']}",
                "CAGR": r["CAGR"],
                "DD": r["DD"],
                "N": r["N"],
            }
            log.append(row)
            print(f"    paper {cfg['tag']:22s} {r['CAGR']:+6.2f}% DD {r['DD']:5.1f} n={r['N']}", flush=True)
            if hit(r):
                hits.append(row)
        for cfg in pause_cfgs:
            r = pause_sim(picked, cfg)
            row = {
                "phase": "pause",
                "parent": parent["tag"],
                "tag": f"{parent['tag']}__pause_{cfg['tag']}",
                "CAGR": r["CAGR"],
                "DD": r["DD"],
                "N": r["N"],
            }
            log.append(row)
            print(f"    pause {cfg['tag']:22s} {r['CAGR']:+6.2f}% DD {r['DD']:5.1f} n={r['N']}", flush=True)
            if hit(r):
                hits.append(row)
        if hits:
            break

    # last shot: frozen high rupee + volume (won't be 2% book but report)
    print("===== phase 5: frozen rupee on best volume ML list =====", flush=True)
    best_vol = vol_ok[0] if vol_ok else None
    if best_vol:
        extra_fn = dict(extras)[best_vol["extra"]]
        sub = extra_fn(scored[scored[best_vol["method"]] >= float(best_vol["min_ratio"])])
        picked = pick(sub, 32, 0.42)
        for risk in (500, 1000, 2800, 4000, 8000):
            fr = run_vol_managed(picked, 50_000.0, float(risk), 0.04)
            row = {
                "phase": "frozen",
                "parent": best_vol["tag"],
                "risk": risk,
                "tag": f"{best_vol['tag']}_rs{risk}",
                "CAGR": round(float(fr["CAGR"]), 2),
                "DD": float(fr["DD"]),
                "N": int(fr["N"]),
            }
            log.append(row)
            print(f"  Rs{risk}: {row['CAGR']:+.2f}% DD {row['DD']} n={row['N']}", flush=True)
            if hit(row):
                hits.append(row)

    dump(OUT / "vol_nifty_2m.json", log)
    _finish(log, hits)


def _finish(log: list[dict], hits: list[dict]) -> None:
    usable = [r for r in log if r.get("CAGR") is not None]
    print("\n===== TARGET CAGR>=50 and DD<25 =====", flush=True)
    if hits:
        for r in hits[:10]:
            print(f"  HIT {r.get('tag')}  {r['CAGR']:+.2f}% DD {r['DD']} n={r.get('N')}", flush=True)
    else:
        print("  no hit", flush=True)
        both = [r for r in usable if r["CAGR"] >= 40]
        both.sort(key=lambda r: (r["DD"], -r["CAGR"]))
        print("  closest by DD among CAGR>=40:", flush=True)
        for r in both[:12]:
            print(f"    {r['CAGR']:+6.2f}%  DD {r['DD']:5.1f}  n={r.get('N')}  {r.get('tag')}", flush=True)
        print("  top CAGR:", flush=True)
        for r in sorted(usable, key=lambda r: -r["CAGR"])[:12]:
            print(f"    {r['CAGR']:+6.2f}%  DD {r['DD']:5.1f}  n={r.get('N')}  {r.get('tag')}", flush=True)
        print("  best score CAGR-0.35*DD:", flush=True)
        for r in sorted(usable, key=lambda r: -(r["CAGR"] - 0.35 * r["DD"]))[:12]:
            print(
                f"    {r['CAGR']:+6.2f}%  DD {r['DD']:5.1f}  score {r['CAGR']-0.35*r['DD']:5.1f}  {r.get('tag')}",
                flush=True,
            )
    print(f"wrote {OUT / 'vol_nifty_2m.json'}  rows={len(log)}", flush=True)


if __name__ == "__main__":
    main()
