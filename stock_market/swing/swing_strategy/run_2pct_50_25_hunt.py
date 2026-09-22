"""
Locked 2% equity. Search until CAGR>50 and DD<25.

Does not raise risk above 2%. Exit mods (BE / scale) and filters only.
Never writes Swing_low / Swing_Live.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from src.analysis.indian_brokerage_calculator import calculate_indian_trade_charges
from swing_strategy.run_ml_next_search import META_FEAT, select_meta, walk_meta
from swing_strategy.run_ml_target_books import CORE_COLS
from swing_strategy.run_ml_top5_selector import _metrics
from swing_strategy.run_swing_low_cagr_hunt import _walk_kind

SCORED = BASE / "Reports" / "SwingLowCagrHunt" / "Scored_hgb.parquet"
FEAT = BASE / "Reports" / "SwingLowLiquidity_v2" / "Features_v6_M22.parquet"
NIFTY = BASE / "data_daily" / "_NSEI_1d.csv"
OUT = BASE / "Reports" / "SwingLow_OldLiquidity"
CACHE_1M = OUT / "Scored_hgb_age1m.parquet"
LOG = OUT / "hunt_2pct_50_25.json"
PCT = 0.02
TV = 0.04
CAPITAL = 50_000.0
HIT_CAGR = 50.0
HIT_DD = 25.0


def age_ok(liq, entry, months: int) -> pd.Series:
    return pd.to_datetime(liq) + pd.DateOffset(months=months) <= pd.to_datetime(entry)


def load_nifty() -> pd.DataFrame:
    n = pd.read_csv(NIFTY)
    n["Date"] = pd.to_datetime(n["Date"])
    n = n.sort_values("Date")
    n["sma50"] = n["Close"].rolling(50).mean()
    n["sma100"] = n["Close"].rolling(100).mean()
    n["sma200"] = n["Close"].rolling(200).mean()
    n["ret20"] = n["Close"].pct_change(20)
    n["vol20"] = n["Close"].pct_change().rolling(20).std()
    return n


def attach_nifty(df: pd.DataFrame, nifty: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Entry_Date"] = pd.to_datetime(out["Entry_Date"])
    out["Exit_Date"] = pd.to_datetime(out["Exit_Date"])
    cols = ["Date", "Close", "sma50", "sma100", "sma200", "ret20", "vol20"]
    n = nifty[cols].rename(columns={"Date": "Entry_Date", "Close": "Nifty"}).sort_values("Entry_Date")
    out = out.sort_values("Entry_Date")
    return pd.merge_asof(out, n, on="Entry_Date", direction="backward")


def apply_exit(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Rewrite Exit_Price from MFE_R. Does not change size."""
    if mode == "raw":
        return df
    out = df.copy()
    entry = pd.to_numeric(out["Entry_Price"], errors="coerce")
    sl = pd.to_numeric(out["SL_Price"], errors="coerce")
    risk = (entry - sl).clip(lower=0.05)
    mfe = pd.to_numeric(out["MFE_R"], errors="coerce").fillna(0.0)
    r = pd.to_numeric(out["Realized_R"], errors="coerce")
    win = r >= 1.5
    if mode.startswith("be"):
        need = float(mode.split("_")[1])
        be = (~win) & (mfe >= need)
        new_r = r.where(~be, 0.0)
    elif mode == "scale50":
        # 50% at 1R if MFE>=1, remainder 2R or SL
        new_r = np.where(win, 1.5, np.where(mfe >= 1.0, 0.0, r))
    elif mode == "scale50_be12":
        new_r = np.where(win, 1.5, np.where(mfe >= 1.2, 0.0, r))
    else:
        return out
    out["Realized_R"] = new_r
    out["Exit_Price"] = (entry + new_r * risk).round(2)
    out["Outcome"] = np.where(out["Realized_R"] >= 1.5, "Success", np.where(out["Realized_R"] >= 0, "BE", "Failure"))
    return out


def sim(picked: pd.DataFrame, cfg: dict) -> dict:
    """2% equity, vol 0.04, rank taper. Pause uses Realized_R<=-0.5 as SL (BE is not an SL)."""
    k = int(cfg.get("k", 99))
    pause_days = int(cfg.get("pause_days", 0))
    skip_trades = int(cfg.get("skip_trades", 0))
    size_cut = float(cfg.get("size_cut", 1.0))
    cut_after = int(cfg.get("cut_after", 99))
    day_loss_n = int(cfg.get("day_loss_n", 99))
    day_loss_pause = int(cfg.get("day_loss_pause", 0))
    max_day = int(cfg.get("max_day", 99))
    dd_pause = float(cfg.get("dd_pause", 0.0))
    dd_days = int(cfg.get("dd_days", 0))
    dd_resume = float(cfg.get("dd_resume", 0.0))
    scale_lo = float(cfg.get("scale_lo", 0.40))
    scale_hi = float(cfg.get("scale_hi", 1.80))
    rank_hi = float(cfg.get("rank_hi", 1.4))
    rank_lo = float(cfg.get("rank_lo", 0.6))

    df = picked.copy()
    df["Entry_Date"] = pd.to_datetime(df["Entry_Date"])
    by_entry = defaultdict(list)
    for row in df.to_dict("records"):
        by_entry[pd.Timestamp(row["Entry_Date"]).normalize()].append(row)
    for day, cands in by_entry.items():
        cands.sort(key=lambda x: float(x.get("Meta_P", 0.0)), reverse=True)
        if max_day < 99:
            del cands[max_day:]

    min_dt = pd.Timestamp("2010-01-01")
    max_dt = max(df["Entry_Date"].max(), pd.to_datetime(df["Exit_Date"]).max())
    cash = CAPITAL
    peak = CAPITAL
    max_dd = 0.0
    open_pos: dict[int, dict] = {}
    executed = 0
    trade_id = 1
    consec = 0
    skip_left = 0
    pause_until = pd.Timestamp("1900-01-01")
    size_mult = 1.0
    event_days = sorted(set(by_entry.keys()) | {pd.Timestamp(r["Exit_Date"]).normalize() for recs in by_entry.values() for r in recs})

    for day in event_days:
        port = cash + sum(p["spend"] for p in open_pos.values())
        if port >= peak:
            peak = port
        dd_now = ((peak - port) / peak) if peak else 0.0
        if dd_pause > 0 and dd_now >= dd_pause:
            pause_until = max(pause_until, day + pd.Timedelta(days=dd_days))
        if dd_resume > 0 and dd_now <= dd_resume:
            pause_until = min(pause_until, day)

        risk = max(port * PCT * size_mult, 100.0)
        avail = cash
        cands = by_entry.get(day, [])
        n = max(len(cands), 1)
        vols = [float(c.get("idio_vol", np.nan)) for c in cands]
        vols = [v for v in vols if np.isfinite(v) and v > 0]
        med = float(np.median(vols)) if vols else TV
        scale = float(np.clip(TV / max(med, 1e-6), scale_lo, scale_hi))
        paused = day < pause_until
        day_losses = 0
        taken = 0
        for i, cand in enumerate(cands):
            if paused or skip_left > 0:
                if skip_left > 0:
                    skip_left -= 1
                continue
            if taken >= max_day:
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
            taken += 1
            rr = float(cand.get("Realized_R", 0.0))
            open_pos[trade_id] = {
                "entry": entry_p,
                "exit": float(cand["Exit_Price"]),
                "qty": qty,
                "spend": spend,
                "xdt": pd.Timestamp(cand["Exit_Date"]).normalize(),
                "rr": rr,
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
            rr = float(pos.get("rr", 0.0))
            if rr >= -0.25:
                consec = 0
                size_mult = 1.0
            else:
                consec += 1
                day_losses += 1
                if consec >= cut_after:
                    size_mult = size_cut
                if consec >= k:
                    if pause_days > 0:
                        pause_until = max(pause_until, day + pd.Timedelta(days=pause_days))
                    if skip_trades > 0:
                        skip_left = skip_trades
                    consec = 0
        if day_losses >= day_loss_n and day_loss_pause > 0:
            pause_until = max(pause_until, day + pd.Timedelta(days=day_loss_pause))

        port = cash + sum(p["spend"] for p in open_pos.values())
        peak = max(peak, port)
        max_dd = max(max_dd, ((peak - port) / peak * 100) if peak else 0)

    years = max((max_dt - min_dt).days / 365.25, 0.01)
    _, cagr = _metrics(CAPITAL, cash, years)
    return {"CAGR": round(float(cagr), 2), "N": executed, "DD": round(max_dd, 2), "Final": round(cash, 2)}


def is_hit(r: dict) -> bool:
    return r.get("CAGR", -999) > HIT_CAGR and r.get("DD", 999) < HIT_DD


def pick(df, top_n, thresh):
    return select_meta(df, top_n, thresh, "Meta_P")


def nifty_filter(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    if kind == "all":
        return df
    if kind == "sma50":
        return df[df["Nifty"] > df["sma50"]]
    if kind == "sma100":
        return df[df["Nifty"] > df["sma100"]]
    if kind == "sma200":
        return df[df["Nifty"] > df["sma200"]]
    if kind == "ret20":
        return df[df["ret20"] > 0]
    if kind == "sma50_ret20":
        return df[(df["Nifty"] > df["sma50"]) & (df["ret20"] > 0)]
    if kind == "not_crash":
        q = df["vol20"].quantile(0.90)
        return df[df["vol20"] <= q]
    if kind == "ym":
        return df[df["TF_Rank"] >= 2]
    return df


def ensure_1m(nifty: pd.DataFrame) -> pd.DataFrame:
    if CACHE_1M.exists():
        s = pd.read_parquet(CACHE_1M)
        s["Entry_Date"] = pd.to_datetime(s["Entry_Date"])
        return attach_nifty(s, nifty)
    print("[train] >=1m HGB+meta ...", flush=True)
    feat = pd.read_parquet(FEAT)
    feat["Entry_Date"] = pd.to_datetime(feat["Entry_Date"])
    feat["Exit_Date"] = pd.to_datetime(feat["Exit_Date"])
    feat["Liquidity_Date"] = pd.to_datetime(feat["Liquidity_Date"])
    feat = feat[age_ok(feat["Liquidity_Date"], feat["Entry_Date"], 1)].copy()
    if "year" not in feat.columns:
        feat["year"] = feat["Entry_Date"].dt.year
    walked = _walk_kind(feat, CORE_COLS, "hgb")
    walked = walk_meta(walked, META_FEAT)
    OUT.mkdir(parents=True, exist_ok=True)
    walked.to_parquet(CACHE_1M, index=False)
    return attach_nifty(walked, nifty)


def dump(rows, hits):
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.write_text(json.dumps({"hits": hits, "rows": rows[-400:], "n": len(rows)}, indent=2), encoding="utf-8")


def main() -> None:
    nifty = load_nifty()
    raw = pd.read_parquet(SCORED)
    raw["Entry_Date"] = pd.to_datetime(raw["Entry_Date"])
    raw["Exit_Date"] = pd.to_datetime(raw["Exit_Date"])
    raw["Liquidity_Date"] = pd.to_datetime(raw["Liquidity_Date"])
    raw = attach_nifty(raw, nifty)
    raw["turnover"] = np.exp(pd.to_numeric(raw["dolvol_log"], errors="coerce").fillna(0).clip(upper=40))
    raw["sh_n"] = raw["turnover"] / pd.to_numeric(raw["Entry_Price"], errors="coerce") / pd.to_numeric(raw["Nifty"], errors="coerce")

    age1 = raw[age_ok(raw["Liquidity_Date"], raw["Entry_Date"], 1)].copy()
    age2 = raw[age_ok(raw["Liquidity_Date"], raw["Entry_Date"], 2)].copy()
    sh15 = float(age2["sh_n"].quantile(0.15)) if len(age2) else 0
    age2v = age2[age2["sh_n"] >= sh15].copy()
    scored_1m = ensure_1m(nifty)
    scored_1m["sh_n"] = scored_1m.get("sh_n", np.nan)
    if "dolvol_log" in scored_1m.columns and "Nifty" in scored_1m.columns:
        scored_1m["turnover"] = np.exp(pd.to_numeric(scored_1m["dolvol_log"], errors="coerce").fillna(0).clip(upper=40))
        scored_1m["sh_n"] = scored_1m["turnover"] / pd.to_numeric(scored_1m["Entry_Price"], errors="coerce") / pd.to_numeric(scored_1m["Nifty"], errors="coerce")

    universes = {
        "full": raw,
        "age1_same": age1,
        "age2_same": age2,
        "age2_sh15": age2v,
        "age1_retrain": scored_1m,
    }
    print("universes", {k: len(v) for k, v in universes.items()}, "sh15", round(sh15, 4), flush=True)

    exits = ["raw", "be_1.0", "be_1.2", "be_1.5", "scale50", "scale50_be12"]
    nfty = ["all", "sma50", "sma200", "ret20", "sma50_ret20", "not_crash", "ym"]
    knobs = [(32, 0.42), (24, 0.42), (16, 0.42), (32, 0.38), (16, 0.48), (12, 0.48), (24, 0.48), (8, 0.50)]
    pauses = [
        {},
        {"k": 6, "pause_days": 15},
        {"k": 5, "pause_days": 20},
        {"k": 6, "pause_days": 15, "cut_after": 3, "size_cut": 0.5},
        {"day_loss_n": 2, "day_loss_pause": 10},
        {"max_day": 8},
        {"max_day": 5},
        {"k": 6, "pause_days": 15, "max_day": 8},
        {"k": 6, "pause_days": 15, "max_day": 5},
        {"cut_after": 3, "size_cut": 0.5},
    ]

    rows = []
    hits = []

    def run_one(tag, picked, pcfg):
        if picked is None or picked.empty:
            return None
        r = sim(picked, pcfg)
        r["tag"] = tag
        r["pct"] = PCT
        rows.append(r)
        mark = " HIT" if is_hit(r) else ""
        if r["CAGR"] >= 40 or r["DD"] <= 28 or mark:
            print(f"  {tag:70s} {r['CAGR']:+6.2f}% DD {r['DD']:5.1f} n={r['N']}{mark}", flush=True)
        if is_hit(r):
            hits.append(r)
            dump(rows, hits)
            print("===== FOUND =====", r, flush=True)
        return r

    # Wave 1: exit mods × universes × default knobs
    print("===== wave 1: exit mods @ 2% =====", flush=True)
    for uname, u in universes.items():
        for em in exits:
            ue = apply_exit(u, em)
            for top_n, thresh in ((32, 0.42), (16, 0.48), (24, 0.42)):
                picked = pick(ue, top_n, thresh)
                run_one(f"{uname}/{em}/t{thresh}/top{top_n}", picked, {})
                if hits:
                    return
            if hits:
                return

    dump(rows, hits)
    print("===== wave 2: nifty regime + pause on best-looking =====", flush=True)
    # rank wave1 by CAGR-0.4*DD among CAGR>=40 or best DD
    ranked = sorted([r for r in rows if r["CAGR"] is not None], key=lambda x: -(x["CAGR"] - 0.4 * x["DD"]))[:12]
    # also force age1_retrain/be combos
    force = [
        ("age1_retrain", "raw"),
        ("age1_retrain", "be_1.0"),
        ("age1_retrain", "be_1.2"),
        ("age1_retrain", "scale50"),
        ("full", "be_1.0"),
        ("age1_same", "be_1.0"),
        ("age2_sh15", "be_1.0"),
        ("age2_same", "be_1.2"),
    ]
    combos = []
    for uname, em in force:
        combos.append((uname, em))
    for nf in nfty:
        for uname, em in (("age1_retrain", "be_1.0"), ("age1_retrain", "raw"), ("full", "be_1.0"), ("age1_retrain", "scale50")):
            combos.append((uname, em, nf))

    seen = set()
    for item in combos:
        if len(item) == 2:
            uname, em = item
            nf = "all"
        else:
            uname, em, nf = item
        key = (uname, em, nf)
        if key in seen:
            continue
        seen.add(key)
        u = nifty_filter(apply_exit(universes[uname], em), nf)
        if len(u) < 500:
            continue
        for top_n, thresh in knobs:
            picked = pick(u, top_n, thresh)
            for pi, pcfg in enumerate(pauses):
                tag = f"{uname}/{em}/{nf}/t{thresh}/top{top_n}/p{pi}"
                run_one(tag, picked, pcfg)
                if hits:
                    return
        dump(rows, hits)

    print("===== wave 3: denser pause on top 2% books =====", flush=True)
    best = sorted(rows, key=lambda x: -(x["CAGR"] - 0.35 * x["DD"]))[:6]
    extra_pauses = []
    for k in (4, 5, 6, 7, 8):
        for d in (8, 10, 12, 15, 18, 20, 25):
            extra_pauses.append({"k": k, "pause_days": d})
    extra_pauses += [
        {"k": 6, "pause_days": 15, "max_day": 6},
        {"k": 6, "pause_days": 15, "cut_after": 2, "size_cut": 0.5},
        {"k": 6, "pause_days": 15, "cut_after": 3, "size_cut": 0.4},
        {"day_loss_n": 2, "day_loss_pause": 8, "k": 6, "pause_days": 15},
        {"max_day": 4},
        {"max_day": 6, "k": 6, "pause_days": 15},
    ]
    # rebuild from tags is hard; re-run the force universe that produced best CAGR>=45
    for uname, em, nf in (
        ("age1_retrain", "be_1.0", "all"),
        ("age1_retrain", "be_1.2", "all"),
        ("age1_retrain", "scale50", "all"),
        ("age1_retrain", "raw", "sma50"),
        ("age1_retrain", "be_1.0", "sma50"),
        ("full", "be_1.0", "all"),
        ("full", "scale50", "all"),
        ("age1_retrain", "be_1.5", "all"),
        ("age1_retrain", "be_1.0", "sma200"),
        ("age1_retrain", "be_1.0", "ret20"),
    ):
        u = nifty_filter(apply_exit(universes[uname], em), nf)
        for top_n, thresh in ((32, 0.42), (16, 0.42), (16, 0.48), (24, 0.38), (12, 0.48)):
            picked = pick(u, top_n, thresh)
            for j, pcfg in enumerate(extra_pauses):
                run_one(f"w3/{uname}/{em}/{nf}/t{thresh}/top{top_n}/x{j}", picked, pcfg)
                if hits:
                    return

    dump(rows, hits)
    print("===== no hit yet. closest =====", flush=True)
    ok = [r for r in rows if r["CAGR"] >= 45]
    for r in sorted(ok, key=lambda x: x["DD"])[:15]:
        print(f"  {r['CAGR']:+6.2f}% DD {r['DD']:5.1f} n={r['N']} {r['tag']}", flush=True)
    for r in sorted(rows, key=lambda x: -x["CAGR"])[:10]:
        print(f"  CAGR {r['CAGR']:+6.2f}% DD {r['DD']:5.1f} {r['tag']}", flush=True)
    print(f"logged {len(rows)} @ {LOG}", flush=True)


if __name__ == "__main__":
    main()
