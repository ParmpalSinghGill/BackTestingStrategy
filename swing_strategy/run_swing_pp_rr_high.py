"""Rare 1:5 / 1:6 upgrades on top of the M47 1R 1:3 switch.

Keep default 1:2. After +1R, if P(1:3)>=0.80 raise to 1:3. Only then, if P(1:5)
or P(1:6) is high enough, upgrade further. Economic vs banking 2R: 1:5 needs
>40%, 1:6 >33%; vs giving up a 1:3 already chosen, 1:5 needs >60%. We still
use tight cuts so capital is not locked on noisy flags.

Do not write Swing_low / Swing_Live.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from swing_strategy.run_pred_paper_gate import M46, load_pred_list
from swing_strategy.run_swing_pp_rr_classifier import (
    BASELINE_CAGR,
    BASELINE_DD,
    LOG,
    OUT,
    PATH1,
    _feat_cols,
    attach_lots,
    build_cache,
    choose_ladder,
    oos_table,
    sim_lots,
    walk_proba,
)

HIGH_LOG = OUT / "swing_pp_rr_high.json"
T3 = 0.80  # M47 economic 1:3 cut


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    paper = load_pred_list()
    paper["Entry_Date"] = pd.to_datetime(paper["Entry_Date"])
    df = build_cache(paper)
    df["Entry_Date"] = pd.to_datetime(df["Entry_Date"])
    for c in ("xdt_2", "xdt_3", "xdt_4", "xdt_5", "xdt_6", "label_date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    df["armed"] = pd.to_numeric(df.get("armed", 0), errors="coerce").fillna(0).astype(int)
    df["hit2"] = pd.to_numeric(df.get("hit2", 0), errors="coerce").fillna(0).astype(int)
    for k in (2, 3, 4, 5, 6):
        df[f"win_{k}"] = pd.to_numeric(df.get(f"win_{k}", 0), errors="coerce").fillna(0).astype(int)
    print(
        f"universe {len(df):,} armed {int(df.armed.sum()):,} "
        f"win3 {int(df.win_3.sum()):,} win5 {int(df.win_5.sum()):,} win6 {int(df.win_6.sum()):,}",
        flush=True,
    )

    arm_cols = _feat_cols(df, PATH1)
    armed_mask = df["armed"] == 1
    print("[wf] at-1R P(win 1:3/5/6 | armed) ...", flush=True)
    df["p3_1r"] = walk_proba(df, "win_3", arm_cols, armed_mask)
    df["p5_1r"] = walk_proba(df, "win_5", arm_cols, armed_mask)
    df["p6_1r"] = walk_proba(df, "win_6", arm_cols, armed_mask)
    # 1:4 kept for ladder completeness
    df["p4_1r"] = walk_proba(df, "win_4", arm_cols, armed_mask)

    th = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95]
    acc = {
        "arm_win5": oos_table(df.loc[armed_mask, "win_5"].to_numpy(), df.loc[armed_mask, "p5_1r"].to_numpy(), th),
        "arm_win6": oos_table(df.loc[armed_mask, "win_6"].to_numpy(), df.loc[armed_mask, "p6_1r"].to_numpy(), th),
        "arm_win3": oos_table(df.loc[armed_mask, "win_3"].to_numpy(), df.loc[armed_mask, "p3_1r"].to_numpy(), th),
    }
    print("OOS P(1:5 | armed):", flush=True)
    for row in acc["arm_win5"]:
        print(f"  t={row['t']:.2f} n={row['n_flag']:4d} prec={row['precision']:.1%} rec={row['recall']:.1%}", flush=True)
    print("OOS P(1:6 | armed):", flush=True)
    for row in acc["arm_win6"]:
        print(f"  t={row['t']:.2f} n={row['n_flag']:4d} prec={row['precision']:.1%} rec={row['recall']:.1%}", flush=True)

    armed = df["armed"].to_numpy() == 1
    bars = pd.to_numeric(df.get("bars_to_1R"), errors="coerce").fillna(99).to_numpy()
    close_r = pd.to_numeric(df.get("arm_close_R"), errors="coerce").fillna(0).to_numpy()
    meta = pd.to_numeric(df["Meta_P"], errors="coerce").fillna(0).to_numpy()
    gate = {"roll_n": 50, "fail_max": 0.74, **M46}
    rows = []

    def run_policy(tag, chosen, scale50=False):
        ch = pd.Series(chosen, index=df.index) if not isinstance(chosen, pd.Series) else chosen
        n5 = int((ch == 5).sum())
        n6 = int((ch == 6).sum())
        fill = attach_lots(df, ch, scale50)
        r = sim_lots(fill, paper, {**gate, "tag": tag})
        r["n_1to5_names"] = n5
        r["n_1to6_names"] = n6
        r["n_flagged_names"] = int((ch > 2).sum())
        rows.append(r)
        has_high = n5 + n6 > 0
        mark = ""
        if r.get("better"):
            mark = " BETTER"
        elif r.get("keep"):
            mark = " KEEP"
        show = has_high or "baseline" in tag or "arm3_t0.80" in tag or r["CAGR"] >= 55
        if show:
            print(
                f"  {tag:48s} CAGR {r['CAGR']:+6.2f}% DD {r['DD']:5.1f} n={r['N']:4d} "
                f"rr={r.get('rr_used')} 5={n5} 6={n6}{mark}",
                flush=True,
            )
        return r

    print("\n===== anchors =====", flush=True)
    run_policy("baseline_1to2", pd.Series(2, index=df.index))
    ch3 = choose_ladder(df["p3_1r"], df["p4_1r"], df["p5_1r"], T3, None, None, armed)
    run_policy("arm3_t0.80_no_upgrade", ch3)
    ora5 = pd.Series(np.where(df["win_5"] == 1, 5, np.where(df["win_3"] == 1, 3, 2)), index=df.index)
    run_policy("ORACLE_5_else_3_else_2", ora5)
    ora6 = pd.Series(np.where(df["win_6"] == 1, 6, np.where(df["win_3"] == 1, 3, 2)), index=df.index)
    run_policy("ORACLE_6_else_3_else_2", ora6)

    print("\n===== 1:5 only (replace 1:2, no 1:3 layer) =====", flush=True)
    for t5 in (0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95):
        ch = choose_ladder(df["p3_1r"], df["p4_1r"], df["p5_1r"], None, None, t5, armed)
        run_policy(f"arm5_only_t{t5:.2f}", ch)

    print("\n===== 1:6 only =====", flush=True)
    for t6 in (0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        ch = choose_ladder(df["p3_1r"], df["p4_1r"], df["p5_1r"], None, None, None, armed, df["p6_1r"], t6)
        run_policy(f"arm6_only_t{t6:.2f}", ch)

    print("\n===== stack on 1:3 t=0.80, upgrade 1:5 / 1:6 =====", flush=True)
    for t5 in (0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95):
        ch = choose_ladder(df["p3_1r"], df["p4_1r"], df["p5_1r"], T3, None, t5, armed)
        run_policy(f"stack_t3={T3:.2f}_t5={t5:.2f}", ch)
    for t6 in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        ch = choose_ladder(df["p3_1r"], df["p4_1r"], df["p5_1r"], T3, None, None, armed, df["p6_1r"], t6)
        run_policy(f"stack_t3={T3:.2f}_t6={t6:.2f}", ch)
    for t5, t6 in (
        (0.80, 0.80),
        (0.85, 0.80),
        (0.85, 0.85),
        (0.90, 0.85),
        (0.90, 0.90),
        (0.75, 0.75),
        (0.80, 0.75),
        (0.92, 0.90),
    ):
        ch = choose_ladder(df["p3_1r"], df["p4_1r"], df["p5_1r"], T3, None, t5, armed, df["p6_1r"], t6)
        run_policy(f"stack_t3={T3:.2f}_t5={t5:.2f}_t6={t6:.2f}", ch)

    print("\n===== fast-arm filter (bars_to_1R <= 3) on 5/6 =====", flush=True)
    fast = armed & (bars <= 3)
    for t5 in (0.70, 0.75, 0.80, 0.85):
        ch = choose_ladder(df["p3_1r"], df["p4_1r"], df["p5_1r"], T3, None, t5, fast)
        # restore 1:3 on slow arms that still clear t3
        ch = ch.copy()
        ch = pd.Series(
            np.where((ch.to_numpy() == 2) & armed & (np.nan_to_num(df["p3_1r"].to_numpy(), nan=-1) >= T3), 3, ch.to_numpy()),
            index=df.index,
        )
        run_policy(f"fast3_t5={t5:.2f}_plus_arm3", ch)
    for t6 in (0.70, 0.80, 0.85):
        ch = choose_ladder(df["p3_1r"], df["p4_1r"], df["p5_1r"], T3, None, None, fast, df["p6_1r"], t6)
        ch = pd.Series(
            np.where((ch.to_numpy() == 2) & armed & (np.nan_to_num(df["p3_1r"].to_numpy(), nan=-1) >= T3), 3, ch.to_numpy()),
            index=df.index,
        )
        run_policy(f"fast3_t6={t6:.2f}_plus_arm3", ch)

    print("\n===== close-through-1R + high P =====", flush=True)
    thru = pd.to_numeric(df.get("arm_close_thru"), errors="coerce").fillna(0).to_numpy() == 1
    strong = armed & thru & (close_r >= 1.15)
    for t5 in (0.70, 0.80, 0.85):
        ch = np.full(len(df), 2, dtype=int)
        ch = np.where(armed & (np.nan_to_num(df["p3_1r"].to_numpy(), nan=-1) >= T3), 3, ch)
        ch = np.where(strong & (np.nan_to_num(df["p5_1r"].to_numpy(), nan=-1) >= t5), 5, ch)
        run_policy(f"thru_t5={t5:.2f}_plus_arm3", ch)
    for t6 in (0.70, 0.80):
        ch = np.full(len(df), 2, dtype=int)
        ch = np.where(armed & (np.nan_to_num(df["p3_1r"].to_numpy(), nan=-1) >= T3), 3, ch)
        ch = np.where(strong & (np.nan_to_num(df["p6_1r"].to_numpy(), nan=-1) >= t6), 6, ch)
        run_policy(f"thru_t6={t6:.2f}_plus_arm3", ch)

    print("\n===== yearly top-N by P5 / P6 (cap how many runners) =====", flush=True)
    df["year"] = df["Entry_Date"].dt.year
    p5 = np.nan_to_num(df["p5_1r"].to_numpy(), nan=-1)
    p6 = np.nan_to_num(df["p6_1r"].to_numpy(), nan=-1)
    p3 = np.nan_to_num(df["p3_1r"].to_numpy(), nan=-1)
    for n_top, col, k in ((2, p5, 5), (3, p5, 5), (1, p6, 6), (2, p6, 6), (2, p5, 5)):
        ch = np.where(armed & (p3 >= T3), 3, 2)
        pick = np.zeros(len(df), dtype=bool)
        for y, idx in df.groupby("year").groups.items():
            ii = np.array(list(idx))
            ok = ii[armed[ii]]
            if len(ok) == 0:
                continue
            order = ok[np.argsort(-col[ok])]
            take = order[:n_top]
            # only upgrade if P at least 0.60
            take = take[col[take] >= 0.60]
            pick[take] = True
        ch = np.where(pick, k, ch)
        run_policy(f"top{n_top}_per_year_1to{k}_plus_arm3", ch)

    print("\n===== scale-out 50/50 1:2 + 1:5/1:6 on stack =====", flush=True)
    for t5 in (0.80, 0.85, 0.90):
        ch = choose_ladder(df["p3_1r"], df["p4_1r"], df["p5_1r"], T3, None, t5, armed)
        run_policy(f"scale50_stack_t5={t5:.2f}", ch, scale50=True)
    for t6 in (0.80, 0.85):
        ch = choose_ladder(df["p3_1r"], df["p4_1r"], df["p5_1r"], T3, None, None, armed, df["p6_1r"], t6)
        run_policy(f"scale50_stack_t6={t6:.2f}", ch, scale50=True)

    print("\n===== rules (no extra ML) on top of arm3 =====", flush=True)
    for close_min, bar_max, meta_min, k in (
        (1.40, 2, 0.60, 5),
        (1.60, 3, 0.60, 5),
        (1.80, 3, 0.55, 5),
        (1.50, 2, 0.65, 6),
        (1.80, 2, 0.60, 6),
        (2.00, 3, 0.55, 6),
    ):
        ch = np.where(armed & (p3 >= T3), 3, 2)
        m = armed & (close_r >= close_min) & (bars <= bar_max) & (meta >= meta_min)
        ch = np.where(m, k, ch)
        run_policy(f"rule_c{close_min:.2f}_b{bar_max}_p{meta_min:.2f}_1to{k}_plus_arm3", ch)

    keep_hi = [
        r
        for r in rows
        if r.get("keep") and (r.get("n_1to5_names", 0) + r.get("n_1to6_names", 0)) > 0
    ]
    better_hi = [
        r
        for r in rows
        if r.get("better") and (r.get("n_1to5_names", 0) + r.get("n_1to6_names", 0)) > 0
    ]
    vs3 = [r for r in rows if r["CAGR"] >= 59.94 - 0.05 and r["DD"] <= 24.8 and (r.get("n_1to5_names", 0) + r.get("n_1to6_names", 0)) > 0]
    print("\n===== 1:5/1:6 that do not worsen vs 1:2 =====", flush=True)
    if not keep_hi:
        print("  none", flush=True)
    for r in sorted(keep_hi, key=lambda x: -x["CAGR"])[:20]:
        print(
            f"  {r['tag']:48s} {r['CAGR']:+6.2f}% DD {r['DD']:5.1f} "
            f"5={r['n_1to5_names']} 6={r['n_1to6_names']} {r.get('rr_used')}",
            flush=True,
        )
    print("also >= M47 1:3 (+59.94% / 24.5%):", flush=True)
    if not vs3:
        print("  none", flush=True)
    for r in vs3:
        print(f"  {r['tag']:48s} {r['CAGR']:+6.2f}% DD {r['DD']:5.1f} 5={r['n_1to5_names']} 6={r['n_1to6_names']}", flush=True)

    HIGH_LOG.write_text(
        json.dumps(
            {
                "baseline": {"CAGR": BASELINE_CAGR, "DD": BASELINE_DD},
                "m47_arm3": {"CAGR": 59.94, "DD": 24.5, "t3": T3},
                "oos": acc,
                "keep_with_5or6": keep_hi,
                "better_with_5or6": better_hi,
                "match_or_beat_m47": vs3,
                "rows": rows,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"wrote {HIGH_LOG}", flush=True)


if __name__ == "__main__":
    main()
