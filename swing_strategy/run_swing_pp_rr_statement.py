"""
21-column account statement + plots for Swing_PP RR stack (M48).

After +1R: P6>=0.85 -> 1:6, else P5>=0.85 -> 1:5, else P3>=0.80 -> 1:3, else 1:2.
Paper-gate still uses original 1:2 paper outcomes. Rs 50,000 start, 2% equity.
Holding_Equity_Value is MTM (qty x daily close) per Guide/Account_Statement_guide.md.
Not a frozen Rs 500 book. Does not write Swing_low / Swing_Live.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.analysis.indian_brokerage_calculator import calculate_indian_trade_charges
from swing_strategy.run_2pct_50_25_hunt import PCT, TV
from swing_strategy.run_ml_top5_selector import _metrics
from swing_strategy.run_pred_paper_gate import exits_by_day, load_pred_list
from swing_strategy.run_swing_pp_rr_classifier import (
    PATH1,
    _feat_cols,
    apply_rr,
    build_cache,
    choose_ladder,
    walk_proba,
)
from swing_strategy.run_swing_pp_statement import CSV_COLS, _hyperlink, _plot_one, _write_excel
from swing_strategy.statement_mtm import CloseCache, expand_daily_mtm, max_drawdown_pct
from swing_strategy.visualizer import generate_all_visualizations

EXP = "SwingPP_RR"
REPORTS = BASE_DIR / "Reports" / EXP
PLOTS = BASE_DIR / "Plots" / EXP
CHARTS = PLOTS / "trade_charts"
CAPITAL = 50_000.0
ROLL_N = 50
FAIL_MAX = 0.74
SCALE_HI = 1.3
RANK_HI = 1.2
RANK_LO = 0.6
SCALE_LO = 0.40
T3, T5, T6 = 0.80, 0.85, 0.85
FEAT_N = 36_542


def _chart_rel(trade_id: int, ticker: str, entry_dt: pd.Timestamp, net_pnl: float) -> str:
    safe = ticker.replace(".NS", "").replace(".BO", "")
    label = "PROFIT" if net_pnl >= 0 else "LOSS"
    name = f"Trade_{trade_id:04d}_{safe}_{entry_dt.strftime('%Y-%m-%d')}_{label}.png"
    return f"../../Plots/{EXP}/trade_charts/{name}"


def _scored_fill() -> pd.DataFrame:
    paper = load_pred_list()
    paper["Entry_Date"] = pd.to_datetime(paper["Entry_Date"])
    df = build_cache(paper)
    df["Entry_Date"] = pd.to_datetime(df["Entry_Date"])
    for c in ("xdt_2", "xdt_3", "xdt_4", "xdt_5", "xdt_6", "label_date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    df["armed"] = pd.to_numeric(df.get("armed", 0), errors="coerce").fillna(0).astype(int)
    for k in (2, 3, 4, 5, 6):
        df[f"win_{k}"] = pd.to_numeric(df.get(f"win_{k}", 0), errors="coerce").fillna(0).astype(int)
    arm_cols = _feat_cols(df, PATH1)
    armed_mask = df["armed"] == 1
    print("[wf] scoring P(1:3/5/6 | +1R) ...", flush=True)
    df["p3_1r"] = walk_proba(df, "win_3", arm_cols, armed_mask)
    df["p5_1r"] = walk_proba(df, "win_5", arm_cols, armed_mask)
    df["p6_1r"] = walk_proba(df, "win_6", arm_cols, armed_mask)
    df["p4_1r"] = 0.0
    armed = df["armed"].to_numpy() == 1
    chosen = choose_ladder(
        df["p3_1r"], df["p4_1r"], df["p5_1r"], T3, None, T5, armed, df["p6_1r"], T6,
    )
    fill = apply_rr(df, chosen)
    fill["chosen_rr"] = chosen.to_numpy()
    print(
        "RR names",
        fill["Target_RR_Mode"].value_counts().to_dict(),
        flush=True,
    )
    return paper, fill


def seed_recent(paper: dict, start_dt: pd.Timestamp) -> deque:
    recent: deque[int] = deque(maxlen=ROLL_N)
    for day in sorted(d for d in paper if d < start_dt):
        for won in paper[day]:
            recent.append(1 if won else 0)
    return recent


def simulate_rr_book(
    by_entry: dict,
    paper: dict,
    event_days: list,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    closes: CloseCache,
    chart_rel_fn=None,
) -> dict:
    """Walk the RR book. Sizing uses cost-basis equity; statement cols 17-18 are MTM."""
    if chart_rel_fn is None:
        chart_rel_fn = _chart_rel
    cash = CAPITAL
    tx_id = 2
    trade_id = 1
    open_pos: dict[int, dict] = {}
    executed = 0
    skipped_days = 0
    total_gross = total_tax_z = total_tax_f = 0.0
    rr_fills = defaultdict(int)
    plot_jobs: list[dict] = []
    buy_row_idx: dict[int, int] = {}
    recent = seed_recent(paper, start_dt)
    rows: list[dict] = [{
        "Transaction_ID": 1, "Trade_ID": 0, "Type": "DEPOSIT",
        "Date": start_dt.strftime("%Y-%m-%d"),
        "Ticker": "N/A", "Liquidity_Source": "N/A", "Support_Price": 0.0, "Quantity": 0,
        "Price": 0.0, "Total_Spend": 0.0, "Gross_PnL": 0.0, "Statutory_Taxes": 0.0,
        "Net_PnL": 0.0, "Return_Pct": 0.0, "Cash_Balance": cash,
        "Active_Position_Count": 0, "Holding_Equity_Value": 0.0,
        "Total_Portfolio_Value": cash, "Target_RR_Mode": "1:2",
        "Outcome": "DEPOSIT", "Chart_PNG_URI": "N/A",
    }]
    snaps: list[dict] = [{
        "Date": start_dt.normalize(),
        "Cash_Balance": cash,
        "Open": [],
        "Active_Positions": 0,
    }]

    def _snap(day) -> None:
        snaps.append({
            "Date": pd.Timestamp(day).normalize(),
            "Cash_Balance": cash,
            "Open": [(p["Ticker"], p["Quantity"], p["Entry_Price"]) for p in open_pos.values()],
            "Active_Positions": len(open_pos),
        })

    for di, day in enumerate(event_days, 1):
        if day < start_dt:
            continue
        if day > end_dt:
            break
        fail_block = False
        if len(recent) >= ROLL_N:
            fail_block = (1.0 - (sum(recent) / len(recent))) >= FAIL_MAX
        if fail_block and day in by_entry:
            skipped_days += 1

        cost_eq = cash + sum(p["Total_Spend"] for p in open_pos.values())
        risk = max(cost_eq * PCT, 100.0)
        avail = cash
        cands = by_entry.get(day, [])
        n = max(len(cands), 1)
        vols = [float(c.get("idio_vol", np.nan)) for c in cands]
        vols = [v for v in vols if np.isfinite(v) and v > 0]
        med = float(np.median(vols)) if vols else TV
        scale = float(np.clip(TV / max(med, 1e-6), SCALE_LO, SCALE_HI))
        for i, cand in enumerate(cands):
            if fail_block:
                continue
            entry_p = float(cand["Entry_Price"])
            sl_p = float(cand["SL_Price"])
            rsk = entry_p - sl_p
            if rsk <= 0.05 or rsk > risk * 1.8:
                continue
            strength = (RANK_HI - (RANK_HI - RANK_LO) * (i / n)) * scale
            qty = min(int((risk * strength) // rsk), int(avail // entry_p))
            spend = round(entry_p * qty, 2)
            while qty >= 1 and spend > avail:
                qty -= 1
                spend = round(entry_p * qty, 2)
            if qty < 1 or spend > cash + 1e-9:
                continue
            cash = max(0.0, round(cash - spend, 2))
            avail = round(avail - spend, 2)
            executed += 1
            mode = str(cand.get("Target_RR_Mode", "1:2"))
            rr_fills[mode] += 1
            k = int(float(cand.get("chosen_rr", 2)))
            tp = round(entry_p + k * rsk, 2)
            open_pos[trade_id] = {
                "Ticker": cand["Ticker"],
                "Liquidity_Source": cand.get("Liquidity_Type", "Weekly"),
                "Support_Price": float(cand.get("Support_Price", 0.0) or 0.0),
                "Liquidity_Date": cand.get("Liquidity_Date"),
                "Entry_Price": entry_p,
                "SL_Price": sl_p,
                "Target_Price": tp,
                "Target_RR_Mode": mode,
                "chosen_rr": k,
                "Exit_Date": pd.Timestamp(cand["Exit_Date"]).normalize(),
                "Exit_Price": float(cand["Exit_Price"]),
                "Quantity": qty,
                "Total_Spend": spend,
                "Entry_Date": day,
            }
            holding = closes.holding_mtm(open_pos, day)
            buy_row_idx[trade_id] = len(rows)
            rows.append({
                "Transaction_ID": tx_id, "Trade_ID": trade_id, "Type": "BUY (ENTRY)",
                "Date": day.strftime("%Y-%m-%d"), "Ticker": cand["Ticker"],
                "Liquidity_Source": cand.get("Liquidity_Type", "Weekly"),
                "Support_Price": float(cand.get("Support_Price", 0.0) or 0.0),
                "Quantity": qty, "Price": entry_p, "Total_Spend": spend,
                "Gross_PnL": 0.0, "Statutory_Taxes": 0.0, "Net_PnL": 0.0, "Return_Pct": 0.0,
                "Cash_Balance": cash, "Active_Position_Count": len(open_pos),
                "Holding_Equity_Value": holding, "Total_Portfolio_Value": round(cash + holding, 2),
                "Target_RR_Mode": mode, "Outcome": "OPEN", "Chart_PNG_URI": "N/A",
            })
            tx_id += 1
            trade_id += 1

        for tid, pos in list(open_pos.items()):
            if pos["Exit_Date"] > day:
                continue
            qty, entry_p, exit_p, spend = pos["Quantity"], pos["Entry_Price"], pos["Exit_Price"], pos["Total_Spend"]
            gross = round((exit_p - entry_p) * qty, 2)
            ch_z = calculate_indian_trade_charges(entry_p, exit_p, qty, 0.0)
            ch_f = calculate_indian_trade_charges(entry_p, exit_p, qty, 20.0)
            tax_z = round(ch_z["total_charges"], 2)
            tax_f = round(ch_f["total_charges"], 2)
            net = round(gross - tax_z, 2)
            total_gross += gross
            total_tax_z += tax_z
            total_tax_f += tax_f
            cash = round(cash + spend + gross - tax_z, 2)
            del open_pos[tid]
            holding = closes.holding_mtm(open_pos, day)
            rel = chart_rel_fn(tid, pos["Ticker"], pos["Entry_Date"], net)
            link = _hyperlink(rel) if rel and rel != "N/A" else "N/A"
            if tid in buy_row_idx:
                rows[buy_row_idx[tid]]["Chart_PNG_URI"] = link
            sell_outcome = "Success" if net >= 0 else "Failure"
            rows.append({
                "Transaction_ID": tx_id, "Trade_ID": tid, "Type": "SELL (EXIT)",
                "Date": pos["Exit_Date"].strftime("%Y-%m-%d"), "Ticker": pos["Ticker"],
                "Liquidity_Source": pos["Liquidity_Source"], "Support_Price": pos["Support_Price"],
                "Quantity": qty, "Price": exit_p, "Total_Spend": spend,
                "Gross_PnL": gross, "Statutory_Taxes": tax_z, "Net_PnL": net,
                "Return_Pct": round((net / spend) * 100, 2) if spend else 0.0,
                "Cash_Balance": cash, "Active_Position_Count": len(open_pos),
                "Holding_Equity_Value": holding, "Total_Portfolio_Value": round(cash + holding, 2),
                "Target_RR_Mode": pos["Target_RR_Mode"], "Outcome": sell_outcome, "Chart_PNG_URI": link,
            })
            plot_jobs.append({
                "Trade_ID": tid, "Ticker": pos["Ticker"],
                "C2_Date": pos["Entry_Date"], "Exit_Date": pos["Exit_Date"],
                "Support_Price": pos["Support_Price"], "Entry_Price": entry_p,
                "SL_Price": pos["SL_Price"],
                "Target_Price": pos["Target_Price"],
                "Exit_Price": exit_p, "Net_PnL": net,
                "Liquidity_Type": pos["Liquidity_Source"],
                "Liquidity_Date": pos.get("Liquidity_Date"),
                "ML_RR_Choice": pos["Target_RR_Mode"],
            })
            tx_id += 1

        for won in paper.get(day, []):
            recent.append(1 if won else 0)

        _snap(day)
        if di % 500 == 0:
            print(
                f"  event day {di}/{len(event_days)} cash={cash:,.0f} open={len(open_pos)} skipd={skipped_days}",
                flush=True,
            )

    df_daily = expand_daily_mtm(snaps, start_dt, end_dt, closes)
    remaining_cost = sum(p["Total_Spend"] for p in open_pos.values())
    remaining_mtm = closes.holding_mtm(open_pos, end_dt)
    net_z = round(float(df_daily["Balance"].iloc[-1]), 2)
    net_f = round(net_z - (total_tax_f - total_tax_z), 2)
    gross_eq = round(CAPITAL + total_gross + (remaining_mtm - remaining_cost), 2)
    max_dd = max_drawdown_pct(df_daily["Balance"])
    sells = [r for r in rows if r["Type"] == "SELL (EXIT)"]
    wins = sum(1 for r in sells if r["Net_PnL"] >= 0)
    win_rate = (wins / len(sells) * 100) if sells else 0.0
    years = max((end_dt - start_dt).days / 365.25, 0.01)
    g_ret, g_cagr = _metrics(CAPITAL, gross_eq, years)
    z_ret, z_cagr = _metrics(CAPITAL, net_z, years)
    f_ret, f_cagr = _metrics(CAPITAL, net_f, years)
    out_daily = df_daily.copy()
    out_daily["Date"] = pd.to_datetime(out_daily["Date"]).dt.strftime("%Y-%m-%d")
    return {
        "rows": rows,
        "df_daily": out_daily,
        "plot_jobs": plot_jobs,
        "executed": executed,
        "skipped_days": skipped_days,
        "rr_fills": dict(rr_fills),
        "total_gross": total_gross,
        "total_tax_z": total_tax_z,
        "total_tax_f": total_tax_f,
        "gross_eq": gross_eq,
        "net_z": net_z,
        "net_f": net_f,
        "g_ret": g_ret, "g_cagr": g_cagr,
        "z_ret": z_ret, "z_cagr": z_cagr,
        "f_ret": f_ret, "f_cagr": f_cagr,
        "win_rate": win_rate,
        "max_dd": max_dd,
        "years": years,
        "open_left": len(open_pos),
    }


def run(max_charts: int) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    CHARTS.mkdir(parents=True, exist_ok=True)

    paper_src, ml = _scored_fill()
    accept_pct = 100.0 * len(ml) / max(FEAT_N, 1)
    print(f"prediction list {len(ml):,}  (~{accept_pct:.1f}% of >=1m swing-low features)", flush=True)

    paper = exits_by_day(paper_src, 1.5)
    by_entry = defaultdict(list)
    for rec in ml.to_dict("records"):
        if pd.isna(rec.get("Exit_Date")):
            continue
        by_entry[pd.Timestamp(rec["Entry_Date"]).normalize()].append(rec)
    for day, cands in by_entry.items():
        cands.sort(key=lambda x: float(x.get("Meta_P", 0.0)), reverse=True)

    min_dt = pd.Timestamp("2010-01-01")
    max_dt = max(ml["Entry_Date"].max(), pd.to_datetime(ml["Exit_Date"]).max()).normalize()
    extra_x = []
    for recs in by_entry.values():
        for r in recs:
            xd = pd.Timestamp(r["Exit_Date"])
            if pd.notna(xd):
                extra_x.append(xd.normalize())
    event_days = sorted(set(by_entry.keys()) | set(paper.keys()) | set(extra_x))

    closes = CloseCache()
    tickers = {str(r.get("Ticker")) for recs in by_entry.values() for r in recs}
    print(f"Loading {len(tickers)} daily close series for MTM ...", flush=True)
    closes.preload(tickers)
    print(f"Simulating {len(event_days):,} event days ...", flush=True)
    book = simulate_rr_book(by_entry, paper, event_days, min_dt, max_dt, closes)

    rows = book["rows"]
    df_daily = book["df_daily"]
    plot_jobs = book["plot_jobs"]
    executed = book["executed"]
    skipped_days = book["skipped_days"]
    rr_fills = book["rr_fills"]
    gross_eq, net_z, net_f = book["gross_eq"], book["net_z"], book["net_f"]
    g_ret, g_cagr = book["g_ret"], book["g_cagr"]
    z_ret, z_cagr = book["z_ret"], book["z_cagr"]
    f_ret, f_cagr = book["f_ret"], book["f_cagr"]
    win_rate, max_dd = book["win_rate"], book["max_dd"]
    total_tax_z, total_tax_f = book["total_tax_z"], book["total_tax_f"]

    df_stmt = pd.DataFrame(rows)[CSV_COLS]
    csv_path = REPORTS / "Swing_Strategy_Account_Statement.csv"
    df_stmt.to_csv(csv_path, index=False)
    df_daily.to_csv(REPORTS / "Daily_Equity.csv", index=False)
    print(f"CSV {len(df_stmt):,} rows -> {csv_path}", flush=True)

    period = f"{min_dt.strftime('%Y-%m-%d')} to {max_dt.strftime('%Y-%m-%d')}"
    rr_txt = ", ".join(f"{k}={v}" for k, v in sorted(rr_fills.items()))
    rule = (
        "Swing_PP RR stack | after +1R: P6>=0.85->1:6 else P5>=0.85->1:5 "
        "else P3>=0.80->1:3 else 1:2 | fair BE | 2% equity | paper-gate last 50 fail%>=74%"
    )
    summary_rows = [
        ("Strategy Rule", rule, rule, rule),
        ("Initial Capital", f"Rs {CAPITAL:,.2f}", f"Rs {CAPITAL:,.2f}", f"Rs {CAPITAL:,.2f}"),
        ("Risk / Trade", "2% of equity", "2% of equity", "2% of equity"),
        ("Final Portfolio Equity", f"Rs {gross_eq:,.2f}", f"Rs {net_z:,.2f}", f"Rs {net_f:,.2f}"),
        ("Total Net Profit (INR)", f"Rs {gross_eq - CAPITAL:,.2f}", f"Rs {net_z - CAPITAL:,.2f}", f"Rs {net_f - CAPITAL:,.2f}"),
        ("Total Return (%)", f"{g_ret:+.2f}%", f"{z_ret:+.2f}%", f"{f_ret:+.2f}%"),
        ("CAGR (%)", f"{g_cagr:+.2f}%", f"{z_cagr:+.2f}%", f"{f_cagr:+.2f}%"),
        ("Executed Trades Count", f"{executed:,}", f"{executed:,}", f"{executed:,}"),
        ("Win Rate (%)", f"{win_rate:.2f}%", f"{win_rate:.2f}%", f"{win_rate:.2f}%"),
        ("Max Drawdown (%)", f"{max_dd:.2f}%", f"{max_dd:.2f}%", f"{max_dd:.2f}%"),
        ("Total Statutory Taxes Paid", "Rs 0.00", f"Rs {total_tax_z:,.2f}", f"Rs {total_tax_f:,.2f}"),
        ("Backtest Period", period, period, period),
        ("Skipped entry days (paper gate)", f"{skipped_days:,}", f"{skipped_days:,}", f"{skipped_days:,}"),
        ("ML accepted / universe", f"{len(ml):,} / {FEAT_N:,}", f"{accept_pct:.1f}%", "Meta_P>=0.48 top16"),
        ("Filled RR mix", rr_txt, rr_txt, rr_txt),
        ("Holding valuation", "MTM qty x close", "MTM qty x close", "MTM qty x close"),
    ]
    xlsx_path = REPORTS / "Swing_Strategy_Account_Statement.xlsx"
    print("Writing Excel ...", flush=True)
    _write_excel(rows, summary_rows, xlsx_path)
    print(f"Excel -> {xlsx_path}", flush=True)

    print("Visualizations ...", flush=True)
    generate_all_visualizations(df_daily, PLOTS, REPORTS, exp_title="Swing_PP RR stack")

    plot_jobs.sort(key=lambda x: x["Trade_ID"])
    if max_charts > 0 and len(plot_jobs) > max_charts:
        half = max_charts // 2
        pick = plot_jobs[:half] + plot_jobs[-(max_charts - half):]
        seen: set[int] = set()
        uniq = []
        for p in pick:
            if p["Trade_ID"] not in seen:
                seen.add(p["Trade_ID"])
                uniq.append(p)
        plot_jobs = uniq
    print(f"Plotting {len(plot_jobs)} trade charts ...", flush=True)
    ok = 0
    payloads = [(p, str(CHARTS)) for p in plot_jobs]
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(_plot_one, pl) for pl in payloads]
        for i, fut in enumerate(as_completed(futs), 1):
            uri = fut.result()
            if uri and uri != "N/A":
                ok += 1
            if i % 100 == 0 or i == len(futs):
                print(f"  plots {i}/{len(futs)}", flush=True)
    print(f"Plots written {ok}", flush=True)
    print(
        f"DONE executed={executed:,} WR={win_rate:.2f}% skipd={skipped_days} "
        f"Gross CAGR {g_cagr:+.2f}%  Zerodha {z_cagr:+.2f}%  FYERS {f_cagr:+.2f}%  "
        f"final Z Rs {net_z:,.0f}  DD {max_dd:.2f}%  fills {dict(rr_fills)}",
        flush=True,
    )


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-charts", type=int, default=0, help="0 = all fills")
    args = ap.parse_args()
    run(args.max_charts)
