"""
Year x month stats and return heatmaps for the three compounding paper-gate books.

Same engine as run_paper_gate.py: Rs 50k start, 2% of current equity, vol 0.04,
Zerodha tax. Paper result applied only after Exit_Date.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.analysis.indian_brokerage_calculator import calculate_indian_trade_charges
from swing_strategy.run_ml_next_search import select_meta
from swing_strategy.run_ml_top5_selector import _metrics
from swing_strategy.run_paper_gate import _exits_by_day
from swing_strategy.visualizer import generate_all_visualizations

SCORED = BASE_DIR / "Reports" / "SwingLowCagrHunt" / "Scored_hgb.parquet"
OUT = BASE_DIR / "Reports" / "SwingLow_HGB_2pctEquity" / "paper_gate_heatmaps"
PLOTS = BASE_DIR / "Plots" / "SwingLow_HGB_2pctEquity" / "paper_gates"
CAPITAL = 50_000.0
PCT = 0.02
TV = 0.04
START = pd.Timestamp("2010-01-01")
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
CFGS = [
    {
        "tag": "keep47",
        "label": "Keep ~47% / cut DD",
        "rule": "All setups, last 20 paper exits, pause while SL ratio >= 75%",
        "shadow": "all",
        "sl_n": 20,
        "sl_max": 0.75,
    },
    {
        "tag": "better_both",
        "label": "Better both",
        "rule": "ML list, last 20 paper exits, pause while WR < 22%",
        "shadow": "ml",
        "roll_n": 20,
        "roll_min": 0.22,
    },
    {
        "tag": "peak_cagr",
        "label": "Peak CAGR",
        "rule": "ML list, last 30 paper exits, pause while SL >= 75%",
        "shadow": "ml",
        "sl_n": 30,
        "sl_max": 0.75,
    },
]


def simulate(ml: pd.DataFrame, paper_exits: dict, cfg: dict) -> dict:
    k = int(cfg.get("k", 99))
    pause_days = int(cfg.get("pause_days", 0))
    roll_n = int(cfg.get("roll_n", 0))
    roll_min = float(cfg.get("roll_min", 0.0))
    sl_n = int(cfg.get("sl_n", 0))
    sl_max = float(cfg.get("sl_max", 1.0))

    by_entry = defaultdict(list)
    for rec in ml.to_dict("records"):
        by_entry[pd.Timestamp(rec["Entry_Date"]).normalize()].append(rec)
    for day, cands in by_entry.items():
        cands.sort(key=lambda x: float(x.get("Meta_P", 0.0)), reverse=True)

    max_dt = max(ml["Entry_Date"].max(), pd.to_datetime(ml["Exit_Date"]).max())
    cash = CAPITAL
    peak = CAPITAL
    max_dd = 0.0
    dd_peak_date = START
    dd_trough_date = START
    peak_date = START
    open_pos: dict[int, dict] = {}
    executed = 0
    wins = 0
    trade_id = 1
    paper_consec = 0
    pause_until = pd.Timestamp("1900-01-01")
    recent: deque[int] = deque(maxlen=max(roll_n, sl_n, 1))
    equity = [{"Date": START, "Balance": CAPITAL, "Cash": CAPITAL, "Holdings": 0.0}]
    exits: list[dict] = []
    event_days = sorted(
        set(by_entry.keys())
        | set(paper_exits.keys())
        | {pd.Timestamp(r["Exit_Date"]).normalize() for recs in by_entry.values() for r in recs}
    )

    for day in event_days:
        paused = day < pause_until
        cold = False
        if roll_n and len(recent) >= roll_n and (sum(recent) / len(recent)) < roll_min:
            cold = True
        sl_ratio_block = False
        if sl_n and len(recent) >= sl_n:
            sl_ratio_block = (1.0 - sum(recent) / len(recent)) >= sl_max

        port = cash + sum(p["spend"] for p in open_pos.values())
        risk = max(port * PCT, 100.0)
        avail = cash
        cands = by_entry.get(day, [])
        n = max(len(cands), 1)
        vols = [float(c.get("idio_vol", np.nan)) for c in cands]
        vols = [v for v in vols if np.isfinite(v) and v > 0]
        med = float(np.median(vols)) if vols else TV
        scale = float(np.clip(TV / max(med, 1e-6), 0.40, 1.80))
        block = paused or cold or sl_ratio_block
        for i, cand in enumerate(cands):
            if block:
                continue
            entry_p = float(cand["Entry_Price"])
            sl_p = float(cand["SL_Price"])
            rsk = entry_p - sl_p
            if rsk <= 0.05 or rsk > risk * 1.8:
                continue
            strength = (1.4 - 0.8 * (i / n)) * scale
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
            net = round(gross - tax, 2)
            cash = round(cash + spend + net, 2)
            if net > 0:
                wins += 1
            exits.append({"Date": day, "Net": net, "Win": int(net > 0)})
            del open_pos[tid]

        for won in paper_exits.get(day, []):
            recent.append(1 if won else 0)
            if won:
                paper_consec = 0
            else:
                paper_consec += 1
                if paper_consec >= k and pause_days > 0:
                    pause_until = max(pause_until, day + pd.Timedelta(days=pause_days))
                    paper_consec = 0

        holding = sum(p["spend"] for p in open_pos.values())
        port = round(cash + holding, 2)
        equity.append({"Date": day, "Balance": port, "Cash": cash, "Holdings": holding})
        if port >= peak:
            peak = port
            peak_date = day
        dd = ((peak - port) / peak * 100) if peak else 0.0
        if dd > max_dd:
            max_dd = dd
            dd_peak_date = peak_date
            dd_trough_date = day

    years = max((max_dt - START).days / 365.25, 0.01)
    _, cagr = _metrics(CAPITAL, cash, years)
    eq = pd.DataFrame(equity)
    eq["Date"] = pd.to_datetime(eq["Date"])
    eq = eq.drop_duplicates("Date", keep="last").sort_values("Date")
    full_idx = pd.date_range(START, max_dt.normalize(), freq="D")
    daily = eq.set_index("Date").reindex(full_idx).ffill().reset_index().rename(columns={"index": "Date"})
    daily["Balance"] = daily["Balance"].astype(float)
    return {
        "CAGR": round(float(cagr), 2),
        "N": executed,
        "wins": wins,
        "WR": round(100.0 * wins / executed, 2) if executed else 0.0,
        "DD": round(max_dd, 2),
        "DD_from": dd_peak_date.strftime("%Y-%m-%d"),
        "DD_to": dd_trough_date.strftime("%Y-%m-%d"),
        "Final": round(float(daily["Balance"].iloc[-1]), 2),
        "daily": daily,
        "exits": pd.DataFrame(exits) if exits else pd.DataFrame(columns=["Date", "Net", "Win"]),
        "end": max_dt,
    }


def monthly_from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    last = df.groupby(["Year", "Month"])["Balance"].last().reset_index()
    last["Prev"] = last["Balance"].shift(1)
    last.loc[0, "Prev"] = float(df["Balance"].iloc[0])
    last["Return_Pct"] = (last["Balance"] - last["Prev"]) / last["Prev"] * 100.0
    last["MonthName"] = last["Month"].map(lambda m: MONTHS[m - 1])
    return last


def yearly_from_monthly(monthly: pd.DataFrame) -> pd.DataFrame:
    y = monthly.groupby("Year").agg(
        End_Equity=("Balance", "last"),
        Months=("Return_Pct", "count"),
        Green=("Return_Pct", lambda s: int((s > 0).sum())),
        Worst_Month=("Return_Pct", "min"),
        Best_Month=("Return_Pct", "max"),
    ).reset_index()
    y["Prev"] = y["End_Equity"].shift(1)
    y.loc[0, "Prev"] = CAPITAL
    y["Return_Pct"] = (y["End_Equity"] - y["Prev"]) / y["Prev"] * 100.0
    return y


def _color_bound(pivot: pd.DataFrame, floor: float = 15.0, cap: float = 40.0) -> float:
    flat = pivot.to_numpy(dtype=float)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return floor
    p95 = float(np.nanpercentile(np.abs(flat), 90))
    return float(np.clip(max(p95, floor), floor, cap))


def save_heatmap(pivot: pd.DataFrame, title: str, path: Path, cbar: str, fmt: str = "+.1f") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(13, max(6.2, len(pivot) * 0.48)))
    sns.set_theme(style="white")
    bound = _color_bound(pivot) if fmt.startswith("+") else None
    ax = sns.heatmap(
        pivot,
        annot=True,
        fmt=fmt,
        cmap=sns.diverging_palette(10, 130, as_cmap=True) if fmt.startswith("+") else "Blues",
        center=0 if fmt.startswith("+") else None,
        vmin=-bound if bound else None,
        vmax=bound if bound else None,
        cbar_kws={"label": cbar},
        linewidths=0.6,
        linecolor="white",
        annot_kws={"size": 8, "weight": "bold"},
        mask=pivot.isna(),
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    print(f"Saved {path}", flush=True)


def save_combined(pivots: list[tuple[str, pd.DataFrame]], path: Path) -> None:
    n = len(pivots)
    fig, axes = plt.subplots(n, 1, figsize=(13, 6.2 * n), sharex=True)
    sns.set_theme(style="white")
    bound = max(_color_bound(p) for _, p in pivots)
    cmap = sns.diverging_palette(10, 130, as_cmap=True)
    for ax, (title, pivot) in zip(axes, pivots):
        sns.heatmap(
            pivot,
            annot=True,
            fmt="+.1f",
            cmap=cmap,
            center=0,
            vmin=-bound,
            vmax=bound,
            cbar_kws={"label": "Monthly return (%)"},
            linewidths=0.5,
            linecolor="white",
            annot_kws={"size": 7, "weight": "bold"},
            mask=pivot.isna(),
            ax=ax,
        )
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel("Year")
        ax.set_xlabel("")
    axes[-1].set_xlabel("Month")
    fig.suptitle(
        "Paper-gate books  |  Rs 50k start, 2% of equity, net Zerodha, 2010-01 to 2026-09",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close()
    print(f"Saved {path}", flush=True)


def save_yearly_bars(rows: list[dict], path: Path) -> None:
    years = sorted({r["Year"] for r in rows})
    tags = ["keep47", "better_both", "peak_cagr"]
    labels = {c["tag"]: c["label"] for c in CFGS}
    x = np.arange(len(years))
    width = 0.26
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ["#2563EB", "#059669", "#D97706"]
    for i, tag in enumerate(tags):
        vals = []
        for y in years:
            hit = next((r for r in rows if r["tag"] == tag and r["Year"] == y), None)
            vals.append(hit["Return_Pct"] if hit else 0.0)
        ax.bar(x + (i - 1) * width, vals, width, label=labels[tag], color=colors[i], edgecolor="none")
    ax.axhline(0, color="#64748B", linewidth=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years], rotation=0)
    ax.set_ylabel("Calendar-year return (%)")
    ax.set_title("Paper-gate books  |  calendar-year net return  |  compounding 2% equity")
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close()
    print(f"Saved {path}", flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(SCORED)
    raw["Entry_Date"] = pd.to_datetime(raw["Entry_Date"])
    raw["Exit_Date"] = pd.to_datetime(raw["Exit_Date"])
    ml = select_meta(raw, 32, 0.42, "Meta_P")
    paper_all = _exits_by_day(raw)
    paper_ml = _exits_by_day(ml)

    month_rows = []
    year_rows = []
    summaries = []
    pivots = []
    canvas = []

    for cfg in CFGS:
        paper = paper_ml if cfg["shadow"] == "ml" else paper_all
        sim = simulate(ml, paper, cfg)
        print(
            f"{cfg['tag']}: CAGR {sim['CAGR']:+.2f} DD {sim['DD']:.1f} n={sim['N']} "
            f"{sim['DD_from']} -> {sim['DD_to']} final={sim['Final']:,.0f}",
            flush=True,
        )
        monthly = monthly_from_daily(sim["daily"])
        exits = sim["exits"].copy()
        if not exits.empty:
            exits["Date"] = pd.to_datetime(exits["Date"])
            exits["Year"] = exits["Date"].dt.year
            exits["Month"] = exits["Date"].dt.month
            g = exits.groupby(["Year", "Month"]).agg(
                Trades=("Net", "count"),
                Wins=("Win", "sum"),
                Net_PnL=("Net", "sum"),
            ).reset_index()
            monthly = monthly.merge(g, on=["Year", "Month"], how="left")
        else:
            monthly["Trades"] = 0
            monthly["Wins"] = 0
            monthly["Net_PnL"] = 0.0
        monthly["Trades"] = monthly["Trades"].fillna(0).astype(int)
        monthly["Wins"] = monthly["Wins"].fillna(0).astype(int)
        monthly["Net_PnL"] = monthly["Net_PnL"].fillna(0.0)
        monthly["tag"] = cfg["tag"]
        monthly["label"] = cfg["label"]
        month_rows.append(monthly)

        yearly = yearly_from_monthly(monthly)
        t_y = monthly.groupby("Year").agg(Trades=("Trades", "sum"), Net_PnL=("Net_PnL", "sum")).reset_index()
        yearly = yearly.merge(t_y, on="Year", how="left")
        yearly["tag"] = cfg["tag"]
        yearly["label"] = cfg["label"]
        year_rows.append(yearly)

        ret_p = monthly.pivot(index="Year", columns="MonthName", values="Return_Pct").reindex(columns=MONTHS)
        tr_p = monthly.pivot(index="Year", columns="MonthName", values="Trades").reindex(columns=MONTHS)
        save_heatmap(
            ret_p,
            f"{cfg['label']}  |  {cfg['rule']}\n"
            f"Net CAGR {sim['CAGR']:+.2f}%  DD {sim['DD']:.1f}%  n={sim['N']:,}  "
            f"Rs 50k start, 2% equity, Zerodha",
            PLOTS / f"Heatmap_Monthly_{cfg['tag']}.png",
            "Monthly return (%)",
        )
        save_heatmap(
            tr_p.astype(float),
            f"{cfg['label']}  |  closed trades per month",
            PLOTS / f"Heatmap_Trades_{cfg['tag']}.png",
            "Closed trades",
            fmt=".0f",
        )
        pivots.append((f"{cfg['label']}  |  {cfg['rule']}", ret_p))

        daily_viz = sim["daily"].copy()
        daily_viz["Cash_Balance"] = daily_viz["Cash"]
        daily_viz["Holding_Equity_Value"] = daily_viz["Holdings"]
        daily_viz["Daily_PnL"] = daily_viz["Balance"].diff().fillna(0.0)
        daily_viz["Daily_Return_Pct"] = daily_viz["Balance"].pct_change().fillna(0.0) * 100.0
        daily_viz["Active_Positions"] = 0
        generate_all_visualizations(
            daily_viz,
            output_plots_dir=PLOTS / cfg["tag"],
            output_reports_dir=OUT / cfg["tag"],
            exp_title=f"{cfg['label']} paper gate",
        )

        summaries.append({
            "tag": cfg["tag"],
            "label": cfg["label"],
            "rule": cfg["rule"],
            "CAGR": sim["CAGR"],
            "DD": sim["DD"],
            "N": sim["N"],
            "WR": sim["WR"],
            "Final": sim["Final"],
            "DD_from": sim["DD_from"],
            "DD_to": sim["DD_to"],
            "green_months": int((monthly["Return_Pct"] > 0).sum()),
            "red_months": int((monthly["Return_Pct"] < 0).sum()),
            "flat_months": int((monthly["Return_Pct"] == 0).sum()),
            "best_month": round(float(monthly["Return_Pct"].max()), 2),
            "worst_month": round(float(monthly["Return_Pct"].min()), 2),
            "best_year": round(float(yearly["Return_Pct"].max()), 2),
            "worst_year": round(float(yearly["Return_Pct"].min()), 2),
        })
        canvas.append({
            "tag": cfg["tag"],
            "label": cfg["label"],
            "rule": cfg["rule"],
            "CAGR": sim["CAGR"],
            "DD": sim["DD"],
            "N": sim["N"],
            "WR": sim["WR"],
            "Final": sim["Final"],
            "DD_from": sim["DD_from"],
            "DD_to": sim["DD_to"],
            "monthly": [
                {
                    "year": int(r.Year),
                    **{MONTHS[int(r.Month) - 1]: round(float(r.Return_Pct), 1)},
                    "year_ret": None,
                    "trades": int(r.Trades),
                }
                for r in monthly.itertuples()
            ],
            "yearly": [
                {
                    "year": int(r.Year),
                    "ret": round(float(r.Return_Pct), 1),
                    "trades": int(r.Trades),
                    "end": round(float(r.End_Equity), 0),
                    "green": int(r.Green),
                }
                for r in yearly.itertuples()
            ],
        })

    months_df = pd.concat(month_rows, ignore_index=True)
    years_df = pd.concat(year_rows, ignore_index=True)
    # attach year return onto canvas monthly rows
    for block in canvas:
        ymap = {int(r["year"]): r["ret"] for r in block["yearly"]}
        by_year: dict[int, dict] = {}
        for rec in block["monthly"]:
            y = rec["year"]
            by_year.setdefault(y, {"year": y, "year_ret": ymap.get(y), "trades": 0})
            for m in MONTHS:
                if m in rec:
                    by_year[y][m] = rec[m]
            by_year[y]["trades"] = by_year[y].get("trades", 0) + rec["trades"]
        block["monthly"] = [by_year[y] for y in sorted(by_year)]

    save_combined(pivots, PLOTS / "Heatmap_Monthly_All3.png")
    save_yearly_bars(years_df.to_dict("records"), PLOTS / "Yearly_Returns_All3.png")

    xlsx = OUT / "Paper_Gate_YearMonth.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        pd.DataFrame(summaries).to_excel(xw, index=False, sheet_name="Summary")
        years_df.to_excel(xw, index=False, sheet_name="Yearly")
        months_df.to_excel(xw, index=False, sheet_name="Monthly_Long")
        for cfg in CFGS:
            sub = months_df[months_df["tag"] == cfg["tag"]]
            sub.pivot(index="Year", columns="MonthName", values="Return_Pct").reindex(columns=MONTHS).to_excel(
                xw, sheet_name=f"Ret_{cfg['tag']}"[:31]
            )
            sub.pivot(index="Year", columns="MonthName", values="Trades").reindex(columns=MONTHS).to_excel(
                xw, sheet_name=f"Trades_{cfg['tag']}"[:31]
            )
    months_df.to_csv(OUT / "monthly_long.csv", index=False)
    years_df.to_csv(OUT / "yearly.csv", index=False)
    (OUT / "canvas_data.json").write_text(json.dumps(canvas, indent=2), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"wrote {xlsx}", flush=True)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
