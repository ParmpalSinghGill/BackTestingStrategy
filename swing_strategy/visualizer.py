"""
Swing Strategy Visualizer Suite

Generates:
1. Monthly Returns Heatmap (PNG): Year vs Month matrix showing percentage returns.
2. Yearly Returns Breakdown (PNG): Bar chart & table of annual performance.
3. Capital Growth Graph (PNG): Monthly line chart of portfolio balance.
4. Interactive Equity Curve (HTML): Standalone HTML chart with interactive date hover tooltips
   showing Date, Account Balance, Active Positions Count, and Daily Net Return.
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PLOTS_DIR = BASE_DIR / "Plots"
DEFAULT_REPORTS_DIR = BASE_DIR / "Reports"


def generate_monthly_heatmap(df_daily_equity: pd.DataFrame, output_png: Path = None):
    if output_png is None:
        output_png = DEFAULT_PLOTS_DIR / "Monthly_Returns_Heatmap.png"
    
    os.makedirs(output_png.parent, exist_ok=True)

    if df_daily_equity.empty:
        return

    df = df_daily_equity.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    monthly_last = df.groupby(["Year", "Month"])["Balance"].last().reset_index()

    monthly_last["Prev_Balance"] = monthly_last["Balance"].shift(1)
    first_bal = df["Balance"].iloc[0]
    monthly_last.loc[0, "Prev_Balance"] = first_bal

    monthly_last["Return_Pct"] = ((monthly_last["Balance"] - monthly_last["Prev_Balance"]) / monthly_last["Prev_Balance"]) * 100.0

    pivot_df = monthly_last.pivot(index="Year", columns="Month", values="Return_Pct")
    month_names = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun", 7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"}
    pivot_df = pivot_df.rename(columns=month_names)

    plt.figure(figsize=(12, max(6, len(pivot_df) * 0.45)))
    sns.set_theme(style="white")
    
    cmap = sns.diverging_palette(10, 130, as_cmap=True)

    ax = sns.heatmap(
        pivot_df,
        annot=True,
        fmt="+.1f",
        cmap=cmap,
        center=0,
        cbar_kws={"label": "Monthly Return (%)"},
        linewidths=0.8,
        linecolor="white",
        annot_kws={"size": 10, "weight": "bold"}
    )

    plt.title("Swing Strategy - Monthly Returns Heatmap (%)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Month", fontsize=11, fontweight="bold")
    plt.ylabel("Year", fontsize=11, fontweight="bold")
    plt.tight_layout()

    plt.savefig(output_png, dpi=300)
    plt.close()


def generate_yearly_breakdown(df_daily_equity: pd.DataFrame, output_png: Path = None):
    if output_png is None:
        output_png = DEFAULT_PLOTS_DIR / "Yearly_Returns_Breakdown.png"

    os.makedirs(output_png.parent, exist_ok=True)

    if df_daily_equity.empty:
        return

    df = df_daily_equity.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df["Year"] = df["Date"].dt.year
    yearly_df = df.groupby("Year")["Balance"].last().reset_index()

    yearly_df["Prev_Balance"] = yearly_df["Balance"].shift(1)
    yearly_df.loc[0, "Prev_Balance"] = df["Balance"].iloc[0]

    yearly_df["Yearly_Return_Pct"] = ((yearly_df["Balance"] - yearly_df["Prev_Balance"]) / yearly_df["Prev_Balance"]) * 100.0

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#10B981" if r >= 0 else "#EF4444" for r in yearly_df["Yearly_Return_Pct"]]
    bars = ax.bar(yearly_df["Year"].astype(str), yearly_df["Yearly_Return_Pct"], color=colors, edgecolor="none", width=0.6)

    for bar, val in zip(bars, yearly_df["Yearly_Return_Pct"]):
        y_pos = bar.get_height() + (1.5 if val >= 0 else -3.5)
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_pos,
            f"{val:+.1f}%",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=10,
            fontweight="bold",
            color="#0F172A"
        )

    ax.set_title("Swing Strategy - Annual Performance Breakdown (%)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Year", fontsize=11, fontweight="bold")
    ax.set_ylabel("Annual Return (%)", fontsize=11, fontweight="bold")
    ax.axhline(0, color="#64748B", linewidth=1.0, linestyle="--")
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()


def generate_capital_growth_chart(df_daily_equity: pd.DataFrame, output_png: Path = None):
    if output_png is None:
        output_png = DEFAULT_PLOTS_DIR / "Capital_Growth_Monthly.png"

    os.makedirs(output_png.parent, exist_ok=True)

    if df_daily_equity.empty:
        return

    df = df_daily_equity.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df["YearMonth"] = df["Date"].dt.to_period("M")
    monthly_df = df.groupby("YearMonth")["Balance"].last().reset_index()
    monthly_df["Date"] = monthly_df["YearMonth"].dt.to_timestamp()

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(monthly_df["Date"], monthly_df["Balance"], color="#2563EB", linewidth=2.5, label="Account Balance (INR)")
    ax1.fill_between(monthly_df["Date"], monthly_df["Balance"], color="#3B82F6", alpha=0.15)

    ax1.set_title("Swing Strategy - Monthly Capital Growth Progression", fontsize=14, fontweight="bold", pad=15)
    ax1.set_xlabel("Date", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Portfolio Equity Balance (INR)", fontsize=11, fontweight="bold", color="#1E293B")
    ax1.yaxis.set_major_formatter("₹{x:,.0f}")
    ax1.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()


def generate_interactive_equity_html(df_daily_equity: pd.DataFrame, output_html: Path = None, exp_title: str = "Swing Strategy"):
    if output_html is None:
        output_html = DEFAULT_REPORTS_DIR / "Interactive_Equity_Curve.html"

    os.makedirs(output_html.parent, exist_ok=True)

    if df_daily_equity.empty:
        return

    df = df_daily_equity.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    dates_list = df["Date"].tolist()
    balance_list = [round(float(b), 2) for b in df["Balance"]]
    active_pos_list = [int(p) for p in df["Active_Positions"]]
    daily_pnl_list = [round(float(p), 2) for p in df["Daily_PnL"]]
    daily_ret_list = [round(float(r), 2) for r in df["Daily_Return_Pct"]]

    data_payload = {
        "dates": dates_list,
        "balances": balance_list,
        "active_positions": active_pos_list,
        "daily_pnl": daily_pnl_list,
        "daily_return": daily_ret_list
    }

    payload_json = json.dumps(data_payload)

    initial_bal = balance_list[0] if balance_list else 100000.0
    final_bal = balance_list[-1] if balance_list else 100000.0
    net_return_pct = ((final_bal - initial_bal) / initial_bal * 100.0) if initial_bal > 0 else 0.0

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{exp_title} - Interactive Equity Curve & Live Position Tracker</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-color: #0F172A;
            --card-bg: #1E293B;
            --text-main: #F8FAFC;
            --text-sub: #94A3B8;
            --accent-blue: #3B82F6;
            --accent-green: #10B981;
            --accent-red: #EF4444;
            --border-color: #334155;
        }}
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            margin: 0;
            padding: 24px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--border-color);
        }}
        .title h1 {{
            margin: 0 0 4px 0;
            font-size: 24px;
            font-weight: 700;
            color: #FFFFFF;
        }}
        .title p {{
            margin: 0;
            font-size: 14px;
            color: var(--text-sub);
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .stat-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px;
        }}
        .stat-label {{
            font-size: 12px;
            font-weight: 600;
            color: var(--text-sub);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .stat-value {{
            font-size: 22px;
            font-weight: 700;
            margin-top: 6px;
        }}
        .stat-value.green {{ color: var(--accent-green); }}
        .stat-value.blue {{ color: var(--accent-blue); }}
        .chart-container {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
            height: 520px;
            position: relative;
        }}
        .footer {{
            margin-top: 24px;
            text-align: center;
            font-size: 12px;
            color: var(--text-sub);
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="title">
            <h1>{exp_title} - Interactive Equity Curve</h1>
            <p>True Realistic Fills Model | Live Hover Date Tracking</p>
        </div>
    </div>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-label">Initial Deposit</div>
            <div class="stat-value blue">₹{initial_bal:,.2f}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Final Equity Balance</div>
            <div class="stat-value green">₹{final_bal:,.2f}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Net Overall Return</div>
            <div class="stat-value green">+{net_return_pct:,.2f}%</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Total Trading Days</div>
            <div class="stat-value">{len(dates_list):,}</div>
        </div>
    </div>

    <div class="chart-container">
        <canvas id="equityChart"></canvas>
    </div>

    <div class="footer">
        Generated automatically by Swing Strategy Engine. Hover over any date on the graph to inspect exact balance, active positions count, and daily PnL.
    </div>

    <script>
        const rawData = {payload_json};

        const ctx = document.getElementById('equityChart').getContext('2d');
        
        const gradient = ctx.createLinearGradient(0, 0, 0, 400);
        gradient.addColorStop(0, 'rgba(59, 130, 246, 0.35)');
        gradient.addColorStop(1, 'rgba(59, 130, 246, 0.0)');

        const chart = new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: rawData.dates,
                datasets: [
                    {{
                        label: 'Account Equity (INR)',
                        data: rawData.balances,
                        borderColor: '#3B82F6',
                        borderWidth: 2,
                        fill: true,
                        backgroundColor: gradient,
                        tension: 0.1,
                        pointRadius: 0,
                        pointHoverRadius: 6,
                        pointHoverBackgroundColor: '#10B981',
                        yAxisID: 'y'
                    }},
                    {{
                        label: 'Active Open Positions',
                        data: rawData.active_positions,
                        borderColor: '#F59E0B',
                        borderWidth: 1.5,
                        borderDash: [4, 4],
                        fill: false,
                        tension: 0.1,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        yAxisID: 'y1'
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                interaction: {{
                    mode: 'index',
                    intersect: false
                }},
                plugins: {{
                    legend: {{
                        labels: {{
                            color: '#F8FAFC',
                            font: {{ size: 12, weight: 'bold' }}
                        }}
                    }},
                    tooltip: {{
                        backgroundColor: '#0F172A',
                        titleColor: '#F8FAFC',
                        bodyColor: '#94A3B8',
                        borderColor: '#334155',
                        borderWidth: 1,
                        padding: 12,
                        displayColors: true,
                        callbacks: {{
                            title: function(items) {{
                                return 'Date: ' + items[0].label;
                            }},
                            label: function(context) {{
                                const idx = context.dataIndex;
                                const bal = rawData.balances[idx];
                                const active = rawData.active_positions[idx];
                                const pnl = rawData.daily_pnl[idx];
                                const ret = rawData.daily_return[idx];

                                if (context.datasetIndex === 0) {{
                                    return [
                                        ' Account Balance: ₹' + bal.toLocaleString('en-IN', {{minimumFractionDigits: 2}}),
                                        ' Daily Net PnL: ₹' + (pnl >= 0 ? '+' : '') + pnl.toLocaleString('en-IN', {{minimumFractionDigits: 2}}) + ' (' + (ret >= 0 ? '+' : '') + ret.toFixed(2) + '%)'
                                    ];
                                }} else {{
                                    return ' Active Positions Count: ' + active;
                                }}
                            }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        grid: {{ color: '#334155', drawBorder: false }},
                        ticks: {{ color: '#94A3B8', maxTicksLimit: 12 }}
                    }},
                    y: {{
                        type: 'linear',
                        display: true,
                        position: 'left',
                        grid: {{ color: '#334155', drawBorder: false }},
                        ticks: {{
                            color: '#3B82F6',
                            callback: function(value) {{
                                return '₹' + value.toLocaleString('en-IN');
                            }}
                        }}
                    }},
                    y1: {{
                        type: 'linear',
                        display: true,
                        position: 'right',
                        grid: {{ drawOnChartArea: false }},
                        ticks: {{ color: '#F59E0B' }},
                        title: {{
                            display: true,
                            text: 'Active Positions Count',
                            color: '#F59E0B'
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)


def generate_all_visualizations(df_daily_equity: pd.DataFrame, output_plots_dir: Path = None, output_reports_dir: Path = None, exp_title: str = "Swing Strategy"):
    if output_plots_dir is None:
        output_plots_dir = DEFAULT_PLOTS_DIR
    if output_reports_dir is None:
        output_reports_dir = DEFAULT_REPORTS_DIR

    os.makedirs(output_plots_dir, exist_ok=True)
    os.makedirs(output_reports_dir, exist_ok=True)

    generate_monthly_heatmap(df_daily_equity, output_png=output_plots_dir / "Monthly_Returns_Heatmap.png")
    generate_yearly_breakdown(df_daily_equity, output_png=output_plots_dir / "Yearly_Returns_Breakdown.png")
    generate_capital_growth_chart(df_daily_equity, output_png=output_plots_dir / "Capital_Growth_Monthly.png")
    generate_interactive_equity_html(df_daily_equity, output_html=output_reports_dir / "Interactive_Equity_Curve.html", exp_title=exp_title)
