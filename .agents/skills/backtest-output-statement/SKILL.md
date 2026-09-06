---
name: backtest-output-statement
description: >-
  Use when generating, formatting, validating, or updating quantitative backtest account statements,
  Excel/CSV reports, master comparison tables, candlestick trade PNG plots with wick-preserving
  green/red entry/exit marker arrows, decreasing holding equity value tracking on sell exits,
  or computing Before-Tax vs After-Tax Returns and CAGR. Always read Guide/OutputFormatGuide.md
  and enforce its checklist before saving or citing any report.
---

# Backtest Output & Account Statement Skill

**Source of truth:** [Guide/OutputFormatGuide.md](../../Guide/OutputFormatGuide.md)  
**Companion guides:** [Guide/Account_Statement_guide.md](../../Guide/Account_Statement_guide.md), [Guide/Realistic_guide.md](../../Guide/Realistic_guide.md)

When generating or reviewing backtest output, **read `Guide/OutputFormatGuide.md` first**, then apply every rule below. Do not cite CAGR, return %, or trade counts from saved reports until this checklist passes.

---

## 0. Pre-Flight (before running or trusting results)

1. **Simulation engine** must follow all 8 rules in `Guide/Realistic_guide.md` (entries-first cash, gap fills, risk-cap sizing, taxes, walk-forward ML).
2. **Do not reuse stale `Reports/` files** — deleted or outdated reports must be regenerated; never copy archived numbers into new comparisons.
3. **ML must actually filter** — if ML accepts >90% of raw Scenario-1 setups, raise `probability_threshold` or fix labels before reporting CAGR.
4. **Dataset must exist** under `Reports/` (not only `Reports/OLD/`) with required columns: `Entry_Price_1to2`, `SL_Price_1to2`, `Target_Price_1to2`, `Target_Price_1to3`, `Exit_Date_1to2`, `Exit_Date_1to3`, `Outcome_1to2`, `Outcome_1to3`.

---

## 1. Required Output Files (every backtest run)

| Artifact | Path pattern |
|----------|--------------|
| Account Statement Excel | `Reports/<exp>/Swing_Strategy_Account_Statement.xlsx` |
| Account Statement CSV | `Reports/<exp>/Swing_Strategy_Account_Statement.csv` |
| Trade PNG charts | `Plots/<exp>/trade_charts/Trade_<id>_<TICKER>.png` |
| Monthly Returns Heatmap | `Plots/<exp>/Monthly_Returns_Heatmap.png` |
| Annual Performance Breakdown | `Plots/<exp>/Yearly_Returns_Breakdown.png` |
| Monthly Capital Growth | `Plots/<exp>/Capital_Growth_Monthly.png` |
| Interactive equity dashboard | `Reports/<exp>/Interactive_Equity_Curve.html` |
| Master comparison (multi-run) | `Reports/Multi_Experiment_Suite/Master_Experiments_Comparison.xlsx` |

---

## 2. Institutional 21-Column Account Statement Schema

CSV column names (machine-readable). Excel headers may use friendly labels but **same order and meaning**.

| # | CSV column | Type | Rule |
|---|------------|------|------|
| 1 | `Transaction_ID` | int | Sequential 1, 2, 3… |
| 2 | `Trade_ID` | int | Unique per trade; `0` for deposit |
| 3 | `Type` | string | `DEPOSIT`, `BUY (ENTRY)`, `SELL (EXIT)` |
| 4 | `Date` | `YYYY-MM-DD` | Execution date |
| 5 | `Ticker` | string | e.g. `RELIANCE.NS` |
| 6 | `Liquidity_Source` | string | `Yearly`, `Monthly`, `Weekly` |
| 7 | `Support_Price` | INR | Support level price |
| 8 | `Quantity` | int | Shares traded |
| 9 | `Price` | INR | Actual fill (gap-up entry ×1.002 or gap exit ×0.999 when applicable) |
| 10 | `Total_Spend` | INR | `Quantity × Price` |
| 11 | `Gross_PnL` | INR | `0.0` on BUY rows |
| 12 | `Statutory_Taxes` | INR | STT + exchange + SEBI + stamp + GST + brokerage |
| 13 | `Net_PnL` | INR | `Gross_PnL − Statutory_Taxes` |
| 14 | `Return_Pct` | % | `Net_PnL / Total_Spend × 100` |
| 15 | `Cash_Balance` | INR | Free liquid cash (code may use `Balance`; export as `Cash_Balance`) |
| 16 | `Active_Position_Count` | int | `len(open_positions)` on every row |
| 17 | `Holding_Equity_Value` | INR | Sum of open position spends; see §3 |
| 18 | `Total_Portfolio_Value` | INR | `Cash_Balance + Holding_Equity_Value` |
| 19 | `Target_RR_Mode` | string | `1:2`, `1:3`, `1:4`, etc. |
| 20 | `Outcome` | string | `DEPOSIT`, `OPEN`, `Success`, `Failure` |
| 21 | `Chart_PNG_URI` | formula | `=HYPERLINK("Plots/.../Trade_N_TICKER.png", "View Plot Chart (PNG)")` |

**Excel column 21** must be a live `=HYPERLINK(...)` formula, not plain text.

---

## 3. Holding Equity Value & Sell Exit Dynamics

Per `OutputFormatGuide.md` §2:

1. **BUY (ENTRY):** append position → `Holding_Equity_Value` **increases** by position spend.
2. **SELL (EXIT):** remove position → `Holding_Equity_Value` **decreases** by that position's spend.
3. Multiple same-day exits: holding equity **keeps decreasing** until all positions closed (`0.0`).
4. `Total_Portfolio_Value = Cash_Balance + Holding_Equity_Value` on every row.

---

## 4. Candlestick Trade Plot PNG Rules

Per `OutputFormatGuide.md` §3 — implement in `swing_strategy/plotter.py`:

1. **No full vertical lines** over candles (`ax.axvline` excluded).
2. Compute average candle height: `H_avg = mean(High − Low)` over chart window.
3. **Profit trade** (`Net_PnL >= 0`):
   - Entry: green `^` at `Entry_High + H_avg` (`#10B981`)
   - Exit: green `v` at `Exit_Low − H_avg` (`#10B981`)
4. **Loss trade** (`Net_PnL < 0`):
   - Entry: green `^` at `Entry_Low − H_avg` (`#10B981`)
   - Exit: red `v` at `Exit_High + H_avg` (`#EF4444`)
5. **Liquidity source candle (required)**:
   - Find the Yearly/Monthly/Weekly bar whose `Low ≈ Support_Price` (or use `Liquidity_Date` / `Sweep_Date` / `Support_Formed_Date`).
   - Blue `v` at `Liquidity_High + H_avg` (`#2563EB`), labeled `Liquidity Source (...)`. No `ax.axvline`.
   - Expand the chart window so that candle is visible.

```python
avg_candle_size = (df_sub["High"] - df_sub["Low"]).mean()
if net_pnl >= 0:
    entry_arrow_y = entry_high + avg_candle_size
    exit_arrow_y = exit_low - avg_candle_size
    exit_color = "#10B981"
else:
    entry_arrow_y = entry_low - avg_candle_size
    exit_arrow_y = exit_high + avg_candle_size
    exit_color = "#EF4444"
```

---

## 5. Before-Tax vs After-Tax Metrics

Per `OutputFormatGuide.md` §4–§5 — **always** report three equity columns:

| Metric | Gross (before tax) | Net Zerodha (₹0 brokerage) | Net FYERS (₹20/order) |
|--------|--------------------|----------------------------|------------------------|

### Formulas

```
Gross_Return_%  = (Gross_Final_Equity − Initial) / Initial × 100
Net_Return_%    = (Net_Final_Equity − Initial) / Initial × 100
Gross_CAGR_%    = ((Gross_Final / Initial)^(1/N) − 1) × 100
Net_CAGR_%      = ((Net_Final / Initial)^(1/N) − 1) × 100
N               = (End_Date − Start_Date).days / 365.25
```

### Statutory charges (Indian NSE/BSE delivery)

- STT: 0.1% buy + 0.1% sell turnover
- Exchange: 0.00345% turnover
- SEBI: 0.0001% turnover
- Stamp duty: 0.015% buy turnover
- GST: 18% on exchange fees + brokerage
- Brokerage: ₹0 (Zerodha) or ₹20/order (FYERS)

Use `src/analysis/indian_brokerage_calculator.calculate_indian_trade_charges`.

### Performance Summary tab (required in every `.xlsx`)

Second sheet **Performance Summary** with rows:

| Performance Metric | Gross (BEFORE TAX) | Net (Zerodha) | Net (Flat ₹20) |
|--------------------|--------------------|---------------|----------------|
| Initial Capital | | | |
| Final Portfolio Equity | | | |
| Total Net Profit (INR) | | | |
| Total Return (%) | | | |
| CAGR (%) | | | |
| Executed Trades Count | | | |
| Win Rate (%) | | | |
| Max Drawdown (%) | | | |
| Total Statutory Taxes Paid | | | |

Reference implementation:

```python
def calculate_before_and_after_tax_metrics(initial_capital, gross_equity,
        net_equity_zerodha, net_equity_fyers, start_date, end_date):
    num_years = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365.25
    gross_return_pct = ((gross_equity - initial_capital) / initial_capital) * 100.0
    gross_cagr_pct = (((gross_equity / initial_capital) ** (1.0 / num_years)) - 1.0) * 100.0
    net_return_zerodha_pct = ((net_equity_zerodha - initial_capital) / initial_capital) * 100.0
    net_cagr_zerodha_pct = (((net_equity_zerodha / initial_capital) ** (1.0 / num_years)) - 1.0) * 100.0
    net_return_fyers_pct = ((net_equity_fyers - initial_capital) / initial_capital) * 100.0
    net_cagr_fyers_pct = (((net_equity_fyers / initial_capital) ** (1.0 / num_years)) - 1.0) * 100.0
    return {...}
```

---

## 6. Master Comparison Table (multi-experiment runs)

When running capital × risk matrices (`run_multi_experiments.py`), export `Master_Experiments_Comparison.xlsx` with columns:

| Column | Example |
|--------|---------|
| Experiment ID | `Exp_100k_0.5k` |
| Initial Capital | `Rs 100,000` |
| Risk Cap / Trade | `Rs 500` |
| Final Account Equity | `Rs 970,467.03` |
| Net Return (%) | `+870.47%` |
| CAGR (%) | `15.26%` |
| Executed Trades | `12,248` |
| Win Rate (%) | `49.03%` |
| Max Drawdown (%) | `79.88%` |
| Taxes Paid | `Rs …` |

Use **Net Zerodha CAGR** as the primary reported CAGR unless user asks for gross.

---

## 7. Implementation Entry Points

| Task | Script |
|------|--------|
| Single backtest + statement | `python swing_strategy/run_strategy.py` |
| Multi capital/risk matrix | `python swing_strategy/run_multi_experiments.py` |
| Statement generator (core) | `swing_strategy/generate_statement.py` |
| Trade PNG plots | `swing_strategy/plotter.py` |
| Portfolio visualizations | `swing_strategy/visualizer.py` |

`generate_swing_strategy_statement()` parameters:
- `initial_deposit` — starting capital (INR)
- `max_risk_per_trade` — fixed risk cap per trade (INR)
- `df_acc` — ML-filtered trade DataFrame (pass explicitly; do not rely on stale cache)
- `custom_reports_dir` / `custom_plots_dir` — per-experiment output folders
- `exp_name` — experiment label (e.g. `Exp_100k_0.5k`)

---

## 8. Validation Checklist (must pass before citing results)

Copy this checklist into your response when delivering reports:

```
[ ] 21-column CSV/Excel schema matches §2 (correct order, Cash_Balance, Chart_PNG_URI hyperlinks)
[ ] Performance Summary tab has Gross + Zerodha + FYERS columns (§5)
[ ] Holding_Equity_Value increases on BUY, decreases on each SELL (§3)
[ ] Trade PNGs use wick-preserving arrow offsets, no ax.axvline on candles (§4)
[ ] All 4 portfolio visualizations + Interactive HTML exist (§1)
[ ] CAGR computed with N = (end − start) / 365.25, not hard-coded 16 years
[ ] Taxes computed via indian_brokerage_calculator (§5)
[ ] Simulation followed Realistic_guide.md entries-first + gap-fill rules
[ ] ML filter acceptance rate is reasonable (<30% of raw setups unless documented)
[ ] Executed trade count is plausible for capital/risk (not 99% of signals with tiny capital)
```

**Reject the run** if any item fails. Fix the pipeline, regenerate, then report.

---

## 9. Common Failure Modes (learned from this repo)

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| CAGR wildly different from expectation | Stale/wrong `Reports/` or ML cache | Delete `Reports/Cache/*.pkl`, regenerate |
| ML accepts ~99% of setups | Threshold too low (`p1+p2 >= 0.42` when min ≈ 0.42) | Raise threshold to 0.55–0.60 |
| Very few executed trades vs ML accepted | Cash constraints under entries-first rule | Normal; report both ML-accepted and executed counts |
| Negative CAGR on all capital levels | Broken ML filter or missing dataset columns | Validate §0 pre-flight |
| `Chart_PNG_URI` not clickable | Plain text instead of `=HYPERLINK(...)` | Fix Excel export in `generate_statement.py` |
| Performance Summary missing 3 tax columns | Only single-column summary exported | Add Gross/Zerodha/FYERS per §5 |

---

## 10. Summary Checklist for AI Assistants

From `OutputFormatGuide.md` — enforce on every output:

1. ✅ Follow 21-column schema from `Account_Statement_guide.md`
2. ✅ Preserve candle wicks; offset arrows by `H_avg`; mark the liquidity-source candle above its High in `#2563EB`
3. ✅ Dynamic holding equity: increase on BUY, decrease on each SELL
4. ✅ Embed `=HYPERLINK(...)` in column 21
5. ✅ Export Gross + Net (Zerodha) + Net (FYERS) return and CAGR side-by-side
