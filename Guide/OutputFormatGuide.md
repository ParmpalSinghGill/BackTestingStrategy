# Master Guide: Output Format Specification, PNG Hyperlinks & Tax-Adjusted CAGR Engine

This document defines the standardized **Output Format Specification** for backtest exports, performance summaries, and statement generation. It extends [Guide/Account_Statement_guide.md](file:///c:/DATA/CODE/Stocks/BackTest/Guide/Account_Statement_guide.md) to ensure all generated statements incorporate working **PNG trade plot hyperlinks**, standardized graphic visualizations, wick-preserving entry/exit marker arrows, decreasing holding equity tracking on sell entries, and side-by-side **Before-Tax vs. After-Tax Return (%) & CAGR (%)** performance metrics.

---

## 📋 1. Alignment with Account Statement Schema

All generated account statements (`Swing_Strategy_Account_Statement.xlsx` and `Swing_Strategy_Account_Statement.csv`) must follow the institutional 21-column layout defined in `Account_Statement_guide.md`:

| Col # | Column Name | Type / Format | Description & Rule |
| :--- | :--- | :--- | :--- |
| **1** | `Transaction_ID` | Integer | Sequential transaction index ($1, 2, 3, \dots$). |
| **2** | `Trade_ID` | Integer | Unique identifier for each trade setup ($0$ for initial deposit). |
| **3** | `Type` | String | Transaction type (`DEPOSIT`, `BUY (ENTRY)`, `SELL (EXIT)`). |
| **4** | `Date` | `YYYY-MM-DD` | Execution date. |
| **5** | `Ticker` | String | Stock symbol (e.g. `RELIANCE.NS`). |
| **6** | `Liquidity_Source` | String | Support timeframe (`Yearly`, `Monthly`, `Weekly`). |
| **7** | `Support_Price` | Currency (INR) | Key support price level. |
| **8** | `Quantity` | Integer | Shares traded. |
| **9** | `Price` | Currency (INR) | Actual execution price ($0.2\%$ gap-up entry or $0.1\%$ gap exit fill). |
| **10** | `Total_Spend` | Currency (INR) | Total position cost ($\text{Quantity} \times \text{Price}$). |
| **11** | `Gross_PnL` | Currency (INR) | Gross profit or loss before taxes & charges ($0.0$ for `BUY` rows). |
| **12** | `Statutory_Taxes` | Currency (INR) | Sum of STT, exchange fees, SEBI charges, stamp duty, GST, brokerage. |
| **13** | `Net_PnL` | Currency (INR) | Pure net profit/loss ($\text{Gross PnL} - \text{Statutory Taxes}$). |
| **14** | `Return_Pct` | Percentage | Net trade return percentage ($\frac{\text{Net PnL}}{\text{Total Spend}} \times 100$). |
| **15** | `Cash_Balance` | Currency (INR) | Free liquid cash balance available in account. |
| **16** | `Active_Position_Count` | Integer | Count of currently open active positions ($\text{len}(\text{open\_positions})$). |
| **17** | `Holding_Equity_Value` | Currency (INR) | **Mark-to-Market Holding Equity**: Sum of position values of currently open positions. Increases on `BUY (ENTRY)` and **decreases continuously as `SELL (EXIT)` entries occur**. |
| **18** | `Total_Portfolio_Value` | Currency (INR) | Total Net Portfolio Value ($\text{Cash Balance} + \text{Holding Equity Value}$). |
| **19** | `Target_RR_Mode` | String | Machine Learning Risk-Reward mode (`1:2`, `1:3`, etc.). |
| **20** | `Outcome` | String | Trade result (`OPEN`, `Success`, `Failure`, `DEPOSIT`). |
| **21** | `Chart_PNG_URI` | Excel Formula | Native Excel formula pointing to valid trade PNG plot chart. |

---

## 📉 2. Holding Equity Value & Sell Exit Dynamics Rule

Holding Equity Value (`Holding_Equity_Value`) tracks the Mark-to-Market equity tied up in active open trades on every transaction row:

1. **On `BUY (ENTRY)`**: The new position is appended to `open_positions`, so `Holding_Equity_Value` **increases** by the total spend of the new trade.
2. **On `SELL (EXIT)`**: The closed position is removed from `open_positions`.
   - **Exit 1**: `Holding_Equity_Value` **decreases** by the value of trade 1.
   - **Subsequent Exits**: `Holding_Equity_Value` **keeps decreasing** as more `SELL (EXIT)` transactions execute.
   - When all open positions are liquidated, `Holding_Equity_Value` reaches `0.0`.
3. **Total Portfolio Liquidation Value**:
   $$\text{Total Portfolio Value} = \text{Cash Balance} + \text{Holding Equity Value}$$

---

## 🖼️ 3. Working PNG Hyperlinks & Wick-Preserving Plotting Rules

### A. Candle Wick Visibility & Marker Arrow Offset Protocol
To ensure full candlestick wicks (High/Low tails) remain 100% visible and un-obscured:

1. **No Wick-Hiding Vertical Lines**: Do NOT plot full vertical lines across candles (`ax.axvline` is excluded).
2. **Average Candle Size Offset Calculation**:
   Calculate average candle range $H_{\text{avg}}$ across the displayed chart window:
   $$H_{\text{avg}} = \text{mean}\left(\text{High} - \text{Low}\right)$$
3. **Profit Trade Marker Positioning (Winning Trade)**:
   - **Entry Candle Marker**: Green Up-Arrow (`^`, `#10B981`) placed **above High of Entry** at $y = \text{Entry High} + H_{\text{avg}}$.
   - **Exit Candle Marker**: Green Down-Arrow (`v`, `#10B981`) placed **below Low of Exit** at $y = \text{Exit Low} - H_{\text{avg}}$.
4. **Loss Trade Marker Positioning (Losing Trade)**:
   - **Entry Candle Marker**: Green Up-Arrow (`^`, `#10B981`) placed **below Low of Entry** at $y = \text{Entry Low} - H_{\text{avg}}$.
   - **Exit Candle Marker**: Red Down-Arrow (`v`, `#EF4444`) placed **above High of Exit** at $y = \text{Exit High} + H_{\text{avg}}$.

```python
# Calculate Average Candle Height
avg_candle_size = (df_sub["High"] - df_sub["Low"]).mean()

if net_pnl >= 0:
    # Profit Trade: Entry above High of Entry, Exit below Low of Exit (empty space)
    entry_arrow_y = entry_high + avg_candle_size
    exit_arrow_y = exit_low - avg_candle_size
    entry_marker, exit_marker = "^", "v"
    exit_color = "#10B981"
else:
    # Loss Trade: Entry below Low of Entry, Exit above High of Exit (empty space)
    entry_arrow_y = entry_low - avg_candle_size
    exit_arrow_y = exit_high + avg_candle_size
    entry_marker, exit_marker = "^", "v"
    exit_color = "#EF4444"

ax.scatter([entry_date], [entry_arrow_y], color="#10B981", s=140, marker=entry_marker, label="Entry Point")
ax.scatter([exit_date], [exit_arrow_y], color=exit_color, s=140, marker=exit_marker, label="Exit Point")
```

5. **Liquidity Source Candle Marker (required)**:
   - Locate the Yearly / Monthly / Weekly bar that **printed the support low** (`Low ≈ Support_Price`) before entry. Prefer an explicit `Liquidity_Date` / `Sweep_Date` / `Support_Formed_Date` when the trade record has one; otherwise search prior bars.
   - Place a **blue down-arrow** (`v`, `#2563EB`) **above High of that candle** at $y = \text{Liquidity High} + H_{\text{avg}}$ so the wick stays fully visible. Do **not** use `ax.axvline`.
   - Label it `Liquidity Source (Yearly|Monthly|Weekly)` and put the date in the info box.
   - Widen the chart window so that candle is visible (at least 15 calendar days before it).

```python
liq_arrow_y = liq_high + avg_candle_size
ax.scatter([liq_date], [liq_arrow_y], color="#2563EB", s=170, marker="v", label="Liquidity Source")
```

### B. Trade Chart PNG Hyperlinks (Column 21)
In Excel exports (`.xlsx`), Column 21 (`Chart_PNG_URI`) must contain active, clickable Excel formulas pointing to the trade plot chart:

```excel
=HYPERLINK("Plots/swing_statement_trades/trade_charts/Trade_1_RELIANCE.NS.png", "View Plot Chart (PNG)")
```

### C. Standard Portfolio Visualization Suite
Every execution run must generate the following 4 core visual report graphics:
1. **Monthly Returns Heatmap PNG**: `Plots/Monthly_Returns_Heatmap.png`
2. **Annual Performance Breakdown PNG**: `Plots/Yearly_Returns_Breakdown.png`
3. **Monthly Capital Growth Line Chart PNG**: `Plots/Capital_Growth_Monthly.png`
4. **Interactive Chart.js Dashboard HTML**: `Reports/Interactive_Equity_Curve.html`

---

## 💰 4. Before-Tax vs. After-Tax Return & CAGR Calculation Engine

Backtest results must explicitly report metrics both **BEFORE TAX** (Gross) and **AFTER TAX** (Net) to quantify the impact of statutory Indian market taxes and brokerage friction.

### A. Mathematical Formulas

#### 1. Individual Trade PnL
$$\text{Gross PnL} = (\text{Exit Price} - \text{Entry Price}) \times \text{Quantity}$$
$$\text{Statutory Taxes} = \text{STT} + \text{Exchange Fee} + \text{SEBI Fee} + \text{Stamp Duty} + \text{GST} + \text{Brokerage}$$
$$\text{Net PnL} = \text{Gross PnL} - \text{Statutory Taxes}$$

#### 2. Statutory Tax Breakdown (Indian NSE/BSE Equity Delivery)
- **STT (Securities Transaction Tax)**: $0.1\%$ on buy & sell turnover.
- **Exchange Transaction Charges**: $0.00345\%$ on total turnover.
- **SEBI Turnover Fee**: $0.0001\%$ on total turnover.
- **Stamp Duty**: $0.015\%$ on buy turnover.
- **GST**: $18\%$ on exchange fees and brokerage.
- **Brokerage**: ₹0 (Zerodha delivery) or flat ₹20/order (FYERS / Discount brokers).

#### 3. Total Return Percentage
$$\text{Gross Return \%} = \left( \frac{\text{Gross Final Equity} - \text{Initial Capital}}{\text{Initial Capital}} \right) \times 100$$
$$\text{Net Return \%} = \left( \frac{\text{Net Final Equity} - \text{Initial Capital}}{\text{Initial Capital}} \right) \times 100$$

#### 4. Compound Annual Growth Rate (CAGR)
For a backtest duration of $N$ years ($N = \frac{\text{End Date} - \text{Start Date}}{365.25}$):

$$\text{Gross CAGR \%} = \left( \left( \frac{\text{Gross Final Equity}}{\text{Initial Capital}} \right)^{\frac{1}{N}} - 1 \right) \times 100$$

$$\text{Net CAGR \%} = \left( \left( \frac{\text{Net Final Equity}}{\text{Initial Capital}} \right)^{\frac{1}{N}} - 1 \right) \times 100$$

---

## 📊 5. Performance Summary Tab & Export Schema

In addition to the main `Account Statement` tab, the generated Excel workbook (`Swing_Strategy_Account_Statement.xlsx`) and master comparison reports (`Master_Experiments_Comparison.xlsx`) must include a dedicated **Performance Summary** table displaying explicit side-by-side Before-Tax and After-Tax metrics:

```markdown
| Performance Metric | Gross Value (BEFORE TAX) | Net Value (AFTER TAX - Zerodha) | Net Value (AFTER TAX - Flat ₹20) |
| :--- | :--- | :--- | :--- |
| **Initial Capital** | ₹100,000.00 | ₹100,000.00 | ₹100,000.00 |
| **Final Portfolio Equity** | ₹2,422,160.00 | ₹2,322,160.00 | ₹2,185,430.00 |
| **Total Net Profit (INR)** | ₹2,322,160.00 | ₹2,222,160.00 | ₹2,085,430.00 |
| **Total Return (%)** | **+2,322.16%** | **+2,222.16%** | **+2,085.43%** |
| **CAGR (%)** | **22.45%** | **22.04%** | **21.52%** |
| **Executed Trades Count**| 4,215 | 4,215 | 4,215 |
| **Win Rate (%)** | 52.40% | 51.85% | 51.20% |
| **Max Drawdown (%)** | 12.80% | 13.15% | 13.60% |
| **Total Statutory Taxes Paid** | ₹0.00 | ₹100,000.00 | ₹236,730.00 |
```

---

## 🐍 6. Python Implementation Example

Below is the standard Python snippet for calculating and formatting Return (%) and CAGR (%) before and after tax when writing outputs:

```python
import math
import pandas as pd
from src.analysis.indian_brokerage_calculator import calculate_indian_trade_charges

def calculate_before_and_after_tax_metrics(initial_capital: float, gross_equity: float, net_equity_zerodha: float, net_equity_fyers: float, start_date: str, end_date: str):
    num_years = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365.25
    
    # 1. Before Tax (Gross)
    gross_return_pct = ((gross_equity - initial_capital) / initial_capital) * 100.0
    gross_cagr_pct = (((gross_equity / initial_capital) ** (1.0 / num_years)) - 1.0) * 100.0 if gross_equity > 0 else 0.0
    
    # 2. After Tax (Zerodha Net)
    net_return_zerodha_pct = ((net_equity_zerodha - initial_capital) / initial_capital) * 100.0
    net_cagr_zerodha_pct = (((net_equity_zerodha / initial_capital) ** (1.0 / num_years)) - 1.0) * 100.0 if net_equity_zerodha > 0 else 0.0
    
    # 3. After Tax (Flat Rs 20 Net)
    net_return_fyers_pct = ((net_equity_fyers - initial_capital) / initial_capital) * 100.0
    net_cagr_fyers_pct = (((net_equity_fyers / initial_capital) ** (1.0 / num_years)) - 1.0) * 100.0 if net_equity_fyers > 0 else 0.0
    
    return {
        "Gross_Return_Pct": round(gross_return_pct, 2),
        "Gross_CAGR_Pct": round(gross_cagr_pct, 2),
        "Net_Return_Zerodha_Pct": round(net_return_zerodha_pct, 2),
        "Net_CAGR_Zerodha_Pct": round(net_cagr_zerodha_pct, 2),
        "Net_Return_FYERS_Pct": round(net_return_fyers_pct, 2),
        "Net_CAGR_FYERS_Pct": round(net_cagr_fyers_pct, 2),
    }
```

---

## 🎯 Summary Checklist for AI Assistants & LLMs

When generating backtest reports or statement files:
1. ✅ **Follow Column Schema**: Enforce all 21 columns from `Account_Statement_guide.md`.
2. ✅ **Preserve Candle Wicks**: Do NOT draw full vertical lines over candles. Offset entry/exit arrows by average candle height $H_{\text{avg}}$ (above High for profit, below Low for loss). Mark the **liquidity-source candle** with a blue `v` (`#2563EB`) above its High at $H_{\text{avg}}$.
3. ✅ **Dynamic Holding Equity**: Increase `Holding_Equity_Value` on BUY entries and decrease it continuously as SELL exits execute.
4. ✅ **Embed Hyperlinks**: Use `=HYPERLINK("Plots/...", "View Plot Chart (PNG)")` in Column 21 (`Chart_PNG_URI`).
5. ✅ **Side-by-Side Tax Metrics**: Always print & export **Gross Return / Gross CAGR (Before Tax)** alongside **Net Return / Net CAGR (After Tax - Zerodha / FYERS)**.
