# Specification Guide: Institutional Excel Account Statement & Portfolio Valuation Engine

This document defines the complete specification and column schema for generating institutional-grade **Account Statements**, **Portfolio Valuations**, and **Mark-to-Market (MTM) Visual Plots**. Any LLM or AI coding assistant following this guide will implement complete transaction logs with real-time holding equity valuations.

---

## 📋 Account Statement Schema & Column Definitions

The generated Excel Account Statement (`Swing_Strategy_Account_Statement.xlsx`) and CSV logs must include all **21 columns** listed below in exact order:

| Col # | Column Name | Format / Type | Description |
| :--- | :--- | :--- | :--- |
| **1** | `Transaction_ID` | Integer | Sequential transaction number (1, 2, 3, ...). |
| **2** | `Trade_ID` | Integer | Unique identifier for each trade setup. |
| **3** | `Type` | String | Transaction type (`DEPOSIT`, `BUY (ENTRY)`, `SELL (EXIT)`). |
| **4** | `Date` | YYYY-MM-DD | Execution date of transaction. |
| **5** | `Ticker` | String | Stock symbol (e.g. `RELIANCE.NS`, `TATASTEEL.NS`). |
| **6** | `Liquidity_Source` | String | Higher timeframe support level (`Yearly`, `Monthly`, `Weekly`). |
| **7** | `Support_Price` | Currency (INR) | Key support price level where pattern formed. |
| **8** | `Quantity` | Integer | Number of shares bought or sold. |
| **9** | `Price` | Currency (INR) | Execution price (reflecting 0.2% gap-up entry or 0.1% gap exit). |
| **10** | `Total_Spend` | Currency (INR) | Total position cost ($\text{Quantity} \times \text{Price}$). |
| **11** | `Gross_PnL` | Currency (INR) | Gross profit or loss before taxes ($0.0$ for BUY rows). |
| **12** | `Statutory_Taxes` | Currency (INR) | Sum of STT, NSE exchange charges, SEBI fees, stamp duty, GST. |
| **13** | `Net_PnL` | Currency (INR) | Pure net profit/loss ($\text{Gross PnL} - \text{Statutory Taxes}$). |
| **14** | `Return_Pct` | Percentage | Net return percentage on trade spend ($\frac{\text{Net PnL}}{\text{Total Spend}} \times 100$). |
| **15** | `Cash_Balance` | Currency (INR) | Free liquid cash balance available in trading account. |
| **16** | `Active_Position_Count` | Integer | Count of currently open active positions ($\text{len}(\text{open\_positions})$). |
| **17** | `Holding_Equity_Value` | Currency (INR) | **Mark-to-Market Valuation** of all currently open positions based on daily Closing Price: $\sum (\text{Quantity} \times \text{Closing Price})$. |
| **18** | `Total_Portfolio_Value` | Currency (INR) | **Total Net Portfolio Liquidation Value**: $\text{Cash Balance} + \text{Holding Equity Value}$. |
| **19** | `Target_RR_Mode` | String | Machine Learning Risk-Reward mode (`1:2` or `1:3`). |
| **20** | `Outcome` | String | Trade result (`OPEN`, `Success`, `Failure`, `DEPOSIT`). |
| **21** | `Chart_PNG_URI` | Excel Formula | Native Excel formula `=HYPERLINK("path/to/chart.png", "View Plot Chart (PNG)")`. |

---

## 💡 Real-Time Mark-to-Market (MTM) Portfolio Equity Math

### 1. Holding Equity Value Calculation
For any given date $D$, inspect all open positions in `open_positions`:
$$\text{Holding Equity Value}_D = \sum_{p \in \text{open\_positions}} \left( \text{Quantity}_p \times \text{Close Price}_{p, D} \right)$$

### 2. Total Net Portfolio Value Calculation
$$\text{Total Portfolio Value}_D = \text{Cash Balance}_D + \text{Holding Equity Value}_D$$

---

## 🎨 Visualization & Plotting Requirements

When generating charts (Heatmaps, Monthly Capital Growth Graphs, and Chart.js HTML Dashboards):

1. **Dual Curve Equity Curves**:
   - Plot **Total Net Portfolio Value (Equity + Cash Balance)** as the primary solid blue line ($2.5\text{px}$ thickness).
   - Plot **Free Cash Balance** as a secondary dashed line to highlight cash utilization and portfolio allocation.
2. **Chart.js Interactive HTML Hover Tooltips**:
   Hovering over any date on the interactive Chart.js HTML widget (`Interactive_Equity_Curve.html`) must display:
   - **Date**: `YYYY-MM-DD`
   - **Total Portfolio Value (Cash + Holdings)**: `₹XX,XXX.XX`
   - **Free Cash Balance**: `₹XX,XXX.XX`
   - **Open Holdings MTM Equity Value**: `₹XX,XXX.XX`
   - **Active Positions Count**: `N`
   - **Daily Net PnL**: `±₹XX,XXX.XX (±X.XX%)`
