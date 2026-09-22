# SwingNoMl Strategy Specification Document

**Strategy Name**: SwingNoMl (Pure Price-Action 1:3 RR Strategy with Case #5 Gap-Up TouchPlannedEntry Rule)  
**Strategy Type**: Quantitative Equity Swing Trading (No Machine Learning Filters)  
**Target Market**: Indian Equities (NSE / BSE Daily Data)  
**Backtest Window**: 2010 to 2026 (16 Years)  

---

## 1. Strategy Overview & Core Philosophy

The **SwingNoMl Strategy** is a pure price-action, quantitative swing trading framework designed to exploit liquidity sweeps below major support levels across higher timeframes (Yearly, Monthly, Weekly). 

The strategy operates on a strict **1:3 Risk-to-Reward (RR)** ratio and applies realistic real-world execution constraints—including intra-day entry/exit priority, strict fixed-risk cash sizing, exact Indian statutory taxes (STT, GST, Stamp Duty, SEBI fees), and gap-up retest filtering (`TouchPlannedEntry`).

---

## 2. Candidate Setup Selection Rules (C1 Candle)

For every stock in the daily dataset, candle $C1$ is evaluated against three core price-action criteria:

1. **Green Candle Requirement**:
   $$C1\text{ Close} > C1\text{ Open}$$
   *(Candle $C1$ must close higher than its open price, indicating buying response after breaking support).*

2. **Under-Liquidity Breakdown Requirement**:
   $$C1\text{ Close} < \text{Support Price} \quad \text{AND} \quad C1\text{ High} < \text{Support Price}$$
   *(Both the close and high of candle $C1$ must stay strictly below the Support Liquidity level, confirming price is swept below liquidity).*

3. **Higher Timeframe Priority Hierarchy**:
   If a stock triggers setup signals across multiple timeframe liquidity levels on the exact same date, enforce strict priority deduplication:
   $$\text{Yearly Liquidity} > \text{Monthly Liquidity} > \text{Weekly Liquidity}$$
   *(Smaller timeframe setups are discarded if a higher timeframe liquidity setup is present).*

---

## 3. Order Price Level Formulas

Once a candidate $C1$ setup is validated:

- **Planned Entry Price**:
  $$\text{Planned Entry Price} = C1\text{ High} \times 1.001 \quad (+0.1\% \text{ buffer})$$

- **Stop-Loss (SL) Price**:
  $$\text{Stop-Loss Price} = \text{Support Price} \times 0.999 \quad (-0.1\% \text{ buffer})$$

- **Planned Risk per Share**:
  $$\text{Planned Risk per Share} = \text{Planned Entry Price} - \text{Stop-Loss Price}$$

- **Target Price (1:3 RR)**:
  $$\text{Target Price} = \text{Planned Entry Price} + (3.0 \times \text{Planned Risk per Share})$$

---

## 4. Execution Day C2 Logic (Case #5 Gap-Up Opening Rule)

On day $C2$ (the day immediately following $C1$ setup):

### Case 5 Rule (`TouchPlannedEntry`):
1. **Gap-Up Opening ($C2\text{ Open} > \text{Planned Entry Price}$)**:
   - Check if $C2\text{ Low} \le \text{Planned Entry Price}$.
   - **If TRUE**: Price pulled back during the day to touch our planned price. Execute fill at:
     $$\text{Actual Entry Price} = \text{Planned Entry Price}$$
     *(Target & Stop-Loss are calculated based on this actual fill price).*
   - **If FALSE**: Price opened above entry and never pulled back. **Skip the trade completely** (do not chase inflated gap-up prices).

2. **Normal / Gap-Down Opening ($C2\text{ Open} \le \text{Planned Entry Price}$)**:
   - Check if $C2\text{ High} \ge \text{Planned Entry Price}$.
   - **If TRUE**: Execute fill at:
     $$\text{Actual Entry Price} = \max(\text{Planned Entry Price}, C2\text{ Open})$$
   - **If FALSE**: No trade executed on day $C2$.

---

## 5. Portfolio Cash Allocation & Risk Management

1. **Intra-Day Order Execution Precedence**:
   - On any trading day $D$, **Entries are evaluated FIRST** using capital available at the start of Day $D$.
   - **Exits are processed SECOND**. Cash liberated from exits on Day $D$ becomes available for new entries on Day $D+1$.

2. **Position Sizing Formula**:
   $$\text{Quantity} = \left\lfloor \min\left( \frac{\text{Fixed Risk Cap per Trade}}{\text{Actual Risk per Share}}, \frac{\text{Available Cash Balance}}{\text{Actual Entry Price}} \right) \right\rfloor$$
   - Total trade cost ($\text{Quantity} \times \text{Actual Entry Price}$) is deducted immediately from cash balance upon entry.

3. **Decreasing Equity & Portfolio Tracking**:
   - **Holding Equity Value**: Sum of current cost value of all open active positions.
   - **Total Portfolio Value**: $\text{Cash Balance} + \text{Holding Equity Value}$.

---

## 6. Daily Lifecycle & Exit Processing

Positions are monitored daily from day $C2$ onwards until closed:

1. **Stop-Loss Exit (SL Hit)**:
   - Condition: $\text{Daily Low} \le \text{Stop-Loss Price}$.
   - Fill Price: $\text{Stop-Loss Price}$.
   - Trade Outcome: `LOSS`.

2. **Target Exit (1:3 Target Hit)**:
   - Condition: $\text{Daily High} \ge \text{Target Price}$.
   - Fill Price: $\text{Target Price}$.
   - Trade Outcome: `PROFIT`.

3. **Statutory Taxes & Charges Deduction**:
   - Calculates exact Indian statutory charges:
     - STT (Securities Transaction Tax): 0.1% on Buy & Sell.
     - Exchange Transaction Fee: 0.00345%.
     - SEBI Turnover Charge: 0.0001%.
     - Stamp Duty: 0.015% (Buy side).
     - GST: 18% on (Exchange Charges + SEBI Fees).
   - $\text{Net PnL} = \text{Gross PnL} - \text{Total Statutory Taxes}$.

---

## 7. Institutional 21-Column Account Statement Schema

The output ledger strictly follows the 21-column institutional format:

| Col # | Column Name | Description |
|---|---|---|
| 1 | `Transaction_ID` | Sequential integer transaction index |
| 2 | `Trade_ID` | Unique trade identifier (0 for DEPOSIT) |
| 3 | `Type` | Transaction type: `DEPOSIT`, `BUY`, or `SELL` |
| 4 | `Date` | Transaction execution date (`YYYY-MM-DD`) |
| 5 | `Ticker` | Stock symbol (e.g., `RELIANCE.NS`) |
| 6 | `Liquidity_Source` | Liquidity timeframe (`Yearly`, `Monthly`, `Weekly`) |
| 7 | `Support_Price` | Historical support liquidity price level |
| 8 | `Quantity` | Number of shares bought / sold |
| 9 | `Price` | Execution price per share |
| 10 | `Total_Spend` | Total capital outflow (Buy) or inflow (Sell) |
| 11 | `Gross_PnL` | Before-tax profit/loss for closed trade |
| 12 | `Statutory_Taxes` | Total STT, GST, Stamp Duty, and exchange fees |
| 13 | `Net_PnL` | After-tax net profit/loss |
| 14 | `Return_Pct` | Percentage net return on trade capital |
| 15 | `Cash_Balance` | Liquid unallocated portfolio cash remaining |
| 16 | `Active_Position_Count` | Number of active open trades in portfolio |
| 17 | `Holding_Equity_Value` | Total capital tied up in open positions |
| 18 | `Total_Portfolio_Value` | Net total portfolio equity (`Cash` + `Holding Equity`) |
| 19 | `Target_RR_Mode` | Target Risk-Reward ratio (`1:3`) |
| 20 | `Outcome` | Trade outcome: `DEPOSIT`, `PROFIT`, or `LOSS` |
| 21 | `Chart_PNG_URI` | Native Excel hyperlink: `=HYPERLINK("Plots/...", "View Plot Chart (PNG)")` |

---

## 8. Candlestick Trade Plot PNG Standards

For every closed trade, a 200 DPI candlestick trade chart PNG is generated:

1. **Horizontal Price Level Lines**:
   - **Support Line**: Dashed Royal Blue (`#2563EB`).
   - **Entry Price Line**: Solid **Dark Green (`#006400`)**.
   - **Planned Entry Line**: Dashed Dark Green (`#047857`) if fill differs.
   - **Exit Price Line**: Solid **Dark Red (`#8B0000`)**.
   - **Stop-Loss Line**: Dotted Red (`#DC2626`).
   - **Target Line**: Dotted Green (`#059669`).
2. **Wick-Preserving Marker Arrows**:
   - Entry arrow (`^`): Solid Dark Green (`#006400`), offset above/below candle wicks.
   - Exit arrow (`v`): Solid Dark Red (`#8B0000`) for loss or Dark Green (`#006400`) for profit.
3. **Trade Details Legend Box**:
   - Displays Trade ID, Ticker, Liquidity Source, Entry Date & Price, Exit Date & Price, Net PnL, and Ending Account Balance.
