"""
Indian Brokerage & Statutory Taxes/Charges Calculator

Calculates exact Indian stock market transaction charges for Equity Delivery trades:
1. Brokerage Fee (Flat INR 20 per order or INR 0 for Zero-Brokerage delivery brokers)
2. STT (Securities Transaction Tax): 0.1% on Buy Value AND 0.1% on Sell Value
3. Exchange Transaction Charge (NSE): 0.00297% of Turnover
4. SEBI Turnover Fee: 0.0001% of Turnover
5. GST: 18% on (Brokerage + Exchange Charge + SEBI Fee)
6. Stamp Duty: 0.015% on Buy Value ONLY
7. DP Charge (CDSL/NSDL): INR 15.93 per stock sell transaction
"""

import math

def calculate_indian_trade_charges(
    entry_price: float,
    exit_price: float,
    position_size: int,
    flat_brokerage_per_order: float = 0.0,  # 0.0 for Zerodha Equity Delivery, 20.0 for Flat Rs 20
) -> dict:
    if position_size <= 0 or entry_price <= 0 or exit_price <= 0:
        return {
            "gross_pnl": 0.0,
            "total_charges": 0.0,
            "net_pnl": 0.0,
            "brokerage": 0.0,
            "stt": 0.0,
            "exchange_charge": 0.0,
            "sebi_fee": 0.0,
            "gst": 0.0,
            "stamp_duty": 0.0,
            "dp_charge": 0.0,
        }

    buy_value = entry_price * position_size
    sell_value = exit_price * position_size
    turnover = buy_value + sell_value
    gross_pnl = sell_value - buy_value

    # 1. Brokerage
    # Max Rs 20 or 0.03% whichever is lower (if flat_brokerage_per_order > 0)
    if flat_brokerage_per_order > 0:
        buy_brokerage = min(flat_brokerage_per_order, 0.0003 * buy_value)
        sell_brokerage = min(flat_brokerage_per_order, 0.0003 * sell_value)
        brokerage = buy_brokerage + sell_brokerage
    else:
        brokerage = 0.0

    # 2. STT (0.1% on Buy + 0.1% on Sell for Equity Delivery)
    stt = 0.001 * buy_value + 0.001 * sell_value

    # 3. NSE Exchange Transaction Charge (0.00297% of Turnover)
    exchange_charge = 0.0000297 * turnover

    # 4. SEBI Fee (0.0001% of Turnover / Rs 10 per Crore)
    sebi_fee = 0.000001 * turnover

    # 5. GST (18% on Brokerage + Exchange Charge + SEBI Fee)
    gst = 0.18 * (brokerage + exchange_charge + sebi_fee)

    # 6. Stamp Duty (0.015% on Buy Value ONLY)
    stamp_duty = 0.00015 * buy_value

    # 7. DP Charge (Rs 13.50 + 18% GST = Rs 15.93 flat per sell transaction)
    dp_charge = 15.93

    total_charges = brokerage + stt + exchange_charge + sebi_fee + gst + stamp_duty + dp_charge
    net_pnl = gross_pnl - total_charges

    return {
        "gross_pnl": round(gross_pnl, 2),
        "total_charges": round(total_charges, 2),
        "net_pnl": round(net_pnl, 2),
        "brokerage": round(brokerage, 2),
        "stt": round(stt, 2),
        "exchange_charge": round(exchange_charge, 2),
        "sebi_fee": round(sebi_fee, 2),
        "gst": round(gst, 2),
        "stamp_duty": round(stamp_duty, 2),
        "dp_charge": round(dp_charge, 2),
    }

if __name__ == "__main__":
    # Test sample trade: Buy 100 shares @ Rs 100, Sell @ Rs 103 (Gross +Rs 300)
    ch = calculate_indian_trade_charges(100.0, 103.0, 100, flat_brokerage_per_order=0.0)
    print("Sample Trade Charges (Zero Brokerage Equity Delivery):", ch)
