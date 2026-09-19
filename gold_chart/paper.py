"""Local paper trading for the gold replay chart.

Fills as CoinDCX B-XAU_USDT market (taker) orders: 0.05% per side + 18% GST
on the fee, plus configurable slippage. Capital and settings persist on disk
until the user changes them.

Two ledgers:
- paper_transactions.csv — every fill (buy entry, sell entry, TP, SL, manual)
- paper_trades.csv — one row per round-trip once the order is fully exited,
  including scale-out legs Exit_1..Exit_5.
"""

from __future__ import annotations

import csv
import json
import math
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

PAPER_DIR = Path(__file__).resolve().parent / "paper"
STATE_FILE = PAPER_DIR / "paper_state.json"
TRADES_FILE = PAPER_DIR / "paper_trades.csv"
TXN_FILE = PAPER_DIR / "paper_transactions.csv"
IST = timezone(timedelta(hours=5, minutes=30))

DEFAULT_TAKER_FEE_PCT = 0.05
DEFAULT_GST_PCT = 18.0
DEFAULT_SLIPPAGE_PCT = 0.02
DEFAULT_SIZE_PCT = 100.0
DEFAULT_LEVERAGE = 1.0
DEFAULT_CAPITAL = 10_000.0
QTY_STEP = 0.001
MAX_EXITS = 5
MAX_MARKERS = 80

TRADE_FIELDS = [
    "Trade_ID",
    "Side",
    "Status",
    "Entry_Date",
    "Entry_Time_IST",
    "Timeframe",
    "Signal_Entry",
    "Entry_Fill",
    "Entry_Qty",
    "Leverage",
    "Money_Used",
    "Notional",
    "Slippage_Pct",
    "Entry_Fee",
    "Entry_GST",
    "Entry_Charges",
]
for _n in range(1, MAX_EXITS + 1):
    TRADE_FIELDS.extend(
        [
            f"Exit_{_n}_Date",
            f"Exit_{_n}_Time_IST",
            f"Exit_{_n}_Signal",
            f"Exit_{_n}_Fill",
            f"Exit_{_n}_Qty",
            f"Exit_{_n}_Reason",
            f"Exit_{_n}_Charges",
        ]
    )
TRADE_FIELDS.extend(
    [
        "Exit_Count",
        "Total_Exit_Qty",
        "Slippage_Cost",
        "Exit_Fee",
        "GST_On_Fees",
        "Total_Charges",
        "Gross_PnL_Before_Charges",
        "Net_PnL_After_Charges",
        "Gross_Return_Pct",
        "Net_Return_Pct",
        "Hold_Minutes",
        "Capital_Before",
        "Capital_After",
        "Taker_Fee_Pct",
        "GST_Pct",
        "Take_Profit",
        "Stop_Loss",
        "Exit_Reason",
    ]
)

TXN_FIELDS = [
    "Txn_ID",
    "Trade_ID",
    "Type",
    "Side",
    "Reason",
    "Date",
    "Time_IST",
    "Timeframe",
    "Quantity",
    "Signal_Price",
    "Fill_Price",
    "Notional",
    "Fee",
    "GST",
    "Charges",
    "Gross_PnL",
    "Net_PnL",
    "Cash_After",
    "Open_Count",
]

_MISSING = object()
_lock = threading.Lock()


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _round_qty(qty: float) -> float:
    steps = math.floor(float(qty) / QTY_STEP + 1e-12)
    return _round(max(0.0, steps * QTY_STEP), 6)


def _ist_parts(unix: int) -> tuple[str, str]:
    stamp = datetime.fromtimestamp(int(unix), tz=timezone.utc).astimezone(IST)
    return stamp.strftime("%Y-%m-%d"), stamp.strftime("%H:%M:%S")


def _default_state() -> dict[str, Any]:
    return {
        "capital": DEFAULT_CAPITAL,
        "cash": DEFAULT_CAPITAL,
        "size_pct": DEFAULT_SIZE_PCT,
        "leverage": DEFAULT_LEVERAGE,
        "slippage_pct": DEFAULT_SLIPPAGE_PCT,
        "taker_fee_pct": DEFAULT_TAKER_FEE_PCT,
        "gst_pct": DEFAULT_GST_PCT,
        "next_id": 1,
        "next_txn_id": 1,
        "open": [],
        "markers": [],
        "last_closed": None,
        "trades_file": str(TRADES_FILE),
        "transactions_file": str(TXN_FILE),
    }


def _ensure_csv(path: Path, fields: list[str]) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writeheader()
        return
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if list(reader.fieldnames or []) == fields:
            return
        rows = list(reader)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _ensure_files() -> None:
    _ensure_csv(TRADES_FILE, TRADE_FIELDS)
    _ensure_csv(TXN_FILE, TXN_FIELDS)


def _count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def _positions(state: dict[str, Any]) -> list[dict[str, Any]]:
    raw = state.get("open")
    if raw is None:
        positions: list[dict[str, Any]] = []
    elif isinstance(raw, dict):
        positions = [raw]
    elif isinstance(raw, list):
        positions = [item for item in raw if isinstance(item, dict)]
    else:
        positions = []
    state["open"] = positions
    return positions


def _load_state() -> dict[str, Any]:
    _ensure_files()
    state = _default_state()
    if STATE_FILE.exists():
        try:
            raw = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                state.update(raw)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    _positions(state)
    state["trades_file"] = str(TRADES_FILE)
    state["transactions_file"] = str(TXN_FILE)
    if "next_txn_id" not in state:
        state["next_txn_id"] = _count_rows(TXN_FILE) + 1
    if "markers" not in state or not isinstance(state["markers"], list):
        state["markers"] = []
    return state


def _save_state(state: dict[str, Any]) -> None:
    _ensure_files()
    blob = {
        key: value
        for key, value in state.items()
        if key not in {"trades_file", "transactions_file"}
    }
    STATE_FILE.write_text(json.dumps(blob, indent=2), encoding="utf-8")


def _fee_on(notional: float, taker_pct: float, gst_pct: float) -> tuple[float, float, float]:
    fee = abs(float(notional)) * (float(taker_pct) / 100.0)
    gst = fee * (float(gst_pct) / 100.0)
    return _round(fee, 6), _round(gst, 6), _round(fee + gst, 6)


def _append_row(path: Path, fields: list[str], row: dict[str, Any]) -> None:
    _ensure_files()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writerow({key: row.get(key, "") for key in fields})


def _add_marker(
    state: dict[str, Any], time_unix: int, buy: bool, text: str, price: float | None = None
) -> None:
    marks = list(state.get("markers") or [])
    marks.append(
        {
            "time": int(time_unix),
            "buy": bool(buy),
            "text": text,
            "price": None if price is None else _round(float(price), 4),
        }
    )
    state["markers"] = marks[-MAX_MARKERS:]


def _public(state: dict[str, Any]) -> dict[str, Any]:
    positions = _positions(state)
    locked = sum(float(pos.get("moneyUsed") or 0) for pos in positions)
    return {
        "capital": _round(state["cash"], 2),
        "setCapital": _round(state["capital"], 2),
        "cash": _round(state["cash"], 2),
        "equityAtCost": _round(float(state["cash"]) + locked, 2),
        "sizePct": float(state["size_pct"]),
        "leverage": float(state["leverage"]),
        "slippagePct": float(state["slippage_pct"]),
        "takerFeePct": float(state["taker_fee_pct"]),
        "gstPct": float(state["gst_pct"]),
        "open": positions,
        "openCount": len(positions),
        "markers": list(state.get("markers") or []),
        "lastClosed": state.get("last_closed"),
        "closedTrades": _count_rows(TRADES_FILE),
        "transactionCount": _count_rows(TXN_FILE),
        "tradesFile": str(TRADES_FILE),
        "transactionsFile": str(TXN_FILE),
        "venue": "CoinDCX B-XAU_USDT",
        "feeNote": (
            "Market fill as CoinDCX taker 0.05%/side + 18% GST on the fee. "
            "USDT-M VIP tiers can differ; edit the fee if your account is not VIP0."
        ),
    }


def get_paper_state() -> dict[str, Any]:
    with _lock:
        return _public(_load_state())


def apply_paper_action(payload: dict[str, Any]) -> dict[str, Any]:
    action = str(payload.get("action") or "").strip().lower()
    with _lock:
        state = _load_state()
        if action == "set_capital":
            _set_capital(state, payload)
        elif action == "settings":
            _set_settings(state, payload)
        elif action in {"stops", "set_stops"}:
            _set_stops(state, payload)
        elif action in {"long", "buy"}:
            _open_position(state, payload, "LONG")
        elif action in {"short"}:
            _open_position(state, payload, "SHORT")
        elif action in {"sell", "exit", "close", "cover"}:
            _close_position(state, payload)
        elif action in {"close_all", "closeall"}:
            _close_all(state, payload)
        elif action in {"reverse", "flip"}:
            _reverse_position(state, payload)
        elif action in {"check", "check_stops"}:
            _check_stops(state, payload)
        else:
            raise ValueError(
                "Unknown paper action. Use set_capital, settings, stops, "
                "long, short, close, close_all, reverse, or check."
            )
        _save_state(state)
        return _public(state)


def _set_capital(state: dict[str, Any], payload: dict[str, Any]) -> None:
    if _positions(state):
        raise ValueError("Close all open trades before changing capital.")
    amount = float(payload.get("capital"))
    if not math.isfinite(amount) or amount <= 0:
        raise ValueError("Capital must be a positive number.")
    state["capital"] = _round(amount, 2)
    state["cash"] = _round(amount, 2)


def _set_settings(state: dict[str, Any], payload: dict[str, Any]) -> None:
    mapping = {
        "size_pct": "sizePct",
        "leverage": "leverage",
        "slippage_pct": "slippagePct",
        "taker_fee_pct": "takerFeePct",
        "gst_pct": "gstPct",
    }
    for key, incoming in mapping.items():
        if incoming not in payload and key not in payload:
            continue
        raw = payload.get(incoming, payload.get(key))
        value = float(raw)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid {incoming}.")
        state[key] = value
    if float(state["size_pct"]) <= 0 or float(state["size_pct"]) > 100:
        raise ValueError("Size % must be between 0 and 100.")
    if float(state["leverage"]) < 1:
        raise ValueError("Leverage must be at least 1.")


def _first_present(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return _MISSING


def _coerce_price(raw: Any) -> float | None:
    if raw is None or raw == "":
        return None
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("TP/SL must be a positive gold price.")
    return _round(value, 4)


def _coerce_pts(raw: Any) -> float | None:
    if raw is None or raw == "":
        return None
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise ValueError("TP/SL points must be a positive distance.")
    if value == 0:
        return None
    return _round(value, 4)


def _parse_tp_sl(
    payload: dict[str, Any],
    *,
    fill: float | None = None,
    side: str = "LONG",
    mark: float | None = None,
    fallback_tp: Any = _MISSING,
    fallback_sl: Any = _MISSING,
) -> tuple[float | None, float | None]:
    raw_tp = _first_present(payload, "tp", "takeProfit")
    raw_sl = _first_present(payload, "sl", "stopLoss")
    tp = fallback_tp if raw_tp is _MISSING else _coerce_price(raw_tp)
    sl = fallback_sl if raw_sl is _MISSING else _coerce_price(raw_sl)
    if tp is _MISSING:
        tp = None
    if sl is _MISSING:
        sl = None
    tp_pts = _coerce_pts(_first_present(payload, "tpPts") if "tpPts" in payload else None)
    sl_pts = _coerce_pts(_first_present(payload, "slPts") if "slPts" in payload else None)
    short = side == "SHORT"
    if fill is not None:
        if tp is None and tp_pts is not None:
            tp = _round(fill - tp_pts if short else fill + tp_pts, 4)
        if sl is None and sl_pts is not None:
            sl = _round(fill + sl_pts if short else fill - sl_pts, 4)
    _validate_stops(tp, sl, fill, side, mark=mark)
    return tp, sl


def _validate_stops(
    tp: float | None,
    sl: float | None,
    fill: float | None,
    side: str = "LONG",
    mark: float | None = None,
) -> None:
    short = side == "SHORT"
    if tp is not None and sl is not None:
        if short and tp >= sl:
            raise ValueError("For a short, TP must be below SL.")
        if not short and sl >= tp:
            raise ValueError("SL must be below TP.")
    if fill is None:
        return
    now = mark if mark is not None and mark > 0 else fill
    if short:
        if tp is not None and tp >= fill:
            raise ValueError("TP must be below the short entry.")
        if sl is not None and sl <= now:
            raise ValueError("SL must be above the current price so it is not already hit.")
        return
    if tp is not None and tp <= fill:
        raise ValueError("TP must be above the long entry.")
    if sl is not None and sl >= now:
        raise ValueError("SL must be below the current price so it is not already hit.")


def _find_position(state: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    positions = _positions(state)
    if not positions:
        raise ValueError("No open trade.")
    raw_id = payload.get("id", payload.get("tradeId"))
    if raw_id in (None, ""):
        if len(positions) == 1:
            return positions[0]
        raise ValueError("Pick a trade from its chart line.")
    want = int(raw_id)
    for pos in positions:
        if int(pos.get("id") or 0) == want:
            return pos
    raise ValueError(f"Trade {want} is not open.")


def _set_stops(state: dict[str, Any], payload: dict[str, Any]) -> None:
    if not _positions(state):
        return
    pos = _find_position(state, payload)
    side = "SHORT" if str(pos.get("side") or "LONG").upper() == "SHORT" else "LONG"
    fill = float(pos["entryFill"])
    mark = None
    raw_mark = payload.get("price", payload.get("mark"))
    if raw_mark not in (None, ""):
        try:
            mark = float(raw_mark)
        except (TypeError, ValueError):
            mark = None
        if mark is not None and (not math.isfinite(mark) or mark <= 0):
            mark = None
    tp, sl = _parse_tp_sl(
        payload,
        fill=fill,
        side=side,
        mark=mark,
        fallback_tp=pos.get("tp"),
        fallback_sl=pos.get("sl"),
    )
    pos["tp"] = tp
    pos["sl"] = sl


def _check_stops(state: dict[str, Any], payload: dict[str, Any]) -> None:
    bars = payload.get("bars") or []
    snapshot = list(_positions(state))
    for pos in snapshot:
        if pos not in _positions(state):
            continue
        try:
            tp = _coerce_price(pos.get("tp")) if pos.get("tp") not in (None, "") else None
            sl = _coerce_price(pos.get("sl")) if pos.get("sl") not in (None, "") else None
        except ValueError:
            continue
        if tp is None and sl is None:
            continue
        short = str(pos.get("side") or "LONG").upper() == "SHORT"
        entry_time = int(pos["entryTime"])
        for bar in bars:
            try:
                time_unix = int(bar.get("time"))
                high = float(bar.get("high"))
                low = float(bar.get("low"))
            except (TypeError, ValueError, AttributeError):
                continue
            if time_unix <= entry_time:
                continue
            if short:
                hit_sl = sl is not None and high >= sl
                hit_tp = tp is not None and low <= tp
            else:
                hit_sl = sl is not None and low <= sl
                hit_tp = tp is not None and high >= tp
            if hit_sl:
                _close_position(
                    state,
                    {
                        "id": pos["id"],
                        "price": sl,
                        "time": time_unix,
                        "reason": "SL",
                        "timeframe": pos.get("timeframe"),
                    },
                )
                break
            if hit_tp:
                _close_position(
                    state,
                    {
                        "id": pos["id"],
                        "price": tp,
                        "time": time_unix,
                        "reason": "TP",
                        "timeframe": pos.get("timeframe"),
                    },
                )
                break


def _open_position(state: dict[str, Any], payload: dict[str, Any], side: str) -> None:
    side = "SHORT" if str(side).upper() == "SHORT" else "LONG"
    signal = float(payload.get("price"))
    time_unix = int(payload.get("time"))
    if not math.isfinite(signal) or signal <= 0:
        raise ValueError("Need a valid gold price to enter.")
    slip = float(state["slippage_pct"]) / 100.0
    fill = signal * (1.0 - slip) if side == "SHORT" else signal * (1.0 + slip)
    cash = float(state["cash"])
    lev_raw = payload.get("leverage")
    leverage = float(lev_raw) if lev_raw not in (None, "") else float(state["leverage"])
    if not math.isfinite(leverage) or leverage < 1:
        raise ValueError("Leverage must be at least 1.")
    taker = float(state["taker_fee_pct"])
    gst_pct = float(state["gst_pct"])
    charge_rate = (taker / 100.0) * (1.0 + gst_pct / 100.0)

    qty_raw = payload.get("qty", payload.get("quantity"))
    if qty_raw not in (None, ""):
        qty = _round_qty(float(qty_raw))
        if qty < QTY_STEP:
            raise ValueError("Position too small. Raise capital or size %.")
        notional = qty * fill
        money = notional / leverage
    else:
        size_raw = payload.get("sizePct", payload.get("size_pct"))
        size_frac = (
            float(size_raw) / 100.0
            if size_raw not in (None, "")
            else float(state["size_pct"]) / 100.0
        )
        if not math.isfinite(size_frac) or size_frac <= 0 or size_frac > 1:
            raise ValueError("Size % must be between 0 and 100.")
        budget = cash * size_frac
        if budget <= 0:
            raise ValueError("No cash left to enter. Lower size % or close a trade.")
        money = budget / (1.0 + leverage * charge_rate)
        notional = money * leverage
        qty = _round_qty(notional / fill)
        if qty < QTY_STEP:
            raise ValueError("Position too small. Raise capital or size %.")
        notional = qty * fill
        money = notional / leverage

    fee, gst, charges = _fee_on(notional, taker, gst_pct)
    if money + charges > cash + 1e-9:
        raise ValueError("Not enough cash after CoinDCX fees.")

    date, clock = _ist_parts(time_unix)
    trade_id = int(state["next_id"])
    try:
        tp, sl = _parse_tp_sl(payload, fill=fill, side=side)
    except ValueError:
        tp, sl = None, None

    state["cash"] = _round(cash - money - charges, 6)
    pos = {
        "id": trade_id,
        "side": side,
        "entryTime": time_unix,
        "entryDate": date,
        "entryTimeIst": clock,
        "timeframe": str(payload.get("timeframe") or ""),
        "signalEntry": _round(signal, 4),
        "entryFill": _round(fill, 4),
        "quantity": qty,
        "originalQuantity": qty,
        "leverage": leverage,
        "moneyUsed": _round(money, 4),
        "notional": _round(notional, 4),
        "capitalBefore": _round(cash, 4),
        "entryFee": fee,
        "entryGst": gst,
        "entryCharges": charges,
        "slippagePct": float(state["slippage_pct"]),
        "takerFeePct": taker,
        "gstPct": gst_pct,
        "tp": tp,
        "sl": sl,
        "exits": [],
        "liqPrice": _round(
            fill * (1.0 + 1.0 / leverage) if side == "SHORT" else fill * (1.0 - 1.0 / leverage),
            4,
        )
        if leverage > 0
        else None,
    }
    _positions(state).append(pos)
    state["next_id"] = trade_id + 1
    _add_marker(state, time_unix, side == "LONG", "B" if side == "LONG" else "S", fill)
    _append_transaction(
        state,
        pos,
        txn_type="BUY" if side == "LONG" else "SELL",
        reason="ENTRY",
        time_unix=time_unix,
        signal=signal,
        fill=fill,
        qty=qty,
        fee=fee,
        gst=gst,
        charges=charges,
        gross=0.0,
        net=-charges,
    )


def _close_all(state: dict[str, Any], payload: dict[str, Any]) -> None:
    for pos in list(_positions(state)):
        _close_position(state, {**payload, "id": pos["id"], "reason": payload.get("reason") or "CLOSE"})


def _reverse_position(state: dict[str, Any], payload: dict[str, Any]) -> None:
    pos = _find_position(state, payload)
    side = "SHORT" if str(pos.get("side") or "LONG").upper() == "SHORT" else "LONG"
    qty = float(pos["quantity"])
    _close_position(state, {**payload, "id": pos["id"], "reason": "REVERSE"})
    opposite = "SHORT" if side == "LONG" else "LONG"
    open_payload = {
        "price": payload.get("price"),
        "time": payload.get("time"),
        "timeframe": payload.get("timeframe") or "",
        "qty": qty,
        "tp": "",
        "sl": "",
    }
    try:
        _open_position(state, open_payload, opposite)
    except ValueError as exc:
        raise ValueError(f"Closed the trade but could not reverse: {exc}") from exc


def _close_position(state: dict[str, Any], payload: dict[str, Any]) -> None:
    pos = _find_position(state, payload)
    signal = float(payload.get("price"))
    time_unix = int(payload.get("time"))
    if not math.isfinite(signal) or signal <= 0:
        raise ValueError("Need a valid gold price to exit.")
    side = "SHORT" if str(pos.get("side") or "LONG").upper() == "SHORT" else "LONG"
    reason = str(payload.get("reason") or "CLOSE").strip().upper()
    if reason in {"SELL", "COVER", "EXIT"}:
        reason = "CLOSE"
    if reason not in {"CLOSE", "TP", "SL", "REVERSE"}:
        reason = "CLOSE"
    remaining = float(pos["quantity"])
    qty_raw = payload.get("qty", payload.get("quantity"))
    if qty_raw in (None, ""):
        close_qty = remaining
    else:
        close_qty = min(remaining, _round_qty(float(qty_raw)))
    if close_qty < QTY_STEP:
        raise ValueError("Close quantity is too small.")

    slip = float(pos["slippagePct"]) / 100.0
    if reason == "TP":
        fill = signal
    elif side == "SHORT":
        fill = signal * (1.0 + slip)
    else:
        fill = signal * (1.0 - slip)

    entry_fill = float(pos["entryFill"])
    signal_entry = float(pos["signalEntry"])
    taker = float(pos["takerFeePct"])
    gst_pct = float(pos["gstPct"])
    exit_notional = close_qty * fill
    exit_fee, exit_gst, exit_charges = _fee_on(exit_notional, taker, gst_pct)
    if side == "SHORT":
        gross = (entry_fill - fill) * close_qty
        slip_cost = close_qty * (signal_entry - entry_fill) + close_qty * (fill - signal)
    else:
        gross = (fill - entry_fill) * close_qty
        slip_cost = close_qty * (entry_fill - signal_entry) + close_qty * (signal - fill)

    frac = close_qty / remaining
    released = float(pos["moneyUsed"]) * frac
    cash_after = float(state["cash"]) + released + gross - exit_charges
    exit_date, exit_clock = _ist_parts(time_unix)
    original_qty = float(pos.get("originalQuantity") or remaining)

    exit_row = {
        "date": exit_date,
        "timeIst": exit_clock,
        "time": time_unix,
        "signal": _round(signal, 4),
        "fill": _round(fill, 4),
        "qty": close_qty,
        "reason": reason,
        "charges": exit_charges,
        "fee": exit_fee,
        "gst": exit_gst,
        "gross": _round(gross, 4),
        "slipCost": _round(slip_cost, 4),
        "released": _round(released, 4),
    }
    pos.setdefault("exits", []).append(exit_row)
    state["cash"] = _round(cash_after, 6)
    _add_marker(state, time_unix, side == "SHORT", "X", fill)
    _append_transaction(
        state,
        pos,
        txn_type="SELL" if side == "LONG" else "BUY",
        reason=reason,
        time_unix=time_unix,
        signal=signal,
        fill=fill,
        qty=close_qty,
        fee=exit_fee,
        gst=exit_gst,
        charges=exit_charges,
        gross=gross,
        net=gross - exit_charges,
    )

    leftover = _round_qty(remaining - close_qty)
    if leftover >= QTY_STEP:
        pos["quantity"] = leftover
        pos["moneyUsed"] = _round(float(pos["moneyUsed"]) - released, 4)
        pos["notional"] = _round(leftover * entry_fill, 4)
        return

    _finalize_trade(state, pos, cash_after, original_qty)
    _positions(state).remove(pos)


def _finalize_trade(
    state: dict[str, Any], pos: dict[str, Any], cash_after: float, original_qty: float
) -> None:
    exits = list(pos.get("exits") or [])
    side = str(pos["side"]).upper()
    entry_charges = float(pos["entryCharges"])
    exit_charges = sum(float(item["charges"]) for item in exits)
    gross = sum(float(item["gross"]) for item in exits)
    slip_cost = sum(float(item["slipCost"]) for item in exits)
    exit_fee = sum(float(item["fee"]) for item in exits)
    exit_gst = sum(float(item["gst"]) for item in exits)
    total_charges = entry_charges + exit_charges
    net = gross - total_charges
    last = exits[-1] if exits else {}
    hold = 0.0
    if last:
        hold = max(0, int(last["time"]) - int(pos["entryTime"])) / 60.0
    money = sum(float(item["released"]) for item in exits)
    money_basis = money if money else 1.0
    reasons = "|".join(str(item["reason"]) for item in exits)

    row: dict[str, Any] = {key: "" for key in TRADE_FIELDS}
    row.update(
        {
            "Trade_ID": pos["id"],
            "Side": side,
            "Status": "CLOSED",
            "Entry_Date": pos["entryDate"],
            "Entry_Time_IST": pos["entryTimeIst"],
            "Timeframe": pos.get("timeframe") or "",
            "Signal_Entry": f"{float(pos['signalEntry']):.4f}",
            "Entry_Fill": f"{float(pos['entryFill']):.4f}",
            "Entry_Qty": f"{original_qty:.6f}",
            "Leverage": f"{float(pos['leverage']):.2f}",
            "Money_Used": f"{money:.4f}",
            "Notional": f"{float(pos.get('notional') or original_qty * float(pos['entryFill'])):.4f}",
            "Slippage_Pct": f"{float(pos['slippagePct']):.4f}",
            "Entry_Fee": f"{float(pos['entryFee']):.6f}",
            "Entry_GST": f"{float(pos['entryGst']):.6f}",
            "Entry_Charges": f"{entry_charges:.6f}",
            "Exit_Count": str(len(exits)),
            "Total_Exit_Qty": f"{sum(float(item['qty']) for item in exits):.6f}",
            "Slippage_Cost": f"{slip_cost:.4f}",
            "Exit_Fee": f"{exit_fee:.6f}",
            "GST_On_Fees": f"{float(pos['entryGst']) + exit_gst:.6f}",
            "Total_Charges": f"{total_charges:.6f}",
            "Gross_PnL_Before_Charges": f"{gross:.4f}",
            "Net_PnL_After_Charges": f"{net:.4f}",
            "Gross_Return_Pct": f"{(gross / money_basis) * 100:.4f}",
            "Net_Return_Pct": f"{(net / money_basis) * 100:.4f}",
            "Hold_Minutes": f"{hold:.2f}",
            "Capital_Before": f"{float(pos['capitalBefore']):.4f}",
            "Capital_After": f"{cash_after:.4f}",
            "Taker_Fee_Pct": f"{float(pos['takerFeePct']):.4f}",
            "GST_Pct": f"{float(pos['gstPct']):.2f}",
            "Take_Profit": (
                f"{float(pos['tp']):.4f}" if pos.get("tp") not in (None, "") else ""
            ),
            "Stop_Loss": (
                f"{float(pos['sl']):.4f}" if pos.get("sl") not in (None, "") else ""
            ),
            "Exit_Reason": reasons,
        }
    )
    for index, item in enumerate(exits[:MAX_EXITS], start=1):
        row[f"Exit_{index}_Date"] = item["date"]
        row[f"Exit_{index}_Time_IST"] = item["timeIst"]
        row[f"Exit_{index}_Signal"] = f"{float(item['signal']):.4f}"
        row[f"Exit_{index}_Fill"] = f"{float(item['fill']):.4f}"
        row[f"Exit_{index}_Qty"] = f"{float(item['qty']):.6f}"
        row[f"Exit_{index}_Reason"] = item["reason"]
        row[f"Exit_{index}_Charges"] = f"{float(item['charges']):.6f}"
    _append_row(TRADES_FILE, TRADE_FIELDS, row)
    state["last_closed"] = {
        "id": pos["id"],
        "side": side,
        "entryTime": int(pos["entryTime"]),
        "exitTime": int(last.get("time") or pos["entryTime"]),
        "entryFill": _round(float(pos["entryFill"]), 4),
        "exitFill": _round(float(last.get("fill") or pos["entryFill"]), 4),
        "quantity": original_qty,
        "reason": reasons,
        "pnl": _round(net, 4),
        "exits": exits,
    }


def _append_transaction(
    state: dict[str, Any],
    pos: dict[str, Any],
    *,
    txn_type: str,
    reason: str,
    time_unix: int,
    signal: float,
    fill: float,
    qty: float,
    fee: float,
    gst: float,
    charges: float,
    gross: float,
    net: float,
) -> None:
    date, clock = _ist_parts(time_unix)
    txn_id = int(state.get("next_txn_id") or 1)
    _append_row(
        TXN_FILE,
        TXN_FIELDS,
        {
            "Txn_ID": txn_id,
            "Trade_ID": pos["id"],
            "Type": txn_type,
            "Side": pos["side"],
            "Reason": reason,
            "Date": date,
            "Time_IST": clock,
            "Timeframe": pos.get("timeframe") or "",
            "Quantity": f"{qty:.6f}",
            "Signal_Price": f"{float(signal):.4f}",
            "Fill_Price": f"{float(fill):.4f}",
            "Notional": f"{qty * fill:.4f}",
            "Fee": f"{fee:.6f}",
            "GST": f"{gst:.6f}",
            "Charges": f"{charges:.6f}",
            "Gross_PnL": f"{gross:.4f}",
            "Net_PnL": f"{net:.4f}",
            "Cash_After": f"{float(state['cash']):.4f}",
            "Open_Count": str(len(_positions(state))),
        },
    )
    state["next_txn_id"] = txn_id + 1
