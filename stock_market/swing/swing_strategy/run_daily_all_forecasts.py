"""Daily forecast job moved to stock_market/forecast. This keeps old imports working."""

from swing_strategy._load_forecast import load

_mod = load("run_daily_all_forecasts.py")
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})
