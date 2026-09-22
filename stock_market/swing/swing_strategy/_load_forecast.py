"""Load a forecast script that now lives in stock_market/forecast."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load(filename: str):
    target = Path(__file__).resolve().parents[2] / "forecast" / filename
    name = "stock_market_forecast_" + Path(filename).stem
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, target)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load forecast script: {target}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]
