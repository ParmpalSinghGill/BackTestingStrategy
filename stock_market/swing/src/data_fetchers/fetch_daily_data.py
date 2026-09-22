"""Daily downloader moved to stock_market/download. This keeps old imports working."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_path = Path(__file__).resolve().parents[3] / "download" / "fetch_daily_data.py"
_download = str(_path.parent)
if _download not in sys.path:
    sys.path.insert(0, _download)

_name = "stock_market_download_fetch_daily_data"
if _name not in sys.modules:
    spec = importlib.util.spec_from_file_location(_name, _path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load downloader: {_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_name] = module
    spec.loader.exec_module(module)

_mod = sys.modules[_name]
get_all_tickers = _mod.get_all_tickers
process_symbol = _mod.process_symbol
main = _mod.main
safe_filename = _mod.safe_filename
