"""Daily downloader entry. The script lives in stock_market/download."""

from pathlib import Path
import runpy

_SCRIPT = Path(__file__).resolve().parents[3] / "download" / "fetch_daily_data.py"


def main() -> None:
    runpy.run_path(str(_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
