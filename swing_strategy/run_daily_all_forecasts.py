"""
One 16:00 job: download daily bars once, then write Swing_Live, Swing_low, Swing_PP.

Does not fetch again inside the three scanners (--skip-fetch).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from swing_strategy.run_daily_swing_forecast import fetch_latest

JOBS = (
    ("Swing_Live", BASE / "swing_strategy" / "run_daily_swing_forecast.py"),
    ("Swing_low", BASE / "swing_strategy" / "run_daily_swing_low_forecast.py"),
    ("Swing_PP", BASE / "swing_strategy" / "run_daily_swing_pp_forecast.py"),
)


def run_one(name: str, script: Path, extra: list[str]) -> int:
    cmd = [sys.executable, str(script), "--skip-fetch", *extra]
    print(f"\n===== {name}  ({' '.join(cmd[1:])}) =====", flush=True)
    r = subprocess.run(cmd, cwd=str(BASE))
    print(f"[{name}] exit {r.returncode}", flush=True)
    return int(r.returncode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--asof", default="", help="YYYY-MM-DD passed through to all three scanners")
    args = ap.parse_args()

    now = datetime.now()
    if now.weekday() >= 5 and not args.asof:
        print(f"[clock] {now:%A %Y-%m-%d} weekend - skip (no Sat/Sun run)", flush=True)
        return

    extra: list[str] = []
    if args.asof:
        extra.extend(["--asof", args.asof])

    if args.skip_fetch:
        print("[fetch] skipped (--skip-fetch)", flush=True)
    else:
        print("[fetch] once for Swing_Live + Swing_low + Swing_PP", flush=True)
        fetch_latest()

    codes = {name: run_one(name, script, extra) for name, script in JOBS}
    failed = [n for n, c in codes.items() if c != 0]
    print("\n===== done =====", flush=True)
    for name, code in codes.items():
        print(f"  {name}: {'ok' if code == 0 else f'FAIL {code}'}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
