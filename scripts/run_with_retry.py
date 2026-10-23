"""Resume-on-failure wrapper for long runs.

Runs a command up to ``--attempts`` times, appending one line per attempt to a log file.
The wrapped commands (``sharpecheck mc run``, ``sharpecheck portfolio run``) checkpoint
every finished unit, so each retry resumes where the previous attempt stopped.

    uv run python scripts/run_with_retry.py --log runs/mc-quick/retry.log -- \
        uv run sharpecheck mc run --config configs/mc_quick.toml --workers 3
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--wait", type=float, default=10.0, help="seconds between attempts")
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        ap.error("no command given")
    args.log.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"{datetime.now(UTC).isoformat(timespec='seconds')} | {msg}"
        print(line, flush=True)
        with args.log.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    for attempt in range(1, args.attempts + 1):
        log(f"attempt {attempt}/{args.attempts}: {' '.join(cmd)}")
        t0 = time.monotonic()
        rc = subprocess.call(cmd)
        log(f"attempt {attempt} exited with {rc} after {time.monotonic() - t0:.0f} s")
        if rc == 0:
            return 0
        if attempt < args.attempts:
            time.sleep(args.wait)
    log("giving up")
    return 1


if __name__ == "__main__":
    sys.exit(main())
