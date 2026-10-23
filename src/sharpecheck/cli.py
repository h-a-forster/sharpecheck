"""Command-line interface: ``sharpecheck sr | compare | mc | data | portfolio``."""

from __future__ import annotations

import argparse
import sys

from sharpecheck import __version__

_PLANNED = {
    "sr": "all single-SR intervals, PSR and MinTRL for a CSV return column (M8)",
    "compare": "Sharpe-ratio difference tests for two CSV return columns (M8)",
    "mc": "Monte Carlo coverage and size study: run | report (M5, M6)",
    "data": "download and cache Ken French data: fetch (M7)",
    "portfolio": "1/N versus optimised rolling study: run | report (M7)",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sharpecheck",
        description="Sharpe-ratio inference with Monte Carlo-checked intervals.",
    )
    parser.add_argument("--version", action="version", version=f"sharpecheck {__version__}")
    sub = parser.add_subparsers(dest="command")
    for name, help_text in _PLANNED.items():
        sub.add_parser(name, help=help_text)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    print(f"sharpecheck {args.command}: not implemented yet", file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
