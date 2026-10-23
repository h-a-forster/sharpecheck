"""Command-line interface: ``sharpecheck sr | compare | mc | data | portfolio``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sharpecheck import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sharpecheck",
        description="Sharpe-ratio inference with Monte Carlo-checked intervals.",
    )
    parser.add_argument("--version", action="version", version=f"sharpecheck {__version__}")
    sub = parser.add_subparsers(dest="command")

    p_sr = sub.add_parser("sr", help="all single-SR intervals, PSR and MinTRL for a CSV column")
    p_sr.add_argument("csv", type=Path, help="CSV file with a header row")
    p_sr.add_argument("--column", help="column name (default: the last column)")
    p_sr.add_argument("--level", type=float, default=0.95)
    p_sr.add_argument("--sr0", type=float, default=0.0, help="benchmark SR for PSR/MinTRL")
    p_sr.add_argument("--n-boot", type=int, default=999)
    p_sr.add_argument("--seed", type=int, default=0)
    p_sr.add_argument("--percent", action="store_true", help="values are in percent")
    p_sr.add_argument("--json", action="store_true", help="print JSON instead of a table")

    p_cmp = sub.add_parser("compare", help="Sharpe-ratio difference tests for two CSV columns")
    p_cmp.add_argument("csv", type=Path)
    p_cmp.add_argument("col1")
    p_cmp.add_argument("col2")
    p_cmp.add_argument("--n-boot", type=int, default=4999)
    p_cmp.add_argument("--block", type=int, default=None)
    p_cmp.add_argument("--seed", type=int, default=0)
    p_cmp.add_argument("--json", action="store_true")

    p_mc = sub.add_parser("mc", help="Monte Carlo coverage and size study")
    mc_sub = p_mc.add_subparsers(dest="action", required=True)
    for name, help_text in (
        ("run", "run or resume all units of a config"),
        ("pilot", "timing pilot: project the run time, write calibration.json"),
        ("report", "aggregate a finished run into results/<name>/"),
    ):
        p = mc_sub.add_parser(name, help=help_text)
        p.add_argument("--config", type=Path, required=True)
        if name != "report":
            p.add_argument("--workers", type=int, default=3)
        if name == "run":
            p.add_argument("--force-new", action="store_true")
            p.add_argument("--max-minutes", type=float, default=None)
            p.add_argument(
                "--experiments",
                default="coverage,difference,dsr",
                help="comma-separated subset of coverage,difference,dsr",
            )
        if name == "report":
            p.add_argument("--out", type=Path, default=None)

    p_data = sub.add_parser("data", help="download and cache Ken French data")
    data_sub = p_data.add_subparsers(dest="action", required=True)
    p = data_sub.add_parser("fetch", help="download the datasets named in a portfolio config")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--force", action="store_true")

    p_pf = sub.add_parser("portfolio", help="1/N versus optimised rolling study")
    pf_sub = p_pf.add_subparsers(dest="action", required=True)
    for name in ("run", "report"):
        p = pf_sub.add_parser(name)
        p.add_argument("--config", type=Path, required=True)
        if name == "run":
            p.add_argument("--workers", type=int, default=3)
        else:
            p.add_argument("--out", type=Path, default=None)
    return parser


def _read_column(path: Path, column: str | None) -> tuple[str, Any]:
    import pandas as pd

    df = pd.read_csv(path)
    col = column if column is not None else str(df.columns[-1])
    if col not in df.columns:
        raise SystemExit(f"column {col!r} not found; available: {list(df.columns)}")
    return col, pd.to_numeric(df[col], errors="raise").dropna().to_numpy(dtype=float)


def _cmd_sr(args: argparse.Namespace) -> int:
    from sharpecheck.moments import sample_moments
    from sharpecheck.single import METHODS, min_track_record_length, psr_test, sharpe_ci

    col, x = _read_column(args.csv, args.column)
    if args.percent:
        x = x / 100.0
    m = sample_moments(x)
    rows = []
    for method in METHODS:
        ci = sharpe_ci(x, method, args.level, n_boot=args.n_boot, seed=args.seed)
        rows.append(ci.to_dict())
    psr_o = psr_test(x, args.sr0, variance_at="observed")
    psr_n = psr_test(x, args.sr0, variance_at="null")
    out = {
        "column": col,
        "moments": m.__dict__,
        "intervals": rows,
        "psr_observed_BLdP2012": psr_o.statistic,
        "psr_null_LZL2025": psr_n.statistic,
        "mintrl_observed": min_track_record_length(m.sr, args.sr0, m.g3, m.g4),
        "mintrl_null": min_track_record_length(m.sr, args.sr0, m.g3, m.g4, variance_at="null"),
    }
    if args.json:
        print(json.dumps(out, indent=2, default=str))
        return 0
    print(f"{col}: T = {m.nobs}, per-period SR = {m.sr:.4f} (mean {m.mean:.4g}, sd {m.sd:.4g})")
    print(f"skewness {m.g3:.3f}, kurtosis {m.g4:.3f} (non-excess), lag-1 autocorr {m.rho1:.3f}")
    print(f"\n{int(args.level * 100)}% two-sided intervals for the per-period SR:")
    print(f"  {'method':<11} {'lower':>9} {'upper':>9} {'se':>9}")
    for r in rows:
        print(f"  {r['method']:<11} {r['lower']:>9.4f} {r['upper']:>9.4f} {r['se']:>9.4f}")
    print(
        f"\nPSR(SR0 = {args.sr0:g}): {out['psr_observed_BLdP2012']:.4f} (BLdP 2012, observed "
        f"variance), {out['psr_null_LZL2025']:.4f} (LZL 2025, null variance)"
    )
    print(
        f"MinTRL at 95%: {out['mintrl_observed']:.1f} (observed), "
        f"{out['mintrl_null']:.1f} (null) periods"
    )
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from sharpecheck.compare import jkm_test, lw_boot_test, lw_hac_test

    _, a = _read_column(args.csv, args.col1)
    _, b = _read_column(args.csv, args.col2)
    if a.size != b.size:
        raise SystemExit("columns have different numbers of non-missing values")
    res = [
        jkm_test(a, b),
        lw_hac_test(a, b, prewhiten=False),
        lw_hac_test(a, b, prewhiten=True),
        lw_boot_test(a, b, block=args.block, n_boot=args.n_boot, seed=args.seed),
    ]
    if args.json:
        print(json.dumps([r.to_dict() for r in res], indent=2, default=str))
        return 0
    d = res[1].details["delta"]
    print(f"SR({args.col1}) - SR({args.col2}) = {d:.4f} per period, T = {a.size}")
    print(f"  {'test':<10} {'statistic':>10} {'p-value':>9}")
    for r in res:
        print(f"  {r.test:<10} {r.statistic:>10.3f} {r.pvalue:>9.4f}")
    bt = res[-1].details
    if "ci_lower" in bt:
        print(
            f"LW bootstrap 95% interval: [{bt['ci_lower']:.4f}, {bt['ci_upper']:.4f}] "
            f"(block {bt['block']}, {bt['n_boot']} resamples)"
        )
    return 0


def _cmd_mc(args: argparse.Namespace) -> int:
    from sharpecheck.montecarlo import report, runner

    if args.action == "run":
        exps = tuple(e.strip() for e in args.experiments.split(",") if e.strip())
        root = runner.run(
            args.config,
            workers=args.workers,
            force_new=args.force_new,
            max_minutes=args.max_minutes,
            experiments=exps,
        )
        print(f"run directory: {root.as_posix()}")
        return 0
    if args.action == "pilot":
        print(json.dumps(runner.pilot(args.config, workers=args.workers), indent=2))
        return 0
    out = report.build_report(args.config, args.out)
    print(f"report written to {out.as_posix()}")
    return 0


def _cmd_data(args: argparse.Namespace) -> int:
    from sharpecheck.portfolio.study import fetch_all

    for name, info in fetch_all(args.config, force=args.force).items():
        print(f"{name}: sha256 {info['sha256'][:16]}..., vintage {info.get('vintage')}")
    return 0


def _cmd_portfolio(args: argparse.Namespace) -> int:
    from sharpecheck.portfolio import study

    if args.action == "run":
        root = study.run(args.config, workers=args.workers)
        print(f"run directory: {root.as_posix()}")
        return 0
    out = study.build_report(args.config, args.out)
    print(f"report written to {out.as_posix()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    handlers = {
        "sr": _cmd_sr,
        "compare": _cmd_compare,
        "mc": _cmd_mc,
        "data": _cmd_data,
        "portfolio": _cmd_portfolio,
    }
    return handlers[args.command](args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
