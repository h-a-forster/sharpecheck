"""Scaffold checks: the package imports, exposes its version, and the CLI starts."""

from __future__ import annotations

import importlib
import math

import pytest

import sharpecheck
from sharpecheck.cli import main
from sharpecheck.results import SharpeInterval, TestResult

MODULES = [
    "sharpecheck.moments",
    "sharpecheck.single",
    "sharpecheck.hac",
    "sharpecheck.bootstrap",
    "sharpecheck.compare",
    "sharpecheck.deflated",
    "sharpecheck.multiple",
    "sharpecheck.checkpoint",
    "sharpecheck.data",
    "sharpecheck.portfolio.weights",
    "sharpecheck.portfolio.backtest",
    "sharpecheck.montecarlo.dgps",
    "sharpecheck.montecarlo.runner",
    "sharpecheck.montecarlo.report",
]


@pytest.mark.parametrize("name", MODULES)
def test_modules_import(name: str) -> None:
    importlib.import_module(name)


def test_version_is_a_string() -> None:
    assert isinstance(sharpecheck.__version__, str) and sharpecheck.__version__


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "sharpecheck" in capsys.readouterr().out


def test_cli_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_test_result_reject() -> None:
    res = TestResult(test="x", statistic=1.0, pvalue=0.01, nobs=10)
    assert res.reject(0.05)
    assert not res.reject(0.001)
    undefined = TestResult(test="x", statistic=math.nan, pvalue=math.nan, nobs=10)
    assert not undefined.reject(0.05)
    with pytest.raises(ValueError):
        res.reject(1.5)
    assert res.to_dict()["pvalue"] == 0.01


def test_sharpe_interval_covers() -> None:
    ci = SharpeInterval("m", estimate=0.1, lower=0.0, upper=0.2, level=0.95, se=0.05, nobs=100)
    assert ci.covers(0.1) and ci.covers(0.0) and not ci.covers(0.25)
    bad = SharpeInterval("m", 0.1, math.nan, math.nan, 0.95, math.nan, 100)
    assert not bad.covers(0.1)
    one_sided = SharpeInterval("m", 0.1, 0.02, math.inf, 0.95, 0.05, 100)
    assert one_sided.covers(5.0) and not one_sided.covers(0.0)
