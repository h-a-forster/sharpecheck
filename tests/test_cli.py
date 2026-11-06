"""CLI smoke tests: ``sr`` and ``compare`` on a small CSV."""

from __future__ import annotations

import json
import math
import re
import shlex
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sharpecheck.cli import main


@pytest.fixture
def csv(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": 0.01 + 0.04 * rng.standard_normal(120)})
    df["b"] = 0.5 * df["a"] + 0.03 * rng.standard_normal(120)
    p = tmp_path / "r.csv"
    df.to_csv(p, index=False)
    return p


def _reject_constant(name: str) -> None:
    raise AssertionError(f"{name} is not valid JSON")


def test_sr_table(csv: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["sr", str(csv), "--column", "a", "--n-boot", "199"]) == 0
    out = capsys.readouterr().out
    for m in ("iid_normal", "exact_nct", "mertens", "ar1", "hac", "hac_pw", "boot_ts"):
        assert m in out
    assert "BLdP 2012" in out and "LZL 2025" in out
    assert main(["sr", str(csv), "--level", "0.975", "--n-boot", "19"]) == 0
    assert "97.5% two-sided" in capsys.readouterr().out


def test_sr_json(csv: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["sr", str(csv), "--json", "--n-boot", "99"]) == 0
    d = json.loads(capsys.readouterr().out, parse_constant=_reject_constant)
    assert d["column"] == "b" and len(d["intervals"]) == 7
    assert d["intervals"][1]["se"] is None  # exact_nct has no standard error
    with pytest.raises(SystemExit):
        main(["sr", str(csv), "--column", "zzz"])


def test_compare(csv: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["compare", str(csv), "a", "b", "--n-boot", "199"]) == 0
    out = capsys.readouterr().out
    assert "jkm" in out and "lw_boot" in out and "interval" in out
    assert main(["compare", str(csv), "a", "b", "--n-boot", "99", "--json"]) == 0
    res = json.loads(capsys.readouterr().out)
    assert [r["test"] for r in res] == ["jkm", "lw_hac", "lw_hac_pw", "lw_boot"]


def test_compare_constant_column(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rng = np.random.default_rng(2)
    p = tmp_path / "cash.csv"
    pd.DataFrame({"cash": 0.1, "fund": 0.01 + 0.04 * rng.standard_normal(60)}).to_csv(p)
    assert main(["compare", str(p), "cash", "fund", "--n-boot", "19", "--json"]) == 0
    res = json.loads(capsys.readouterr().out, parse_constant=_reject_constant)
    assert [r["pvalue"] for r in res] == [None] * 4
    assert main(["compare", str(p), "cash", "fund", "--n-boot", "19"]) == 0
    assert all(math.isnan(float(ln.split()[-1])) for ln in capsys.readouterr().out.split("\n")[2:6])


def test_readme_console_example(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    readme = (root / "README.md").read_text(encoding="utf-8")
    block = re.search(r"```console\n(.*?)```", readme, re.DOTALL)
    assert block is not None
    commands = re.split(r"^\$ ", block.group(1), flags=re.MULTILINE)[1:]
    assert len(commands) == 2
    for chunk in commands:
        cmd, _, expected = chunk.partition("\n")
        assert main(shlex.split(cmd)[1:]) == 0
        assert capsys.readouterr().out.strip() == expected.strip()


def test_mc_requires_action() -> None:
    with pytest.raises(SystemExit):
        main(["mc"])


def test_compare_drops_rows_jointly(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # Missing values in different rows must not shift one series against the other.
    rng = np.random.default_rng(5)
    a = 0.01 + 0.04 * rng.standard_normal(240)
    df = pd.DataFrame({"a": a, "b": 0.6 * a + 0.03 * rng.standard_normal(240)})
    clean = df.drop(index=[1, 200])
    df.loc[1, "a"] = np.nan
    df.loc[200, "b"] = np.nan
    p_mis, p_clean = tmp_path / "mis.csv", tmp_path / "clean.csv"
    df.to_csv(p_mis, index=False)
    clean.to_csv(p_clean, index=False)
    args = ["a", "b", "--n-boot", "99", "--seed", "1", "--json"]
    assert main(["compare", str(p_mis), *args]) == 0
    got = json.loads(capsys.readouterr().out)
    assert main(["compare", str(p_clean), *args]) == 0
    want = json.loads(capsys.readouterr().out)
    assert [r["pvalue"] for r in got] == pytest.approx([r["pvalue"] for r in want])
    assert got[0]["nobs"] == 238
