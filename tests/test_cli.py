"""CLI smoke tests: ``sr`` and ``compare`` on a small CSV."""

from __future__ import annotations

import json
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


def test_sr_table(csv: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["sr", str(csv), "--column", "a", "--n-boot", "199"]) == 0
    out = capsys.readouterr().out
    for m in ("iid_normal", "exact_nct", "mertens", "ar1", "hac", "hac_pw", "boot_ts"):
        assert m in out
    assert "BLdP 2012" in out and "LZL 2025" in out


def test_sr_json(csv: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["sr", str(csv), "--json", "--n-boot", "99"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["column"] == "b" and len(d["intervals"]) == 7
    with pytest.raises(SystemExit):
        main(["sr", str(csv), "--column", "zzz"])


def test_compare(csv: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["compare", str(csv), "a", "b", "--n-boot", "199"]) == 0
    out = capsys.readouterr().out
    assert "jkm" in out and "lw_boot" in out and "interval" in out
    assert main(["compare", str(csv), "a", "b", "--n-boot", "99", "--json"]) == 0
    res = json.loads(capsys.readouterr().out)
    assert [r["test"] for r in res] == ["jkm", "lw_hac", "lw_hac_pw", "lw_boot"]


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
