"""The committed run configs parse and respect the budget rules of docs/DESIGN.md."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _load(name: str) -> dict:
    with (CONFIGS / name).open("rb") as fh:
        return tomllib.load(fh)


@pytest.mark.parametrize("name", ["mc_quick.toml", "mc_full.toml", "portfolio.toml"])
def test_config_parses_and_caps_workers(name: str) -> None:
    cfg = _load(name)
    assert cfg["run"]["workers"] <= 3
    assert cfg["run"]["out_dir"].startswith("runs/")


@pytest.mark.parametrize("name", ["mc_quick.toml", "mc_full.toml"])
def test_monte_carlo_floors(name: str) -> None:
    cov = _load(name)["coverage"]
    assert cov["reps"] >= 1000
    assert cov["boot_reps"] >= 500
    assert len(cov["dgps"]) == 6
    assert set(cov["methods"]) == {
        "iid_normal",
        "exact_nct",
        "mertens",
        "ar1",
        "hac",
        "hac_pw",
        "boot_ts",
    }
