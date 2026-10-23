"""Monte Carlo DGPs, runner, resume, determinism and report (DESIGN 6, 7.2)."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sharpecheck.montecarlo import report, runner
from sharpecheck.montecarlo.dgps import (
    DGPS,
    PAIR_DGPS,
    population_moments,
    simulate,
    simulate_pair,
    simulate_trials,
)

ROOT = Path(__file__).resolve().parents[1]


def test_population_moments_closed_forms() -> None:
    assert population_moments("garch-t7", 0.1).g4 == pytest.approx(6.689189, abs=1e-6)
    assert population_moments("t5", 0.3).g4 == 9.0
    assert population_moments("ar1-0.2", 0.0).rho1 == 0.2
    # Regime: pi1 = 0.1 / 0.12; mixture with means (0, -0.5), sds (0.8, 2.0).
    pi1 = 0.1 / 0.12
    pi2 = 1 - pi1
    var = pi1 * 0.64 + pi2 * 4.0 + pi1 * pi2 * 0.25
    rg = population_moments("regime", 0.1)
    assert rg.rho1 == pytest.approx(0.88 * pi1 * pi2 * 0.25 / var)
    assert rg.g3 < 0 and rg.g4 > 3
    cr = population_moments("crash", 0.1)
    assert cr.g3 < -2 and cr.g4 > 15
    for d in DGPS:
        pm = population_moments(d, 0.3)
        assert pm.sr == 0.3 and pm.sd == 1.0 and pm.mean == 0.3
    with pytest.raises(ValueError):
        population_moments("bogus", 0.1)  # type: ignore[arg-type]


@pytest.mark.parametrize("dgp", DGPS)
def test_simulate_shape_and_seed(dgp: str) -> None:
    a = simulate(dgp, 0.1, 30, 4, np.random.default_rng(1))  # type: ignore[arg-type]
    b = simulate(dgp, 0.1, 30, 4, np.random.default_rng(1))  # type: ignore[arg-type]
    assert a.shape == (4, 30) and np.array_equal(a, b) and np.all(np.isfinite(a))


@pytest.mark.parametrize("name", PAIR_DGPS)
def test_simulate_pair_shapes(name: str) -> None:
    x = simulate_pair(name, 50, 3, np.random.default_rng(2))
    assert x.shape == (3, 50, 2) and np.all(np.isfinite(x))
    with pytest.raises(ValueError):
        simulate_pair("bogus-iid", 50, 3, np.random.default_rng(2))


def test_simulate_trials_correlation() -> None:
    x = simulate_trials(5, 200_000, 0.3, np.random.default_rng(3))
    c = np.corrcoef(x.T)
    off = c[~np.eye(5, dtype=bool)]
    assert np.allclose(off, 0.3, atol=0.01)
    assert np.allclose(x.std(axis=0), 1.0, atol=0.01)
    y = simulate_trials(3, 200_000, 0.0, np.random.default_rng(4), dist="t5")
    assert np.allclose(y.std(axis=0), 1.0, atol=0.02)
    with pytest.raises(ValueError):
        simulate_trials(3, 10, 1.0, np.random.default_rng(4))


@pytest.mark.slow
@pytest.mark.parametrize("dgp", DGPS)
def test_dgp_moments_vs_simulation(dgp: str) -> None:
    rng = np.random.default_rng(20261024)
    x = simulate(dgp, 0.1, 10_000, 100, rng)  # type: ignore[arg-type]
    pm = population_moments(dgp, 0.1)  # type: ignore[arg-type]
    mu = x.mean(axis=1)
    d = x - mu[:, None]
    m2 = (d**2).mean(axis=1)
    stats_ = {
        "mean": mu,
        "sd": np.sqrt(m2),
        "g3": (d**3).mean(axis=1) / m2**1.5,
        "g4": (d**4).mean(axis=1) / m2**2,
        "rho1": (d[:, 1:] * d[:, :-1]).sum(axis=1) / (d**2).sum(axis=1),
    }
    for k, v in stats_.items():
        target = getattr(pm, k)
        se = v.std(ddof=1) / math.sqrt(v.size)
        if dgp == "t5" and k in ("g3", "g4"):
            continue  # infinite higher moments: batch SEs are unreliable
        assert abs(v.mean() - target) < 4 * se + 0.01 * abs(target) + 1e-3, (k, v.mean(), target)


def _tiny_config(out_dir: Path) -> dict:
    cfg = runner.load_config(ROOT / "configs" / "mc_quick.toml")
    cfg["name"] = "mc-tiny"
    cfg["chunk"] = 10
    cfg["coverage"].update(
        {
            "dgps": ["normal", "ar1-0.2"],
            "sr": [0.0, 0.3],
            "nobs": [60],
            "reps": 20,
            "boot_reps": 10,
            "n_boot": 49,
        }
    )
    cfg["difference"].update({"dgps": ["normal-iid"], "reps": 10, "n_boot": 49})
    cfg["dsr"].update(
        {"n_trials": [10], "rho": [0.3], "nobs": [60], "dists": ["normal"], "reps": 10}
    )
    cfg["run"] = {"workers": 3, "out_dir": str(out_dir.as_posix())}
    return runner.validate_config(cfg)


def test_build_units_and_boot_allocation(tmp_path: Path) -> None:
    cfg = _tiny_config(tmp_path / "r")
    units = runner.build_units(cfg)
    cov = [a for _, a in units if a["experiment"] == "coverage"]
    assert len(cov) == 2 * 2 * 1 * 2
    assert [a["boot_reps"] for a in cov[:2]] == [10, 0]
    ids = [u for u, _ in units]
    assert len(ids) == len(set(ids))


def test_validate_config_rejects_bad_values(tmp_path: Path) -> None:
    cfg = _tiny_config(tmp_path / "r")
    bad = json.loads(json.dumps(cfg))
    bad["coverage"]["methods"] = ["nope"]
    with pytest.raises(ValueError):
        runner.validate_config(bad)
    bad = json.loads(json.dumps(cfg))
    bad["run"]["workers"] = 8
    with pytest.raises(ValueError):
        runner.validate_config(bad)
    bad = json.loads(json.dumps(cfg))
    bad["coverage"]["boot_reps"] = 100
    with pytest.raises(ValueError):
        runner.validate_config(bad)


def test_unit_seed_stable() -> None:
    a = np.random.default_rng(runner.unit_seed(1, "x")).random()
    b = np.random.default_rng(runner.unit_seed(1, "x")).random()
    c = np.random.default_rng(runner.unit_seed(1, "y")).random()
    assert a == b != c


def test_run_resume_and_worker_invariance(tmp_path: Path) -> None:
    cfg1 = _tiny_config(tmp_path / "w1")
    cfg3 = _tiny_config(tmp_path / "w3")
    runner.run_config(cfg1, tmp_path / "w1", workers=1)
    runner.run_config(cfg3, tmp_path / "w3", workers=3)
    from sharpecheck.checkpoint import UnitStore, config_hash

    s1 = UnitStore(tmp_path / "w1", config_hash(cfg1), read_only=True)
    s3 = UnitStore(tmp_path / "w3", config_hash(cfg3), read_only=True)
    for u, _ in runner.build_units(cfg1):
        r1, r3 = s1.read(u), s3.read(u)
        for key in ("methods", "rules", "psr", "sr_hat"):
            if key in r1:
                assert r1[key] == r3[key], u
    # Resume: nothing to do the second time.
    log = (tmp_path / "w1" / "progress.log").read_text(encoding="utf-8")
    runner.run_config(cfg1, tmp_path / "w1", workers=1)
    log2 = (tmp_path / "w1" / "progress.log").read_text(encoding="utf-8")
    new_lines = log2[len(log) :].strip().splitlines()
    assert len(new_lines) == 1 and new_lines[0].startswith("run finished")
    assert '"ran": 0' in new_lines[0]


def test_report_on_tiny_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_root = tmp_path / "run"
    cfg = _tiny_config(run_root)
    runner.run_config(cfg, run_root, workers=1)
    cfg_path = tmp_path / "tiny.toml"
    monkeypatch.setattr(runner, "load_config", lambda p: cfg)
    out = report.build_report(cfg_path, tmp_path / "res")
    cov = pd.read_csv(out / "coverage.csv")
    assert set(cov["method"]) == set(runner.COVERAGE_METHODS)
    assert cov["coverage"].between(0, 1).all()
    assert (cov["coverage_lo"] <= cov["coverage"]).all()
    dif = pd.read_csv(out / "difference.csv")
    assert "lw2008" in dif.columns
    dsr = pd.read_csv(out / "dsr.csv")
    assert set(dsr["rule"]) == set(cfg["dsr"]["rules"])
    summary = json.loads((out / "summary.json").read_text())
    assert summary["n_units"] == len(runner.build_units(cfg))
    one = pd.read_csv(out / "onesided.csv")
    assert {"psr_observed", "psr_null"} <= set(one["method"])


def test_report_refuses_unfinished(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_root = tmp_path / "run"
    cfg = _tiny_config(run_root)
    runner.run_config(cfg, run_root, workers=1, experiments=("dsr",))
    monkeypatch.setattr(runner, "load_config", lambda p: cfg)
    with pytest.raises(RuntimeError, match="not finished"):
        report.build_report(tmp_path / "x.toml", tmp_path / "res")


def test_wilson_interval() -> None:
    lo, hi = report.wilson_interval(95, 100)
    assert lo == pytest.approx(0.8882, abs=1e-4) and hi == pytest.approx(0.9785, abs=1e-4)
    assert report.wilson_interval(0, 10)[0] == 0.0
    assert report.wilson_interval(10, 10)[1] == pytest.approx(1.0)
    assert all(math.isnan(v) for v in report.wilson_interval(0, 0))
    with pytest.raises(ValueError):
        report.wilson_interval(11, 10)


def test_lw_table_matches_reference_script() -> None:
    spec = importlib.util.spec_from_file_location("rv", ROOT / "scripts" / "reference_values.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    src = (ROOT / "scripts" / "reference_values.py").read_text(encoding="utf-8")
    keys = {
        "Normal-IID": "normal-iid",
        "t6-IID": "t6-iid",
        "Normal-GARCH": "normal-garch",
        "t6-GARCH": "t6-garch",
        "Normal-VAR": "normal-var",
        "t6-VAR": "t6-var",
    }
    for paper, ours in keys.items():
        row = report.LW_TABLE1_5PCT[ours]
        vals = (row["jkm"], row["hac"], row["hac_pw"], row["boot_iid"], row["boot_ts"])
        assert f'"{paper}": ({", ".join(f"{v}" for v in vals)})' in src
