"""Ken French parser (synthetic fixture), allocation rules, backtest and joint test (DESIGN 7.2)."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import ledoit_wolf

from sharpecheck import data
from sharpecheck.portfolio.backtest import rolling_backtest, turnover
from sharpecheck.portfolio.study import joint_diff_test
from sharpecheck.portfolio.weights import (
    RULES,
    equal_weight,
    gmv,
    hrp,
    inverse_vol,
    ledoit_wolf_cov,
    weights,
)

FIX = Path(__file__).resolve().parent / "fixtures"


def _text(name: str) -> str:
    return (FIX / name).read_bytes().decode("latin-1")


def test_parse_value_weighted_section() -> None:
    df = data.parse_monthly(_text("kf_portfolios.csv"), "value-weighted")
    assert list(df.columns) == ["NoDur", "Durbl", "Manuf"]
    assert len(df) == 4
    assert df.index[0] == pd.Timestamp("1963-07-31")
    assert df.loc["1963-08-31", "Durbl"] == 6.42
    assert np.isnan(df.loc["1963-09-30", "Durbl"])  # -99.99 is missing


def test_parse_equal_weighted_and_factors() -> None:
    ew = data.parse_monthly(_text("kf_portfolios.csv"), "equal-weighted")
    assert ew.iloc[0, 0] == -1.0 and len(ew) == 4
    ff = data.parse_monthly(_text("kf_factors.csv"), factors=True)
    assert list(ff.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert len(ff) == 5  # stops at the blank line before the annual block


def test_parse_errors() -> None:
    with pytest.raises(ValueError):
        data.parse_monthly("no marker here", "value-weighted")
    with pytest.raises(ValueError):
        data.parse_monthly(_text("kf_portfolios.csv"), "bogus")
    bad = _text("kf_portfolios.csv").replace("196308,", "196309,", 1)
    with pytest.raises(ValueError, match="consecutive"):
        data.parse_monthly(bad)
    bad = _text("kf_portfolios.csv").replace("196308,    4.89,", "196308,", 1)
    with pytest.raises(ValueError, match="width"):
        data.parse_monthly(bad)


def test_vintage_and_excess_returns() -> None:
    assert data.vintage(_text("kf_factors.csv")) == "202608"
    port = data.parse_monthly(_text("kf_portfolios.csv"))
    ff = data.parse_monthly(_text("kf_factors.csv"), factors=True)
    ex = data.excess_returns(port, ff)
    assert len(ex) == 4
    assert ex.loc["1963-07-31", "NoDur"] == pytest.approx((-0.49 - 0.27) / 100)


def test_load_monthly_from_zip(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("10_Industry_Portfolios.CSV", (FIX / "kf_portfolios.csv").read_bytes())
    (raw / data.DATASETS["ind10"]).write_bytes(buf.getvalue())
    df = data.load_monthly("ind10", tmp_path)
    assert df.shape == (4, 3)
    with pytest.raises(ValueError):
        data.fetch("bogus", tmp_path)


@pytest.fixture
def window() -> np.ndarray:
    rng = np.random.default_rng(0)
    a = rng.standard_normal((120, 6)) @ np.diag([1, 2, 3, 1.5, 0.5, 1]) * 0.01
    a[:, 1] += 0.7 * a[:, 0]
    return a


def test_ledoit_wolf_matches_sklearn(window: np.ndarray) -> None:
    sigma, delta = ledoit_wolf_cov(window)
    ref, shrink = ledoit_wolf(window)
    assert delta == pytest.approx(shrink, abs=1e-12)
    assert np.allclose(sigma, ref, atol=1e-12, rtol=0)


def test_simple_rules(window: np.ndarray) -> None:
    assert np.allclose(equal_weight(window), 1 / 6)
    iv = inverse_vol(window)
    sd = window.std(axis=0, ddof=1)
    assert np.allclose(iv, (1 / sd) / (1 / sd).sum())
    cov = np.cov(window, rowvar=False)
    w = gmv(cov)
    assert w.sum() == pytest.approx(1.0)
    # GMV has minimal variance: any perturbation along the budget constraint increases it.
    v0 = w @ cov @ w
    e = np.zeros(6)
    e[0], e[1] = 0.01, -0.01
    assert (w + e) @ cov @ (w + e) > v0
    for rule in RULES:
        assert weights(rule, window).sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        weights("bogus", window)  # type: ignore[arg-type]


def test_hrp_properties() -> None:
    cov2 = np.array([[0.04, 0.01], [0.01, 0.01]])
    w = hrp(cov2)
    ivp = (1 / np.diag(cov2)) / (1 / np.diag(cov2)).sum()
    assert np.allclose(w, ivp)
    # Block-diagonal: two uncorrelated pairs with unit variances inside each block.
    c = np.eye(4)
    c[0, 1] = c[1, 0] = 0.9
    c[2, 3] = c[3, 2] = 0.9
    c = c * np.array([1.0, 1.0, 4.0, 4.0])[:, None] ** 0.5 * np.array([1, 1, 4, 4])[None, :] ** 0.5
    w = hrp(c)
    assert np.all(w > 0) and w.sum() == pytest.approx(1.0)
    # Hand computation: cluster {0,1} variance 0.95, cluster {2,3} variance 3.8.
    v1, v2 = 0.95, 3.8
    a = 1 - v1 / (v1 + v2)
    assert w[0] == pytest.approx(a / 2) and w[1] == pytest.approx(a / 2)
    assert w[2] == pytest.approx((1 - a) / 2) and w[3] == pytest.approx((1 - a) / 2)
    assert hrp(np.array([[2.0]]))[0] == 1.0


def test_backtest_no_look_ahead() -> None:
    rng = np.random.default_rng(1)
    idx = pd.period_range("2000-01", periods=40, freq="M").to_timestamp(how="end")
    ex = pd.DataFrame(rng.standard_normal((40, 4)) * 0.03, index=idx, columns=list("abcd"))
    res = rolling_backtest(ex, ("gmv_lw", "hrp"), window=24)
    assert len(res["hrp"]) == 16
    ex2 = ex.copy()
    ex2.iloc[30:] += 0.5  # perturb months 30.. onward
    res2 = rolling_backtest(ex2, ("gmv_lw", "hrp"), window=24)
    for rule in ("gmv_lw", "hrp"):
        w1 = res[rule].drop(columns="return")
        w2 = res2[rule].drop(columns="return")
        # weights for months up to and including index 30 use months <= 29 only
        assert np.allclose(w1.iloc[: 30 - 24 + 1], w2.iloc[: 30 - 24 + 1])
        assert not np.allclose(w1.iloc[-1], w2.iloc[-1])
    r = res["hrp"]
    assert np.allclose(
        r["return"], (r.drop(columns="return").to_numpy() * ex.iloc[24:].to_numpy()).sum(axis=1)
    )
    with pytest.raises(ValueError):
        rolling_backtest(ex.iloc[:10], ("equal",), window=24)


def test_turnover() -> None:
    idx = pd.period_range("2000-01", periods=3, freq="M").to_timestamp(how="end")
    w = pd.DataFrame([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], index=idx, columns=["a", "b"])
    r = pd.DataFrame([[0.1, -0.1], [0.0, 0.0], [0.0, 0.0]], index=idx, columns=["a", "b"])
    to = turnover(w, r)
    # After month 1 the weights drift to (0.55, 0.45); rebalancing costs 0.1.
    assert to.iloc[0] == pytest.approx(0.1)
    assert to.iloc[1] == pytest.approx(0.0)


def test_joint_diff_test_matches_single_pair() -> None:
    from sharpecheck.compare import lw_boot_test

    rng = np.random.default_rng(2)
    x = rng.standard_normal((200, 3)) * 0.04 + 0.005
    res = joint_diff_test(x, [(0, 1), (0, 2)], 5, 499, 7)
    single = lw_boot_test(x[:, 1], x[:, 0], block=5, n_boot=499, seed=7)
    assert res["delta"][0] == pytest.approx(single.details["delta"])
    assert res["se"][0] == pytest.approx(single.details["se"])
    assert np.all(res["p_rw"] >= res["p_boot"] - 1e-12)
    assert json.dumps({k: v.tolist() for k, v in res.items()})


def _kf_zip(
    path: Path, header: str, cols: list[str], values: np.ndarray, months: list[str]
) -> None:
    lines = ["This file was created using the 202608 CRSP database.", "", header]
    lines.append("," + ",".join(cols))
    lines += [
        m + "," + ",".join(f"{v:.2f}" for v in row) for m, row in zip(months, values, strict=True)
    ]
    lines += ["", ""]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(path.stem.replace("_CSV", "") + ".csv", "\r\n".join(lines))
    path.write_bytes(buf.getvalue())


def test_study_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hashlib

    from sharpecheck.portfolio import study

    rng = np.random.default_rng(3)
    months = [str(p).replace("-", "") for p in pd.period_range("1990-01", periods=60, freq="M")]
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    manifest = {}
    specs = {
        "ind10": ("  Average Value Weighted Returns -- Monthly", ["A", "B", "C"]),
        "ff25": ("  Average Value Weighted Returns -- Monthly", ["D", "E", "F", "G"]),
    }
    for name, (hdr, cols) in specs.items():
        p = raw / data.DATASETS[name]
        _kf_zip(p, hdr, cols, 1 + 4 * rng.standard_normal((60, len(cols))), months)
        manifest[name] = {
            "file": p.name,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "vintage": "202608",
            "downloaded": "test",
        }
    p = raw / data.DATASETS["ff3"]
    fac = np.column_stack([rng.standard_normal((60, 3)), np.full(60, 0.3)])
    _kf_zip(p, "", ["Mkt-RF", "SMB", "HML", "RF"], fac, months)
    manifest["ff3"] = {
        "file": p.name,
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "vintage": "202608",
        "downloaded": "test",
    }
    (tmp_path / "data" / "manifest.json").write_text(json.dumps(manifest))
    cfg_text = f"""
name = "pf-test"
seed = 1
[data]
datasets = ["ind10", "ff25"]
factors = "ff3"
section = "value-weighted"
start = "1990-01"
cache_dir = "{(tmp_path / "data").as_posix()}"
[backtest]
window = 24
rules = ["equal", "inverse_vol", "gmv_lw", "hrp"]
benchmark = "equal"
[inference]
periods = [{{ name = "full", start = "1992-01", end = "last" }},
           {{ name = "late", start = "1993-01", end = "last" }}]
primary_period = "full"
n_boot = 99
block = "cube-root"
block_sensitivity = [1, 3]
calibrate_block = false
alpha = 0.05
[run]
workers = 1
out_dir = "{(tmp_path / "run").as_posix()}"
"""
    cfg_path = tmp_path / "pf.toml"
    cfg_path.write_text(cfg_text)
    study.run(cfg_path, workers=1)
    out = study.build_report(cfg_path, tmp_path / "res")
    tests = pd.read_csv(out / "tests.csv")
    assert len(tests) == 2 * 3 * 2  # periods x rules-vs-1/N x datasets
    assert (tests["p_lw_boot_romano_wolf"] >= tests["p_lw_boot"] - 1e-12).all()
    rules = pd.read_csv(out / "rules.csv")
    assert (rules["sr_ci_lower"] < rules["sr_monthly"]).all()
    sens = pd.read_csv(out / "block_sensitivity.csv")
    assert set(sens["block"]) == {1, 3}
    summary = json.loads((out / "summary.json").read_text())
    assert summary["data"]["ind10"]["vintage"] == "202608"
