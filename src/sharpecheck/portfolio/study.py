"""The "1/N versus optimised" study on Ken French data (milestone M7). DESIGN 5.

``run`` downloads (or reuses) the data, runs one checkpointed backtest unit per
dataset x rule, and stores the out-of-sample returns. ``build_report`` applies the
pre-registered inference (DESIGN 5.4) and writes ``results/<name>/``:

- primary: LW studentized circular block bootstrap, two-sided, Romano-Wolf adjusted over the
  4 rules x 2 datasets comparisons against 1/N from one joint bootstrap of all series
- secondary: JKM and LW HAC p-values with Holm, block-size sensitivity, per-rule summaries
"""

from __future__ import annotations

import json
import math
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sharpecheck.checkpoint import UnitStore, config_hash, run_units

FACTORS = {"ff3": "ff3"}


def load_config(path: Path) -> dict[str, Any]:
    with Path(path).open("rb") as fh:
        cfg = tomllib.load(fh)
    for key in ("name", "seed", "data", "backtest", "inference", "run"):
        if key not in cfg:
            raise ValueError(f"portfolio config is missing '{key}'")
    if cfg["backtest"]["benchmark"] not in cfg["backtest"]["rules"]:
        raise ValueError("backtest.benchmark must be one of backtest.rules")
    return cfg


def fetch_all(config_path: Path, *, force: bool = False) -> dict[str, dict[str, Any]]:
    """Download every dataset a portfolio config needs; return the manifest entries."""
    from sharpecheck import data

    cfg = load_config(config_path)
    cache = Path(cfg["data"]["cache_dir"])
    names = [*cfg["data"]["datasets"], cfg["data"]["factors"]]
    for name in names:
        data.fetch(name, cache, force=force)
    manifest = data.read_manifest(cache)
    return {n: manifest[n] for n in names}


def load_excess(cfg: dict[str, Any], name: str) -> pd.DataFrame:
    """Excess returns (decimal) of one dataset from the configured start month."""
    from sharpecheck import data

    cache = Path(cfg["data"]["cache_dir"])
    port = data.load_monthly(name, cache, cfg["data"]["section"])
    fac = data.load_monthly(cfg["data"]["factors"], cache)
    ex = data.excess_returns(port, fac)
    start = pd.Period(cfg["data"]["start"], "M").to_timestamp(how="end").normalize()
    ex = ex[ex.index >= start]
    if ex.isna().to_numpy().any():
        raise ValueError(f"{name} has missing values after {cfg['data']['start']}")
    return ex


def _backtest_unit(a: dict[str, Any]) -> dict[str, Any]:
    from sharpecheck.portfolio.backtest import rolling_backtest, turnover

    ex = a["excess"]
    res = rolling_backtest(ex, (a["rule"],), int(a["window"]))[a["rule"]]
    w = res.drop(columns="return")
    to = turnover(w, ex)
    return {
        "months": [d.strftime("%Y-%m") for d in res.index],
        "returns": res["return"].tolist(),
        "mean_turnover": float(to.mean()),
    }


def _store(cfg: dict[str, Any]) -> tuple[UnitStore, dict[str, Any]]:
    from sharpecheck import __version__, data

    manifest = data.read_manifest(Path(cfg["data"]["cache_dir"]))
    names = [*cfg["data"]["datasets"], cfg["data"]["factors"]]
    missing = [n for n in names if n not in manifest]
    if missing:
        raise RuntimeError(f"data not fetched: {missing}; run `sharpecheck data fetch` first")
    inputs = {n: manifest[n]["sha256"] for n in names}
    inputs["package_version"] = __version__
    store = UnitStore(Path(cfg["run"]["out_dir"]), config_hash(cfg), inputs=inputs)
    return store, {n: manifest[n] for n in names}


def run(config_path: Path, *, workers: int = 3) -> Path:
    """Run (or resume) all backtest units; return the run directory."""
    cfg = load_config(config_path)
    store, _ = _store(cfg)
    units = []
    for name in cfg["data"]["datasets"]:
        ex = load_excess(cfg, name)
        for rule in cfg["backtest"]["rules"]:
            units.append(
                (
                    f"backtest|{name}|{rule}",
                    {"excess": ex, "rule": rule, "window": cfg["backtest"]["window"]},
                )
            )
    stats_ = run_units(store, units, _backtest_unit, workers=workers, on_error="log")
    store._append_log(f"run finished: {json.dumps(stats_)}")
    if stats_["failed"]:
        raise RuntimeError(f"{stats_['failed']} backtest unit(s) failed")
    return Path(cfg["run"]["out_dir"])


def _period_mask(months: pd.PeriodIndex, start: str, end: str) -> np.ndarray:
    lo = pd.Period(start, "M")
    hi = months.max() if end == "last" else pd.Period(end, "M")
    return np.asarray((months >= lo) & (months <= hi))


def joint_diff_test(
    x: np.ndarray, pairs: list[tuple[int, int]], block: int, n_boot: int, seed: int
) -> dict[str, np.ndarray]:
    """LW studentized bootstrap for several (benchmark, rule) column pairs of ``x``.

    All pairs share the resampled time indices. Returns ``delta, se, t_obs, p_boot,
    p_rw`` (Romano-Wolf adjusted) arrays over pairs.
    """
    from sharpecheck.bootstrap import _projected_block_sums, circular_block_indices
    from sharpecheck.compare import lw_hac_se
    from sharpecheck.multiple import romano_wolf

    t = x.shape[0]
    rng = np.random.default_rng(seed)
    k = len(pairs)
    delta, se = np.empty(k), np.empty(k)
    for i, (b0, j) in enumerate(pairs):
        delta[i], se[i] = lw_hac_se(x[:, j], x[:, b0], prewhiten=True)
    idx = circular_block_indices(t, block, n_boot, rng)
    sr_star, h = _projected_block_sums(x[idx], block)
    tstar = np.empty((n_boot, k))
    for i, (b0, j) in enumerate(pairs):
        d_star = sr_star[:, j] - sr_star[:, b0]
        hd = h[:, :, j] - h[:, :, b0]
        se_star = np.sqrt(np.mean(hd * hd, axis=1) / t)
        tstar[:, i] = (d_star - delta[i]) / se_star
    t_obs = delta / se
    a = np.abs(np.where(np.isfinite(tstar), tstar, 0.0))
    p_boot = (1 + np.sum(a >= np.abs(t_obs)[None, :], axis=0)) / (n_boot + 1)
    return {
        "delta": delta,
        "se": se,
        "t_obs": t_obs,
        "p_boot": p_boot,
        "p_rw": romano_wolf(t_obs, tstar),
    }


def build_report(config_path: Path, out_dir: Path | None = None) -> Path:
    """Apply the pre-registered inference and write ``results/<name>/``."""
    from sharpecheck.bootstrap import default_block
    from sharpecheck.compare import jkm_test, lw_hac_test
    from sharpecheck.multiple import holm
    from sharpecheck.single import sharpe_ci

    cfg = load_config(config_path)
    store, manifest = _store(cfg)
    rules = list(cfg["backtest"]["rules"])
    bench = cfg["backtest"]["benchmark"]
    datasets = list(cfg["data"]["datasets"])
    series: dict[str, pd.Series] = {}
    turn: dict[str, float] = {}
    for name in datasets:
        for rule in rules:
            uid = f"backtest|{name}|{rule}"
            if not store.done(uid):
                raise RuntimeError(f"run not finished: {uid} missing")
            rec = store.read(uid)
            idx = pd.PeriodIndex(rec["months"], freq="M")  # type: ignore[arg-type]
            series[f"{name}:{rule}"] = pd.Series(rec["returns"], index=idx)  # type: ignore[arg-type]
            turn[f"{name}:{rule}"] = float(rec["mean_turnover"])  # type: ignore[arg-type]
    panel = pd.DataFrame(series)
    if panel.isna().to_numpy().any():
        raise RuntimeError("out-of-sample months differ across datasets")
    cols = list(panel.columns)
    pairs = [
        (cols.index(f"{d}:{bench}"), cols.index(f"{d}:{r}"))
        for d in datasets
        for r in rules
        if r != bench
    ]
    labels = [cols[j] for _, j in pairs]
    inf = cfg["inference"]
    n_boot, seed = int(inf["n_boot"]), int(cfg["seed"])
    out = Path(out_dir) if out_dir is not None else Path("results") / str(cfg["name"])
    out.mkdir(parents=True, exist_ok=True)
    test_rows, summary_rows, sens_rows = [], [], []
    for period in inf["periods"]:
        mask = _period_mask(panel.index, period["start"], period["end"])
        x = panel.to_numpy()[mask]
        t = x.shape[0]
        blk = default_block(t)
        res = joint_diff_test(x, pairs, blk, n_boot, seed)
        jkm_p = np.array([jkm_test(x[:, j], x[:, b]).pvalue for b, j in pairs])
        hac_p = np.array([lw_hac_test(x[:, j], x[:, b]).pvalue for b, j in pairs])
        jkm_h, hac_h = holm(jkm_p), holm(hac_p)
        months = panel.index[mask]
        for i, lab in enumerate(labels):
            d, r = lab.split(":")
            test_rows.append(
                {
                    "period": period["name"],
                    "start": str(months[0]),
                    "end": str(months[-1]),
                    "months": t,
                    "dataset": d,
                    "rule": r,
                    "delta_sr_monthly": res["delta"][i],
                    "se": res["se"][i],
                    "t_stat": res["t_obs"][i],
                    "p_lw_boot": res["p_boot"][i],
                    "p_lw_boot_romano_wolf": res["p_rw"][i],
                    "p_jkm": jkm_p[i],
                    "p_jkm_holm": jkm_h[i],
                    "p_lw_hac_pw": hac_p[i],
                    "p_lw_hac_pw_holm": hac_h[i],
                    "block": blk,
                    "n_boot": n_boot,
                    "primary": period["name"] == inf["primary_period"],
                }
            )
        for c_i, col in enumerate(cols):
            v = x[:, c_i]
            ci = sharpe_ci(v, "boot_ts", n_boot=n_boot, seed=seed)
            d, r = col.split(":")
            summary_rows.append(
                {
                    "period": period["name"],
                    "dataset": d,
                    "rule": r,
                    "months": t,
                    "mean_monthly": float(v.mean()),
                    "sd_monthly": float(v.std()),
                    "sr_monthly": ci.estimate,
                    "sr_ci_lower": ci.lower,
                    "sr_ci_upper": ci.upper,
                    "sr_annualized_x_sqrt12": ci.estimate * math.sqrt(12),
                    "mean_turnover_full_oos": turn[col],
                }
            )
        if period["name"] == inf["primary_period"]:
            for b in inf.get("block_sensitivity", []):
                rb = joint_diff_test(x, pairs, int(b), n_boot, seed)
                for i, lab in enumerate(labels):
                    d, r = lab.split(":")
                    sens_rows.append(
                        {
                            "block": int(b),
                            "dataset": d,
                            "rule": r,
                            "p_lw_boot": rb["p_boot"][i],
                            "p_lw_boot_romano_wolf": rb["p_rw"][i],
                        }
                    )
    tests = pd.DataFrame(test_rows)
    tests.to_csv(out / "tests.csv", index=False, float_format="%.6g")
    pd.DataFrame(summary_rows).to_csv(out / "rules.csv", index=False, float_format="%.6g")
    if sens_rows:
        pd.DataFrame(sens_rows).to_csv(
            out / "block_sensitivity.csv", index=False, float_format="%.6g"
        )
    prim = tests[tests["primary"]]
    alpha = float(inf["alpha"])
    sig = prim[prim["p_lw_boot_romano_wolf"] < alpha]
    summary = {
        "primary_claim": {
            "period": inf["primary_period"],
            "alpha": alpha,
            "rules_different_from_1N": [
                f"{a}:{b}" for a, b in zip(sig["dataset"], sig["rule"], strict=True)
            ],
        },
        "data": {
            k: {kk: v[kk] for kk in ("file", "sha256", "vintage", "downloaded")}
            for k, v in manifest.items()
        },
        "versions": store.manifest.get("versions"),
        "config": cfg,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return out
