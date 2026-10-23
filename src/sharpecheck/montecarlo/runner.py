"""Monte Carlo runner (milestones M5, M6). DESIGN 6.

Three experiments, each split into checkpointed units:

- ``coverage``: single-SR intervals, unit = (dgp, sr, T, chunk)
- ``difference``: Ledoit-Wolf (2008) Table 1 replication, unit = (dgp, chunk)
- ``dsr``: DSR size under correlated and miscounted trials, unit = (N, rho, T, dist, chunk)

Seeds derive from ``numpy.random.SeedSequence(config seed, spawn_key=...)`` keyed by a
SHA-256 hash of the unit id, so a unit's result does not depend on which other units ran or
on the worker count. Within a unit all methods see the same simulated data.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from sharpecheck.checkpoint import MAX_WORKERS, UnitStore, config_hash, run_units

COVERAGE_METHODS = ("iid_normal", "exact_nct", "mertens", "ar1", "hac", "hac_pw", "boot_ts")
DIFF_METHODS = (
    "jkm",
    "hac",
    "hac_pw",
    "boot_iid",
    "boot_ts_b1",
    "boot_ts_b2",
    "boot_ts_b4",
    "boot_ts_b6",
    "boot_ts_b8",
    "boot_ts_b10",
)
DSR_RULES = (
    "dsr_trueN_estV",
    "dsr_trueN_oracleV",
    "dsr_N/10",
    "dsr_impliedN",
    "psr_best",
    "holm_psr",
    "lzl_max",
)
EXPERIMENTS = ("coverage", "difference", "dsr")


# --------------------------------------------------------------------------- config


def load_config(path: Path) -> dict[str, Any]:
    """Read and validate a Monte Carlo TOML config."""
    with Path(path).open("rb") as fh:
        cfg = tomllib.load(fh)
    return validate_config(cfg)


def validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Check required keys and value ranges; return the config unchanged."""
    from sharpecheck.montecarlo.dgps import DGPS, PAIR_DGPS

    for key in ("name", "seed", "level", "chunk", "run"):
        if key not in cfg:
            raise ValueError(f"config is missing '{key}'")
    if int(cfg["run"].get("workers", 1)) > MAX_WORKERS:
        raise ValueError(f"run.workers must be <= {MAX_WORKERS}")
    if not 0 < float(cfg["level"]) < 1:
        raise ValueError("level must be in (0, 1)")
    if int(cfg["chunk"]) < 1:
        raise ValueError("chunk must be positive")
    cov = cfg.get("coverage")
    if cov is not None:
        bad = set(cov["dgps"]) - set(DGPS)
        if bad:
            raise ValueError(f"unknown coverage dgps {sorted(bad)}")
        bad = set(cov["methods"]) - set(COVERAGE_METHODS)
        if bad:
            raise ValueError(f"unknown coverage methods {sorted(bad)}")
        if int(cov["boot_reps"]) > int(cov["reps"]):
            raise ValueError("coverage.boot_reps must not exceed coverage.reps")
        if cov.get("block", "cube-root") != "cube-root":
            raise ValueError("coverage.block must be 'cube-root'")
    dif = cfg.get("difference")
    if dif is not None:
        bad = set(dif["dgps"]) - set(PAIR_DGPS)
        if bad:
            raise ValueError(f"unknown difference dgps {sorted(bad)}")
        bad = set(dif["methods"]) - set(DIFF_METHODS)
        if bad:
            raise ValueError(f"unknown difference methods {sorted(bad)}")
    dsr = cfg.get("dsr")
    if dsr is not None:
        bad = set(dsr["rules"]) - set(DSR_RULES)
        if bad:
            raise ValueError(f"unknown dsr rules {sorted(bad)}")
        if any(not 0 <= r < 1 for r in dsr["rho"]):
            raise ValueError("dsr.rho values must be in [0, 1)")
    return cfg


def unit_seed(seed: int, unit_id: str) -> np.random.SeedSequence:
    """Seed sequence for one unit: the config seed plus a stable hash of the unit id."""
    h = hashlib.sha256(unit_id.encode()).digest()
    key = tuple(int.from_bytes(h[i : i + 4], "little") for i in range(0, 16, 4))
    return np.random.SeedSequence(int(seed), spawn_key=key)


def _chunks(total: int, chunk: int) -> list[int]:
    sizes = [chunk] * (total // chunk)
    if total % chunk:
        sizes.append(total % chunk)
    return sizes


def build_units(cfg: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """All ``(unit_id, args)`` pairs of a config, in a stable order."""
    units: list[tuple[str, dict[str, Any]]] = []
    seed, level, chunk = int(cfg["seed"]), float(cfg["level"]), int(cfg["chunk"])
    cov = cfg.get("coverage")
    if cov is not None:
        for dgp in cov["dgps"]:
            for sr in cov["sr"]:
                for t in cov["nobs"]:
                    done = 0
                    for c, n in enumerate(_chunks(int(cov["reps"]), chunk)):
                        n_boot_reps = max(0, min(n, int(cov["boot_reps"]) - done))
                        done += n
                        uid = f"coverage|{dgp}|sr={sr}|T={t}|c={c}"
                        units.append(
                            (
                                uid,
                                {
                                    "experiment": "coverage",
                                    "uid": uid,
                                    "seed": seed,
                                    "dgp": dgp,
                                    "sr": float(sr),
                                    "nobs": int(t),
                                    "reps": n,
                                    "boot_reps": n_boot_reps,
                                    "n_boot": int(cov["n_boot"]),
                                    "methods": list(cov["methods"]),
                                    "level": level,
                                },
                            )
                        )
    dif = cfg.get("difference")
    if dif is not None:
        for dgp in dif["dgps"]:
            for c, n in enumerate(_chunks(int(dif["reps"]), chunk)):
                uid = f"difference|{dgp}|c={c}"
                units.append(
                    (
                        uid,
                        {
                            "experiment": "difference",
                            "uid": uid,
                            "seed": seed,
                            "dgp": dgp,
                            "nobs": int(dif["nobs"]),
                            "reps": n,
                            "n_boot": int(dif["n_boot"]),
                            "methods": list(dif["methods"]),
                            "alphas": [float(a) for a in dif["alphas"]],
                        },
                    )
                )
    dsr = cfg.get("dsr")
    if dsr is not None:
        for n_trials in dsr["n_trials"]:
            for rho in dsr["rho"]:
                for t in dsr["nobs"]:
                    for dist in dsr["dists"]:
                        for c, n in enumerate(_chunks(int(dsr["reps"]), chunk)):
                            uid = f"dsr|N={n_trials}|rho={rho}|T={t}|{dist}|c={c}"
                            units.append(
                                (
                                    uid,
                                    {
                                        "experiment": "dsr",
                                        "uid": uid,
                                        "seed": seed,
                                        "n_trials": int(n_trials),
                                        "rho": float(rho),
                                        "nobs": int(t),
                                        "dist": dist,
                                        "reps": n,
                                        "rules": list(dsr["rules"]),
                                        "alpha": float(dsr["alpha"]),
                                    },
                                )
                            )
    return units


# --------------------------------------------------------------------------- coverage


def _zero_counts() -> dict[str, float]:
    return {
        "n": 0,
        "cover": 0,
        "miss_low": 0,
        "miss_high": 0,
        "lower_viol": 0,
        "nan": 0,
        "width_sum": 0.0,
        "ratio_sum": 0.0,
        "ratio_n": 0,
    }


def _tally(
    counts: dict[str, float],
    truth: float,
    lo2: np.ndarray,
    hi2: np.ndarray,
    lo1: np.ndarray,
    ref_width: np.ndarray,
) -> None:
    """Accumulate coverage counts; ``nan`` intervals count as non-covering."""
    ok = np.isfinite(lo2) & np.isfinite(hi2)
    counts["n"] += int(lo2.size)
    counts["nan"] += int(np.sum(~ok))
    counts["cover"] += int(np.sum(ok & (lo2 <= truth) & (truth <= hi2)))
    counts["miss_low"] += int(np.sum(ok & (hi2 < truth)))  # interval entirely below truth
    counts["miss_high"] += int(np.sum(ok & (lo2 > truth)))  # interval entirely above truth
    counts["lower_viol"] += int(np.sum(np.isfinite(lo1) & (lo1 > truth)))
    w = hi2 - lo2
    counts["width_sum"] += float(np.sum(w[ok]))
    r_ok = ok & np.isfinite(ref_width) & (ref_width > 0)
    counts["ratio_sum"] += float(np.sum(w[r_ok] / ref_width[r_ok]))
    counts["ratio_n"] += int(np.sum(r_ok))


def _coverage_unit(a: Mapping[str, Any]) -> dict[str, Any]:
    from sharpecheck.bootstrap import default_block, sr_tstar
    from sharpecheck.hac import long_run_cov
    from sharpecheck.moments import batch_moments
    from sharpecheck.montecarlo.dgps import simulate
    from sharpecheck.single import RHO_CLIP, nct_delta_roots, variance_terms

    rng = np.random.default_rng(unit_seed(a["seed"], a["uid"]))
    t, truth, level = int(a["nobs"]), float(a["sr"]), float(a["level"])
    reps, boot_reps = int(a["reps"]), int(a["boot_reps"])
    methods = list(a["methods"])
    x = simulate(a["dgp"], truth, t, reps, rng)
    boot_rng = np.random.default_rng(rng.integers(2**63))
    m = batch_moments(x)
    sr, g3, g4, rho1 = m["sr"], m["g3"], m["g4"], m["rho1"]
    z2 = float(stats.norm.ppf(1 - (1 - level) / 2))
    z1 = float(stats.norm.ppf(level))
    out: dict[str, Any] = {"methods": {}, "timing": {}}
    t0 = time.perf_counter()

    def normal(se: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        se = np.where(np.isfinite(se) & (se > 0), se, np.nan)
        return sr - z2 * se, sr + z2 * se, sr - z1 * se

    with np.errstate(invalid="ignore", divide="ignore"):
        se_iid = np.sqrt((1 + sr * sr / 2) / t)
        lo_iid, hi_iid, l1_iid = normal(se_iid)
        ref_w = hi_iid - lo_iid
        intervals: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        if "iid_normal" in methods:
            intervals["iid_normal"] = (lo_iid, hi_iid, l1_iid)
        if "mertens" in methods:
            v = 1 - g3 * sr + (g4 - 1) / 4 * sr * sr
            intervals["mertens"] = normal(np.sqrt(np.where(v > 0, v, np.nan) / t))
        if "ar1" in methods:
            rho = np.clip(rho1, -RHO_CLIP, RHO_CLIP)
            aa, bb, cc = (np.empty_like(rho) for _ in range(3))
            for i, r in enumerate(rho):
                if np.isfinite(r):
                    aa[i], bb[i], cc[i] = variance_terms(float(r))
                else:
                    aa[i] = bb[i] = cc[i] = np.nan
            v = aa - bb * g3 * sr + cc * (g4 - 1) / 4 * sr * sr
            intervals["ar1"] = normal(np.sqrt(np.where(v > 0, v, np.nan) / t))
    out["timing"]["analytic"] = time.perf_counter() - t0
    if "exact_nct" in methods:
        t0 = time.perf_counter()
        tstat = np.where(np.isfinite(sr), sr, 0.0) * math.sqrt(t - 1)
        probs = np.array([1 - (1 - level) / 2, (1 - level) / 2, level])
        d, ok = nct_delta_roots(tstat[:, None], t - 1, probs[None, :], iterations=50)
        b = d / math.sqrt(t)
        # Fallback: normal approximation where the root was not bracketed.
        fb = ~ok
        b[:, 0] = np.where(fb[:, 0], lo_iid, b[:, 0])
        b[:, 1] = np.where(fb[:, 1], hi_iid, b[:, 1])
        b[:, 2] = np.where(fb[:, 2], l1_iid, b[:, 2])
        bad = ~np.isfinite(sr)
        b[bad] = np.nan
        intervals["exact_nct"] = (b[:, 0], b[:, 1], b[:, 2])
        out["nct_fallback"] = int(np.sum(fb))
        out["timing"]["exact_nct"] = time.perf_counter() - t0
    se_pw = np.full(reps, np.nan)
    need_pw = "hac_pw" in methods or ("boot_ts" in methods and boot_reps > 0)
    t0 = time.perf_counter()
    for name, pw in (("hac", False), ("hac_pw", True)):
        if name not in methods and not (pw and need_pw):
            continue
        se = np.full(reps, np.nan)
        for i in range(reps):
            r = x[i]
            mu = r.mean()
            gam = np.mean(r * r)
            var = gam - mu * mu
            if not var > 0:
                continue
            grad = np.array([gam / var**1.5, -mu / (2 * var**1.5)])
            y = np.column_stack([r - mu, r * r - gam])
            try:
                psi = long_run_cov(y, "qs", prewhiten=pw, dof_adjust=2)
            except np.linalg.LinAlgError:
                continue
            q = float(grad @ psi @ grad)
            if math.isfinite(q) and q > 0:
                se[i] = math.sqrt(q / t)
        if pw:
            se_pw = se
        if name in methods:
            intervals[name] = normal(se)
    out["timing"]["hac"] = time.perf_counter() - t0
    if "boot_ts" in methods and boot_reps > 0:
        t0 = time.perf_counter()
        blk = default_block(t)
        lo2 = np.full(boot_reps, np.nan)
        hi2 = np.full(boot_reps, np.nan)
        lo1 = np.full(boot_reps, np.nan)
        for i in range(boot_reps):
            if not (np.isfinite(se_pw[i]) and np.isfinite(sr[i])):
                continue
            ts = sr_tstar(x[i], blk, int(a["n_boot"]), boot_rng, float(sr[i]))
            ts = ts[np.isfinite(ts)]
            if ts.size == 0:
                continue
            q2 = float(np.quantile(np.abs(ts), level))
            q1 = float(np.quantile(ts, level))
            lo2[i] = sr[i] - q2 * se_pw[i]
            hi2[i] = sr[i] + q2 * se_pw[i]
            lo1[i] = sr[i] - q1 * se_pw[i]
        intervals["boot_ts"] = (lo2, hi2, lo1)
        out["timing"]["boot_ts"] = time.perf_counter() - t0
        out["timing"]["boot_per_rep"] = out["timing"]["boot_ts"] / boot_reps
    for name, (lo2, hi2, lo1) in intervals.items():
        c = _zero_counts()
        n = lo2.size
        _tally(c, truth, lo2, hi2, lo1, ref_w[:n])
        out["methods"][name] = c
    # PSR one-sided tests of H0: SR <= truth, both conventions (reject when PSR > level).
    with np.errstate(invalid="ignore", divide="ignore"):
        v_obs = 1 - g3 * sr + (g4 - 1) / 4 * sr * sr
        p_obs = stats.norm.cdf((sr - truth) * np.sqrt((t - 1) / np.where(v_obs > 0, v_obs, np.nan)))
        v_null = 1 - g3 * truth + (g4 - 1) / 4 * truth * truth
        p_null = stats.norm.cdf((sr - truth) * np.sqrt(t / np.where(v_null > 0, v_null, np.nan)))
    out["psr"] = {
        "observed": {
            "n": reps,
            "reject": int(np.sum(p_obs > level)),
            "nan": int(np.sum(np.isnan(p_obs))),
        },
        "null": {
            "n": reps,
            "reject": int(np.sum(p_null > level)),
            "nan": int(np.sum(np.isnan(p_null))),
        },
    }
    fin = np.isfinite(sr)
    out["sr_hat"] = {
        "n": int(fin.sum()),
        "sum": float(sr[fin].sum()),
        "sumsq": float((sr[fin] ** 2).sum()),
    }
    return out


# --------------------------------------------------------------------------- difference


def _difference_unit(a: Mapping[str, Any]) -> dict[str, Any]:
    from sharpecheck.bootstrap import diff_tstar
    from sharpecheck.compare import jkm_test, lw_hac_se
    from sharpecheck.montecarlo.dgps import simulate_pair

    rng = np.random.default_rng(unit_seed(a["seed"], a["uid"]))
    t, reps, n_boot = int(a["nobs"]), int(a["reps"]), int(a["n_boot"])
    methods, alphas = list(a["methods"]), list(a["alphas"])
    x = simulate_pair(a["dgp"], t, reps, rng)
    boot_rng = np.random.default_rng(rng.integers(2**63))
    pv = {m: np.full(reps, np.nan) for m in methods}
    for i in range(reps):
        r1, r2 = x[i, :, 0], x[i, :, 1]
        if "jkm" in methods:
            pv["jkm"][i] = jkm_test(r1, r2).pvalue
        if "hac" in methods:
            d, se = lw_hac_se(r1, r2, prewhiten=False)
            if math.isfinite(se) and se > 0:
                pv["hac"][i] = 2 * stats.norm.sf(abs(d / se))
        boot = [m for m in methods if m.startswith("boot")]
        if "hac_pw" in methods or boot:
            d, se = lw_hac_se(r1, r2, prewhiten=True)
            ok = math.isfinite(se) and se > 0
            if "hac_pw" in methods and ok:
                pv["hac_pw"][i] = 2 * stats.norm.sf(abs(d / se))
            if ok:
                pair = np.column_stack([r2, r1])  # SR(col 1) - SR(col 0) = SR1 - SR2
                for m in boot:
                    blk = 1 if m == "boot_iid" else int(m.rsplit("_b", 1)[1])
                    ts = np.abs(diff_tstar(pair, blk, n_boot, boot_rng, np.array([d]))[:, 0])
                    ts = ts[np.isfinite(ts)]
                    if ts.size:
                        pv[m][i] = (1 + np.sum(ts >= abs(d) / se)) / (ts.size + 1)
    out: dict[str, Any] = {"methods": {}}
    for m, p in pv.items():
        fin = np.isfinite(p)
        out["methods"][m] = {
            "n": reps,
            "nan": int(np.sum(~fin)),
            "reject": {f"{al:g}": int(np.sum(fin & (p <= al))) for al in alphas},
        }
    return out


# --------------------------------------------------------------------------- dsr


def _dsr_unit(a: Mapping[str, Any]) -> dict[str, Any]:
    from sharpecheck.deflated import expected_max_sr, implied_trials, var_max_sr
    from sharpecheck.montecarlo.dgps import simulate_trials

    rng = np.random.default_rng(unit_seed(a["seed"], a["uid"]))
    n, rho, t, reps = int(a["n_trials"]), float(a["rho"]), int(a["nobs"]), int(a["reps"])
    alpha, rules = float(a["alpha"]), list(a["rules"])
    z = float(stats.norm.ppf(1 - alpha))
    e_lzl = expected_max_sr(n, 1.0 / t, rho=rho, method="exact")
    sd_lzl = math.sqrt(var_max_sr(n, 1.0 / t, rho=rho))
    rej = dict.fromkeys(rules, 0)
    nan = dict.fromkeys(rules, 0)
    implied = 0.0

    def psr_obs(srv: np.ndarray, sr0: float, g3v: np.ndarray, g4v: np.ndarray) -> np.ndarray:
        v = 1 - g3v * srv + (g4v - 1) / 4 * srv * srv
        with np.errstate(invalid="ignore", divide="ignore"):
            return stats.norm.cdf((srv - sr0) * np.sqrt((t - 1) / np.where(v > 0, v, np.nan)))

    for _ in range(reps):
        x = simulate_trials(n, t, rho, rng, dist=a["dist"])
        mu = x.mean(axis=0)
        d = x - mu
        d2 = d * d
        m2 = d2.mean(axis=0)
        sd = np.sqrt(m2)
        srs = mu / sd
        g3 = np.einsum("ij,ij->j", d2, d) / t / m2**1.5
        g4 = np.einsum("ij,ij->j", d2, d2) / t / m2**2
        k = int(np.argmax(srs))
        sb, g3b, g4b = srs[k : k + 1], g3[k : k + 1], g4[k : k + 1]
        v_est = float(np.var(srs, ddof=1)) if n > 1 else 1.0 / t
        zs = (d / sd).sum(axis=1)
        avg_corr = (float(zs @ zs) / t - n) / (n * (n - 1)) if n > 1 else 0.0
        n_hat = implied_trials(n, avg_corr)
        implied += n_hat

        def decide(rule: str, value: float) -> None:
            if math.isnan(value):
                nan[rule] += 1
            elif value:
                rej[rule] += 1

        for rule in rules:
            if rule == "dsr_trueN_estV":
                p = psr_obs(sb, expected_max_sr(n, v_est), g3b, g4b)[0]
                decide(rule, math.nan if math.isnan(p) else float(p > 1 - alpha))
            elif rule == "dsr_trueN_oracleV":
                p = psr_obs(sb, expected_max_sr(n, 1.0 / t), g3b, g4b)[0]
                decide(rule, math.nan if math.isnan(p) else float(p > 1 - alpha))
            elif rule == "dsr_N/10":
                p = psr_obs(sb, expected_max_sr(max(1, n // 10), v_est), g3b, g4b)[0]
                decide(rule, math.nan if math.isnan(p) else float(p > 1 - alpha))
            elif rule == "dsr_impliedN":
                p = psr_obs(sb, expected_max_sr(n_hat, v_est), g3b, g4b)[0]
                decide(rule, math.nan if math.isnan(p) else float(p > 1 - alpha))
            elif rule == "psr_best":
                p = psr_obs(sb, 0.0, g3b, g4b)[0]
                decide(rule, math.nan if math.isnan(p) else float(p > 1 - alpha))
            elif rule == "holm_psr":
                pv = 1 - psr_obs(srs, 0.0, g3, g4)
                pmin = float(np.nanmin(pv)) if np.any(np.isfinite(pv)) else math.nan
                decide(rule, math.nan if math.isnan(pmin) else float(pmin <= alpha / n))
            elif rule == "lzl_max":
                decide(rule, float((srs[k] - e_lzl) / sd_lzl > z))
    return {
        "rules": {r: {"n": reps, "reject": rej[r], "nan": nan[r]} for r in rules},
        "implied_n_mean": implied / reps,
    }


def work(a: Mapping[str, Any]) -> dict[str, Any]:
    """Run one unit (module level so worker processes can pickle it)."""
    exp = a["experiment"]
    if exp == "coverage":
        return _coverage_unit(a)
    if exp == "difference":
        return _difference_unit(a)
    if exp == "dsr":
        return _dsr_unit(a)
    raise ValueError(f"unknown experiment {exp!r}")


# --------------------------------------------------------------------------- driver


def run_dir(cfg: Mapping[str, Any], base: Path | None = None) -> Path:
    out = Path(cfg["run"]["out_dir"])
    return out if base is None or out.is_absolute() else Path(base) / out


def open_store(cfg: Mapping[str, Any], root: Path, *, force_new: bool = False) -> UnitStore:
    from sharpecheck import __version__

    return UnitStore(
        root,
        config_hash(cfg),
        force_new=force_new,
        inputs={"package_version": __version__},
    )


def run(
    config_path: Path,
    *,
    workers: int = 3,
    force_new: bool = False,
    max_minutes: float | None = None,
    experiments: tuple[str, ...] = EXPERIMENTS,
    out_dir: Path | None = None,
) -> Path:
    """Run (or resume) all units of a config; return the run directory."""
    cfg = load_config(config_path)
    root = Path(out_dir) if out_dir is not None else run_dir(cfg)
    return run_config(
        cfg,
        root,
        workers=workers,
        force_new=force_new,
        max_minutes=max_minutes,
        experiments=experiments,
    )


def run_config(
    cfg: dict[str, Any],
    root: Path,
    *,
    workers: int = 3,
    force_new: bool = False,
    max_minutes: float | None = None,
    experiments: tuple[str, ...] = EXPERIMENTS,
) -> Path:
    """Run (or resume) the units of an already-loaded config into ``root``."""
    store = open_store(cfg, root, force_new=force_new)
    units = [(u, a) for u, a in build_units(cfg) if a["experiment"] in experiments]
    stats_ = run_units(
        store,
        units,
        work,
        workers=workers,
        on_error="log",
        max_seconds=None if max_minutes is None else 60.0 * max_minutes,
    )
    store._append_log(f"run finished: {json.dumps(stats_)}")
    if stats_["failed"]:
        raise RuntimeError(f"{stats_['failed']} unit(s) failed; see {root / 'progress.log'}")
    return root


def pilot_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Shrunken copy of a config for the timing pilot (DESIGN 6.4)."""
    p = copy.deepcopy(dict(cfg))
    p["name"] = f"{cfg['name']}-pilot"
    p["chunk"] = 50
    p["run"] = {**cfg["run"], "out_dir": f"{cfg['run']['out_dir']}-pilot"}
    if "coverage" in p:
        p["coverage"]["reps"] = 50
        p["coverage"]["boot_reps"] = 50
    if "difference" in p:
        p["difference"]["reps"] = 20
    if "dsr" in p:
        p["dsr"]["reps"] = 10
    return p


def pilot(
    config_path: Path, *, workers: int = 3, results_dir: Path | None = None
) -> dict[str, Any]:
    """Timing pilot: run a shrunken config, project the full run, write calibration.json."""
    cfg = load_config(config_path)
    pcfg = pilot_config(cfg)
    root = run_dir(pcfg)
    t0 = time.perf_counter()
    run_config(pcfg, root, workers=workers)
    wall = time.perf_counter() - t0
    store = UnitStore(root, config_hash(pcfg), read_only=True)
    per_unit: dict[str, float] = {}
    for u, _a in build_units(pcfg):
        rec = store.read(u)
        per_unit[u] = float(rec["seconds"])  # type: ignore[arg-type]
    projection = 0.0
    detail: dict[str, float] = {}
    for _u, a in build_units(cfg):
        exp = a["experiment"]
        if exp == "coverage":
            pu = f"coverage|{a['dgp']}|sr={a['sr']}|T={a['nobs']}|c=0"
            rec = store.read(pu)
            tm = rec["timing"]  # type: ignore[index]
            boot = float(tm.get("boot_per_rep", 0.0))  # type: ignore[union-attr]
            other = (float(rec["seconds"]) - boot * 50) / 50  # type: ignore[arg-type]
            est = a["reps"] * max(other, 0.0) + a["boot_reps"] * boot
        elif exp == "difference":
            pu = f"difference|{a['dgp']}|c=0"
            est = per_unit[pu] / 20 * a["reps"]
        else:
            pu = f"dsr|N={a['n_trials']}|rho={a['rho']}|T={a['nobs']}|{a['dist']}|c=0"
            est = per_unit[pu] / 10 * a["reps"]
        projection += est
        detail[exp] = detail.get(exp, 0.0) + est
    w = max(1, min(int(workers), MAX_WORKERS))
    out = {
        "config": str(Path(config_path).as_posix()),
        "pilot_wall_seconds": round(wall, 1),
        "workers": w,
        "projected_cpu_minutes": round(projection / 60, 1),
        "projected_cpu_minutes_by_experiment": {k: round(v / 60, 1) for k, v in detail.items()},
        "projected_wall_minutes": round(projection / 60 / w, 1),
        "note": "per-unit seconds measured in worker processes on a shared machine",
    }
    rdir = Path(results_dir) if results_dir is not None else Path("results") / str(cfg["name"])
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "calibration.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out
