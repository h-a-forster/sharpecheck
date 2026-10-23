"""Data-generating processes with known population Sharpe ratio (milestone M5). DESIGN 6.2.

Every DGP simulates a (reps x T) batch, vectorised over replications, and is calibrated so
that the stationary mean divided by the stationary standard deviation equals the target
per-period SR exactly (closed form; no simulated truth). All coverage DGPs have unit
stationary variance, so the mean equals the SR. Burn-in for dependent processes is drawn
from the same seeded generator: GARCH uses 1000 periods; AR(1) and the regime chain start
in their stationary laws (no burn-in needed).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from sharpecheck._validation import FloatArray

DgpName = Literal["normal", "t5", "garch-t7", "ar1-0.2", "regime", "crash"]

DGPS: tuple[DgpName, ...] = ("normal", "t5", "garch-t7", "ar1-0.2", "regime", "crash")

PAIR_DGPS: tuple[str, ...] = (
    "normal-iid",
    "t6-iid",
    "normal-garch",
    "t6-garch",
    "normal-var",
    "t6-var",
)

GARCH = {"omega": 0.02, "alpha": 0.05, "beta": 0.93, "nu": 7.0, "burn": 1000}
AR_PHI = 0.2
REGIME = {"p11": 0.98, "p22": 0.90, "sd": (0.8, 2.0), "gap": -0.5}
CRASH = {"p": 0.01, "mean": -8.0, "sd": 2.0}


@dataclass(frozen=True)
class PopulationMoments:
    """Stationary moments of a DGP (closed form)."""

    mean: float
    sd: float
    sr: float
    g3: float
    g4: float
    rho1: float


def _mixture(
    probs: tuple[float, float], means: tuple[float, float], sds: tuple[float, float]
) -> tuple[float, float, float, float]:
    """Mean, variance, skewness and non-excess kurtosis of a two-component normal mixture."""
    m = sum(p * mu for p, mu in zip(probs, means, strict=True))
    d = [mu - m for mu in means]
    var = sum(p * (di * di + s * s) for p, di, s in zip(probs, d, sds, strict=True))
    m3 = sum(p * (di**3 + 3 * di * s * s) for p, di, s in zip(probs, d, sds, strict=True))
    m4 = sum(
        p * (di**4 + 6 * di * di * s * s + 3 * s**4) for p, di, s in zip(probs, d, sds, strict=True)
    )
    return m, var, m3 / var**1.5, m4 / var**2


def _regime_raw() -> tuple[float, float, float, float, float]:
    p11, p22 = REGIME["p11"], REGIME["p22"]
    pi1 = (1 - p22) / (2 - p11 - p22)
    pi2 = 1 - pi1
    sds = REGIME["sd"]
    means = (0.0, REGIME["gap"])
    m, var, g3, g4 = _mixture((pi1, pi2), means, sds)  # type: ignore[arg-type]
    acov1 = (p11 + p22 - 1) * pi1 * pi2 * (means[0] - means[1]) ** 2
    return m, var, g3, g4, acov1 / var


def _crash_raw() -> tuple[float, float, float, float]:
    p = CRASH["p"]
    return _mixture((1 - p, p), (0.0, CRASH["mean"]), (1.0, CRASH["sd"]))


def garch_t_kurtosis(alpha: float, beta: float, nu: float) -> float:
    """Unconditional non-excess kurtosis of GARCH(1,1) with standardized t(nu) innovations."""
    kz = 3 * (nu - 2) / (nu - 4)
    denom = 1 - beta**2 - 2 * alpha * beta - kz * alpha**2
    return math.inf if denom <= 0 else kz * (1 - (alpha + beta) ** 2) / denom


def population_moments(name: DgpName, sr: float) -> PopulationMoments:
    """Closed-form stationary moments of DGP ``name`` calibrated to per-period ``sr``."""
    if name == "normal":
        return PopulationMoments(sr, 1.0, sr, 0.0, 3.0, 0.0)
    if name == "t5":
        return PopulationMoments(sr, 1.0, sr, 0.0, 9.0, 0.0)
    if name == "garch-t7":
        k = garch_t_kurtosis(GARCH["alpha"], GARCH["beta"], GARCH["nu"])
        return PopulationMoments(sr, 1.0, sr, 0.0, k, 0.0)
    if name == "ar1-0.2":
        return PopulationMoments(sr, 1.0, sr, 0.0, 3.0, AR_PHI)
    if name == "regime":
        _, _, g3, g4, rho1 = _regime_raw()
        return PopulationMoments(sr, 1.0, sr, g3, g4, rho1)
    if name == "crash":
        _, _, g3, g4 = _crash_raw()
        return PopulationMoments(sr, 1.0, sr, g3, g4, 0.0)
    raise ValueError(f"unknown DGP {name!r}; choose from {DGPS}")


def _std_t(rng: np.random.Generator, nu: float, size: tuple[int, ...]) -> FloatArray:
    return rng.standard_t(nu, size=size) * math.sqrt((nu - 2) / nu)


def simulate(
    name: DgpName, sr: float, nobs: int, reps: int, rng: np.random.Generator
) -> FloatArray:
    """(reps x nobs) excess returns from DGP ``name`` with population SR ``sr``."""
    shape = (reps, nobs)
    if name == "normal":
        return sr + rng.standard_normal(shape)
    if name == "t5":
        return sr + _std_t(rng, 5.0, shape)
    if name == "garch-t7":
        om, al, be, nu = GARCH["omega"], GARCH["alpha"], GARCH["beta"], GARCH["nu"]
        burn = int(GARCH["burn"])
        z = _std_t(rng, nu, (nobs + burn, reps))
        h = np.full(reps, om / (1 - al - be))
        out = np.empty((nobs + burn, reps))
        for t in range(nobs + burn):
            e = np.sqrt(h) * z[t]
            out[t] = e
            h = om + al * e * e + be * h
        return sr + out[burn:].T
    if name == "ar1-0.2":
        phi = AR_PHI
        e = rng.standard_normal((nobs, reps))
        out = np.empty((nobs, reps))
        out[0] = e[0]
        s = math.sqrt(1 - phi * phi)
        for t in range(1, nobs):
            out[t] = phi * out[t - 1] + s * e[t]
        return sr + out.T
    if name == "regime":
        m, var, *_ = _regime_raw()
        p11, p22 = REGIME["p11"], REGIME["p22"]
        pi1 = (1 - p22) / (2 - p11 - p22)
        sd1, sd2 = REGIME["sd"]
        mu1, mu2 = 0.0, REGIME["gap"]
        u = rng.random((nobs, reps))
        state = np.empty((nobs, reps), dtype=bool)  # True = state 1
        state[0] = u[0] < pi1
        for t in range(1, nobs):
            stay1 = u[t] < p11
            go1 = u[t] >= p22  # from state 2, move to 1 with probability 1 - p22
            state[t] = np.where(state[t - 1], stay1, go1)
        z = rng.standard_normal((nobs, reps))
        x = np.where(state, mu1 + sd1 * z, mu2 + sd2 * z)
        return (sr + (x - m) / math.sqrt(var)).T
    if name == "crash":
        m, var, *_ = _crash_raw()
        crash = rng.random(shape) < CRASH["p"]
        z = rng.standard_normal(shape)
        x = np.where(crash, CRASH["mean"] + CRASH["sd"] * z, z)
        return sr + (x - m) / math.sqrt(var)
    raise ValueError(f"unknown DGP {name!r}; choose from {DGPS}")


def _bivariate(
    rng: np.random.Generator, size: tuple[int, ...], corr: float, dist: str
) -> FloatArray:
    """Draws of shape size + (2,) with unit variances and correlation ``corr``."""
    z1 = rng.standard_normal(size)
    z2 = corr * z1 + math.sqrt(1 - corr * corr) * rng.standard_normal(size)
    z = np.stack([z1, z2], axis=-1)
    if dist == "t6":
        w = rng.chisquare(6.0, size=size) / 6.0
        z = z / np.sqrt(w)[..., None] * math.sqrt(4.0 / 6.0)
    return z


# Ledoit and Wolf (2008) section 4.2 diagonal-vech GARCH(1,1) parameters.
LW_GARCH = {
    "c": ((0.15, 0.13), (0.13, 0.15)),
    "a": ((0.075, 0.05), (0.05, 0.075)),
    "b": ((0.90, 0.89), (0.89, 0.90)),
    "mean": 16.5 / 52,
    "burn": 1000,
}


def simulate_pair(name: str, nobs: int, reps: int, rng: np.random.Generator) -> FloatArray:
    """(reps x nobs x 2) paired returns with equal SR: the six Ledoit-Wolf (2008) DGPs.

    IID: means 1, variances 1, correlation 0.5 (normal or bivariate t6 scaled to unit
    variance). VAR: the IID cases plus AR(1) coefficient 0.2 in each series, scaled so the
    *stationary* law has means 1, variances 1 and correlation 0.5 (DESIGN 6.3.2 reading).
    GARCH: diagonal-vech bivariate GARCH(1,1) with LW's C, A, B, mean 16.5/52, innovations
    bivariate normal or standardized t6 with identity scale, 1000 burn-in periods.
    """
    if name not in PAIR_DGPS:
        raise ValueError(f"unknown pair DGP {name!r}; choose from {PAIR_DGPS}")
    dist, kind = name.split("-")
    if kind == "iid":
        return 1.0 + _bivariate(rng, (reps, nobs), 0.5, dist)
    if kind == "var":
        phi = 0.2
        u = _bivariate(rng, (nobs, reps), 0.5, dist) * math.sqrt(1 - phi * phi)
        y = np.empty_like(u)
        y[0] = _bivariate(rng, (reps,), 0.5, "normal")  # stationary start (Gaussian)
        for t in range(1, nobs):
            y[t] = phi * y[t - 1] + u[t]
        return 1.0 + np.transpose(y, (1, 0, 2))
    c, a, b = (np.array(LW_GARCH[k]) for k in ("c", "a", "b"))
    burn = int(LW_GARCH["burn"])
    total = nobs + burn
    if dist == "normal":
        z = rng.standard_normal((total, reps, 2))
    else:
        z = _std_t(rng, 6.0, (total, reps, 2))
    h11 = np.full(reps, c[0, 0] / (1 - a[0, 0] - b[0, 0]))
    h22 = np.full(reps, c[1, 1] / (1 - a[1, 1] - b[1, 1]))
    h12 = np.full(reps, c[0, 1] / (1 - a[0, 1] - b[0, 1]))
    out = np.empty((total, reps, 2))
    for t in range(total):
        # Cholesky of [[h11, h12], [h12, h22]].
        l11 = np.sqrt(h11)
        l21 = h12 / l11
        l22 = np.sqrt(np.maximum(h22 - l21 * l21, 0.0))
        e1 = l11 * z[t, :, 0]
        e2 = l21 * z[t, :, 0] + l22 * z[t, :, 1]
        out[t, :, 0] = e1
        out[t, :, 1] = e2
        h11 = c[0, 0] + a[0, 0] * e1 * e1 + b[0, 0] * h11
        h22 = c[1, 1] + a[1, 1] * e2 * e2 + b[1, 1] * h22
        h12 = c[0, 1] + a[0, 1] * e1 * e2 + b[0, 1] * h12
    return LW_GARCH["mean"] + np.transpose(out[burn:], (1, 0, 2))


def simulate_trials(
    n_trials: int,
    nobs: int,
    rho: float,
    rng: np.random.Generator,
    *,
    dist: Literal["normal", "t5"] = "normal",
) -> FloatArray:
    """(nobs x n_trials) zero-SR trial returns with equicorrelation ``rho`` (one factor).

    ``X_i = sqrt(rho) F + sqrt(1 - rho) e_i`` with ``F`` and ``e_i`` iid standard normal or
    standardized t5, so every trial has mean 0, variance 1 and pairwise correlation ``rho``.
    """
    if not 0.0 <= rho < 1.0:
        raise ValueError(f"rho must be in [0, 1), got {rho}")
    if dist == "normal":
        f = rng.standard_normal((nobs, 1))
        e = rng.standard_normal((nobs, n_trials))
    elif dist == "t5":
        f = _std_t(rng, 5.0, (nobs, 1))
        e = _std_t(rng, 5.0, (nobs, n_trials))
    else:
        raise ValueError(f"dist must be 'normal' or 't5', got {dist!r}")
    return math.sqrt(rho) * f + math.sqrt(1 - rho) * e
