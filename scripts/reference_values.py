"""Independent recomputation of the reference values quoted in docs/DESIGN.md section 7.

This script deliberately does not import sharpecheck. It recomputes each value from the
published formula with numpy/scipy only, so the package tests have an oracle that was not
written from the package code. Run it with:

    uv run python scripts/reference_values.py

Published values that the script reproduces are marked "paper" in the output; values that
have no published counterpart are marked "derived" (closed form, quoted to 1e-6 or better).
"""

from __future__ import annotations

import math

from scipy import integrate, optimize, stats

EULER_GAMMA = 0.5772156649015329


def sr_var_lzl(sr: float, t: float, g3: float = 0.0, g4: float = 3.0, rho: float = 0.0) -> float:
    """Asymptotic variance of the Sharpe ratio estimator (Lopez de Prado, Lipton and
    Zoonekynd 2025, eq. for AR(1) returns; rho = 0 gives Mertens 2002 / Opdyke 2007).

    g4 is the non-excess kurtosis.
    """
    a = 1 + 2 * rho / (1 - rho)
    b = 1 + rho / (1 - rho) + rho**2 / (1 - rho**2)
    c = 1 + 2 * rho**2 / (1 - rho**2)
    return (a - b * g3 * sr + c * (g4 - 1) / 4 * sr**2) / t


def psr_null(sr: float, sr0: float, t: float, g3: float, g4: float) -> float:
    """PSR with the variance evaluated at SR0 and divisor T (2025 convention)."""
    return float(stats.norm.cdf((sr - sr0) / math.sqrt(sr_var_lzl(sr0, t, g3, g4))))


def psr_observed(sr: float, sr0: float, t: float, g3: float, g4: float) -> float:
    """PSR with the variance evaluated at the observed SR and divisor T - 1 (2012 convention)."""
    v = 1 - g3 * sr + (g4 - 1) / 4 * sr**2
    return float(stats.norm.cdf((sr - sr0) * math.sqrt(t - 1) / math.sqrt(v)))


def emax_gumbel(n: int, var: float) -> float:
    """Expected maximum of n iid N(0, var) (Bailey and Lopez de Prado 2014, eq. 1)."""
    z1 = stats.norm.ppf(1 - 1 / n)
    z2 = stats.norm.ppf(1 - 1 / (n * math.e))
    return math.sqrt(var) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def emax_exact(n: int) -> tuple[float, float]:
    """E[max] and Var[max] of n iid N(0, 1) by adaptive quadrature of n z phi(z) Phi(z)^(n-1)."""

    def dens(z: float) -> float:
        return n * stats.norm.pdf(z) * stats.norm.cdf(z) ** (n - 1)

    m1 = integrate.quad(lambda z: z * dens(z), -12, 12, limit=200)[0]
    m2 = integrate.quad(lambda z: z * z * dens(z), -12, 12, limit=200)[0]
    return m1, m2 - m1 * m1


def dsr(sr: float, sr0: float, t: int, g3: float, g4: float) -> float:
    """Deflated Sharpe ratio (Bailey and Lopez de Prado 2014, eq. 2)."""
    return psr_observed(sr, sr0, t, g3, g4)


def nct_ci(sr_hat: float, t: int, level: float = 0.95) -> tuple[float, float]:
    """Exact CI for SR under iid normal returns: sqrt(T) * SR_hat ~ nct(T - 1, sqrt(T) * SR)."""
    x = math.sqrt(t) * sr_hat
    lo_q, hi_q = (1 + level) / 2, (1 - level) / 2

    def root(q: float) -> float:
        f = lambda d: stats.nct.cdf(x, t - 1, d) - q  # noqa: E731
        return optimize.brentq(f, x - 20, x + 20, xtol=1e-12) / math.sqrt(t)

    return root(lo_q), root(hi_q)


def memmel_var(sr1: float, sr2: float, rho: float, t: int) -> float:
    """Jobson-Korkie variance of SR1 - SR2 with the Memmel (2003) correction."""
    return (2 - 2 * rho + 0.5 * (sr1**2 + sr2**2 - 2 * sr1 * sr2 * rho**2)) / t


def lo_eta(q: int, rho: float) -> float:
    """Lo (2002) annualisation factor for AR(1) returns: SR(q) = eta(q) * SR."""
    s = sum((q - k) * rho**k for k in range(1, q))
    return q / math.sqrt(q + 2 * s)


def garch_t_kurtosis(alpha: float, beta: float, nu: float) -> float:
    """Unconditional non-excess kurtosis of GARCH(1,1) with standardized t(nu) innovations."""
    kz = 3 * (nu - 2) / (nu - 4)
    denom = 1 - beta**2 - 2 * alpha * beta - kz * alpha**2
    if denom <= 0:
        return math.inf
    return kz * (1 - (alpha + beta) ** 2) / denom


def main() -> None:
    print("== PSR / MinTRL / FDR: Lopez de Prado, Lipton, Zoonekynd (2025) numeric example ==")
    sr, g3, g4, t = 0.036 / 0.079, -2.448, 10.164, 24
    print(f"SR_hat = {sr:.6f}")
    print(f"paper  sd(SR_hat) non-normal T=24       = {math.sqrt(sr_var_lzl(sr, t, g3, g4)):.6f}")
    print(f"paper  sd(SR_hat) normal T=24           = {math.sqrt(sr_var_lzl(sr, t)):.6f}")
    print(f"paper  PSR(SR0=0)  [null variance, T]   = {psr_null(sr, 0.0, t, g3, g4):.6f}")
    print(f"paper  PSR(SR0=.1) [null variance, T]   = {psr_null(sr, 0.1, t, g3, g4):.6f}")
    z = stats.norm.ppf(0.95)
    mintrl0 = sr_var_lzl(0.0, 1, g3, g4) * (z / sr) ** 2
    print(f"paper  MinTRL(SR0=0) [null var, no +1]  = {mintrl0:.6f}")
    sr_c = 0.0 + z * math.sqrt(sr_var_lzl(0.0, t, g3, g4))
    beta = stats.norm.cdf((sr_c - 0.5) / math.sqrt(sr_var_lzl(0.5, t, g3, g4)))
    print(f"paper  beta (SR0=0, SR1=.5, T=24)       = {beta:.6f}")
    p_h1 = 0.05
    pfdr = 1 / (1 + (1 - 0.315) * p_h1 / 0.05 / (1 - p_h1))
    print(f"paper  pFDR(p_H1=.05, a=.05, b=.315)    = {pfdr:.6f}")
    p0 = 1 - psr_null(sr, 0.0, t, g3, g4)
    p1 = 1 - psr_null(sr, 0.5, t, g3, g4)
    ofdr = p0 * (1 - p_h1) / (p0 * (1 - p_h1) + p1 * p_h1)
    print(f"paper  oFDR(SR0=0, SR1=.5, p_H1=.05)    = {ofdr:.6f}")
    print(f"derived PSR(0) [observed var, T-1]      = {psr_observed(sr, 0.0, t, g3, g4):.6f}")
    print(f"derived PSR(0) normal [observed, T-1]   = {psr_observed(sr, 0.0, t, 0.0, 3.0):.6f}")

    print("\n== PSR / MinTRL: Bailey and Lopez de Prado (2012) appendix A.3 inputs ==")
    mu, sd, g3b, g4b = 2.0, 12**0.5, -0.72, 5.78
    srb, sr_ref, obs = mu / sd, 1 / 12**0.5, 59.895
    v = 1 - g3b * srb + (g4b - 1) / 4 * srb**2
    mintrl = 1 + v * (z / (srb - sr_ref)) ** 2
    print(f"paper  MinTRL (4 moments)               = {mintrl:.6f}   (paper: 59.895 months)")
    psr_b = psr_observed(srb, sr_ref, obs, g3b, g4b)
    print(f"paper  PSR at T=59.895 (4 moments)      = {psr_b:.6f}")

    print("\n== DSR: Bailey and Lopez de Prado (2014) numeric example ==")
    sr_d = 2.5 / math.sqrt(250)
    for n, g3d, g4d in [(100, -3.0, 10.0), (46, -3.0, 10.0), (88, 0.0, 3.0)]:
        sr0 = emax_gumbel(n, 0.5 / 250)
        val = dsr(sr_d, sr0, 1250, g3d, g4d)
        print(f"paper  N={n:4d} g3={g3d:+.0f} g4={g4d:.0f}: SR0={sr0:.6f}  DSR={val:.6f}")

    print("\n== Expected maximum of N iid N(0,1): exact vs Gumbel approximation ==")
    for n in [10, 100, 1000]:
        m1, var = emax_exact(n)
        gum = emax_gumbel(n, 1.0)
        print(f"derived N={n:5d}: exact E={m1:.6f} Var={var:.6f}   Gumbel E={gum:.6f}")
    rho = 0.5
    m1, var = emax_exact(100)
    print(
        f"derived equicorrelated rho=0.5, N=100: E={math.sqrt(1 - rho) * m1:.6f} "
        f"Var={rho + (1 - rho) * var:.6f}"
    )

    print("\n== Single-SR asymptotic variances (T * Var) ==")
    print(f"derived iid normal, SR=0.1             = {1 + 0.1**2 / 2:.6f}")
    kt5 = 3 + 6 / (5 - 4)
    print(f"derived iid t5 (kurt {kt5:.0f}), SR=0.1      = {sr_var_lzl(0.1, 1, 0.0, kt5):.6f}")
    print(f"derived Gaussian AR(1) rho=.2, SR=0.1  = {sr_var_lzl(0.1, 1, 0.0, 3.0, 0.2):.6f}")
    check = 1.2 / 0.8 + (1 + 2 * 0.04 / 0.96) * 0.01 / 2
    print(f"derived   check a + c*SR^2/2           = {check:.6f}")

    print("\n== Exact noncentral-t CI (iid normal) ==")
    for t_, s_ in [(60, 0.2), (250, 0.05)]:
        lo, hi = nct_ci(s_, t_)
        print(f"derived T={t_}, SR_hat={s_}: 95% CI = ({lo:.6f}, {hi:.6f})")

    print("\n== Jobson-Korkie-Memmel variance of SR1 - SR2 ==")
    print(f"derived SR=(0.2, 0.1), rho=0.5, T=120: Var = {memmel_var(0.2, 0.1, 0.5, 120):.8f}")

    print("\n== Lo (2002) annualisation, monthly -> annual (q=12) ==")
    for r in [0.0, 0.1, 0.2, 0.5]:
        print(f"derived rho={r}: eta(12) = {lo_eta(12, r):.6f}  (sqrt(12) = {math.sqrt(12):.6f})")

    print("\n== GARCH(1,1)-t unconditional kurtosis (Monte Carlo DGP) ==")
    print(f"derived alpha=.05 beta=.93 nu=7: kurtosis = {garch_t_kurtosis(0.05, 0.93, 7):.6f}")

    print("\n== Ledoit and Wolf (2008) Table 1, alpha = 5%, T = 120, 5000 reps (paper) ==")
    table1 = {
        "Normal-IID": (5.0, 5.3, 5.4, 4.9, 4.8),
        "t6-IID": (10.7, 6.7, 6.9, 5.2, 5.0),
        "Normal-GARCH": (7.2, 7.1, 7.2, 6.0, 5.5),
        "t6-GARCH": (7.4, 7.7, 7.5, 6.9, 5.7),
        "Normal-VAR": (9.5, 6.9, 6.1, 8.5, 5.0),
        "t6-VAR": (14.5, 7.9, 7.3, 7.3, 5.1),
    }
    print("paper  DGP           JKM  HAC  HACPW BootIID BootTS")
    for k, row in table1.items():
        print(f"paper  {k:13s} " + " ".join(f"{x:5.1f}" for x in row))


if __name__ == "__main__":
    main()
