# SciPy `stats.nct.cdf`: NaN and noise in the far left tail

Checked 2026-10-30. Status: **known upstream, fixed in Boost.Math `develop`, not yet in
SciPy.** No new report is needed; sharpecheck works around it in `single.nct_cdf`.

## Reproduction (SciPy 1.18.1, the latest release; NumPy 2.5.4; Windows x86-64)

```python
from scipy import stats, special

stats.nct.cdf(-1.37, 249, 11.95)  # nan
stats.nct.cdf(-1.38, 249, 11.95)  # 3.47e-40, true value 1.10e-40
stats.nct.cdf(-1.36, 249, 11.95)  # 1.06e-40, true value 1.43e-40
with special.errstate(all="raise"):
    special.nctdtr(249, 11.95, -1.37)  # SpecialFunctionError: no result obtained
```

True values come from one-dimensional quadrature of
`F(x) = E[Phi(x sqrt(V / df) - nc)]`, `V ~ chi2(df)`, in log space (relative tolerance
1e-10). Between x = -1.45 and -1.30 the SciPy values are not monotone in x, off by up to a
factor of 6, and NaN at 3 of 16 grid points.

On a grid of df in {59, 119, 249, 999, 2519}, nc in [5, 20] (step 0.25) and x in [-5, 5]
(step 0.01), 52,930 of 305,305 values (17%) are NaN. Every NaN sits where the true CDF is
below about 1e-16: the left tail is computed with an absolute, not relative, error near
machine epsilon, and a result that comes out negative is turned into NaN by SciPy's
`nct_cdf_wrap` range check. `nct.sf` at the reflected arguments, `sf(-x, df, -nc)`, has
the same values (0.0 instead of NaN at x = -1.37). Values above about 1e-16 are accurate,
so confidence-interval inversion at ordinary levels is unaffected once the NaN is handled.

## Where the defect lives

- SciPy `main` (commit `7490ad2d`, 2026-10-29): `nct_gen._cdf` calls `special.nctdtr`,
  which is `nct_cdf_wrap` in `scipy/special/boost_special_functions.h`, i.e.
  `boost::math::cdf(non_central_t_distribution(v, l), x)`; an out-of-range result becomes
  NaN with `SF_ERROR_NO_RESULT`. Same code path as 1.18.1.
- SciPy `main` pins Boost.Math at `7ff2d2b4` (Boost 1.92, 2026-08-13).
- Boost.Math issue [#1524](https://github.com/boostorg/math/issues/1524) ("non_central_t
  cdf returns a negative probability") was fixed by PR
  [#1529](https://github.com/boostorg/math/pull/1529) (merged into `develop` on
  2026-10-10: the left tail is computed by quadrature when the series cancel). That commit
  is not an ancestor of the Boost.Math commit SciPy `main` pins, so SciPy `main` still has
  the defect until its next Boost bump. Not verified by building SciPy `main`.
- SciPy issue [#25470](https://github.com/scipy/scipy/issues/25470) ("`nct.cdf` return
  `nan`", open since 2026-06-22) reports the same symptom. It is platform dependent: NaN on
  Windows and Linux, while a maintainer got a finite value on macOS.

A useful upstream action would be a comment on SciPy #25470 linking Boost.Math #1529 and
the reproduction above.

## Workaround in sharpecheck

`single.nct_cdf` replaces non-finite entries by the reflected survival function and then
by the limit (0 or 1), and clips to [0, 1]. The regression grid in
`tests/test_single.py` covers negative t statistics; the Monte Carlo reports
`fallback_bounds` in `results/mc-quick/coverage.csv` (0 after the fix).
