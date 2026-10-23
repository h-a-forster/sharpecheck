"""The independent reference script reproduces the published numeric examples.

These checks pin the oracle itself (scripts/reference_values.py), which later milestones use
to test the package. Published values are rounded in the papers, hence the tolerances.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "reference_values.py"


@pytest.fixture(scope="module")
def ref() -> ModuleType:
    spec = importlib.util.spec_from_file_location("reference_values", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SR, G3, G4, T = 0.036 / 0.079, -2.448, 10.164, 24


def test_lzl_2025_numeric_example(ref: ModuleType) -> None:
    assert round(math.sqrt(ref.sr_var_lzl(SR, T, G3, G4)), 3) == 0.329
    assert round(math.sqrt(ref.sr_var_lzl(SR, T)), 3) == 0.214
    assert round(ref.psr_null(SR, 0.0, T, G3, G4), 3) == 0.987
    assert round(ref.psr_null(SR, 0.1, T, G3, G4), 3) == 0.939


def test_bldp_2014_dsr_example(ref: ModuleType) -> None:
    sr = 2.5 / math.sqrt(250)
    sr0 = ref.emax_gumbel(100, 0.5 / 250)
    assert round(sr0, 4) == 0.1132
    assert round(ref.dsr(sr, sr0, 1250, -3.0, 10.0), 4) == 0.9004
    assert round(ref.dsr(sr, ref.emax_gumbel(46, 0.5 / 250), 1250, -3.0, 10.0), 4) == 0.9505


def test_bldp_2012_mintrl_example(ref: ModuleType) -> None:
    sr, sr_ref = 2.0 / 12**0.5, 1 / 12**0.5
    assert ref.psr_observed(sr, sr_ref, 59.895, -0.72, 5.78) == pytest.approx(0.95, abs=1e-5)


def test_expected_max_exact(ref: ModuleType) -> None:
    # Known values of E[max of N iid N(0,1)].
    assert ref.emax_exact(10)[0] == pytest.approx(1.538753, abs=1e-6)
    assert ref.emax_exact(100)[0] == pytest.approx(2.507594, abs=1e-6)


def test_ar1_variance_matches_delta_method(ref: ModuleType) -> None:
    rho, sr = 0.2, 0.1
    a = (1 + rho) / (1 - rho)
    c = (1 + rho**2) / (1 - rho**2)
    assert ref.sr_var_lzl(sr, 1, 0.0, 3.0, rho) == pytest.approx(a + c * sr**2 / 2, rel=1e-12)


def test_lo_eta_iid_is_sqrt_q(ref: ModuleType) -> None:
    assert ref.lo_eta(12, 0.0) == pytest.approx(math.sqrt(12), rel=1e-12)
