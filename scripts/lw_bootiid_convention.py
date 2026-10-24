"""Boot-IID with the two original-data standard errors (Ledoit and Wolf 2008, Table 1).

LW 2008 section 3.2.1 studentize the original Boot-IID statistic with the iid standard
error (sample covariance of the moment vectors); the main run now does the same. Up to the
0.1.0 corrections the main run used the prewhitened QS HAC standard error instead, which
removed Boot-IID's liberal size under the VAR DGPs. This script runs both variants on the
same draws to show the effect. Same DGPs, T = 120, M = 499, 2000 replications per DGP,
seeded.

    uv run python scripts/lw_bootiid_convention.py   # about 1-2 CPU-minutes

Writes results/mc-quick/bootiid_convention.json.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from sharpecheck.bootstrap import diff_tstar, iid_se_diff
from sharpecheck.compare import lw_hac_se
from sharpecheck.montecarlo.dgps import PAIR_DGPS, simulate_pair
from sharpecheck.montecarlo.report import LW_TABLE1_5PCT, wilson_interval

REPS, T, N_BOOT, SEED = 2000, 120, 499, 20261024


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out: dict[str, dict[str, object]] = {}
    for k, dgp in enumerate(PAIR_DGPS):
        rng = np.random.default_rng([SEED, k])
        x = simulate_pair(dgp, T, REPS, rng)
        rej_iid = rej_hac = 0
        for i in range(REPS):
            pair = np.column_stack([x[i, :, 1], x[i, :, 0]])
            d, se_hac = lw_hac_se(x[i, :, 0], x[i, :, 1], prewhiten=True)
            ts = np.abs(diff_tstar(pair, 1, N_BOOT, rng, np.array([d]))[:, 0])
            ts = ts[np.isfinite(ts)]
            for se, which in ((float(iid_se_diff(pair)[0]), "iid"), (se_hac, "hac")):
                if not (math.isfinite(se) and se > 0) or ts.size == 0:
                    continue
                p = (1 + np.sum(ts >= abs(d) / se)) / (ts.size + 1)
                if p <= 0.05:
                    if which == "iid":
                        rej_iid += 1
                    else:
                        rej_hac += 1
        out[dgp] = {
            "reps": REPS,
            "boot_iid_iid_se": rej_iid / REPS,
            "boot_iid_iid_se_wilson": wilson_interval(rej_iid, REPS),
            "boot_iid_hac_pw_se": rej_hac / REPS,
            "boot_iid_hac_pw_se_wilson": wilson_interval(rej_hac, REPS),
            "lw2008": LW_TABLE1_5PCT[dgp]["boot_iid"] / 100,
        }
        print(dgp, out[dgp])
    dest = root / "results" / "mc-quick" / "bootiid_convention.json"
    meta = {"T": T, "n_boot": N_BOOT, "seed": SEED, "alpha": 0.05}
    dest.write_text(json.dumps({"meta": meta, "results": out}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
