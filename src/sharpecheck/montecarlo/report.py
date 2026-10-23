"""Monte Carlo report: tables with Wilson intervals and the coverage heatmap (M6). DESIGN 6.5.

Read-only on the run directory; writes ``results/<name>/`` (CSV, JSON summary, PNG
heatmap when matplotlib is available). Fails cleanly when the run is not finished.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from sharpecheck.checkpoint import UnitStore, config_hash
from sharpecheck.multiple import holm

#: Ledoit and Wolf (2008) Table 1, nominal 5%, T = 120, 5000 replications (percent).
#: Columns: JKM, HAC, HACPW, Boot-IID, Boot-TS (calibrated block size).
LW_TABLE1_5PCT: dict[str, dict[str, float]] = {
    "normal-iid": {"jkm": 5.0, "hac": 5.3, "hac_pw": 5.4, "boot_iid": 4.9, "boot_ts": 4.8},
    "t6-iid": {"jkm": 10.7, "hac": 6.7, "hac_pw": 6.9, "boot_iid": 5.2, "boot_ts": 5.0},
    "normal-garch": {"jkm": 7.2, "hac": 7.1, "hac_pw": 7.2, "boot_iid": 6.0, "boot_ts": 5.5},
    "t6-garch": {"jkm": 7.4, "hac": 7.7, "hac_pw": 7.5, "boot_iid": 6.9, "boot_ts": 5.7},
    "normal-var": {"jkm": 9.5, "hac": 6.9, "hac_pw": 6.1, "boot_iid": 8.5, "boot_ts": 5.0},
    "t6-var": {"jkm": 14.5, "hac": 7.9, "hac_pw": 7.3, "boot_iid": 7.3, "boot_ts": 5.1},
}
LW_REPS = 5000

UNDER_MATERIAL = 0.93
OVER_MATERIAL = 0.97


def wilson_interval(successes: int, trials: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if trials <= 0:
        return math.nan, math.nan
    if not 0 <= successes <= trials:
        raise ValueError("successes must be in [0, trials]")
    z = float(stats.norm.ppf(1 - (1 - level) / 2))
    p = successes / trials
    den = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / den
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def _rate(k: int, n: int, prefix: str) -> dict[str, float]:
    lo, hi = wilson_interval(k, n)
    return {prefix: k / n if n else math.nan, f"{prefix}_lo": lo, f"{prefix}_hi": hi}


def _load(config_path: Path) -> tuple[dict[str, Any], UnitStore, list[tuple[str, dict[str, Any]]]]:
    from sharpecheck.montecarlo.runner import build_units, load_config, run_dir

    cfg = load_config(config_path)
    store = UnitStore(run_dir(cfg), config_hash(cfg), read_only=True)
    units = build_units(cfg)
    missing = [u for u, _ in units if not store.done(u)]
    if missing:
        raise RuntimeError(
            f"run not finished: {len(missing)} of {len(units)} units missing (e.g. {missing[0]})"
        )
    return cfg, store, units


def coverage_table(store: UnitStore, units: list[tuple[str, dict[str, Any]]]) -> pd.DataFrame:
    """One row per (dgp, sr, T, method) with coverage, one-sided size, widths and flags."""
    agg: dict[tuple[Any, ...], dict[str, float]] = {}
    psr: dict[tuple[Any, ...], dict[str, float]] = {}
    for uid, a in units:
        if a["experiment"] != "coverage":
            continue
        rec = store.read(uid)
        cell = (a["dgp"], a["sr"], a["nobs"])
        for m, c in rec["methods"].items():  # type: ignore[union-attr]
            acc = agg.setdefault((*cell, m), {})
            for k, v in c.items():
                acc[k] = acc.get(k, 0) + v
        for conv, c in rec["psr"].items():  # type: ignore[union-attr]
            acc = psr.setdefault((*cell, f"psr_{conv}"), {})
            for k, v in c.items():
                acc[k] = acc.get(k, 0) + v
    rows = []
    for (dgp, sr, t, m), c in agg.items():
        n = int(c["n"])
        row: dict[str, Any] = {"dgp": dgp, "sr": sr, "nobs": t, "method": m, "reps": n}
        row.update(_rate(int(c["cover"]), n, "coverage"))
        row["miss_low"] = c["miss_low"] / n
        row["miss_high"] = c["miss_high"] / n
        row.update(_rate(int(c["lower_viol"]), n, "onesided_size"))
        row["nan"] = int(c["nan"])
        ok = n - c["nan"]
        row["mean_width"] = c["width_sum"] / ok if ok else math.nan
        row["mean_width_ratio_vs_iid"] = c["ratio_sum"] / c["ratio_n"] if c["ratio_n"] else math.nan
        row["p_under"] = float(stats.binom.cdf(int(c["cover"]), n, 0.95))
        row["p_over"] = float(stats.binom.sf(int(c["cover"]) - 1, n, 0.95))
        rows.append(row)
    df = pd.DataFrame(rows)
    df["p_under_holm"] = holm(df["p_under"].to_numpy())
    df["p_over_holm"] = holm(df["p_over"].to_numpy())
    df["flag_under"] = (df["p_under_holm"] < 0.05) & (df["coverage"] < UNDER_MATERIAL)
    df["flag_over"] = (df["p_over_holm"] < 0.05) & (df["coverage"] > OVER_MATERIAL)
    psr_rows = []
    for (dgp, sr, t, m), c in psr.items():
        n = int(c["n"])
        r: dict[str, Any] = {"dgp": dgp, "sr": sr, "nobs": t, "method": m, "reps": n}
        r.update(_rate(int(c["reject"]), n, "onesided_size"))
        r["nan"] = int(c["nan"])
        psr_rows.append(r)
    df.attrs["psr"] = pd.DataFrame(psr_rows)
    return df.sort_values(["dgp", "sr", "nobs", "method"]).reset_index(drop=True)


def onesided_table(cov: pd.DataFrame) -> pd.DataFrame:
    """One-sided 95% lower-bound violation rates (size of H0: SR <= SR_true), incl. PSR."""
    a = cov[
        [
            "dgp",
            "sr",
            "nobs",
            "method",
            "reps",
            "onesided_size",
            "onesided_size_lo",
            "onesided_size_hi",
            "nan",
        ]
    ]
    b = cov.attrs["psr"]
    out = pd.concat([a, b], ignore_index=True)
    return out.sort_values(["dgp", "sr", "nobs", "method"]).reset_index(drop=True)


def difference_table(store: UnitStore, units: list[tuple[str, dict[str, Any]]]) -> pd.DataFrame:
    agg: dict[tuple[str, str], dict[str, Any]] = {}
    for uid, a in units:
        if a["experiment"] != "difference":
            continue
        rec = store.read(uid)
        for m, c in rec["methods"].items():  # type: ignore[union-attr]
            acc = agg.setdefault((a["dgp"], m), {"n": 0, "nan": 0, "reject": {}})
            acc["n"] += c["n"]
            acc["nan"] += c["nan"]
            for al, k in c["reject"].items():
                acc["reject"][al] = acc["reject"].get(al, 0) + k
    rows = []
    for (dgp, m), c in agg.items():
        for al, k in c["reject"].items():
            row: dict[str, Any] = {"dgp": dgp, "method": m, "alpha": float(al), "reps": c["n"]}
            row.update(_rate(int(k), int(c["n"]), "rejection_rate"))
            row["nan"] = c["nan"]
            lw_key = "boot_ts" if m.startswith("boot_ts") else m
            lw = LW_TABLE1_5PCT.get(dgp, {}).get(lw_key)
            if lw is not None and abs(float(al) - 0.05) < 1e-12:
                ours = k / c["n"]
                p = (ours + lw / 100) / 2
                tol = 3 * math.sqrt(p * (1 - p) * (1 / c["n"] + 1 / LW_REPS))
                row["lw2008"] = lw / 100
                row["agree_tol"] = tol
                row["agree"] = bool(abs(ours - lw / 100) <= tol)
                row["note"] = (
                    "LW Boot-TS used a calibrated block; ours is fixed"
                    if lw_key == "boot_ts"
                    else ""
                )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["dgp", "method", "alpha"]).reset_index(drop=True)


def dsr_table(store: UnitStore, units: list[tuple[str, dict[str, Any]]]) -> pd.DataFrame:
    agg: dict[tuple[Any, ...], dict[str, float]] = {}
    imp: dict[tuple[Any, ...], list[float]] = {}
    for uid, a in units:
        if a["experiment"] != "dsr":
            continue
        rec = store.read(uid)
        cell = (a["n_trials"], a["rho"], a["nobs"], a["dist"])
        imp.setdefault(cell, []).append(float(rec["implied_n_mean"]) * a["reps"])  # type: ignore[arg-type]
        imp.setdefault((*cell, "n"), []).append(a["reps"])
        for r, c in rec["rules"].items():  # type: ignore[union-attr]
            acc = agg.setdefault((*cell, r), {})
            for k, v in c.items():
                acc[k] = acc.get(k, 0) + v
    rows = []
    for (n_t, rho, t, dist, rule), c in agg.items():
        cell = (n_t, rho, t, dist)
        row: dict[str, Any] = {
            "n_trials": n_t,
            "rho": rho,
            "nobs": t,
            "dist": dist,
            "rule": rule,
            "reps": int(c["n"]),
        }
        row.update(_rate(int(c["reject"]), int(c["n"]), "fwer"))
        row["nan"] = int(c["nan"])
        row["implied_n_mean"] = sum(imp[cell]) / sum(imp[(*cell, "n")])
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["n_trials", "rho", "nobs", "dist", "rule"])
        .reset_index(drop=True)
    )


def heatmap(cov: pd.DataFrame, path: Path) -> bool:
    """Coverage heatmap: rows method, columns dgp x sr, one panel per T. Returns success."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    except ImportError:  # pragma: no cover - optional dependency
        return False
    from sharpecheck.montecarlo.runner import COVERAGE_METHODS

    cmap = LinearSegmentedColormap.from_list("cov", ["#e34948", "#f0efec", "#2a78d6"])
    norm = TwoSlopeNorm(vmin=0.80, vcenter=0.95, vmax=1.0)
    methods = [m for m in COVERAGE_METHODS if m in set(cov["method"])]
    dgps = list(dict.fromkeys(cov["dgp"]))
    srs = sorted(set(cov["sr"]))
    cols = [(d, s) for d in dgps for s in srs]
    ts = sorted(set(cov["nobs"]))
    fig, axes = plt.subplots(len(ts), 1, figsize=(13, 2.35 * len(ts) + 1.2), squeeze=False)
    idx = cov.set_index(["dgp", "sr", "nobs", "method"])
    im = None
    for ax, t in zip(axes[:, 0], ts, strict=True):
        z = np.full((len(methods), len(cols)), np.nan)
        flag = np.zeros_like(z, dtype=bool)
        for i, m in enumerate(methods):
            for j, (d, s) in enumerate(cols):
                key = (d, s, t, m)
                if key in idx.index:
                    z[i, j] = idx.loc[key, "coverage"]
                    flag[i, j] = bool(idx.loc[key, "flag_under"])
        im = ax.imshow(np.clip(z, 0.80, 1.0), cmap=cmap, norm=norm, aspect="auto")
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                if np.isfinite(z[i, j]):
                    txt = f"{100 * z[i, j]:.0f}" + ("*" if flag[i, j] else "")
                    ax.text(j, i, txt, ha="center", va="center", fontsize=6.5, color="#2b2b29")
        ax.set_yticks(range(len(methods)), methods, fontsize=7.5)
        ax.set_xticks(range(len(cols)), [f"{d}\nSR={s:g}" for d, s in cols], fontsize=6.5)
        ax.set_title(f"T = {t}", fontsize=9, loc="left", color="#2b2b29")
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(cols), len(srs)), minor=True)
        ax.grid(which="minor", axis="x", color="#fcfcfb", linewidth=2)
    fig.suptitle(
        "Two-sided 95% coverage (%) of SR intervals; * = significant (Holm) and below 93%",
        fontsize=10,
        x=0.01,
        ha="left",
        color="#2b2b29",
    )
    if im is not None:
        cb = fig.colorbar(im, ax=axes[:, 0].tolist(), fraction=0.015, pad=0.01)
        cb.set_label("coverage", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    return True


def build_report(config_path: Path, out_dir: Path | None = None) -> Path:
    """Aggregate a finished run into ``results/<name>/``; return that directory."""
    cfg, store, units = _load(config_path)
    out = Path(out_dir) if out_dir is not None else Path("results") / str(cfg["name"])
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "config": cfg,
        "config_hash": store.manifest.get("config_hash"),
        "versions": store.manifest.get("versions"),
        "started": store.manifest.get("started"),
    }
    seconds = sum(float(store.read(u)["seconds"]) for u, _ in units)  # type: ignore[arg-type]
    summary["cpu_seconds_in_units"] = round(seconds, 1)
    summary["n_units"] = len(units)
    if "coverage" in cfg:
        cov = coverage_table(store, units)
        cov.to_csv(out / "coverage.csv", index=False, float_format="%.6g")
        onesided_table(cov).to_csv(out / "onesided.csv", index=False, float_format="%.6g")
        summary["coverage_cells"] = int(len(cov))
        summary["flag_under"] = int(cov["flag_under"].sum())
        summary["flag_over"] = int(cov["flag_over"].sum())
        summary["heatmap"] = heatmap(cov, out / "coverage_heatmap.png")
    if "difference" in cfg:
        dif = difference_table(store, units)
        dif.to_csv(out / "difference.csv", index=False, float_format="%.6g")
        if "agree" in dif:
            sub = dif.dropna(subset=["agree"])
            summary["lw_table1_agree"] = f"{int(sub['agree'].sum())}/{len(sub)}"
    if "dsr" in cfg:
        dsr = dsr_table(store, units)
        dsr.to_csv(out / "dsr.csv", index=False, float_format="%.6g")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return out
