"""Ken French data library download, cache and parsing (milestone M7).

Source: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
(free CSV zips under ``.../ken.french/ftp/<name>_CSV.zip``). Downloads use a timeout, are
written atomically to ``<cache_dir>/raw/``, and their SHA-256 and the CRSP vintage line
("This file was created using the YYYYMM CRSP database.") are recorded in
``<cache_dir>/manifest.json``. Raw data are never committed. Parsing: DESIGN 5.2.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

DATASETS: dict[str, str] = {
    "ind10": "10_Industry_Portfolios_CSV.zip",
    "ind48": "48_Industry_Portfolios_CSV.zip",
    "ff25": "25_Portfolios_5x5_CSV.zip",
    "ff3": "F-F_Research_Data_Factors_CSV.zip",
}

SECTION_MARKERS = {
    "value-weighted": "Average Value Weighted Returns -- Monthly",
    "equal-weighted": "Average Equal Weighted Returns -- Monthly",
}
MISSING = (-99.99, -999.0)
_VINTAGE = re.compile(r"using the (\d{6}) CRSP database", re.IGNORECASE)


def _manifest_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / "manifest.json"


def read_manifest(cache_dir: Path) -> dict[str, Any]:
    p = _manifest_path(cache_dir)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _csv_text(zip_bytes: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"expected one CSV member, found {members}")
        return zf.read(members[0]).decode("latin-1")


def vintage(text: str) -> str | None:
    """The CRSP vintage (``YYYYMM``) named in a Ken French file header, if any."""
    m = _VINTAGE.search(text[:2000])
    return m.group(1) if m else None


def fetch(name: str, cache_dir: Path, *, timeout: float = 60.0, force: bool = False) -> Path:
    """Download one dataset zip into ``cache_dir/raw`` (atomic write) and record its hash."""
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    cache_dir = Path(cache_dir)
    path = cache_dir / "raw" / DATASETS[name]
    manifest = read_manifest(cache_dir)
    if path.exists() and not force and name in manifest:
        return path
    url = BASE_URL + DATASETS[name]
    req = urllib.request.Request(url, headers={"User-Agent": "sharpecheck data fetch"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https host
        data = resp.read()
    text = _csv_text(data)  # validates the archive before it is cached
    _atomic_write(path, data)
    manifest[name] = {
        "file": DATASETS[name],
        "url": url,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "vintage": vintage(text),
        "downloaded": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    _atomic_write(_manifest_path(cache_dir), (json.dumps(manifest, indent=2) + "\n").encode())
    return path


def parse_monthly(
    text: str, section: str = "value-weighted", *, factors: bool = False
) -> pd.DataFrame:
    """Parse the monthly block of a Ken French CSV text into percent values.

    Portfolio files: the block after the line containing the section marker. Factor files
    (``factors=True``): the first block, whose header starts with ``,Mkt-RF``. The block
    ends at the first blank line. Index: month-end ``Timestamp``; -99.99/-999 become NaN.
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if factors:
        start = next((i for i, ln in enumerate(lines) if ln.strip().startswith(",Mkt-RF")), None)
        if start is None:
            raise ValueError("factor header ',Mkt-RF' not found")
        header_idx = start
    else:
        if section not in SECTION_MARKERS:
            raise ValueError(f"section must be one of {sorted(SECTION_MARKERS)}")
        marker = SECTION_MARKERS[section]
        pos = next((i for i, ln in enumerate(lines) if marker in ln), None)
        if pos is None:
            raise ValueError(f"section marker {marker!r} not found")
        header_idx = pos + 1
    header = [h.strip() for h in lines[header_idx].split(",")]
    if header[0] != "" or len(header) < 2:
        raise ValueError(f"unexpected header line: {lines[header_idx]!r}")
    cols = header[1:]
    dates, rows = [], []
    for ln in lines[header_idx + 1 :]:
        if not ln.strip():
            break
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) != len(header):
            raise ValueError(f"row width {len(parts)} != header width {len(header)}: {ln!r}")
        if not re.fullmatch(r"\d{6}", parts[0]):
            raise ValueError(f"expected a YYYYMM date, got {parts[0]!r}")
        dates.append(parts[0])
        rows.append([float(v) for v in parts[1:]])
    if not rows:
        raise ValueError("empty monthly block")
    idx = pd.to_datetime(dates, format="%Y%m") + pd.offsets.MonthEnd(0)
    df = pd.DataFrame(np.array(rows), index=idx, columns=cols)
    df = df.mask(df.isin(MISSING))
    periods = idx.to_period("M")
    if not (np.diff(periods.asi8) == 1).all():
        raise ValueError("months are not consecutive")
    if not np.isfinite(df.to_numpy()[~df.isna().to_numpy()]).all():
        raise ValueError("non-finite values")
    df.index.name = "month"
    return df


def load_monthly(name: str, cache_dir: Path, section: str = "value-weighted") -> pd.DataFrame:
    """Parse the monthly section of a cached zip into percent returns indexed by month end.

    ``section`` selects "value-weighted" or "equal-weighted" for portfolio files. Missing
    values (-99.99, -999) become NaN; the caller decides how to handle them.
    """
    path = Path(cache_dir) / "raw" / DATASETS[name]
    text = _csv_text(path.read_bytes())
    return parse_monthly(text, section, factors=name.startswith("ff3"))


def excess_returns(portfolios: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Subtract the one-month T-bill rate (``RF``) month by month; inner join on dates.

    Inputs in percent; the output is in decimals.
    """
    rf = factors["RF"]
    joined = portfolios.join(rf.rename("__rf__"), how="inner")
    out = joined.drop(columns="__rf__").sub(joined["__rf__"], axis=0) / 100.0
    return out
