"""Ken French data library download, cache and parsing (milestone M7).

Source: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
(free CSV zips under ``.../ken.french/ftp/<name>_CSV.zip``). Downloads use a timeout, are
written atomically to ``data/raw/``, and their SHA-256 and the CRSP vintage line
("This file was created using the YYYYMM CRSP database.") are recorded in
``data/manifest.json``. Raw data are never committed. Parsing: DESIGN 5.2.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

DATASETS: dict[str, str] = {
    "ind10": "10_Industry_Portfolios_CSV.zip",
    "ind48": "48_Industry_Portfolios_CSV.zip",
    "ff25": "25_Portfolios_5x5_CSV.zip",
    "ff3": "F-F_Research_Data_Factors_CSV.zip",
}


def fetch(name: str, cache_dir: Path, *, timeout: float = 60.0, force: bool = False) -> Path:
    """Download one dataset zip into ``cache_dir`` (atomic write) and record its hash."""
    raise NotImplementedError("M7")


def load_monthly(name: str, cache_dir: Path, section: str = "value-weighted") -> pd.DataFrame:
    """Parse the monthly section of a cached zip into percent returns indexed by month end.

    ``section`` selects "value-weighted" or "equal-weighted" for portfolio files. Missing
    values (-99.99, -999) become NaN; the caller decides how to handle them.
    """
    raise NotImplementedError("M7")


def excess_returns(portfolios: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Subtract the one-month T-bill rate (``RF``) month by month; inner join on dates."""
    raise NotImplementedError("M7")
