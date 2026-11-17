"""Daily market series (S&P 500, VIX, short rate) from free public sources.

Every series is cached as a CSV in ``data/raw/`` so the analysis runs offline.
The cache holds the raw published numbers (index levels, percent quotes);
unit conversions happen in the loaders and are documented there.

Cache directory resolution: explicit ``cache_dir`` argument, else the
``VOLKIT_DATA_DIR`` environment variable, else ``<repo>/data/raw``.
"""

from __future__ import annotations

import io
import os
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "raw"
ENV_CACHE_DIR = "VOLKIT_DATA_DIR"

START_DATE = "2004-01-01"
CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
TBILL_DAYS = 91  # nominal maturity of the 13-week bill behind DTB3 / ^IRX

SPX_FILE = "spx.csv"
VIX_FILE = "vix.csv"
RATES_FILE = "rates.csv"
VIX_FAMILY_FILE = "vix_family.csv"


def cache_dir_path(cache_dir: str | Path | None = None) -> Path:
    """Resolve the cache directory: argument > $VOLKIT_DATA_DIR > <repo>/data/raw."""
    if cache_dir is not None:
        return Path(cache_dir)
    env = os.environ.get(ENV_CACHE_DIR)
    return Path(env) if env else DEFAULT_CACHE_DIR


def require_cached(*paths: str | Path, cache_dir: str | Path | None = None, hint: str = "") -> None:
    """Exit with a clear message if any needed data file is missing.

    The raw market data is not part of the git repository (third-party data, not
    redistributed). Analysis scripts call this first so that a fresh clone fails with
    instructions instead of silently downloading new data and producing different numbers.
    Bare file names are looked up in the cache directory; other paths are used as given.
    """
    base = cache_dir_path(cache_dir)
    missing = []
    for p in paths:
        p = Path(p)
        full = p if p.is_absolute() or p.parent != Path(".") else base / p
        if not full.exists():
            missing.append(str(full))
    if missing:
        raise SystemExit(
            "Missing market data file(s):\n  " + "\n  ".join(missing) + "\n"
            "The raw data (Yahoo Finance, CBOE, FRED) is not included in this repository.\n"
            "Run `python scripts/download_data.py` first to download it into data/raw/.\n"
            "A fresh download gives different numbers from those in the README, which come "
            "from the 15-17 Nov 2026 snapshots." + (f"\n{hint}" if hint else "")
        )


# --------------------------------------------------------------------------
# download helpers (network)
# --------------------------------------------------------------------------

def _http_get(url: str, user_agent: str | None, timeout: float = 60.0) -> str:
    """GET a URL as text. FRED drops spoofed browser agents, CBOE accepts either."""
    headers = {"User-Agent": user_agent} if user_agent else {}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def _yf_close(ticker: str, start: str = START_DATE) -> pd.Series:
    """Unadjusted daily close of a Yahoo ticker as a tz-naive, date-indexed Series."""
    import yfinance as yf

    hist = yf.Ticker(ticker).history(start=start, auto_adjust=False)
    if hist.empty:
        raise RuntimeError(f"yfinance returned no rows for {ticker}")
    close = hist["Close"].copy()
    close.index = pd.DatetimeIndex(close.index.tz_localize(None).normalize())
    return close


def _cboe_close(name: str) -> pd.Series:
    """Daily close of a CBOE volatility index from the official history CSV."""
    text = _http_get(CBOE_URL.format(name=name), user_agent="Mozilla/5.0")
    raw = pd.read_csv(io.StringIO(text))
    idx = pd.to_datetime(raw["DATE"], format="%m/%d/%Y")
    return pd.Series(raw["CLOSE"].to_numpy(dtype=float), index=pd.DatetimeIndex(idx))


def _fred_series(series: str) -> pd.Series:
    """A FRED daily series via the no-login CSV endpoint (missing values -> NaN)."""
    text = _http_get(FRED_URL.format(series=series), user_agent=None)
    raw = pd.read_csv(io.StringIO(text), na_values=["."])
    if raw.shape[1] != 2 or raw.columns[1] != series:
        raise RuntimeError(f"unexpected FRED payload for {series}: {list(raw.columns)}")
    idx = pd.to_datetime(raw.iloc[:, 0])
    return pd.Series(raw.iloc[:, 1].to_numpy(dtype=float), index=pd.DatetimeIndex(idx))


def _first_working(sources: list[tuple[str, callable]]) -> tuple[pd.Series, str]:
    """Try each (label, fetcher) in order; return the first non-empty result."""
    errors = []
    for label, fetch in sources:
        try:
            s = fetch()
            if s.dropna().empty:
                raise RuntimeError("empty series")
            return s, label
        except Exception as exc:  # recorded and re-raised below if all fail
            errors.append(f"{label}: {exc!r}")
    raise RuntimeError("all sources failed -> " + " | ".join(errors))


# --------------------------------------------------------------------------
# cache I/O
# --------------------------------------------------------------------------

def _tidy(s: pd.Series, name: str) -> pd.Series:
    """Sorted, de-duplicated (keep last), NaN-free float Series named ``name``."""
    s = s.astype(float).dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s.index.name = "date"
    return s.rename(name)


def _write_cache(obj: pd.Series | pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj.to_csv(path, index_label="date")


def _read_cache_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, comment="#", parse_dates=["date"], index_col="date")
    if df.empty:
        raise ValueError(f"cache file {path} has no rows")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.astype(float)


def _read_cache_series(path: Path) -> pd.Series:
    df = _read_cache_frame(path)
    if df.shape[1] != 1:
        raise ValueError(f"cache file {path} must have exactly one value column")
    return df.iloc[:, 0].dropna()


def _load(path: Path, refresh: bool, download: callable) -> pd.Series:
    """Read ``path``; if missing or refresh=True, call download() and rewrite it."""
    if refresh or not path.exists():
        series, source = download()
        _write_cache(_tidy(series, source), path)
    return _read_cache_series(path)


# --------------------------------------------------------------------------
# public loaders
# --------------------------------------------------------------------------

def load_spx(refresh: bool = False, cache_dir: str | Path | None = None) -> pd.Series:
    """S&P 500 index daily close (^GSPC, yfinance, price index, no dividends) from 2004-01-01.

    Returns the raw index level indexed by trading date; cache file ``spx.csv``.
    """
    path = cache_dir_path(cache_dir) / SPX_FILE
    s = _load(path, refresh, lambda: (_yf_close("^GSPC"), "gspc_close"))
    return s.rename("spx")


def load_vix(refresh: bool = False, cache_dir: str | Path | None = None) -> pd.Series:
    """VIX daily close as a DECIMAL annualised volatility (index level / 100, 18.5 -> 0.185).

    Source: CBOE history CSV (from 1990), fallback yfinance ^VIX (from 2004).
    The cache ``vix.csv`` keeps the raw index level; the column name records the source.
    """
    path = cache_dir_path(cache_dir) / VIX_FILE
    sources = [("cboe_vix_close", lambda: _cboe_close("VIX")),
               ("yf_vix_close", lambda: _yf_close("^VIX"))]
    s = _load(path, refresh, lambda: _first_working(sources))
    return (s / 100.0).rename("vix")


def discount_to_continuous(discount_pct: pd.Series | float, days: int = TBILL_DAYS):
    """T-bill bank-discount quote (percent) -> continuously compounded ACT/365 rate (decimal).

    Bill price per 1 of face: P = 1 - (d/100) * days/360;  r = -ln(P) * 365/days.
    """
    price = 1.0 - (np.asarray(discount_pct, dtype=float) / 100.0) * days / 360.0
    if np.any(price <= 0.0):
        raise ValueError("discount rate implies a non-positive bill price")
    r = -np.log(price) * 365.0 / days
    if isinstance(discount_pct, pd.Series):
        return pd.Series(r, index=discount_pct.index, name=discount_pct.name)
    return float(r)


def load_rates(refresh: bool = False, cache_dir: str | Path | None = None) -> pd.Series:
    """3-month T-bill rate as a continuously compounded decimal (ACT/365), daily.

    Source: FRED DTB3 (secondary-market bank-discount rate, percent), fallback
    yfinance ^IRX (same quote convention). The cache ``rates.csv`` keeps the raw
    percent discount quote; conversion is ``discount_to_continuous`` with 91 days.
    Days with no quote (bond-market holidays) are absent, not filled: use
    ``align_to_calendar`` to forward-fill onto the SPX trading calendar.
    """
    path = cache_dir_path(cache_dir) / RATES_FILE
    sources = [("fred_dtb3_discount_pct", lambda: _fred_series("DTB3")),
               ("yf_irx_discount_pct", lambda: _yf_close("^IRX", start="1990-01-01"))]
    s = _load(path, refresh, lambda: _first_working(sources))
    return discount_to_continuous(s).rename("r")


def load_vix_family(refresh: bool = False, cache_dir: str | Path | None = None) -> pd.DataFrame:
    """VIX9D, VIX, VIX3M daily closes as decimal vols (columns vix9d, vix, vix3m).

    Source: CBOE history CSVs, fallback yfinance per index. Outer-joined on date,
    so early rows are NaN where an index did not exist yet (VIX3M starts 2009,
    VIX9D 2011); this frame is the one place NaN is allowed. Cache keeps raw levels.
    """
    path = cache_dir_path(cache_dir) / VIX_FAMILY_FILE
    if refresh or not path.exists():
        cols = {}
        for col, cboe_name, yf_name in [("vix9d", "VIX9D", "^VIX9D"),
                                        ("vix", "VIX", "^VIX"),
                                        ("vix3m", "VIX3M", "^VIX3M")]:
            s, _ = _first_working([(f"cboe_{col}", lambda n=cboe_name: _cboe_close(n)),
                                   (f"yf_{col}", lambda n=yf_name: _yf_close(n))])
            cols[col] = _tidy(s, col)
        _write_cache(pd.DataFrame(cols).sort_index(), path)
    return _read_cache_frame(path) / 100.0


def align_to_calendar(series: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """Reindex onto ``calendar`` carrying the last earlier observation forward.

    Used to put rates on the SPX trading calendar (bond holidays get the prior
    quote). Dates before the first observation raise rather than return NaN.
    """
    calendar = pd.DatetimeIndex(calendar)
    if not series.index.is_monotonic_increasing:
        raise ValueError("series index must be sorted")
    out = series.reindex(series.index.union(calendar)).ffill().reindex(calendar)
    if out.isna().any():
        raise ValueError("calendar starts before the first observation of the series")
    return out
