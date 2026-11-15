"""Option-chain snapshot: download (yfinance), save/load (CSV) and quote cleaning.

A snapshot is one point in time. Yahoo quotes are delayed and, outside trading
hours, bids/asks are frequently zero or stale: always check the filter counts
that ``clean_chain`` reports before trusting a surface built from it.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

CHAIN_COLUMNS = [
    "ticker", "snapshot_time", "spot", "market_state", "contract_symbol",
    "expiry", "kind", "strike", "bid", "ask", "last", "volume",
    "open_interest", "last_trade_time", "yf_iv",
]
_RENAME = {
    "contractSymbol": "contract_symbol", "lastPrice": "last",
    "openInterest": "open_interest", "lastTradeDate": "last_trade_time",
    "impliedVolatility": "yf_iv",
}
EXPIRY_TZ = "America/New_York"
EXPIRY_HOUR = 16
SECONDS_PER_YEAR = 365.0 * 24 * 3600


def fetch_chain(ticker: str = "^SPX", max_expiries: int | None = None,
                pause: float = 0.25) -> pd.DataFrame:
    """Download the listed option chain of ``ticker`` from yfinance, long format.

    One row per contract with columns ``CHAIN_COLUMNS``. ``snapshot_time`` is the
    UTC wall-clock time of the download. ``spot`` is Yahoo's ``regularMarketPrice``
    of the underlying, returned in the same response as the first expiry (for
    indices a 15-minute-delayed quote; when the market is closed, the last
    regular-session price). ``market_state`` is Yahoo's flag (REGULAR, CLOSED, ...).
    ``yf_iv`` is Yahoo's own implied vol, kept for comparison only.
    """
    import yfinance as yf

    if max_expiries is not None and max_expiries < 1:
        raise ValueError("max_expiries must be >= 1 or None")
    tk = yf.Ticker(ticker)
    expiries = list(tk.options)
    if not expiries:
        raise RuntimeError(f"yfinance lists no option expiries for {ticker}")
    if max_expiries is not None:
        expiries = expiries[:max_expiries]

    snapshot_time = pd.Timestamp.now(tz="UTC").floor("s")
    spot, market_state, frames = np.nan, "", []
    for i, expiry in enumerate(expiries):
        oc = tk.option_chain(expiry)
        if i == 0:
            spot = float(oc.underlying["regularMarketPrice"])
            market_state = str(oc.underlying.get("marketState", ""))
        for kind, part in (("call", oc.calls), ("put", oc.puts)):
            part = part.rename(columns=_RENAME)
            part["expiry"] = pd.Timestamp(expiry)
            part["kind"] = kind
            frames.append(part)
        time.sleep(pause)

    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    df["ticker"] = ticker
    df["snapshot_time"] = snapshot_time
    df["spot"] = spot
    df["market_state"] = market_state
    df["last_trade_time"] = pd.to_datetime(df["last_trade_time"], utc=True)
    for col in ("strike", "bid", "ask", "last", "volume", "open_interest", "yf_iv"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[CHAIN_COLUMNS].sort_values(["expiry", "kind", "strike"], ignore_index=True)
    return df


def save_chain(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a chain snapshot to CSV (timestamps as ISO-8601 strings)."""
    missing = [c for c in CHAIN_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"chain frame is missing columns: {missing}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_chain(path: str | Path) -> pd.DataFrame:
    """Read a chain CSV written by ``save_chain``, restoring datetime columns."""
    df = pd.read_csv(path)
    missing = [c for c in CHAIN_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    df["expiry"] = pd.to_datetime(df["expiry"])
    df["snapshot_time"] = pd.to_datetime(df["snapshot_time"], utc=True)
    df["last_trade_time"] = pd.to_datetime(df["last_trade_time"], utc=True)
    df["market_state"] = df["market_state"].fillna("").astype(str)
    return df


def time_to_expiry(expiry: pd.Series, snapshot_time: pd.Series) -> pd.Series:
    """T in years = (expiry date at 16:00 New York - snapshot time) / 365 calendar days.

    Simplification: every contract is treated as PM-settled. AM-settled SPX
    monthlies actually fix at the open of the third Friday, about 6.5h earlier.
    """
    exp = pd.to_datetime(expiry).dt.normalize() + pd.Timedelta(hours=EXPIRY_HOUR)
    exp_utc = exp.dt.tz_localize(EXPIRY_TZ).dt.tz_convert("UTC")
    snap = pd.to_datetime(snapshot_time, utc=True)
    return (exp_utc - snap).dt.total_seconds() / SECONDS_PER_YEAR


def clean_chain(
    df: pd.DataFrame,
    spot: float,
    min_bid: float = 0.05,
    max_rel_spread: float = 0.25,
    min_dte: float = 7.0,
    max_dte: float = 365.0,
    moneyness_range: tuple[float, float] = (-0.30, 0.30),
) -> pd.DataFrame:
    """Add mid, T, dte, log_moneyness, rel_spread and drop unusable quotes.

    mid = (bid+ask)/2; T per ``time_to_expiry``; dte = 365*T;
    log_moneyness = ln(strike/spot) (spot, not forward); rel_spread = (ask-bid)/mid.
    Filters run in this order and each row is counted under the first one it fails:
    missing_quote (NaN bid/ask), low_bid (bid < min_bid, so zero bids),
    crossed (ask < bid), wide_spread (rel_spread > max_rel_spread),
    dte_too_short (< min_dte), dte_too_long (> max_dte),
    moneyness (log_moneyness outside moneyness_range).
    Counts are in ``result.attrs["filter_counts"]`` (keys: input, each filter,
    output); ``filter_report(result)`` renders them as a table.
    """
    if not np.isfinite(spot) or spot <= 0:
        raise ValueError("spot must be a positive finite number")
    if min_bid < 0 or max_rel_spread <= 0:
        raise ValueError("min_bid must be >= 0 and max_rel_spread > 0")
    if min_dte > max_dte:
        raise ValueError("min_dte must not exceed max_dte")
    lo, hi = moneyness_range
    if lo >= hi:
        raise ValueError("moneyness_range must be (low, high) with low < high")
    need = ["expiry", "kind", "strike", "bid", "ask", "snapshot_time"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"chain frame is missing columns: {missing}")
    bad_kind = set(df["kind"].unique()) - {"call", "put"}
    if bad_kind:
        raise ValueError(f"kind must be 'call' or 'put', got {sorted(bad_kind)}")

    out = df.copy()
    out["mid"] = 0.5 * (out["bid"] + out["ask"])
    out["T"] = time_to_expiry(out["expiry"], out["snapshot_time"])
    out["dte"] = out["T"] * 365.0
    out["log_moneyness"] = np.log(out["strike"] / spot)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["rel_spread"] = (out["ask"] - out["bid"]) / out["mid"]

    # each mask is True for rows that FAIL the filter
    filters = [
        ("missing_quote", out["bid"].isna() | out["ask"].isna()),
        ("low_bid", ~(out["bid"] >= min_bid) | ~(out["bid"] > 0)),
        ("crossed", out["ask"] < out["bid"]),
        ("wide_spread", ~(out["rel_spread"] <= max_rel_spread)),
        ("dte_too_short", out["dte"] < min_dte),
        ("dte_too_long", out["dte"] > max_dte),
        ("moneyness", ~out["log_moneyness"].between(lo, hi)),
    ]
    counts = {"input": int(len(out))}
    keep = pd.Series(True, index=out.index)
    for name, fails in filters:
        hit = keep & fails
        counts[name] = int(hit.sum())
        keep &= ~fails
    out = out[keep].reset_index(drop=True)
    counts["output"] = int(len(out))
    out.attrs["filter_counts"] = counts
    out.attrs["filter_params"] = {
        "spot": float(spot), "min_bid": min_bid, "max_rel_spread": max_rel_spread,
        "min_dte": min_dte, "max_dte": max_dte, "moneyness_range": (lo, hi),
    }
    return out


def filter_report(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Table of rows removed by each ``clean_chain`` filter and rows remaining after it."""
    counts = cleaned.attrs.get("filter_counts")
    if counts is None:
        raise ValueError("frame has no filter_counts; pass the output of clean_chain")
    rows, remaining = [], counts["input"]
    for name, n in counts.items():
        if name in ("input", "output"):
            continue
        remaining -= n
        rows.append({"filter": name, "removed": n, "remaining": remaining})
    return pd.DataFrame(rows)
