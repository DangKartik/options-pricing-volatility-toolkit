"""Refresh every cached dataset in data/raw/ and save an option-chain snapshot.

Run from anywhere:  .venv/bin/python scripts/download_data.py
Prints what was downloaded, from which source, and how usable the chain quotes are.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from volkit.data import chain, market  # noqa: E402

USABLE_BID_FRACTION = 0.5  # below this the SPX snapshot is flagged and SPY is also saved


def describe_series(label: str, s: pd.Series, path: Path) -> None:
    source = pd.read_csv(path, nrows=0).columns[1]
    print(f"{label:<6} rows={len(s):>5}  {s.index[0].date()} -> {s.index[-1].date()}  "
          f"last={s.iloc[-1]:.4f}  source={source}  file={path.name}")


def us_market_open(ts: pd.Timestamp) -> bool:
    """Clock test only (Mon-Fri 09:30-16:15 New York); ignores exchange holidays."""
    ny = ts.tz_convert("America/New_York")
    minutes = ny.hour * 60 + ny.minute
    return ny.weekday() < 5 and 9 * 60 + 30 <= minutes <= 16 * 60 + 15


def describe_chain(df: pd.DataFrame, path: Path) -> float:
    """Print snapshot quality numbers; return the fraction of rows with bid > 0."""
    snap, spot = df["snapshot_time"].iloc[0], float(df["spot"].iloc[0])
    state = df["market_state"].iloc[0]
    has_bid = df["bid"] > 0
    two_sided = has_bid & (df["ask"] > df["bid"])
    print(f"\nchain {df['ticker'].iloc[0]} -> {path.name}")
    print(f"  snapshot_time (UTC) : {snap}   New York: {snap.tz_convert('America/New_York')}")
    print(f"  market open at snapshot: {us_market_open(snap)} (clock)   yahoo marketState={state!r}")
    print(f"  spot (yahoo regularMarketPrice): {spot:.2f}")
    print(f"  rows={len(df)}  expiries={df['expiry'].nunique()}  "
          f"{df['expiry'].min().date()} -> {df['expiry'].max().date()}")
    print(f"  rows with bid > 0          : {has_bid.sum()} ({has_bid.mean():.1%})")
    print(f"  rows with ask > 0          : {(df['ask'] > 0).sum()} ({(df['ask'] > 0).mean():.1%})")
    print(f"  rows with 0 < bid < ask    : {two_sided.sum()} ({two_sided.mean():.1%})")
    print(f"  rows with volume > 0       : {(df['volume'] > 0).sum()}")
    last_trade = df["last_trade_time"].max()
    print(f"  most recent last_trade_time: {last_trade}")

    # spreads near the money: |ln(K/S)| <= 5%, 7 to 90 days, two-sided quotes only
    mid = 0.5 * (df["bid"] + df["ask"])
    dte = chain.time_to_expiry(df["expiry"], df["snapshot_time"]) * 365.0
    near = (np.log(df["strike"] / spot).abs() <= 0.05) & dte.between(7, 90)
    rel = ((df["ask"] - df["bid"]) / mid)[near & two_sided]
    print(f"  near-the-money rows (|ln(K/S)|<=5%, 7-90 dte): {near.sum()}, two-sided: {len(rel)}")
    if len(rel):
        q = rel.quantile([0.25, 0.5, 0.75])
        print(f"  relative spread (ask-bid)/mid there: "
              f"q25={q[0.25]:.3%}  median={q[0.5]:.3%}  q75={q[0.75]:.3%}")

    cleaned = chain.clean_chain(df, spot)
    print("  clean_chain with default parameters:", cleaned.attrs["filter_params"])
    print("   ", cleaned.attrs["filter_counts"])
    return float(has_bid.mean())


def snapshot(ticker: str, stem: str, cache: Path) -> float:
    df = chain.fetch_chain(ticker)
    stamp = df["snapshot_time"].iloc[0].strftime("%Y%m%d")
    path = chain.save_chain(df, cache / f"{stem}_chain_{stamp}.csv")
    return describe_chain(chain.load_chain(path), path)


def main() -> int:
    cache = market.cache_dir_path()
    print(f"cache directory: {cache}\n")
    failures = []

    spx = None
    for label, loader, fname in [("SPX", market.load_spx, market.SPX_FILE),
                                 ("VIX", market.load_vix, market.VIX_FILE),
                                 ("RATES", market.load_rates, market.RATES_FILE)]:
        try:
            s = loader(refresh=True)
            describe_series(label, s, cache / fname)
            if label == "SPX":
                spx = s
            if label == "RATES" and spx is not None:
                first = max(s.index[0], spx.index[0])
                cal = spx.index[spx.index >= first]
                aligned = market.align_to_calendar(s, cal)
                print(f"       on SPX calendar: {len(aligned)} days, "
                      f"{len(cal.difference(s.index))} forward-filled")
        except Exception as exc:
            failures.append(label)
            print(f"{label:<6} FAILED: {exc!r}")

    try:
        fam = market.load_vix_family(refresh=True)
        print(f"VIXFAM rows={len(fam)}  first valid: "
              + ", ".join(f"{c}={fam[c].first_valid_index().date()}" for c in fam.columns)
              + f"  -> {fam.index[-1].date()}  file={market.VIX_FAMILY_FILE}")
    except Exception as exc:
        failures.append("VIXFAM")
        print(f"VIXFAM FAILED: {exc!r}")

    try:
        frac = snapshot("^SPX", "spx", cache)
    except Exception as exc:
        frac = 0.0
        failures.append("SPX chain")
        print(f"\nchain ^SPX FAILED: {exc!r}")
    if frac < USABLE_BID_FRACTION:
        print(f"\n^SPX snapshot has bid > 0 on only {frac:.1%} of rows: also saving SPY "
              "(ETF, American-style, pays dividends) for comparison.")
        try:
            snapshot("SPY", "spy", cache)
        except Exception as exc:
            failures.append("SPY chain")
            print(f"\nchain SPY FAILED: {exc!r}")

    print("\nfailures:", failures if failures else "none")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
