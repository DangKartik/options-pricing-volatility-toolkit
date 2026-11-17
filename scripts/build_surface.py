"""Build the implied-volatility surface of one option-chain snapshot.

    .venv/bin/python scripts/build_surface.py                      # SPY (default)
    .venv/bin/python scripts/build_surface.py --snapshot data/raw/spx_chain_20261115.csv

Steps: load snapshot -> valuation time -> clean_chain (filter counts printed)
-> build_surface (parity forward, OTM mid IVs, outlier rule) -> sanity checks,
European-vs-American IV comparison, synthetic American parity-bias study,
arbitrage diagnostics -> figures/*_<ticker>.png, results/surface_iv_<ticker>.csv
and the <ticker> entry of results/surface_summary.json.

Cleaning thresholds (defaults below) and why:
  min_bid 0.05      SPY ticks are $0.01; under 5c one tick is >20% of the price,
                    so the mid (and its IV) is mostly rounding.
  max_rel_spread    0.25: spread/mid above 25% means the mid is not a price
                    anyone agreed on; IV of such a mid is noise.
  min_dte 7         under a week T is a few trading days, the weekend /
                    calendar-vs-trading-time choice changes IV by several vol
                    points, and event (earnings, data) days dominate.
  max_dte 800       keep the LEAPS (~2.2y) so the term structure has a long end;
                    their spreads are wider, which the spread filter handles.
  moneyness (-0.40, 0.30) in ln(K/S): the put wing is quoted much further out
                    than the call wing; outside this range quotes are minimum-tick.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import norm  # noqa: E402

from volkit.data.chain import clean_chain, filter_report, load_chain  # noqa: E402
from volkit.data.market import (  # noqa: E402
    RATES_FILE, VIX_FAMILY_FILE, load_rates, load_vix_family, require_cached,
)
from volkit.iv.solver import implied_vol_american  # noqa: E402
from volkit.iv.surface import (  # noqa: E402
    build_surface,
    butterfly_violations,
    calendar_violations,
    european_vs_american,
    synthetic_chain,
    valuation_time,
)

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR, RES_DIR = ROOT / "figures", ROOT / "results"
DPI = 160
MIN_CLEAN_ROWS = 100       # below this there is no surface worth building
MIN_SLICES = 3
SYN_STEPS = 1000           # synthetic American chain priced on a fine tree
K_BUCKETS = [-0.45, -0.20, -0.10, -0.03, 0.03, 0.10, 0.30]


def nearest_rows(df: pd.DataFrame, col: str, targets: list[float]) -> pd.DataFrame:
    """Unique rows of df whose col is nearest each target."""
    idx = [int((df[col] - t).abs().idxmin()) for t in targets]
    return df.loc[sorted(set(idx))]


def r2(x: float, n: int = 4) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def update_summary(ticker: str, entry: dict) -> Path:
    RES_DIR.mkdir(exist_ok=True)
    path = RES_DIR / "surface_summary.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[ticker] = entry
    path.write_text(json.dumps(data, indent=2, default=str))
    return path


# --------------------------------------------------------------------------
# data quality
# --------------------------------------------------------------------------

def data_quality(raw: pd.DataFrame, vt: pd.Timestamp) -> dict:
    """Spread / staleness numbers on the raw snapshot."""
    spot = float(raw["spot"].iloc[0])
    lt = pd.to_datetime(raw["last_trade_time"], utc=True)
    session_day = vt.tz_convert("America/New_York").normalize()
    lt_ny = lt.dt.tz_convert("America/New_York")
    from volkit.data.chain import time_to_expiry
    dte = time_to_expiry(raw["expiry"], pd.Series(vt, index=raw.index)) * 365
    k = np.log(raw["strike"] / spot)
    otm = ((raw["kind"] == "put") & (k < 0)) | ((raw["kind"] == "call") & (k >= 0))
    two_sided = (raw["bid"] > 0) & (raw["ask"] > raw["bid"])
    mid = 0.5 * (raw["bid"] + raw["ask"])
    near = two_sided & otm & (k.abs() < 0.02) & dte.between(7, 60)
    # contracts that traded in the last 15 minutes of the session: is last inside the quote?
    recent = two_sided & (lt >= vt - pd.Timedelta(minutes=30))
    last_in = (raw["last"] >= raw["bid"]) & (raw["last"] <= raw["ask"])
    rel = (raw["ask"] - raw["bid"]) / mid
    out = {
        "rows": int(len(raw)),
        "frac_bid_positive": r2((raw["bid"] > 0).mean()),
        "n_crossed_bid_gt_ask": int((raw["ask"] < raw["bid"]).sum()),
        "frac_last_trade_on_session_day": r2((lt_ny.dt.normalize() == session_day).mean()),
        "near_atm_otm_7_60d_n": int(near.sum()),
        "near_atm_otm_7_60d_median_spread_usd": r2(
            (raw["ask"] - raw["bid"])[near].median()),
        "near_atm_otm_7_60d_median_rel_spread": r2(rel[near].median()),
        "traded_last_30min_n": int(recent.sum()),
        "traded_last_30min_frac_last_within_bid_ask": r2(last_in[recent].mean()),
        "traded_last_30min_median_abs_last_minus_mid_over_spread": r2(
            ((raw["last"] - mid).abs() / (raw["ask"] - raw["bid"]))[recent].median()),
    }
    return out


# --------------------------------------------------------------------------
# sanity numbers
# --------------------------------------------------------------------------

def forward_at(surface, T: float) -> float:
    """F(T) by linear interpolation of the slice forwards in T."""
    return float(np.interp(T, surface.slices["T"], surface.slices["F"]))


def delta_risk_reversal(surface, dte: float, delta: float = 0.25) -> dict:
    """IV(put, forward delta -0.25) - IV(call, forward delta +0.25) on the dte smile.

    Forward (undiscounted Black) delta: call N(d1), put N(d1) - 1,
    d1 = (-k + sigma^2 T / 2) / (sigma sqrt T).
    """
    T = dte / 365.0
    k = np.linspace(-0.45, 0.35, 1601)
    sig = surface.implied_vol(k, T)
    ok = np.isfinite(sig)
    k, sig = k[ok], sig[ok]
    if len(k) < 10:
        return {"iv_put": None, "iv_call": None, "rr": None}
    d1 = (-k + 0.5 * sig**2 * T) / (sig * np.sqrt(T))
    call_d, put_d = norm.cdf(d1), norm.cdf(d1) - 1.0
    res = {}
    # call delta falls with k; put delta also falls with k (from ~0 to -1)
    for name, dd, target, side in (("iv_call", call_d, delta, k >= 0),
                                   ("iv_put", put_d, -delta, k < 0)):
        x, y, s = dd[side], k[side], sig[side]
        if len(x) < 2 or not (x.min() <= target <= x.max()):
            res[name] = None
            continue
        order = np.argsort(x)
        res[name] = float(np.interp(target, x[order], s[order]))
        res[name.replace("iv", "k")] = float(np.interp(target, x[order], y[order]))
    if res.get("iv_put") is not None and res.get("iv_call") is not None:
        res["rr"] = res["iv_put"] - res["iv_call"]
    else:
        res["rr"] = None
    return res


# --------------------------------------------------------------------------
# synthetic American parity-bias study
# --------------------------------------------------------------------------

def american_bias_study(spot: float, r: float, vt: pd.Timestamp, q: float = 0.012,
                        steps: int = 200) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Price an American chain on a 1000-step CRR tree from a known skewed vol,
    run the European pipeline on it and measure the forward / IV errors it
    makes; American re-inversion uses ``steps`` (so it carries tree error too)."""
    def vol(k, T):
        return float(np.clip(0.17 - 0.45 * k + 1.0 * k * k, 0.08, 1.0))

    strikes = np.unique(np.concatenate([np.arange(0.70, 0.95, 0.025) * spot,
                                        np.arange(0.95, 1.05, 0.005) * spot,
                                        np.arange(1.05, 1.21, 0.025) * spot])).round(0)
    days = [14, 30, 90, 180, 365, 730]
    syn = synthetic_chain(spot, r, q, vol, days, strikes, vt, american=True,
                          steps=SYN_STEPS, ticker="SYN")
    syn["mid"] = 0.5 * (syn["bid"] + syn["ask"])
    from volkit.data.chain import time_to_expiry
    syn["T"] = time_to_expiry(syn["expiry"], syn["snapshot_time"])
    surf = build_surface(syn[syn["mid"] > 1e-4], spot, r, bid_ask_iv=False, drop_outliers=False)
    sl = surf.slices.copy()
    sl["F_true"] = spot * np.exp((r - q) * sl["T"])
    sl["F_err"] = sl["F"] - sl["F_true"]
    sl["q_err"] = sl["q"] - q
    pts = surf.points.copy()
    # true sigma uses the TRUE forward's k
    pts["k_true"] = np.log(pts["strike"] / (spot * np.exp((r - q) * pts["T"])))
    pts["iv_true"] = [vol(k, T) for k, T in zip(pts["k_true"], pts["T"])]
    pts["err_eu"] = pts["iv"] - pts["iv_true"]
    sub = pts.groupby("expiry", group_keys=False).apply(
        lambda g: g.iloc[np.unique(np.linspace(0, len(g) - 1, min(12, len(g))).round().astype(int))])
    sub = sub.copy()
    sub["iv_am_trueq"] = [implied_vol_american(m, spot, K, T, r, q, kd, steps=steps)
                          for m, K, T, kd in zip(sub["mid"], sub["strike"], sub["T"], sub["kind"])]
    sub["iv_am_parityq"] = [implied_vol_american(m, spot, K, T, r, qq, kd, steps=steps)
                            for m, K, T, qq, kd in zip(sub["mid"], sub["strike"], sub["T"],
                                                       sub["q"], sub["kind"])]
    sub["err_am_trueq"] = sub["iv_am_trueq"] - sub["iv_true"]
    sub["err_am_parityq"] = sub["iv_am_parityq"] - sub["iv_true"]
    return sl, sub


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def fig_smiles(surface, ticker, date, path):
    sl = surface.slices
    chosen = nearest_rows(sl, "dte", [7, 30, 90, 215, 760])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(chosen)))
    for c, (_, s) in zip(colors, chosen.iterrows()):
        p = surface.points[surface.points["expiry"] == s["expiry"]].sort_values("k")
        ax.plot(p["k"], 100 * p["iv"], "-", color=c, lw=1.2,
                label=f"{pd.Timestamp(s['expiry']).date()} ({s['dte']:.0f}d)")
        if "iv_bid" in p:
            ax.fill_between(p["k"], 100 * p["iv_bid"], 100 * p["iv_ask"], color=c, alpha=0.25, lw=0)
        o = surface.outliers
        if len(o):
            o = o[o["expiry"] == s["expiry"]]
            ax.plot(o["k"], 100 * o["iv"], "x", color=c, ms=6)
    ax.axvline(0, color="grey", lw=0.6, ls=":")
    ax.set_xlabel("log-moneyness k = ln(K / F)")
    ax.set_ylabel("implied volatility (%)")
    ax.set_title(f"{ticker} implied-vol smiles, OTM mid quotes, {date}\n"
                 "line = mid IV, band = bid-ask IV, x = quotes removed by outlier rule")
    ax.legend(title="expiry (days)", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_term(atm, vixf, vix_date, ticker, date, path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(atm["dte"], 100 * atm["atm_iv"], "o-", ms=4, label=f"{ticker} ATM IV (k = 0)")
    for d, col, lab, m in ((9, "vix9d", "VIX9D", "s"), (30, "vix", "VIX", "D"),
                           (93, "vix3m", "VIX3M", "^")):
        v = vixf.get(col)
        if v is not None and np.isfinite(v):
            ax.plot([d], [100 * v], m, ms=9, color="crimson", mfc="none", mew=2,
                    label=f"{lab} = {100 * v:.2f} ({vix_date})")
    ax.set_xscale("log")
    ax.set_xticks([7, 14, 30, 60, 90, 180, 365, 730])
    ax.set_xticklabels(["7", "14", "30", "60", "90", "180", "365", "730"])
    ax.set_xlabel("days to expiry (log scale)")
    ax.set_ylabel("implied volatility (%)")
    ax.set_title(f"{ticker} ATM implied-vol term structure, {date}\n"
                 "red: CBOE S&P 500 vol indices (model-free variance-swap style, include skew; "
                 "not ATM IV)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_surface(surface, ticker, date, path):
    k = np.linspace(-0.35, 0.20, 111)
    dte = np.geomspace(surface.slices["dte"].min(), surface.slices["dte"].max(), 120)
    K, D = np.meshgrid(k, dte)
    iv = surface.implied_vol(K, D / 365.0)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    m = ax.pcolormesh(K, D, 100 * np.ma.masked_invalid(iv), shading="auto", cmap="magma")
    cb = fig.colorbar(m, ax=ax)
    cb.set_label("implied volatility (%)")
    ax.plot(surface.points["k"], surface.points["dte"], ",", color="cyan", alpha=0.4)
    ax.set_yscale("log")
    ax.set_xlabel("log-moneyness k = ln(K / F)")
    ax.set_ylabel("days to expiry (log scale)")
    ax.set_title(f"{ticker} implied-vol surface, {date}\n"
                 "linear in total variance in k and T; white = outside quoted range (no extrapolation); "
                 "dots = quotes", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_total_variance(surface, ticker, date, path):
    sl = surface.slices
    chosen = nearest_rows(sl, "dte", [7, 14, 30, 60, 90, 140, 215, 320, 430, 580, 760])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = plt.cm.viridis(np.linspace(0, 0.95, len(chosen)))
    for c, (_, s) in zip(colors, chosen.iterrows()):
        p = surface.points[surface.points["expiry"] == s["expiry"]].sort_values("k")
        ax.plot(p["k"], p["w"], "-", color=c, lw=1.2, label=f"{s['dte']:.0f}d")
    ax.set_xlabel("log-moneyness k = ln(K / F)")
    ax.set_ylabel("total implied variance w = sigma^2 T (years x vol^2)")
    ax.set_title(f"{ticker} total implied variance by expiry, {date}\n"
                 "no calendar arbitrage <=> curves do not cross (w non-decreasing in T)")
    ax.legend(title="days to expiry", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_yf(points, ticker, date, path):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    p = points[points["yf_iv"] > 0.001]
    sc = ax.scatter(100 * p["iv"], 100 * p["yf_iv"], c=np.log10(p["dte"]), s=4, cmap="viridis")
    lim = [0, min(120, 100 * max(p["iv"].max(), p["yf_iv"].max()) * 1.05)]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("log10(days to expiry)")
    ax.set_xlabel("volkit IV from mid, parity forward (%)")
    ax.set_ylabel("Yahoo impliedVolatility column (%)")
    ax.set_title(f"{ticker}: volkit IV vs Yahoo IV, same OTM contracts, {date}", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshot", default=str(ROOT / "data/raw/spy_chain_20261115.csv"))
    ap.add_argument("--min-bid", type=float, default=0.05)
    ap.add_argument("--max-rel-spread", type=float, default=0.25)
    ap.add_argument("--min-dte", type=float, default=7.0)
    ap.add_argument("--max-dte", type=float, default=800.0)
    ap.add_argument("--k-lo", type=float, default=-0.40)
    ap.add_argument("--k-hi", type=float, default=0.30)
    ap.add_argument("--am-steps", type=int, default=200)
    ap.add_argument("--am-per-expiry", type=int, default=15)
    ap.add_argument("--skip-bias-study", action="store_true")
    args = ap.parse_args(argv)

    require_cached(args.snapshot, RATES_FILE, VIX_FAMILY_FILE,
                   hint="Then pass the chain file the download script saved, e.g. "
                        "`--snapshot data/raw/spx_chain_<date>.csv` (or spy_chain_<date>.csv).")
    raw = load_chain(args.snapshot)
    ticker = str(raw["ticker"].iloc[0]).lstrip("^")
    spot = float(raw["spot"].iloc[0])
    snap = pd.Timestamp(raw["snapshot_time"].iloc[0])
    vt = valuation_time(raw)
    date = str(vt.tz_convert("America/New_York").date())
    print(f"== {ticker}: {Path(args.snapshot).name}  spot={spot}  "
          f"market_state={raw['market_state'].iloc[0]}  snapshot={snap}  valuation_time={vt}")

    rates = load_rates()
    r_series = rates[rates.index <= vt.tz_convert("America/New_York").tz_localize(None)]
    r, r_date = float(r_series.iloc[-1]), str(r_series.index[-1].date())
    print(f"r = {r:.5f} (3m T-bill, continuous, quote of {r_date}; one flat rate for all T)")

    dq = data_quality(raw, vt)
    print("data quality:", json.dumps(dq, indent=1))

    chain = raw.copy()
    chain["snapshot_time"] = vt   # measure T from the time the quotes refer to
    cleaned = clean_chain(chain, spot, min_bid=args.min_bid, max_rel_spread=args.max_rel_spread,
                          min_dte=args.min_dte, max_dte=args.max_dte,
                          moneyness_range=(args.k_lo, args.k_hi))
    rep = filter_report(cleaned)
    print("cleaning filters (each row counted under the first filter it fails):")
    print(rep.to_string(index=False))
    entry = {
        "ticker": ticker, "snapshot_file": Path(args.snapshot).name, "spot": spot,
        "snapshot_time_utc": str(snap), "valuation_time_utc": str(vt),
        "market_state": str(raw["market_state"].iloc[0]),
        "r": r2(r, 5), "r_quote_date": r_date, "data_quality": dq,
        "filter_params": {k: (list(v) if isinstance(v, tuple) else v)
                          for k, v in cleaned.attrs["filter_params"].items()},
        "filter_counts": cleaned.attrs["filter_counts"],
    }

    surface = None
    if len(cleaned) >= MIN_CLEAN_ROWS:
        surface = build_surface(cleaned, spot, r)
    n_slices = 0 if surface is None else len(surface.slices)
    if surface is None or n_slices < MIN_SLICES:
        msg = (f"INSUFFICIENT DATA for {ticker}: {len(cleaned)} of {len(raw)} rows survive cleaning "
               f"({cleaned.attrs['filter_counts']['low_bid']} removed for bid < {args.min_bid}); "
               f"{n_slices} usable expiries (need {MIN_SLICES}). "
               f"Snapshot market_state={raw['market_state'].iloc[0]}: re-download during US market "
               f"hours (09:30-16:00 New York) and re-run with --snapshot <file>.")
        print(msg)
        entry.update(status="insufficient_data", message=msg,
                     dropped_expiries=[] if surface is None else surface.dropped)
        print("summary ->", update_summary(ticker, entry))
        return 0

    sl, pts = surface.slices, surface.points
    print(f"surface: {len(sl)} expiries, {len(pts)} OTM quotes, "
          f"{len(surface.outliers)} outliers removed, dropped expiries: {surface.dropped or 'none'}")
    pd.set_option("display.width", 200)
    num = ["dte", "F", "q", "F_pair_range", "F_reg", "r_reg"]
    show = sl[["expiry"] + num + ["n_otm", "n_iv_nan", "n_outliers"]].copy()
    show[num] = show[num].round(4)
    print(show.to_string(index=False))

    # ---------- implied spot from the forwards (assumes no dividend before expiry)
    short = sl[sl["dte"] <= 30]
    implied_spot = (short["F"] * short["DF"]).to_numpy()
    print(f"parity forward x DF for expiries <= 30d (= spot if no dividend before expiry): "
          f"{np.round(implied_spot, 2)} vs reported spot {spot}")

    # ---------- sanity checks
    atm = surface.atm_term_structure()
    vixf_all = load_vix_family()
    vix_row = vixf_all[vixf_all.index <= vt.tz_convert("America/New_York").tz_localize(None)].iloc[-1]
    vix_date = str(vix_row.name.date())
    vixf = {c: float(vix_row[c]) for c in ("vix9d", "vix", "vix3m")}
    atm30 = float(surface.implied_vol(0.0, 30 / 365))
    atm9 = float(surface.implied_vol(0.0, 9 / 365))
    atm93 = float(surface.implied_vol(0.0, 93 / 365))
    atm365 = float(surface.implied_vol(0.0, 365 / 365))
    skew_k = float(surface.implied_vol(-0.10, 30 / 365) - surface.implied_vol(0.10, 30 / 365))
    rr = delta_risk_reversal(surface, 30)
    term = {"atm_iv_9d": r2(atm9), "atm_iv_30d": r2(atm30), "atm_iv_93d": r2(atm93),
            "atm_iv_365d": r2(atm365),
            "slope_93d_minus_30d_volpts": r2(100 * (atm93 - atm30), 2),
            "slope_365d_minus_30d_volpts": r2(100 * (atm365 - atm30), 2)}
    # log-linear fit of ATM IV vs log(dte) over all expiries
    ok = atm["atm_iv"].notna()
    b = np.polyfit(np.log(atm.loc[ok, "dte"]), 100 * atm.loc[ok, "atm_iv"], 1)[0]
    term["fit_volpts_per_log_dte"] = r2(b, 3)
    # T measured from the Sunday snapshot instead of Friday close: effect on 30d ATM IV
    alt = raw.copy()
    alt_clean = clean_chain(alt, spot, min_bid=args.min_bid, max_rel_spread=args.max_rel_spread,
                            min_dte=args.min_dte, max_dte=args.max_dte,
                            moneyness_range=(args.k_lo, args.k_hi))
    alt_surf = build_surface(alt_clean[alt_clean["dte"] < 60], spot, r, bid_ask_iv=False,
                             drop_outliers=False)
    alt_atm = alt_surf.atm_term_structure()
    t_effect = None
    if (vt != snap) and len(alt_atm):
        m = atm.merge(alt_atm, on="expiry", suffixes=("", "_snap"))
        t_effect = [{"expiry": str(pd.Timestamp(e).date()), "dte_valuation": r2(d, 2),
                     "atm_iv_T_from_close": r2(a), "atm_iv_T_from_snapshot": r2(bb)}
                    for e, d, a, bb in zip(m["expiry"], m["dte"], m["atm_iv"], m["atm_iv_snap"])][:6]
    sanity = {
        "vix_family_date": vix_date, "vix9d": r2(vixf["vix9d"]), "vix": r2(vixf["vix"]),
        "vix3m": r2(vixf["vix3m"]), "atm_iv_30d_minus_vix_volpts": r2(100 * (atm30 - vixf["vix"]), 2),
        "skew_iv_k-0.10_minus_k+0.10_30d": r2(skew_k),
        "rr25_forward_delta_30d": {k: r2(v) for k, v in rr.items()},
        "term_structure": term,
        "T_measurement_effect": t_effect,
    }
    print("sanity:", json.dumps(sanity, indent=1))

    # ---------- arbitrage diagnostics
    cal = calendar_violations(surface, n_grid=41, tol=0.0)
    bf0 = butterfly_violations(surface, tol=0.0)
    bf1 = butterfly_violations(surface, tol=0.01)
    print("calendar violations per consecutive pair (41-point k grid on overlap):",
          int(cal["n_violations"].sum()), "of", int(cal["n_grid"].sum()))
    bad_cal = cal[cal["n_violations"] > 0]
    print(bad_cal.to_string(index=False) if len(bad_cal) else "  none")
    print("butterfly (convexity) / monotonicity violations, tol=$0 and tol=$0.01 (one tick):")
    bf = bf0[["expiry", "dte", "n_strikes", "n_convexity_violations", "max_butterfly_deficit",
              "n_monotonicity_violations"]].copy()
    bf["conv_viol_tol_1c"] = bf1["n_convexity_violations"]
    bf["mono_viol_tol_1c"] = bf1["n_monotonicity_violations"]
    bf[["dte", "max_butterfly_deficit"]] = bf[["dte", "max_butterfly_deficit"]].round(4)
    print(bf.to_string(index=False))
    arb = {
        "calendar_grid_points": int(cal["n_grid"].sum()),
        "calendar_violations": int(cal["n_violations"].sum()),
        "calendar_detail": [{"expiry_1": str(pd.Timestamp(a).date()), "expiry_2": str(pd.Timestamp(b).date()),
                             "n": int(n), "max_w_shortfall": r2(s, 6), "k": r2(kk, 3)}
                            for a, b, n, s, kk in cal[cal["n_violations"] > 0][
                                ["expiry_1", "expiry_2", "n_violations", "max_shortfall", "k_worst"]].itertuples(index=False)],
        "butterfly_triples": int((bf0["n_strikes"] - 2).clip(lower=0).sum()),
        "convexity_violations_tol0": int(bf0["n_convexity_violations"].sum()),
        "convexity_violations_tol_1cent": int(bf1["n_convexity_violations"].sum()),
        "monotonicity_violations_tol0": int(bf0["n_monotonicity_violations"].sum()),
        "monotonicity_violations_tol_1cent": int(bf1["n_monotonicity_violations"].sum()),
        "expiries_with_deficit_over_10c": [
            {"expiry": str(pd.Timestamp(e).date()), "max_deficit_usd": r2(v, 3)}
            for e, v in zip(bf0["expiry"], bf0["max_butterfly_deficit"]) if v > 0.10],
    }

    # ---------- European vs American inversion on a subset
    am_exp = nearest_rows(sl, "dte", [7, 30, 63, 126, 215, 320, 430, 760])["expiry"].tolist()
    eva = european_vs_american(surface, am_exp, per_expiry=args.am_per_expiry, steps=args.am_steps)
    eva["k_bucket"] = pd.cut(eva["k"], K_BUCKETS)
    eva["dte_label"] = eva["dte"].round(0).astype(int)
    tab = eva.pivot_table(index="k_bucket", columns="dte_label", values="diff",
                          aggfunc="mean", observed=False) * 100
    tab_eep = eva.pivot_table(index="k_bucket", columns="dte_label", values="diff_eep",
                              aggfunc="mean", observed=False) * 100
    print("Black-Scholes IV minus American-tree IV, mean, vol points (rows k bucket, cols dte):")
    print(tab.round(3).to_string())
    print("European-tree IV minus American-tree IV (early exercise only), mean, vol points:")
    print(tab_eep.round(3).to_string())
    # tree discretisation check on the 5 largest differences
    worst = eva.reindex(eva["diff"].abs().sort_values(ascending=False).index).head(5)
    conv = []
    for _, w in worst.iterrows():
        q_row = float(pts[(pts["expiry"] == w["expiry"]) & (pts["strike"] == w["strike"])]["q"].iloc[0])
        T_row = float(w["dte"]) / 365
        iv800 = implied_vol_american(w["mid"], spot, w["strike"], T_row, r, q_row, w["kind"], steps=800)
        conv.append({"expiry": str(pd.Timestamp(w["expiry"]).date()), "strike": w["strike"],
                     "kind": w["kind"], "iv_am_200": r2(w["iv_am"], 5), "iv_am_800": r2(iv800, 5),
                     "iv_eu": r2(w["iv_eu"], 5)})
    print("largest differences, tree steps 200 vs 800:", json.dumps(conv, indent=1))
    eu_am = {
        "n_options": int(len(eva)), "steps": args.am_steps,
        "mean_diff_volpts": r2(100 * eva["diff"].mean(), 3),
        "max_abs_diff_volpts": r2(100 * eva["diff"].abs().max(), 3),
        "median_abs_diff_volpts": r2(100 * eva["diff"].abs().median(), 3),
        "by_kind_mean_volpts": {k: r2(100 * v, 3) for k, v in eva.groupby("kind")["diff"].mean().items()},
        "table_mean_volpts": {str(i): {str(c): r2(v, 3) for c, v in row.items()} for i, row in tab.iterrows()},
        "table_eep_only_mean_volpts": {str(i): {str(c): r2(v, 3) for c, v in row.items()}
                                       for i, row in tab_eep.iterrows()},
        "max_abs_eep_only_volpts": r2(100 * eva["diff_eep"].abs().max(), 3),
        "tree_convergence_check": conv,
    }

    # ---------- synthetic American parity-bias study
    bias = None
    if not args.skip_bias_study:
        syn_sl, syn_pts = american_bias_study(spot, r, vt, steps=args.am_steps)
        print("synthetic American chain (true q=0.012, skewed vol): parity forward error")
        print(syn_sl[["dte", "F", "F_true", "F_err", "q", "q_err", "F_reg", "r_reg"]].round(4).to_string(index=False))
        syn_pts["k_bucket"] = pd.cut(syn_pts["k_true"], K_BUCKETS)
        g = syn_pts.groupby([syn_pts["dte"].round(0).astype(int)])
        print("IV error (vol points): European inversion with parity F | American tree with true q | "
              "American tree with parity q")
        err = g[["err_eu", "err_am_trueq", "err_am_parityq"]].agg(lambda s: 100 * s.abs().max())
        print("max |error| by dte:\n", err.round(3).to_string())
        bias = {
            "true_q": 0.012,
            "forward_error_usd": {str(int(round(d))): r2(e, 4) for d, e in zip(syn_sl["dte"], syn_sl["F_err"])},
            "q_error": {str(int(round(d))): r2(e, 5) for d, e in zip(syn_sl["dte"], syn_sl["q_err"])},
            "r_reg_vs_true_r": {str(int(round(d))): r2(x, 4) for d, x in zip(syn_sl["dte"], syn_sl["r_reg"])},
            "max_abs_iv_error_volpts_by_dte": {str(i): {c: r2(v, 3) for c, v in row.items()}
                                               for i, row in err.iterrows()},
        }

    # ---------- yahoo comparison
    yf_cmp = pts[pts["yf_iv"] > 0.001]
    yf_diff = 100 * (yf_cmp["yf_iv"] - yf_cmp["iv"])
    yf = {"n": int(len(yf_cmp)), "median_yf_minus_ours_volpts": r2(yf_diff.median(), 2),
          "median_abs_volpts": r2(yf_diff.abs().median(), 2),
          "by_kind_median_volpts": {k: r2(v, 2) for k, v in yf_diff.groupby(yf_cmp["kind"]).median().items()}}
    print("Yahoo IV minus ours:", yf)

    # ---------- outputs
    FIG_DIR.mkdir(exist_ok=True)
    RES_DIR.mkdir(exist_ok=True)
    t = ticker.lower()
    fig_smiles(surface, ticker, date, FIG_DIR / f"smile_{t}.png")
    fig_term(atm, vixf, vix_date, ticker, date, FIG_DIR / f"term_structure_{t}.png")
    fig_surface(surface, ticker, date, FIG_DIR / f"surface_{t}.png")
    fig_total_variance(surface, ticker, date, FIG_DIR / f"total_variance_{t}.png")
    fig_yf(pts, ticker, date, FIG_DIR / f"iv_vs_yahoo_{t}.png")

    cols = ["expiry", "dte", "T", "kind", "strike", "bid", "ask", "mid", "volume", "open_interest",
            "F", "DF", "q", "k", "iv", "iv_bid", "iv_ask", "w", "yf_iv"]
    table = pts[cols].assign(used=True)
    if len(surface.outliers):
        o = surface.outliers.copy()
        o["dte"] = o["T"] * 365
        o["F"] = o["expiry"].map(sl.set_index("expiry")["F"])
        o["DF"] = o["expiry"].map(sl.set_index("expiry")["DF"])
        o["q"] = o["expiry"].map(sl.set_index("expiry")["q"])
        o["w"] = o["iv"] ** 2 * o["T"]
        table = pd.concat([table, o[cols].assign(used=False)], ignore_index=True)
    table["expiry"] = pd.to_datetime(table["expiry"]).dt.date
    table.sort_values(["expiry", "strike"]).to_csv(RES_DIR / f"surface_iv_{t}.csv", index=False)

    entry.update(
        status="ok",
        n_expiries=int(len(sl)), n_quotes=int(len(pts)), n_outliers_removed=int(len(surface.outliers)),
        outliers_by_expiry={str(pd.Timestamp(e).date()): int(n) for e, n in zip(sl["expiry"], sl["n_outliers"]) if n},
        dropped_expiries=surface.dropped,
        expiries=[{"expiry": str(pd.Timestamp(s["expiry"]).date()), "dte": r2(s["dte"], 2),
                   "F": r2(s["F"], 3), "q": r2(s["q"], 5), "F_reg": r2(s["F_reg"], 3),
                   "r_reg": r2(s["r_reg"], 4), "n_otm": int(s["n_otm"]),
                   "atm_iv": r2(a)} for (_, s), a in zip(sl.iterrows(), atm["atm_iv"])],
        implied_spot_short_expiries=[r2(x, 3) for x in implied_spot],
        sanity=sanity, arbitrage=arb, european_vs_american=eu_am,
        synthetic_american_bias=bias, yahoo_iv_comparison=yf,
    )
    print("summary ->", update_summary(ticker, entry))
    print("figures ->", FIG_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
