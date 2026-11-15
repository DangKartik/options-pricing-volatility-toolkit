"""Implied-volatility surface from an option-chain snapshot.

Pipeline (``build_surface``), one expiry at a time:

1. Forward and discount factor from put-call parity, C - P = DF (F - K).
   DF = exp(-r T) is taken from the T-bill rate (``load_rates``); F is the
   median over the ``n_pairs`` strikes with the smallest |C - P| (the pairs
   nearest the money) of F_i = K_i + (C_i - P_i) / DF. The implied continuous
   dividend yield is q = r - ln(F / S) / T. As a diagnostic, an OLS fit of
   C - P = a - b K over the ``n_reg`` nearest pairs also estimates DF (= b)
   and F (= a / b) without using the T-bill rate.
   Parity is exact for European options only. For American options (SPY) the
   early-exercise premia of the call and the put do not cancel, so the F found
   this way is biased; ``scripts/build_surface.py`` measures that bias on a
   synthetic American chain.
2. Implied vol from MID quotes of out-of-the-money options only: puts with
   K < F, calls with K >= F. Reasons: OTM options are the liquid ones (tighter
   relative spreads, most volume); their price is all time value, so a quote
   error is not swamped by intrinsic value the inversion has to subtract; and
   their early-exercise premium is small, so a European inversion of an
   American OTM quote is a close approximation.
3. The surface lives in (k, T) with k = ln(K / F) and stores total implied
   variance w = sigma^2 T. Interpolation is linear in k within an expiry and
   linear in T at fixed k between expiries. Total variance rather than vol:
   absence of calendar arbitrage means w is non-decreasing in T at fixed k, and
   linear interpolation of a non-decreasing quantity stays non-decreasing,
   while linear interpolation in sigma does not have that property; w is also
   the quantity that scales with time (constant vol gives w linear in T).
   No smoothing or parametric fit (e.g. SVI) is applied: the surface passes
   through every retained quote, noise included.

Arbitrage diagnostics count violations rather than repair them:
``calendar_violations`` (w decreasing in T at fixed k) and
``butterfly_violations`` (call price not convex / not monotone in strike).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from volkit.data.chain import CHAIN_COLUMNS, EXPIRY_HOUR, EXPIRY_TZ, time_to_expiry
from volkit.iv.solver import implied_vol, implied_vol_american, implied_vol_tree
from volkit.pricing.binomial import crr_price
from volkit.pricing.black_scholes import bs_price

__all__ = [
    "valuation_time",
    "forward_from_parity",
    "flag_outliers",
    "otm_quotes",
    "VolSurface",
    "build_surface",
    "calendar_violations",
    "butterfly_violations",
    "european_vs_american",
    "synthetic_chain",
]


# --------------------------------------------------------------------------
# valuation time
# --------------------------------------------------------------------------

def valuation_time(chain: pd.DataFrame) -> pd.Timestamp:
    """Time the quotes refer to: snapshot_time if market_state is REGULAR, else
    16:00 New York on the date of the most recent trade (the last session close).

    A snapshot taken on a weekend carries Friday's closing quotes; measuring T
    from the download time would shorten every T by the weekend and inflate IVs.
    """
    snap = pd.Timestamp(chain["snapshot_time"].iloc[0])
    state = str(chain["market_state"].iloc[0]) if "market_state" in chain else ""
    if state == "REGULAR":
        return snap
    last = pd.to_datetime(chain["last_trade_time"], utc=True).max()
    if pd.isna(last):
        return snap
    day = last.tz_convert(EXPIRY_TZ).normalize()
    close = (day + pd.Timedelta(hours=EXPIRY_HOUR)).tz_convert("UTC")
    return min(close, snap)


# --------------------------------------------------------------------------
# forward / discount factor
# --------------------------------------------------------------------------

def forward_from_parity(strike: np.ndarray, call_mid: np.ndarray, put_mid: np.ndarray,
                        T: float, r: float, spot: float, n_pairs: int = 6,
                        n_reg: int = 20) -> dict:
    """Forward from parity on same-strike (call, put) mids: F = median(K + (C - P) e^{rT})
    over the n_pairs strikes with the smallest |C - P|; q = r - ln(F/S)/T.

    Also returns an OLS estimate C - P = a - b K over the n_reg nearest pairs
    (DF_reg = b, F_reg = a/b, r_reg = -ln(b)/T) as a check on the T-bill DF.
    """
    K = np.asarray(strike, float)
    cp = np.asarray(call_mid, float) - np.asarray(put_mid, float)
    ok = np.isfinite(K) & np.isfinite(cp)
    K, cp = K[ok], cp[ok]
    if T <= 0 or len(K) < 1:
        raise ValueError("need T > 0 and at least one call/put pair")
    order = np.argsort(np.abs(cp))
    DF = math.exp(-r * T)
    near = order[:n_pairs]
    F_i = K[near] + cp[near] / DF
    F = float(np.median(F_i))
    out = {
        "F": F, "DF": DF, "q": r - math.log(F / spot) / T, "n_pairs": int(len(near)),
        "F_pair_range": float(F_i.max() - F_i.min()),
        "DF_reg": math.nan, "F_reg": math.nan, "r_reg": math.nan, "n_reg": 0,
    }
    reg = order[:n_reg]
    if len(reg) >= 3 and np.ptp(K[reg]) > 0:
        slope, intercept = np.polyfit(K[reg], cp[reg], 1)
        b = -slope
        if b > 0:
            out.update(DF_reg=float(b), F_reg=float(intercept / b),
                       r_reg=float(-math.log(b) / T), n_reg=int(len(reg)))
    return out


def _neighbour_prediction(k: np.ndarray, iv: np.ndarray, i: int, m: int = 3) -> float:
    """Median, over pairs (j, l) with j among the m quotes left of i and l among
    the m quotes right of i, of the straight line through (k_j, iv_j), (k_l, iv_l)
    evaluated at k_i. At an end of the smile the pairs come from the m nearest
    quotes on the one available side (extrapolation)."""
    n = len(k)
    left = list(range(max(0, i - m), i))
    right = list(range(i + 1, min(n, i + m + 1)))
    if left and right:
        pairs = [(j, l) for j in left for l in right]
    else:
        side = left or right
        pairs = [(a, b) for x, a in enumerate(side) for b in side[x + 1:]]
    if not pairs:
        return math.nan
    preds = [iv[j] + (iv[l] - iv[j]) * (k[i] - k[j]) / (k[l] - k[j]) for j, l in pairs]
    return float(np.median(preds))


def flag_outliers(k: np.ndarray, iv: np.ndarray, iv_width: np.ndarray,
                  min_jump: float = 0.01) -> np.ndarray:
    """Iteratively flag isolated bad quotes in one smile (k sorted ascending).

    Deviation d_i = |iv_i - neighbour prediction|, the prediction being the
    median of straight lines in k through one of the 3 quotes left and one of
    the 3 quotes right of i (9 lines; robust to a bad neighbour and to uneven
    strike spacing). While max_i of d_i - (min_jump +
    iv_width_i / 2) is positive, remove that worst quote and recompute; i.e. a
    quote is dropped when even its own bid-ask IV range misses the neighbours'
    prediction by more than min_jump (1 vol point). iv_width = iv(ask) - iv(bid).
    Catches stale quotes, which sit off the line through their neighbours.
    """
    k, iv = np.asarray(k, float), np.asarray(iv, float)
    if np.any(np.diff(k) <= 0):
        raise ValueError("k must be strictly increasing")
    width = np.nan_to_num(np.asarray(iv_width, float), nan=0.0)
    thresh = min_jump + 0.5 * width
    alive = np.arange(len(k))
    flags = np.zeros(len(k), dtype=bool)
    while len(alive) >= 5:
        kk, vv = k[alive], iv[alive]
        excess = np.array([abs(vv[i] - _neighbour_prediction(kk, vv, i)) for i in range(len(alive))])
        excess = excess - thresh[alive]
        worst = int(np.nanargmax(excess))
        if not excess[worst] > 0:
            break
        flags[alive[worst]] = True
        alive = np.delete(alive, worst)
    return flags


def otm_quotes(slice_df: pd.DataFrame, F: float) -> pd.DataFrame:
    """Out-of-the-money rows of one expiry: puts with strike < F, calls with strike >= F."""
    put = (slice_df["kind"] == "put") & (slice_df["strike"] < F)
    call = (slice_df["kind"] == "call") & (slice_df["strike"] >= F)
    return slice_df[put | call].sort_values("strike").reset_index(drop=True)


# --------------------------------------------------------------------------
# surface object
# --------------------------------------------------------------------------

@dataclass
class VolSurface:
    """Per-expiry total-variance slices w(k) on k = ln(K/F), linear in k and in T.

    ``slices``: one row per kept expiry (expiry, T, dte, F, DF, q, ...).
    ``points``: one row per OTM quote used (expiry, T, strike, kind, mid, k, iv, w, ...).
    ``dropped``: expiries left out and why; ``outliers``: quotes removed by flag_outliers.
    """
    slices: pd.DataFrame
    points: pd.DataFrame
    dropped: list[dict] = field(default_factory=list)
    spot: float = math.nan
    r: float = math.nan
    outliers: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        self.slices = self.slices.sort_values("T").reset_index(drop=True)
        self._k, self._w = [], []
        for exp in self.slices["expiry"]:
            p = self.points[self.points["expiry"] == exp].sort_values("k")
            self._k.append(p["k"].to_numpy(float))
            self._w.append(p["w"].to_numpy(float))
        self._T = self.slices["T"].to_numpy(float)

    def slice_total_variance(self, i: int, k: np.ndarray | float) -> np.ndarray:
        """w on slice i, linear in k between quotes, nan outside the quoted k range."""
        return np.interp(k, self._k[i], self._w[i], left=np.nan, right=np.nan)

    def total_variance(self, k: np.ndarray | float, T: np.ndarray | float) -> np.ndarray:
        """w(k, T): linear in k on each slice, then linear in T at fixed k.

        nan outside [first expiry T, last expiry T] or outside either bracketing
        slice's quoted k range (no extrapolation).
        """
        k, T = np.broadcast_arrays(np.asarray(k, float), np.asarray(T, float))
        out = np.full(k.shape, np.nan)
        if len(self._T) == 0:
            return out
        for idx in np.ndindex(k.shape):
            t = T[idx]
            if not self._T[0] <= t <= self._T[-1]:
                continue
            j = int(np.searchsorted(self._T, t))
            if self._T[j] == t:
                out[idx] = self.slice_total_variance(j, k[idx])
                continue
            t0, t1 = self._T[j - 1], self._T[j]
            w0 = self.slice_total_variance(j - 1, k[idx])
            w1 = self.slice_total_variance(j, k[idx])
            out[idx] = w0 + (w1 - w0) * (t - t0) / (t1 - t0)
        return out

    def implied_vol(self, k: np.ndarray | float, T: np.ndarray | float) -> np.ndarray:
        """sigma(k, T) = sqrt(w(k, T) / T)."""
        w = self.total_variance(k, T)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.sqrt(w / np.asarray(T, float))

    def atm_term_structure(self) -> pd.DataFrame:
        """ATM (k = 0, i.e. K = F) implied vol of each expiry, from the slice interpolation."""
        rows = []
        for i, s in self.slices.iterrows():
            w = float(self.slice_total_variance(i, 0.0))
            rows.append({"expiry": s["expiry"], "T": s["T"], "dte": s["dte"],
                         "atm_iv": math.sqrt(w / s["T"]) if np.isfinite(w) else math.nan})
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------

def _iv_or_nan(price: float, spot: float, K: float, T: float, r: float, q: float,
               kind: str) -> float:
    if not (np.isfinite(price) and price > 0):
        return math.nan
    return implied_vol(price, spot, K, T, r, q, kind)


def build_surface(cleaned: pd.DataFrame, spot: float, r: float, n_pairs: int = 6,
                  n_reg: int = 20, min_pairs: int = 3, min_otm: int = 8,
                  bid_ask_iv: bool = True, drop_outliers: bool = True) -> VolSurface:
    """Build a VolSurface from a cleaned chain (columns expiry, kind, strike, bid, ask, mid, T).

    Per expiry: forward_from_parity -> otm_quotes -> European implied_vol of
    the mid (also of bid and ask when bid_ask_iv) -> optional flag_outliers
    (removed quotes are kept in ``outliers``). Expiries with fewer than
    min_pairs same-strike call/put pairs or fewer than min_otm valid OTM IVs
    are dropped and recorded in ``dropped``. r is one flat rate for all T.
    """
    need = ["expiry", "kind", "strike", "bid", "ask", "mid", "T"]
    missing = [c for c in need if c not in cleaned.columns]
    if missing:
        raise ValueError(f"cleaned chain is missing columns: {missing}")
    if not (np.isfinite(spot) and spot > 0):
        raise ValueError("spot must be positive")

    slices, points, dropped, outliers = [], [], [], []
    for expiry, g in cleaned.groupby("expiry", sort=True):
        T = float(g["T"].iloc[0])
        calls = g[g["kind"] == "call"].set_index("strike")["mid"]
        puts = g[g["kind"] == "put"].set_index("strike")["mid"]
        common = calls.index.intersection(puts.index)
        if len(common) < min_pairs:
            dropped.append({"expiry": str(pd.Timestamp(expiry).date()), "T": T,
                            "reason": f"only {len(common)} call/put pairs (< {min_pairs})"})
            continue
        fw = forward_from_parity(common.to_numpy(float), calls[common].to_numpy(float),
                                 puts[common].to_numpy(float), T, r, spot, n_pairs, n_reg)
        otm = otm_quotes(g, fw["F"]).copy()
        q = fw["q"]
        otm["iv"] = [_iv_or_nan(p, spot, K, T, r, q, kd)
                     for p, K, kd in zip(otm["mid"], otm["strike"], otm["kind"])]
        if bid_ask_iv:
            for side in ("bid", "ask"):
                otm[f"iv_{side}"] = [_iv_or_nan(p, spot, K, T, r, q, kd)
                                     for p, K, kd in zip(otm[side], otm["strike"], otm["kind"])]
        n_nan = int(otm["iv"].isna().sum())
        otm = otm[otm["iv"].notna()].reset_index(drop=True)
        otm["k"] = np.log(otm["strike"] / fw["F"])
        n_out = 0
        if drop_outliers and bid_ask_iv:
            bad = flag_outliers(otm["k"].to_numpy(), otm["iv"].to_numpy(),
                                (otm["iv_ask"] - otm["iv_bid"]).to_numpy())
            n_out = int(bad.sum())
            if n_out:
                outliers.append(otm[bad].assign(expiry=expiry, T=T))
            otm = otm[~bad].reset_index(drop=True)
        if len(otm) < min_otm:
            dropped.append({"expiry": str(pd.Timestamp(expiry).date()), "T": T,
                            "reason": f"only {len(otm)} valid OTM IVs (< {min_otm})"})
            continue
        otm["expiry"] = expiry
        otm["T"] = T
        otm["dte"] = T * 365.0
        otm["F"], otm["DF"], otm["q"] = fw["F"], fw["DF"], q
        otm["w"] = otm["iv"] ** 2 * T
        points.append(otm)
        slices.append({"expiry": expiry, "T": T, "dte": T * 365.0, **fw,
                       "n_otm": int(len(otm)), "n_iv_nan": n_nan, "n_outliers": n_out,
                       "k_min": float(otm["k"].min()), "k_max": float(otm["k"].max())})

    pts = pd.concat(points, ignore_index=True) if points else pd.DataFrame(
        columns=["expiry", "T", "dte", "strike", "kind", "mid", "k", "iv", "w"])
    sl = pd.DataFrame(slices) if slices else pd.DataFrame(columns=["expiry", "T", "dte", "F", "DF", "q"])
    out = pd.concat(outliers, ignore_index=True) if outliers else pd.DataFrame()
    return VolSurface(sl, pts, dropped, spot=float(spot), r=float(r), outliers=out)


# --------------------------------------------------------------------------
# no-arbitrage diagnostics
# --------------------------------------------------------------------------

def calendar_violations(surface: VolSurface, n_grid: int = 41, tol: float = 0.0) -> pd.DataFrame:
    """For each pair of consecutive expiries, count k-grid points (over the
    overlap of their quoted k ranges) where w(k, T2) < w(k, T1) - tol."""
    rows = []
    sl = surface.slices
    for i in range(len(sl) - 1):
        lo = max(sl.loc[i, "k_min"], sl.loc[i + 1, "k_min"])
        hi = min(sl.loc[i, "k_max"], sl.loc[i + 1, "k_max"])
        row = {"expiry_1": sl.loc[i, "expiry"], "expiry_2": sl.loc[i + 1, "expiry"],
               "k_lo": lo, "k_hi": hi, "n_grid": 0, "n_violations": 0,
               "max_shortfall": 0.0, "k_worst": math.nan}
        if hi > lo:
            k = np.linspace(lo, hi, n_grid)
            d = surface.slice_total_variance(i + 1, k) - surface.slice_total_variance(i, k)
            bad = d < -tol
            row.update(n_grid=n_grid, n_violations=int(bad.sum()))
            if bad.any():
                row.update(max_shortfall=float(-d.min()), k_worst=float(k[np.argmin(d)]))
        rows.append(row)
    return pd.DataFrame(rows)


def _black_call(F: float, K: np.ndarray, T: float, sigma: np.ndarray, DF: float) -> np.ndarray:
    """Discounted Black call: DF * [F N(d1) - K N(d2)], via bs_price with S=F, r=q=0."""
    return DF * np.asarray(bs_price(F, K, T, 0.0, sigma, 0.0, "call"), float)


def butterfly_violations(surface: VolSurface, tol: float = 0.0) -> pd.DataFrame:
    """Per expiry, on quoted strikes K_1 < ... < K_n with call prices C_i = DF*Black(F, K_i, iv_i):
    butterfly B_i = a C_{i-1} + (1-a) C_{i+1} - C_i, a = (K_{i+1}-K_i)/(K_{i+1}-K_{i-1});
    convexity violation if B_i < -tol. Monotonicity violation if a call
    spread C_{i+1} - C_i is outside [-DF (K_{i+1} - K_i) - tol, tol]. tol is in dollars."""
    rows = []
    for _, s in surface.slices.iterrows():
        p = surface.points[surface.points["expiry"] == s["expiry"]].sort_values("strike")
        K = p["strike"].to_numpy(float)
        C = _black_call(s["F"], K, s["T"], p["iv"].to_numpy(float), s["DF"])
        n_conv, worst, k_worst, n_mono = 0, 0.0, math.nan, 0
        if len(K) >= 3:
            a = (K[2:] - K[1:-1]) / (K[2:] - K[:-2])
            B = a * C[:-2] + (1 - a) * C[2:] - C[1:-1]
            bad = B < -tol
            n_conv = int(bad.sum())
            if bad.any():
                j = int(np.argmin(B))
                worst, k_worst = float(-B[j]), float(np.log(K[j + 1] / s["F"]))
        if len(K) >= 2:
            dC, dK = np.diff(C), np.diff(K)
            n_mono = int(((dC > tol) | (dC < -s["DF"] * dK - tol)).sum())
        rows.append({"expiry": s["expiry"], "dte": s["dte"], "n_strikes": int(len(K)),
                     "n_convexity_violations": n_conv, "max_butterfly_deficit": worst,
                     "k_worst": k_worst, "n_monotonicity_violations": n_mono})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# European vs American inversion
# --------------------------------------------------------------------------

def european_vs_american(surface: VolSurface, expiries: list | None = None,
                         per_expiry: int = 15, steps: int = 200) -> pd.DataFrame:
    """Re-invert a subset of the surface's OTM mids with the American CRR tree
    (same S, r, parity q). Columns: iv_eu (Black-Scholes, as in the surface),
    iv_am (American tree), iv_eu_tree (European tree, same steps);
    diff = iv_eu - iv_am (what the choice of model changes);
    diff_eep = iv_eu_tree - iv_am (early exercise only: tree error cancels).

    per_expiry quotes are taken evenly spaced in strike order on each chosen expiry.
    """
    pts = surface.points
    if expiries is None:
        expiries = list(surface.slices["expiry"])
    rows = []
    for exp in expiries:
        p = pts[pts["expiry"] == exp].sort_values("strike").reset_index(drop=True)
        if p.empty:
            continue
        idx = np.unique(np.linspace(0, len(p) - 1, min(per_expiry, len(p))).round().astype(int))
        for _, row in p.iloc[idx].iterrows():
            iv_am = implied_vol_american(row["mid"], surface.spot, row["strike"], row["T"],
                                         surface.r, row["q"], row["kind"], steps=steps)
            iv_et = implied_vol_tree(row["mid"], surface.spot, row["strike"], row["T"],
                                     surface.r, row["q"], row["kind"], steps=steps,
                                     american=False)
            rows.append({"expiry": exp, "dte": row["dte"], "kind": row["kind"],
                         "strike": row["strike"], "k": row["k"], "mid": row["mid"],
                         "iv_eu": row["iv"], "iv_am": iv_am, "iv_eu_tree": iv_et,
                         "diff": row["iv"] - iv_am, "diff_eep": iv_et - iv_am})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# synthetic chains (tests and bias studies)
# --------------------------------------------------------------------------

def synthetic_chain(spot: float, r: float, q: float, vol_fn, days: list[int],
                    strikes: np.ndarray, snapshot_time: str | pd.Timestamp,
                    american: bool = False, steps: int = 200, half_spread: float = 0.0,
                    ticker: str = "SYN") -> pd.DataFrame:
    """Chain in CHAIN_COLUMNS format priced from sigma = vol_fn(k, T), k = ln(K/F),
    F = S e^{(r-q)T}; European BS prices, or CRR American prices when american=True.
    bid/ask = price -/+ half_spread (bid floored at 0)."""
    snap = pd.Timestamp(snapshot_time)
    snap = snap.tz_localize("UTC") if snap.tzinfo is None else snap.tz_convert("UTC")
    snap_ny = snap.tz_convert(EXPIRY_TZ)
    rows = []
    for d in days:
        expiry = pd.Timestamp(snap_ny.date()) + pd.Timedelta(days=d)   # naive date
        T = float(time_to_expiry(pd.Series([expiry]), pd.Series([snap])).iloc[0])
        F = spot * math.exp((r - q) * T)
        for K in strikes:
            sig = float(vol_fn(math.log(K / F), T))
            for kind in ("call", "put"):
                if american:
                    px = crr_price(spot, K, T, r, sig, q, kind, steps=steps, american=True)
                else:
                    px = float(bs_price(spot, K, T, r, sig, q, kind))
                rows.append({"ticker": ticker, "snapshot_time": snap, "spot": spot,
                             "market_state": "REGULAR", "contract_symbol": f"{ticker}{d}{kind[0]}{K}",
                             "expiry": expiry, "kind": kind, "strike": float(K),
                             "bid": max(px - half_spread, 0.0), "ask": px + half_spread,
                             "last": px, "volume": 1.0, "open_interest": 1.0,
                             "last_trade_time": snap, "yf_iv": sig})
    return pd.DataFrame(rows)[CHAIN_COLUMNS]
