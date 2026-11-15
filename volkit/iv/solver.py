"""Implied volatility: invert Black-Scholes (European) or a CRR tree (American).

European: safeguarded Newton-Raphson. Newton uses the analytic vega and starts
at the inflection point of price in sigma, sigma0 = sqrt(2 |ln(F/K)| / T)
(Manaster-Koehler; vega is largest there), floored at 0.1 so an at-the-money
start is not zero. A bracket [lo, hi] that always contains the root is
shrunk after every evaluation; if a Newton step leaves it, vega is tiny, or
Newton has not converged after ``max_iter`` steps, the solver switches to
``scipy.optimize.brentq`` on the current bracket.

nan is returned (never an exception) for prices the model cannot produce:
outside the no-arbitrage bounds, or so close to a bound that sigma is not
identifiable inside [SIGMA_LO, SIGMA_HI]. Invalid inputs (kind, S <= 0, K <= 0)
raise ValueError, per the project conventions.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.optimize import brentq

from volkit.pricing.binomial import crr_price
from volkit.pricing.black_scholes import bs_price, bs_vega

__all__ = [
    "SIGMA_LO",
    "SIGMA_HI",
    "implied_vol",
    "implied_vol_american",
    "implied_vol_tree",
    "implied_vol_array",
    "implied_vol_frame",
]

SIGMA_LO = 1e-6
SIGMA_HI = 5.0
_MIN_VEGA = 1e-10      # below this a Newton step is numerically meaningless


def _check_common(S: float, K: float, kind: str) -> None:
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    if not (S > 0 and K > 0):
        raise ValueError("S and K must be strictly positive")


def european_bounds(S: float, K: float, T: float, r: float, q: float = 0.0,
                    kind: str = "call") -> tuple[float, float]:
    """No-arbitrage bounds of a European option: (discounted intrinsic, S e^{-qT} or K e^{-rT})."""
    fwd_s, pv_k = S * math.exp(-q * T), K * math.exp(-r * T)
    if kind == "call":
        return max(fwd_s - pv_k, 0.0), fwd_s
    return max(pv_k - fwd_s, 0.0), pv_k


def _implied_vol_info(price: float, S: float, K: float, T: float, r: float,
                      q: float = 0.0, kind: str = "call", tol: float = 1e-8,
                      max_iter: int = 100) -> tuple[float, str]:
    """implied_vol plus the path taken: 'newton', 'brent', or 'nan:<reason>'."""
    _check_common(S, K, kind)
    if not all(map(math.isfinite, (price, S, K, T, r, q))):
        return math.nan, "nan:non_finite"
    if T <= 0:
        return math.nan, "nan:expired"
    lower, upper = european_bounds(S, K, T, r, q, kind)
    if price < lower or price >= upper:
        return math.nan, "nan:arbitrage_bounds"

    def f(sig: float) -> float:
        return float(bs_price(S, K, T, r, sig, q, kind)) - price

    lo, hi = SIGMA_LO, SIGMA_HI
    f_lo, f_hi = f(lo), f(hi)
    if f_lo >= 0:   # price indistinguishable from the zero-vol value
        return math.nan, "nan:below_sigma_lo"
    if f_hi < 0:
        return math.nan, "nan:above_sigma_hi"

    x = math.log(S / K) + (r - q) * T            # ln(F/K)
    sig = min(max(math.sqrt(2.0 * abs(x) / T), 0.1), hi)
    for _ in range(max_iter):
        g = f(sig)
        if g == 0.0:
            return sig, "newton"
        if g > 0:
            hi = min(hi, sig)
        else:
            lo = max(lo, sig)
        vega = float(bs_vega(S, K, T, r, sig, q, kind))
        if vega < _MIN_VEGA:
            break
        step = g / vega
        new = sig - step
        if not lo < new < hi:
            break
        if abs(step) < tol:
            return new, "newton"
        sig = new
    root = brentq(f, lo, hi, xtol=tol, rtol=4 * np.finfo(float).eps, maxiter=500)
    return float(root), "brent"


def implied_vol(price: float, S: float, K: float, T: float, r: float,
                q: float = 0.0, kind: str = "call", tol: float = 1e-8,
                max_iter: int = 100) -> float:
    """Black-Scholes implied vol: solve bs_price(S,K,T,r,sigma,q,kind) = price for sigma.

    Safeguarded Newton (analytic vega) with brentq fallback on [1e-6, 5]; nan
    if T <= 0, price is outside [discounted intrinsic, upper bound), or the
    root lies outside [1e-6, 5].
    """
    return _implied_vol_info(price, S, K, T, r, q, kind, tol, max_iter)[0]


def american_sigma_floor(T: float, r: float, q: float, steps: int) -> float:
    """Smallest sigma for which the CRR tree is arbitrage-free: 1.01 |r-q| sqrt(T/steps), >= 1e-4."""
    return max(1e-4, 1.01 * abs(r - q) * math.sqrt(T / steps))


def implied_vol_tree(price: float, S: float, K: float, T: float, r: float,
                     q: float = 0.0, kind: str = "call", steps: int = 200,
                     american: bool = True, tol: float = 1e-6,
                     sigma_hi: float = SIGMA_HI) -> float:
    """CRR-tree implied vol: brentq on crr_price(..., steps, american) = price.

    nan if T <= 0, price is below the immediate-exercise value (American) or
    the discounted intrinsic (European), price >= S (call) / K (put), or the
    root lies outside [american_sigma_floor, sigma_hi]. Bracketing only: the
    tree has no analytic vega and its price is not smooth in sigma.
    """
    _check_common(S, K, kind)
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if not all(map(math.isfinite, (price, S, K, T, r, q))):
        return math.nan
    if T <= 0:
        return math.nan
    if american:
        lower = max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
    else:
        lower = european_bounds(S, K, T, r, q, kind)[0]
    upper = S if kind == "call" else K
    if price < lower or price >= upper:
        return math.nan

    def f(sig: float) -> float:
        return crr_price(S, K, T, r, sig, q, kind, steps=steps, american=american) - price

    lo = american_sigma_floor(T, r, q, steps)
    f_lo, f_hi = f(lo), f(sigma_hi)
    if f_lo >= 0 or f_hi < 0:
        return math.nan
    return float(brentq(f, lo, sigma_hi, xtol=tol, maxiter=200))


def implied_vol_american(price: float, S: float, K: float, T: float, r: float,
                         q: float = 0.0, kind: str = "call", steps: int = 200,
                         tol: float = 1e-6, sigma_hi: float = SIGMA_HI) -> float:
    """American implied vol: brentq on crr_price(..., steps, american=True) = price.

    nan if T <= 0, price < immediate-exercise value, price >= S (call) / K (put),
    or the root lies outside [american_sigma_floor, sigma_hi].
    """
    return implied_vol_tree(price, S, K, T, r, q, kind, steps, True, tol, sigma_hi)


def implied_vol_array(price: ArrayLike, S: ArrayLike, K: ArrayLike, T: ArrayLike,
                      r: ArrayLike, q: ArrayLike = 0.0, kind: ArrayLike = "call",
                      american: bool = False, **kw) -> np.ndarray:
    """Element-wise implied_vol (or implied_vol_american) over broadcast arrays."""
    arrs = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (price, S, K, T, r, q)))
    kinds = np.broadcast_to(np.asarray(kind, dtype=object), arrs[0].shape)
    fn = implied_vol_american if american else implied_vol
    out = np.empty(arrs[0].shape, dtype=float)
    for idx in np.ndindex(out.shape):
        p, s, k, t, rr, qq = (float(a[idx]) for a in arrs)
        out[idx] = fn(p, s, k, t, rr, qq, str(kinds[idx]), **kw)
    return out


def implied_vol_frame(df: pd.DataFrame, price_col: str = "mid", S: float | str = "spot",
                      r: float | str = "r", q: float | str = "q",
                      american: bool = False, **kw) -> pd.Series:
    """Implied vol for every row of ``df`` (needs strike, T, kind); scalars or column names for S, r, q."""
    def col(v):
        return df[v].to_numpy(dtype=float) if isinstance(v, str) else v

    for c in ("strike", "T", "kind", price_col):
        if c not in df.columns:
            raise ValueError(f"frame is missing column {c!r}")
    iv = implied_vol_array(df[price_col].to_numpy(dtype=float), col(S),
                           df["strike"].to_numpy(dtype=float), df["T"].to_numpy(dtype=float),
                           col(r), col(q), df["kind"].to_numpy(dtype=object),
                           american=american, **kw)
    return pd.Series(iv, index=df.index, name="iv")
