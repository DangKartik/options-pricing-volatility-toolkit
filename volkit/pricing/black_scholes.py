"""Black-Scholes-Merton closed-form prices and Greeks for European options.

Model: dS = (r - q) S dt + sigma S dW under the risk-neutral measure, so

    d1 = [ln(S/K) + (r - q + sigma^2 / 2) T] / (sigma sqrt(T))
    d2 = d1 - sigma sqrt(T)
    call = S e^{-qT} N(d1) - K e^{-rT} N(d2)
    put  = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)

Every formula below is written once for both option kinds using the sign
w = +1 (call) or w = -1 (put):  price = w [S e^{-qT} N(w d1) - K e^{-rT} N(w d2)].

Degenerate case sigma * sqrt(T) == 0 (expiry, or zero volatility): the
terminal price is known with certainty, S_T = S e^{(r-q)T}, so the option is
worth its discounted intrinsic value on the forward. This is handled by setting
d1 = d2 = +inf (forward above strike), -inf (below) or 0 (exactly at the
strike), which makes N(.) equal to 1, 0 or 0.5. The terms that contain the
normal density (gamma, vega, the diffusion part of theta) are set to 0 there;
exactly at the strike the true gamma and theta are infinite, and 0 is returned
instead of inf by convention.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

__all__ = [
    "bs_price",
    "bs_delta",
    "bs_gamma",
    "bs_vega",
    "bs_theta",
    "bs_rho",
    "bs_greeks",
]


def _kind_sign(kind: str) -> float:
    """Return +1.0 for a call and -1.0 for a put; anything else is an error."""
    if kind == "call":
        return 1.0
    if kind == "put":
        return -1.0
    raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def _prepare(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike,
             sigma: ArrayLike, q: ArrayLike) -> tuple[np.ndarray, ...]:
    """Validate the inputs and return (S, K, T, r, sigma, q, d1, d2) as float arrays.

    The arrays are NOT broadcast to a common shape here; numpy broadcasting in
    the calling formula does that. d1 and d2 already have the broadcast shape.
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    q = np.asarray(q, dtype=float)

    if np.any(S <= 0.0):
        raise ValueError("S must be strictly positive")
    if np.any(K <= 0.0):
        raise ValueError("K must be strictly positive")
    if np.any(T < 0.0):
        raise ValueError("T must be non-negative")
    if np.any(sigma < 0.0):
        raise ValueError("sigma must be non-negative")

    total_vol = sigma * np.sqrt(T)                 # sigma * sqrt(T)
    log_moneyness = np.log(S / K) + (r - q) * T    # ln(F / K), F = S e^{(r-q)T}
    positive = total_vol > 0.0

    # Regular case. Where total_vol == 0 divide by 1 instead (value discarded).
    safe_vol = np.where(positive, total_vol, 1.0)
    d1_regular = (log_moneyness + 0.5 * total_vol**2) / safe_vol

    # Degenerate case: +inf, -inf or 0 according to the sign of ln(F / K).
    with np.errstate(invalid="ignore"):            # inf * 0 = nan is handled next line
        d1_degenerate = np.where(log_moneyness == 0.0, 0.0,
                                 np.inf * np.sign(log_moneyness))

    d1 = np.where(positive, d1_regular, d1_degenerate)
    d2 = d1 - total_vol
    return S, K, T, r, sigma, q, d1, d2


def _density_over(d1: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Return n(d1) / denominator, defined as 0 where the denominator is 0."""
    nonzero = denominator > 0.0
    safe = np.where(nonzero, denominator, 1.0)
    return np.where(nonzero, norm.pdf(d1) / safe, 0.0)


def _output(value: np.ndarray) -> float | np.ndarray:
    """Scalars in -> Python float out; arrays in -> ndarray out."""
    value = np.asarray(value, dtype=float)
    return float(value) if value.ndim == 0 else value


def bs_price(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike,
             sigma: ArrayLike, q: ArrayLike = 0.0,
             kind: str = "call") -> float | np.ndarray:
    """European option price: w [S e^{-qT} N(w d1) - K e^{-rT} N(w d2)], w = +1 call, -1 put."""
    w = _kind_sign(kind)
    S, K, T, r, sigma, q, d1, d2 = _prepare(S, K, T, r, sigma, q)
    price = w * (S * np.exp(-q * T) * norm.cdf(w * d1)
                 - K * np.exp(-r * T) * norm.cdf(w * d2))
    return _output(price)


def bs_delta(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike,
             sigma: ArrayLike, q: ArrayLike = 0.0,
             kind: str = "call") -> float | np.ndarray:
    """delta = dV/dS = w e^{-qT} N(w d1)."""
    w = _kind_sign(kind)
    S, K, T, r, sigma, q, d1, d2 = _prepare(S, K, T, r, sigma, q)
    return _output(w * np.exp(-q * T) * norm.cdf(w * d1))


def bs_gamma(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike,
             sigma: ArrayLike, q: ArrayLike = 0.0,
             kind: str = "call") -> float | np.ndarray:
    """gamma = d2V/dS2 = e^{-qT} n(d1) / (S sigma sqrt(T)); same for call and put."""
    _kind_sign(kind)  # validated for a consistent interface; gamma does not depend on it
    S, K, T, r, sigma, q, d1, d2 = _prepare(S, K, T, r, sigma, q)
    return _output(np.exp(-q * T) * _density_over(d1, S * sigma * np.sqrt(T)))


def bs_vega(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike,
            sigma: ArrayLike, q: ArrayLike = 0.0,
            kind: str = "call") -> float | np.ndarray:
    """vega = dV/dsigma = S e^{-qT} n(d1) sqrt(T), per 1.00 of vol; same for call and put."""
    _kind_sign(kind)  # validated for a consistent interface; vega does not depend on it
    S, K, T, r, sigma, q, d1, d2 = _prepare(S, K, T, r, sigma, q)
    # n(+-inf) = 0 and sqrt(0) = 0, so the degenerate cases need no special handling,
    # except sigma == 0 at the strike (d1 = 0), where the convention is vega = 0.
    density = np.where(sigma * np.sqrt(T) > 0.0, norm.pdf(d1), 0.0)
    return _output(S * np.exp(-q * T) * density * np.sqrt(T))


def bs_theta(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike,
             sigma: ArrayLike, q: ArrayLike = 0.0,
             kind: str = "call") -> float | np.ndarray:
    """theta = dV/dt = -dV/dT, per year.

    theta = -S e^{-qT} n(d1) sigma / (2 sqrt(T))
            - w r K e^{-rT} N(w d2) + w q S e^{-qT} N(w d1)
    """
    w = _kind_sign(kind)
    S, K, T, r, sigma, q, d1, d2 = _prepare(S, K, T, r, sigma, q)
    # sigma / (2 sqrt(T)) = sigma^2 / (2 sigma sqrt(T)), written so that the
    # denominator is the total vol and the term is 0 when that is 0.
    diffusion = -0.5 * S * np.exp(-q * T) * sigma**2 * _density_over(d1, sigma * np.sqrt(T))
    carry = (-w * r * K * np.exp(-r * T) * norm.cdf(w * d2)
             + w * q * S * np.exp(-q * T) * norm.cdf(w * d1))
    return _output(diffusion + carry)


def bs_rho(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike,
           sigma: ArrayLike, q: ArrayLike = 0.0,
           kind: str = "call") -> float | np.ndarray:
    """rho = dV/dr = w K T e^{-rT} N(w d2), per 1.00 of rate."""
    w = _kind_sign(kind)
    S, K, T, r, sigma, q, d1, d2 = _prepare(S, K, T, r, sigma, q)
    return _output(w * K * T * np.exp(-r * T) * norm.cdf(w * d2))


def bs_greeks(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike,
              sigma: ArrayLike, q: ArrayLike = 0.0,
              kind: str = "call") -> dict[str, float | np.ndarray]:
    """All five analytic Greeks as a dict with keys delta, gamma, vega, theta, rho."""
    args = (S, K, T, r, sigma, q, kind)
    return {
        "delta": bs_delta(*args),
        "gamma": bs_gamma(*args),
        "vega": bs_vega(*args),
        "theta": bs_theta(*args),
        "rho": bs_rho(*args),
    }
