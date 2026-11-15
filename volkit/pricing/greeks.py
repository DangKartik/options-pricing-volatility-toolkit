"""Finite-difference Greeks for any pricer with the standard volkit signature.

Central differences, with truncation error O(h^2):

    first derivative   f'(x)  ~ [f(x + h) - f(x - h)] / (2h)
    second derivative  f''(x) ~ [f(x + h) - 2 f(x) + f(x - h)] / h^2

Bump sizes are a trade-off. A smaller h reduces truncation error, but a
numerical pricer is not smooth at small scales: the terminal nodes of an
N-step CRR tree are spaced 2 sigma sqrt(T/N) apart in log-price, and a spot
bump smaller than that spacing gives a meaningless gamma (the second
difference just measures whether a node crossed the strike). The price also
only curves over a distance of about S sigma sqrt(T), so that is the natural
unit for the spot bump. The default spot bump is therefore

    h_S = 0.2 * sigma * sqrt(T) * S

which is about 2.2 node spacings for N = 500 (0.2 * sqrt(500) / 2) and small
enough that the truncation error on a smooth pricer is below 1% of gamma.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

__all__ = ["fd_greeks"]

# Default spot bump as a fraction of one standard deviation of ln(S_T).
_SPOT_BUMP_IN_STDEVS = 0.2
# Fallback relative spot bump when sigma * sqrt(T) is zero.
_FALLBACK_REL_BUMP_S = 1e-2


def fd_greeks(price_fn: Callable[..., Any], S: float, K: float, T: float,
              r: float, sigma: float, q: float = 0.0, kind: str = "call", *,
              rel_bump_S: float | None = None, rel_bump_sigma: float = 1e-2,
              rel_bump_T: float = 1e-2, bump_r: float = 1e-4,
              **kw: Any) -> dict[str, float]:
    """Delta, gamma, vega, theta, rho of ``price_fn`` by central finite differences.

    ``price_fn(S, K, T, r, sigma, q, kind, **kw)`` must return a number.
    Bumps: S, sigma and T are bumped by a fraction of their own value
    (``rel_bump_*``); r is bumped by the absolute amount ``bump_r`` because r
    can be zero or negative. ``rel_bump_S=None`` means 0.2 * sigma * sqrt(T)
    (see the module docstring); pass a small number such as 1e-4 for a smooth
    closed-form pricer. Theta is -dV/dT. If sigma or T is zero a central
    difference would need a negative value, so a one-sided (forward)
    difference is used for that Greek instead. Scalar inputs only.
    """
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    if S <= 0.0:
        raise ValueError("S must be strictly positive")
    if T < 0.0 or sigma < 0.0:
        raise ValueError("T and sigma must be non-negative")
    if rel_bump_S is None:
        rel_bump_S = _SPOT_BUMP_IN_STDEVS * sigma * math.sqrt(T)
        if rel_bump_S == 0.0:
            rel_bump_S = _FALLBACK_REL_BUMP_S
    for name, bump in (("rel_bump_S", rel_bump_S), ("rel_bump_sigma", rel_bump_sigma),
                       ("rel_bump_T", rel_bump_T)):
        if not 0.0 < bump < 1.0:
            raise ValueError(f"{name} must be in (0, 1), got {bump}")
    if bump_r <= 0.0:
        raise ValueError(f"bump_r must be positive, got {bump_r}")

    def price(S_: float = S, T_: float = T, r_: float = r, sigma_: float = sigma) -> float:
        """Price with some of the inputs replaced by bumped values."""
        return float(price_fn(S_, K, T_, r_, sigma_, q, kind, **kw))

    base = price()

    # Delta and gamma share the same two bumped prices.
    h_S = rel_bump_S * S
    up_S, down_S = price(S_=S + h_S), price(S_=S - h_S)
    delta = (up_S - down_S) / (2.0 * h_S)
    gamma = (up_S - 2.0 * base + down_S) / h_S**2

    # Vega: central if sigma > 0, forward difference from sigma = 0 otherwise.
    if sigma > 0.0:
        h_sigma = rel_bump_sigma * sigma
        vega = (price(sigma_=sigma + h_sigma) - price(sigma_=sigma - h_sigma)) / (2.0 * h_sigma)
    else:
        h_sigma = rel_bump_sigma
        vega = (price(sigma_=h_sigma) - base) / h_sigma

    # Theta = -dV/dT: central if T > 0, forward difference from T = 0 otherwise.
    if T > 0.0:
        h_T = rel_bump_T * T
        theta = -(price(T_=T + h_T) - price(T_=T - h_T)) / (2.0 * h_T)
    else:
        h_T = rel_bump_T
        theta = -(price(T_=h_T) - base) / h_T

    rho = (price(r_=r + bump_r) - price(r_=r - bump_r)) / (2.0 * bump_r)

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
