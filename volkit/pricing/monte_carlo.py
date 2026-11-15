"""Monte Carlo pricing of European options under geometric Brownian motion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MCResult:
    """Monte Carlo estimate: price, its standard error, and paths used."""

    price: float
    stderr: float
    n_paths: int

    def conf_int(self, z: float = 1.96) -> tuple[float, float]:
        """Normal confidence interval price +/- z * stderr (z=1.96 -> 95%)."""
        return (self.price - z * self.stderr, self.price + z * self.stderr)


def _check_market_inputs(S: float, T: float, sigma: float) -> None:
    """Raise ValueError unless S > 0, T >= 0 and sigma >= 0."""
    if S <= 0:
        raise ValueError(f"S must be positive, got {S}")
    if T < 0:
        raise ValueError(f"T must be non-negative, got {T}")
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")


def _payoff(S_T: np.ndarray | float, K: float, kind: str) -> np.ndarray:
    """Vanilla payoff: max(S_T - K, 0) for a call, max(K - S_T, 0) for a put."""
    if kind == "call":
        return np.maximum(S_T - K, 0.0)
    return np.maximum(K - S_T, 0.0)


def mc_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    kind: str = "call",
    n_paths: int = 200_000,
    antithetic: bool = True,
    seed: int | None = None,
) -> MCResult:
    """European option price by Monte Carlo under risk-neutral GBM.

    Exact one-step terminal draw S_T = S exp((r - q - sigma^2/2) T + sigma sqrt(T) Z),
    price = exp(-r T) * mean(payoff). With antithetic=True, n_paths/2 normals Z are
    drawn and used as (Z, -Z); n_paths must then be even (odd raises ValueError).
    """
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    _check_market_inputs(S, T, sigma)
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    if n_paths < 2:
        raise ValueError(f"n_paths must be at least 2, got {n_paths}")
    if antithetic and n_paths % 2 != 0:
        raise ValueError(f"n_paths must be even when antithetic=True, got {n_paths}")

    if T == 0:
        # Expiry is now: the option is worth its intrinsic value with certainty.
        return MCResult(price=float(_payoff(S, K, kind)), stderr=0.0, n_paths=n_paths)

    rng = np.random.default_rng(seed)
    drift = (r - q - 0.5 * sigma**2) * T
    vol = sigma * np.sqrt(T)
    discount = np.exp(-r * T)

    if antithetic:
        Z = rng.standard_normal(n_paths // 2)
        payoff_plus = _payoff(S * np.exp(drift + vol * Z), K, kind)
        payoff_minus = _payoff(S * np.exp(drift - vol * Z), K, kind)
        # The two payoffs in a pair share the same Z, so they are (negatively)
        # correlated, not independent. The pair AVERAGES are independent across
        # pairs, so those are the samples whose mean and standard error we take.
        samples = discount * 0.5 * (payoff_plus + payoff_minus)
    else:
        Z = rng.standard_normal(n_paths)
        samples = discount * _payoff(S * np.exp(drift + vol * Z), K, kind)

    # Standard error of a mean of independent samples: sample std / sqrt(count).
    # len(samples) is n_paths/2 with antithetic, n_paths without.
    price = float(np.mean(samples))
    stderr = float(np.std(samples, ddof=1) / np.sqrt(len(samples)))
    return MCResult(price=price, stderr=stderr, n_paths=n_paths)


def mc_paths(
    S: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    n_steps: int = 252,
    n_paths: int = 10_000,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate risk-neutral GBM paths; returns shape (n_paths, n_steps + 1).

    Exact scheme on a uniform grid dt = T / n_steps (no discretisation bias):
    S_{t+dt} = S_t exp((r - q - sigma^2/2) dt + sigma sqrt(dt) Z). Column 0 equals S.
    """
    _check_market_inputs(S, T, sigma)
    if n_steps < 1:
        raise ValueError(f"n_steps must be at least 1, got {n_steps}")
    if n_paths < 1:
        raise ValueError(f"n_paths must be at least 1, got {n_paths}")

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    Z = rng.standard_normal((n_paths, n_steps))
    log_increments = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    # Log-price is a cumulative sum of independent normal increments.
    log_paths = np.zeros((n_paths, n_steps + 1))
    log_paths[:, 1:] = np.cumsum(log_increments, axis=1)
    return S * np.exp(log_paths)
