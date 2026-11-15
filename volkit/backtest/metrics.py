"""Realised volatility and per-trade performance statistics.

Realised vol is ZERO-MEAN by default: sigma_RV = sqrt(252/n * sum_i ln(S_i/S_{i-1})^2).
That is the quantity a variance swap (and hence VIX) and the gamma P&L of a
delta hedge are written on; demeaning would subtract the drift of a ~21-day
window, which is mostly noise and is not what the short option pays for.
Use demean=True for the textbook sample-std estimator.

Sharpe ratios are computed from non-overlapping trades and annualised with
sqrt(periods_per_year) (12 for monthly trades). With ~100-200 monthly trades
the standard error of an annualised Sharpe is ~0.25-0.35, so every Sharpe is
reported with a bootstrap confidence interval.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

TRADING_DAYS = 252


def log_returns(prices: ArrayLike) -> np.ndarray:
    """Close-to-close log returns ln(S_i / S_{i-1})."""
    p = np.asarray(prices, dtype=float)
    if p.ndim != 1 or p.size < 2:
        raise ValueError("prices must be 1-D with at least two values")
    if np.any(p <= 0.0) or not np.all(np.isfinite(p)):
        raise ValueError("prices must be finite and positive")
    return np.diff(np.log(p))


def realised_vol(prices: ArrayLike, periods_per_year: int = TRADING_DAYS,
                 demean: bool = False) -> float:
    """Annualised close-to-close vol: sqrt(ppy/n * sum r^2) (zero mean) or sqrt(ppy) * std(r, ddof=1)."""
    r = log_returns(prices)
    if demean:
        if r.size < 2:
            raise ValueError("demeaned vol needs at least two returns")
        return float(np.sqrt(periods_per_year) * np.std(r, ddof=1))
    return float(np.sqrt(periods_per_year * np.mean(r**2)))


def _clean(x: ArrayLike) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1 or a.size == 0:
        raise ValueError("input must be a non-empty 1-D array")
    if not np.all(np.isfinite(a)):
        raise ValueError("input contains NaN or inf")
    return a


def t_stat(x: ArrayLike) -> float:
    """t = mean / (std(ddof=1) / sqrt(n)); nan for n < 2 or zero dispersion."""
    a = _clean(x)
    if a.size < 2:
        return float("nan")
    s = np.std(a, ddof=1)
    return float("nan") if s == 0 else float(np.mean(a) / (s / np.sqrt(a.size)))


def sharpe(x: ArrayLike, periods_per_year: float = 12.0) -> float:
    """Annualised Sharpe = mean / std(ddof=1) * sqrt(periods_per_year) (P&L already excess of cash)."""
    a = _clean(x)
    if a.size < 2:
        return float("nan")
    s = np.std(a, ddof=1)
    return float("nan") if s == 0 else float(np.mean(a) / s * np.sqrt(periods_per_year))


def skewness(x: ArrayLike) -> float:
    """Sample skewness m3 / m2^1.5 with population moments (no small-sample correction)."""
    a = _clean(x)
    d = a - a.mean()
    m2 = np.mean(d**2)
    return float("nan") if m2 == 0 else float(np.mean(d**3) / m2**1.5)


def max_drawdown(x: ArrayLike) -> float:
    """Largest peak-to-trough fall of the cumulative sum of x, starting from 0 (>= 0)."""
    cum = np.concatenate([[0.0], np.cumsum(_clean(x))])
    return float(np.max(np.maximum.accumulate(cum) - cum))


def hit_rate(x: ArrayLike) -> float:
    """Fraction of strictly positive entries."""
    return float(np.mean(_clean(x) > 0.0))


def bootstrap_ci(x: ArrayLike, stat=np.mean, n_boot: int = 10_000, level: float = 0.95,
                 seed: int = 12345) -> tuple[float, float]:
    """Percentile bootstrap CI of stat(x), iid resampling of trades with a fixed seed."""
    a = _clean(x)
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    samples = np.array([stat(a[row]) for row in idx]) if stat is not np.mean \
        else a[idx].mean(axis=1)
    lo, hi = np.nanquantile(samples, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


def summarize(x: ArrayLike, periods_per_year: float = 12.0, n_boot: int = 10_000,
              seed: int = 12345) -> dict[str, float]:
    """Aggregate statistics of per-trade P&L: n, mean, median, std, t, Sharpe (+CI), skew, worst, MDD, hit rate, mean CI."""
    a = _clean(x)
    mean_lo, mean_hi = bootstrap_ci(a, np.mean, n_boot=n_boot, seed=seed)
    sr_lo, sr_hi = bootstrap_ci(a, lambda v: sharpe(v, periods_per_year), n_boot=n_boot,
                                seed=seed)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "std": float(np.std(a, ddof=1)) if a.size > 1 else float("nan"),
        "t_stat": t_stat(a),
        "sharpe_ann": sharpe(a, periods_per_year),
        "sharpe_ci95": [sr_lo, sr_hi],
        "skew": skewness(a),
        "worst": float(a.min()),
        "best": float(a.max()),
        "max_drawdown": max_drawdown(a),
        "hit_rate": hit_rate(a),
        "mean_ci95": [mean_lo, mean_hi],
    }
