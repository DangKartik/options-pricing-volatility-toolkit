"""Tests for volkit.pricing.monte_carlo. Deterministic (fixed seeds), no network."""

import dataclasses

import numpy as np
import pytest
from scipy.stats import norm

from volkit.pricing.monte_carlo import MCResult, mc_paths, mc_price


def _ref_bsm(S, K, T, r, sigma, q=0.0, kind="call"):
    """Private reference Black-Scholes-Merton price (independent of volkit)."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


S0, T, R, SIGMA = 100.0, 1.0, 0.05, 0.20


# ---------------------------------------------------------------- accuracy

@pytest.mark.parametrize("antithetic", [True, False])
@pytest.mark.parametrize("q", [0.0, 0.03])
@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
def test_price_within_three_stderr_of_black_scholes(K, kind, q, antithetic):
    res = mc_price(S0, K, T, R, SIGMA, q=q, kind=kind, antithetic=antithetic, seed=42)
    exact = _ref_bsm(S0, K, T, R, SIGMA, q=q, kind=kind)
    assert res.stderr > 0
    assert abs(res.price - exact) < 3.0 * res.stderr


def test_put_call_parity_same_seed():
    # Same seed -> same Z for call and put, so C - P is an MC estimate of the forward.
    call = mc_price(S0, 100.0, T, R, SIGMA, q=0.02, kind="call", seed=7)
    put = mc_price(S0, 100.0, T, R, SIGMA, q=0.02, kind="put", seed=7)
    forward_value = S0 * np.exp(-0.02 * T) - 100.0 * np.exp(-R * T)
    assert call.price - put.price == pytest.approx(forward_value, abs=0.1)


# ------------------------------------------------------- stderr is honest

@pytest.mark.parametrize("antithetic", [True, False])
@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
def test_reported_stderr_matches_empirical_spread(K, antithetic):
    """Std of the estimator across independent seeds must equal the reported stderr.

    This fails if the antithetic stderr is computed from all n_paths payoffs as if
    they were independent (that overstates the stderr by a large factor).
    """
    results = [
        mc_price(S0, K, T, R, SIGMA, n_paths=2_000, antithetic=antithetic, seed=s)
        for s in range(200)
    ]
    empirical_std = np.std([res.price for res in results], ddof=1)
    mean_reported = np.mean([res.stderr for res in results])
    assert mean_reported == pytest.approx(empirical_std, rel=0.20)


# ------------------------------------------------------ variance reduction

@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
def test_antithetic_reduces_stderr(K):
    anti = mc_price(S0, K, T, R, SIGMA, n_paths=200_000, antithetic=True, seed=1)
    plain = mc_price(S0, K, T, R, SIGMA, n_paths=200_000, antithetic=False, seed=1)
    assert anti.stderr < plain.stderr


def test_stderr_scales_like_inverse_sqrt_n():
    small = mc_price(S0, 100.0, T, R, SIGMA, n_paths=50_000, seed=3)
    large = mc_price(S0, 100.0, T, R, SIGMA, n_paths=200_000, seed=3)
    assert small.stderr / large.stderr == pytest.approx(2.0, rel=0.05)


# --------------------------------------------------------- reproducibility

def test_same_seed_same_result_different_seed_different():
    a = mc_price(S0, 100.0, T, R, SIGMA, n_paths=10_000, seed=123)
    b = mc_price(S0, 100.0, T, R, SIGMA, n_paths=10_000, seed=123)
    c = mc_price(S0, 100.0, T, R, SIGMA, n_paths=10_000, seed=124)
    assert a == b
    assert a.price != c.price


# ---------------------------------------------------------------- MCResult

def test_result_fields_conf_int_and_frozen():
    res = mc_price(S0, 100.0, T, R, SIGMA, n_paths=10_000, seed=0)
    assert isinstance(res, MCResult)
    assert res.n_paths == 10_000
    lo, hi = res.conf_int()
    assert lo == pytest.approx(res.price - 1.96 * res.stderr)
    assert hi == pytest.approx(res.price + 1.96 * res.stderr)
    lo3, hi3 = res.conf_int(z=3.0)
    assert hi3 - lo3 == pytest.approx(6.0 * res.stderr)
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.price = 0.0


# ------------------------------------------------------------- edge cases

@pytest.mark.parametrize(
    "K, kind, intrinsic",
    [(90.0, "call", 10.0), (110.0, "call", 0.0), (110.0, "put", 10.0), (90.0, "put", 0.0)],
)
def test_zero_expiry_returns_intrinsic(K, kind, intrinsic):
    res = mc_price(S0, K, 0.0, R, SIGMA, kind=kind, seed=0)
    assert res.price == intrinsic
    assert res.stderr == 0.0


def test_zero_vol_is_deterministic_forward_payoff():
    res = mc_price(S0, 100.0, T, R, 0.0, n_paths=1_000, seed=0)
    assert res.price == pytest.approx(S0 - 100.0 * np.exp(-R * T))
    assert res.stderr == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(kind="straddle"),
        dict(S=-100.0),
        dict(S=0.0),
        dict(K=-100.0),
        dict(T=-1.0),
        dict(sigma=-0.2),
        dict(n_paths=1),
        dict(n_paths=0),
        dict(n_paths=1_001, antithetic=True),
    ],
)
def test_mc_price_invalid_inputs_raise(kwargs):
    args = dict(S=S0, K=100.0, T=T, r=R, sigma=SIGMA)
    args.update(kwargs)
    with pytest.raises(ValueError):
        mc_price(**args)


def test_odd_n_paths_allowed_without_antithetic():
    res = mc_price(S0, 100.0, T, R, SIGMA, n_paths=1_001, antithetic=False, seed=0)
    assert res.n_paths == 1_001


# ---------------------------------------------------------------- mc_paths

Q = 0.02
N_STEPS, N_PATHS = 50, 40_000


@pytest.fixture(scope="module")
def paths():
    return mc_paths(S0, T, R, SIGMA, q=Q, n_steps=N_STEPS, n_paths=N_PATHS, seed=11)


def test_paths_shape_start_and_positivity(paths):
    assert paths.shape == (N_PATHS, N_STEPS + 1)
    assert np.all(paths[:, 0] == S0)
    assert np.all(paths > 0)


def test_paths_terminal_mean_is_forward(paths):
    terminal = paths[:, -1]
    stderr = terminal.std(ddof=1) / np.sqrt(N_PATHS)
    assert abs(terminal.mean() - S0 * np.exp((R - Q) * T)) < 3.0 * stderr


def test_paths_log_return_moments(paths):
    log_ret = np.log(paths[:, -1] / S0)
    assert log_ret.std(ddof=1) == pytest.approx(SIGMA * np.sqrt(T), rel=0.02)
    assert log_ret.mean() == pytest.approx((R - Q - 0.5 * SIGMA**2) * T, abs=0.005)


def test_paths_discounted_price_is_martingale(paths):
    # E[exp(-(r - q) t) S_t] = S at every grid time, not only at T.
    t = np.linspace(0.0, T, N_STEPS + 1)
    discounted = paths * np.exp(-(R - Q) * t)
    stderr = discounted.std(axis=0, ddof=1) / np.sqrt(N_PATHS)
    assert np.all(np.abs(discounted.mean(axis=0) - S0) <= 3.5 * stderr + 1e-12)


def test_paths_price_matches_black_scholes(paths):
    # Pricing off the multi-step terminal column agrees with the closed form.
    payoff = np.exp(-R * T) * np.maximum(paths[:, -1] - 100.0, 0.0)
    stderr = payoff.std(ddof=1) / np.sqrt(N_PATHS)
    assert abs(payoff.mean() - _ref_bsm(S0, 100.0, T, R, SIGMA, q=Q)) < 3.0 * stderr


def test_paths_reproducible():
    a = mc_paths(S0, T, R, SIGMA, n_steps=10, n_paths=100, seed=5)
    b = mc_paths(S0, T, R, SIGMA, n_steps=10, n_paths=100, seed=5)
    c = mc_paths(S0, T, R, SIGMA, n_steps=10, n_paths=100, seed=6)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize(
    "kwargs",
    [dict(S=-1.0), dict(T=-1.0), dict(sigma=-0.2), dict(n_steps=0), dict(n_paths=0)],
)
def test_mc_paths_invalid_inputs_raise(kwargs):
    args = dict(S=S0, T=T, r=R, sigma=SIGMA)
    args.update(kwargs)
    with pytest.raises(ValueError):
        mc_paths(**args)
