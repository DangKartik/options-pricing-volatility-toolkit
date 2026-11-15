"""Tests for the Cox-Ross-Rubinstein binomial tree."""

import numpy as np
import pytest
from scipy.stats import norm

from volkit.pricing.binomial import crr_price


def _bsm(S, K, T, r, sigma, q=0.0, kind="call"):
    """Reference Black-Scholes-Merton price, kept local so this file is self-contained."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


S, T, R, SIGMA = 100.0, 1.0, 0.05, 0.20
STRIKES = [80.0, 100.0, 120.0]
YIELDS = [0.0, 0.03]
KINDS = ["call", "put"]


# ---------------------------------------------------------------- European


@pytest.mark.parametrize("K", STRIKES)
@pytest.mark.parametrize("q", YIELDS)
@pytest.mark.parametrize("kind", KINDS)
def test_european_converges_to_black_scholes(K, q, kind):
    tree = crr_price(S, K, T, R, SIGMA, q, kind, steps=2000)
    exact = _bsm(S, K, T, R, SIGMA, q, kind)
    # Measured worst case on this grid is 1.0e-3 (ATM); 2e-3 leaves a 2x margin.
    assert abs(tree - exact) < 2e-3


def _mean_abs_error(K, q, kind, steps):
    """Average |tree - BSM| over steps and steps + 1.

    The CRR error flips sign between odd and even step counts, so a single
    step count can look accidentally good; averaging a neighbouring pair
    measures the size of the oscillation envelope instead.
    """
    exact = _bsm(S, K, T, R, SIGMA, q, kind)
    errors = [abs(crr_price(S, K, T, R, SIGMA, q, kind, steps=n) - exact) for n in (steps, steps + 1)]
    return sum(errors) / 2


@pytest.mark.parametrize("K", STRIKES)
@pytest.mark.parametrize("q", YIELDS)
@pytest.mark.parametrize("kind", KINDS)
def test_european_error_is_first_order_in_steps(K, q, kind):
    coarse = _mean_abs_error(K, q, kind, 100)
    fine = _mean_abs_error(K, q, kind, 1000)
    # O(1/steps) predicts a 10x reduction for 10x the steps (measured 10x to 15x).
    # O(1/sqrt(steps)) would give only 3.2x, so a factor of 5 separates the two.
    assert coarse / fine > 5.0


@pytest.mark.parametrize("K", STRIKES)
@pytest.mark.parametrize("q", YIELDS)
def test_put_call_parity_is_exact_on_the_tree(K, q):
    # The tree's risk-neutral mean of S_T is exactly S exp((r-q)T), so
    # C - P = S exp(-qT) - K exp(-rT) holds to float error at any step count.
    steps = 300
    call = crr_price(S, K, T, R, SIGMA, q, "call", steps=steps)
    put = crr_price(S, K, T, R, SIGMA, q, "put", steps=steps)
    forward_value = S * np.exp(-q * T) - K * np.exp(-R * T)
    assert call - put == pytest.approx(forward_value, abs=1e-9)


# ---------------------------------------------------------------- American


@pytest.mark.parametrize("K", STRIKES)
def test_american_call_without_dividends_equals_european(K):
    # With q = 0 and r >= 0 early exercise of a call is never optimal.
    european = crr_price(S, K, T, R, SIGMA, 0.0, "call", steps=500)
    american = crr_price(S, K, T, R, SIGMA, 0.0, "call", steps=500, american=True)
    assert american == pytest.approx(european, abs=1e-10)


@pytest.mark.parametrize("K", STRIKES)
@pytest.mark.parametrize("q", YIELDS)
@pytest.mark.parametrize("kind", KINDS)
def test_american_is_worth_at_least_european(K, q, kind):
    european = crr_price(S, K, T, R, SIGMA, q, kind, steps=300)
    american = crr_price(S, K, T, R, SIGMA, q, kind, steps=300, american=True)
    assert american >= european - 1e-12


def test_american_itm_put_is_strictly_worth_more_than_european():
    european = crr_price(S, 120.0, T, R, SIGMA, 0.0, "put", steps=500)
    american = crr_price(S, 120.0, T, R, SIGMA, 0.0, "put", steps=500, american=True)
    assert american > european + 0.5  # measured premium is about 1.9


def test_american_call_with_large_dividend_yield_is_worth_more_than_european():
    european = crr_price(S, 100.0, T, R, SIGMA, 0.10, "call", steps=500)
    american = crr_price(S, 100.0, T, R, SIGMA, 0.10, "call", steps=500, american=True)
    assert american > european + 0.3  # measured premium is about 0.63


@pytest.mark.parametrize("spot", [60.0, 80.0, 100.0, 120.0, 140.0])
@pytest.mark.parametrize("K", STRIKES)
@pytest.mark.parametrize("q", [0.0, 0.08])
@pytest.mark.parametrize("kind", KINDS)
def test_american_is_worth_at_least_intrinsic(spot, K, q, kind):
    american = crr_price(spot, K, T, R, SIGMA, q, kind, steps=200, american=True)
    intrinsic = max(spot - K, 0.0) if kind == "call" else max(K - spot, 0.0)
    assert american >= intrinsic - 1e-12


def test_american_put_reference_value():
    # Standard textbook benchmark: S=K=100, T=1, r=5%, sigma=20%, q=0 gives an
    # American put of about 6.090. There is no closed form; the figure is the
    # value high-resolution lattices converge to. Checked independently here by
    # running this tree at 2000 / 4000 / 8000 steps: 6.08999 / 6.09018 / 6.09028.
    american = crr_price(100.0, 100.0, 1.0, 0.05, 0.20, 0.0, "put", steps=2000, american=True)
    assert american == pytest.approx(6.090, abs=2e-3)


# ---------------------------------------------------------------- edge cases


@pytest.mark.parametrize("american", [False, True])
def test_zero_time_to_expiry_returns_intrinsic(american):
    assert crr_price(110.0, 100.0, 0.0, R, SIGMA, kind="call", american=american) == 10.0
    assert crr_price(110.0, 100.0, 0.0, R, SIGMA, kind="put", american=american) == 0.0
    assert crr_price(90.0, 100.0, 0.0, R, SIGMA, kind="put", american=american) == 10.0


def test_single_step_tree_matches_hand_calculation():
    u = np.exp(SIGMA)
    d = 1.0 / u
    p = (np.exp(R) - d) / (u - d)
    expected = np.exp(-R) * (p * max(S * u - 100.0, 0.0) + (1 - p) * max(S * d - 100.0, 0.0))
    assert crr_price(S, 100.0, 1.0, R, SIGMA, steps=1) == pytest.approx(expected, abs=1e-12)


def test_returns_python_float():
    assert type(crr_price(S, 100.0, T, R, SIGMA, steps=10)) is float


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "straddle"},
        {"steps": 0},
        {"S": -1.0},
        {"K": -1.0},
        {"T": -0.5},
        {"sigma": -0.2},
        {"sigma": 0.0},  # u = d = 1: no tree can be built when T > 0
    ],
)
def test_invalid_inputs_raise(kwargs):
    args = {"S": S, "K": 100.0, "T": T, "r": R, "sigma": SIGMA}
    args.update(kwargs)
    with pytest.raises(ValueError):
        crr_price(**args)


def test_probability_outside_unit_interval_raises():
    # One step, r = 50%, sigma = 10%: exp(r dt) = 1.65 > u = 1.11, so p > 1.
    with pytest.raises(ValueError, match="outside"):
        crr_price(S, 100.0, 1.0, 0.50, 0.10, steps=1)
    # More steps shrink the drift per step faster than the vol per step, fixing it.
    assert crr_price(S, 100.0, 1.0, 0.50, 0.10, steps=100) > 0.0
