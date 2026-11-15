"""Tests for volkit.pricing.black_scholes prices (Greeks are in test_greeks.py)."""

import itertools
import math

import numpy as np
import pytest
from scipy.integrate import quad

from volkit.pricing.black_scholes import bs_greeks, bs_price

# A grid that covers in/at/out of the money, short and long maturities,
# negative rates and a dividend yield.
GRID = list(itertools.product(
    (60.0, 90.0, 100.0, 110.0, 160.0),   # S
    (100.0,),                            # K
    (0.02, 0.25, 1.0, 3.0),              # T
    (-0.01, 0.0, 0.05),                  # r
    (0.05, 0.2, 0.6),                    # sigma
    (0.0, 0.03),                         # q
))


def integrated_price(S, K, T, r, sigma, q, kind):
    """Independent check: discounted expected payoff under the lognormal density.

    S_T = S exp((r - q - sigma^2/2) T + sigma sqrt(T) z), z ~ N(0, 1), so
    price = e^{-rT} * integral of payoff(S_T(z)) * n(z) dz.
    """
    def integrand(z):
        s_T = S * math.exp((r - q - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * z)
        payoff = max(s_T - K, 0.0) if kind == "call" else max(K - s_T, 0.0)
        return payoff * math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)

    # z at which S_T = K: the payoff has a kink there, so split the integral.
    z_star = (math.log(K / S) - (r - q - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    left, _ = quad(integrand, -12.0, z_star, epsabs=1e-12, epsrel=1e-12)
    right, _ = quad(integrand, z_star, 12.0, epsabs=1e-12, epsrel=1e-12)
    return math.exp(-r * T) * (left + right)


# ---------------------------------------------------------------- reference values

def test_textbook_reference_values():
    assert bs_price(100, 100, 1, 0.05, 0.2, 0.0, "call") == pytest.approx(10.4506, abs=5e-5)
    assert bs_price(100, 100, 1, 0.05, 0.2, 0.0, "put") == pytest.approx(5.5735, abs=5e-5)


def test_hull_reference_value():
    # Hull, "Options, Futures and Other Derivatives", Black-Scholes example:
    # S=42, K=40, r=10%, sigma=20%, T=0.5 -> call 4.76, put 0.81.
    assert bs_price(42, 40, 0.5, 0.10, 0.2, 0.0, "call") == pytest.approx(4.76, abs=5e-3)
    assert bs_price(42, 40, 0.5, 0.10, 0.2, 0.0, "put") == pytest.approx(0.81, abs=5e-3)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("S, K, T, r, sigma, q", [
    (100.0, 95.0, 0.5, 0.03, 0.25, 0.02),
    (100.0, 120.0, 2.0, 0.01, 0.40, 0.05),
    (80.0, 100.0, 0.25, 0.05, 0.30, 0.00),
])
def test_matches_numerical_integration_with_dividend_yield(S, K, T, r, sigma, q, kind):
    expected = integrated_price(S, K, T, r, sigma, q, kind)
    assert bs_price(S, K, T, r, sigma, q, kind) == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------- no-arbitrage relations

@pytest.mark.parametrize("S, K, T, r, sigma, q", GRID)
def test_put_call_parity(S, K, T, r, sigma, q):
    call = bs_price(S, K, T, r, sigma, q, "call")
    put = bs_price(S, K, T, r, sigma, q, "put")
    forward_value = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert call - put == pytest.approx(forward_value, abs=1e-10)


@pytest.mark.parametrize("S, K, T, r, sigma, q", GRID)
def test_no_arbitrage_bounds(S, K, T, r, sigma, q):
    discounted_spot = S * math.exp(-q * T)
    discounted_strike = K * math.exp(-r * T)
    tol = 1e-10
    call = bs_price(S, K, T, r, sigma, q, "call")
    put = bs_price(S, K, T, r, sigma, q, "put")
    assert max(discounted_spot - discounted_strike, 0.0) - tol <= call <= discounted_spot + tol
    assert max(discounted_strike - discounted_spot, 0.0) - tol <= put <= discounted_strike + tol


def test_call_increasing_and_put_decreasing_in_spot():
    S = np.linspace(50.0, 150.0, 201)
    assert np.all(np.diff(bs_price(S, 100.0, 1.0, 0.03, 0.2, 0.01, "call")) > 0.0)
    assert np.all(np.diff(bs_price(S, 100.0, 1.0, 0.03, 0.2, 0.01, "put")) < 0.0)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_price_increasing_in_volatility(kind):
    sigma = np.linspace(0.01, 1.5, 150)
    assert np.all(np.diff(bs_price(100.0, 105.0, 1.0, 0.03, sigma, 0.01, kind)) > 0.0)


def test_call_decreasing_and_put_increasing_in_strike():
    K = np.linspace(50.0, 150.0, 201)
    assert np.all(np.diff(bs_price(100.0, K, 1.0, 0.03, 0.2, 0.01, "call")) < 0.0)
    assert np.all(np.diff(bs_price(100.0, K, 1.0, 0.03, 0.2, 0.01, "put")) > 0.0)


# ---------------------------------------------------------------- edge cases

@pytest.mark.parametrize("S, call_payoff, put_payoff", [
    (120.0, 20.0, 0.0), (100.0, 0.0, 0.0), (70.0, 0.0, 30.0),
])
def test_expiry_returns_intrinsic_value(S, call_payoff, put_payoff):
    assert bs_price(S, 100.0, 0.0, 0.05, 0.2, 0.02, "call") == call_payoff
    assert bs_price(S, 100.0, 0.0, 0.05, 0.2, 0.02, "put") == put_payoff


@pytest.mark.parametrize("S", [70.0, 100.0, 130.0])
def test_zero_volatility_returns_discounted_forward_intrinsic(S):
    K, T, r, q = 100.0, 2.0, 0.05, 0.01
    forward_value = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert bs_price(S, K, T, r, 0.0, q, "call") == pytest.approx(max(forward_value, 0.0), abs=1e-12)
    assert bs_price(S, K, T, r, 0.0, q, "put") == pytest.approx(max(-forward_value, 0.0), abs=1e-12)


def test_zero_volatility_with_forward_exactly_at_strike_is_worthless():
    # r = q = 0 and S = K: the forward equals the strike exactly.
    assert bs_price(100.0, 100.0, 1.0, 0.0, 0.0, 0.0, "call") == 0.0
    assert bs_price(100.0, 100.0, 1.0, 0.0, 0.0, 0.0, "put") == 0.0


def test_price_is_continuous_as_volatility_and_maturity_go_to_zero():
    args_limit = bs_price(110.0, 100.0, 1.0, 0.05, 0.0, 0.0, "call")
    assert bs_price(110.0, 100.0, 1.0, 0.05, 1e-8, 0.0, "call") == pytest.approx(args_limit, abs=1e-9)
    assert bs_price(110.0, 100.0, 1e-12, 0.05, 0.2, 0.0, "call") == pytest.approx(10.0, abs=1e-6)


def test_greeks_at_expiry():
    in_the_money = bs_greeks(120.0, 100.0, 0.0, 0.05, 0.2, 0.0, "call")
    assert in_the_money["delta"] == 1.0
    assert in_the_money["gamma"] == 0.0
    assert in_the_money["vega"] == 0.0
    assert in_the_money["rho"] == 0.0
    assert in_the_money["theta"] == pytest.approx(-0.05 * 100.0)  # only -r K remains

    out_of_the_money = bs_greeks(80.0, 100.0, 0.0, 0.05, 0.2, 0.0, "call")
    assert all(value == 0.0 for value in out_of_the_money.values())

    assert bs_greeks(80.0, 100.0, 0.0, 0.05, 0.2, 0.0, "put")["delta"] == -1.0
    # Exactly at the strike the delta jumps from 0 to 1; the midpoint is returned.
    assert bs_greeks(100.0, 100.0, 0.0, 0.05, 0.2, 0.0, "call")["delta"] == 0.5


def test_edge_case_results_are_finite():
    for T, sigma in [(0.0, 0.2), (1.0, 0.0), (0.0, 0.0)]:
        for S in (80.0, 100.0, 120.0):
            for kind in ("call", "put"):
                assert math.isfinite(bs_price(S, 100.0, T, 0.0, sigma, 0.0, kind))
                assert all(math.isfinite(v) for v in bs_greeks(S, 100.0, T, 0.0, sigma, 0.0, kind).values())


# ---------------------------------------------------------------- vectorisation

def test_scalar_input_returns_python_float():
    assert type(bs_price(100.0, 100.0, 1.0, 0.05, 0.2)) is float
    assert all(type(v) is float for v in bs_greeks(100.0, 100.0, 1.0, 0.05, 0.2).values())


def test_array_input_matches_scalar_calls():
    S = np.array([80.0, 100.0, 120.0])
    prices = bs_price(S, 100.0, 1.0, 0.05, 0.2, 0.01, "put")
    assert isinstance(prices, np.ndarray) and prices.shape == (3,)
    for spot, price in zip(S, prices):
        assert price == pytest.approx(bs_price(float(spot), 100.0, 1.0, 0.05, 0.2, 0.01, "put"), abs=1e-14)


def test_broadcasting_strikes_against_maturities():
    K = np.array([90.0, 100.0, 110.0])[:, None]      # shape (3, 1)
    T = np.array([0.0, 0.5, 1.0, 2.0])[None, :]      # shape (1, 4), includes expiry
    prices = bs_price(100.0, K, T, 0.03, 0.2)
    assert prices.shape == (3, 4)
    assert prices[0, 0] == 10.0 and prices[2, 0] == 0.0          # intrinsic at T = 0
    assert prices[1, 2] == pytest.approx(bs_price(100.0, 100.0, 1.0, 0.03, 0.2))
    greeks = bs_greeks(100.0, K, T, 0.03, 0.2)
    assert all(v.shape == (3, 4) and np.all(np.isfinite(v)) for v in greeks.values())


# ---------------------------------------------------------------- invalid input

@pytest.mark.parametrize("S, K, T, sigma", [
    (-100.0, 100.0, 1.0, 0.2),
    (0.0, 100.0, 1.0, 0.2),
    (100.0, -100.0, 1.0, 0.2),
    (100.0, 0.0, 1.0, 0.2),
    (100.0, 100.0, -1.0, 0.2),
    (100.0, 100.0, 1.0, -0.2),
])
def test_invalid_numbers_raise(S, K, T, sigma):
    with pytest.raises(ValueError):
        bs_price(S, K, T, 0.05, sigma)
    with pytest.raises(ValueError):
        bs_greeks(S, K, T, 0.05, sigma)


def test_one_invalid_element_in_an_array_raises():
    with pytest.raises(ValueError):
        bs_price(np.array([100.0, -1.0]), 100.0, 1.0, 0.05, 0.2)


@pytest.mark.parametrize("kind", ["Call", "c", "", "straddle", None])
def test_invalid_kind_raises(kind):
    with pytest.raises(ValueError):
        bs_price(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, kind)
    with pytest.raises(ValueError):
        bs_greeks(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, kind)
