"""Tests for the analytic Black-Scholes Greeks and for finite-difference Greeks."""

import itertools
import math

import numpy as np
import pytest

from volkit.pricing.black_scholes import (
    bs_delta, bs_gamma, bs_greeks, bs_price, bs_rho, bs_theta, bs_vega,
)
from volkit.pricing.greeks import fd_greeks

GREEK_NAMES = ("delta", "gamma", "vega", "theta", "rho")

# Moneyness x maturity grid (K = 100), with a non-zero rate and dividend yield.
R, SIGMA, Q = 0.03, 0.25, 0.01
GRID = list(itertools.product(
    (70.0, 90.0, 100.0, 110.0, 140.0),   # S
    (1.0 / 12.0, 0.5, 1.0, 3.0),         # T
    ("call", "put"),
))

# Small bumps: appropriate for a smooth closed-form pricer.
SMALL_BUMPS = dict(rel_bump_S=1e-4, rel_bump_sigma=1e-4, rel_bump_T=1e-4, bump_r=1e-5)


# ---------------------------------------------------------------- analytic vs finite difference

@pytest.mark.parametrize("S, T, kind", GRID)
def test_analytic_greeks_match_finite_differences(S, T, kind):
    analytic = bs_greeks(S, 100.0, T, R, SIGMA, Q, kind)
    numeric = fd_greeks(bs_price, S, 100.0, T, R, SIGMA, Q, kind, **SMALL_BUMPS)
    assert set(numeric) == set(GREEK_NAMES)
    assert numeric["delta"] == pytest.approx(analytic["delta"], abs=1e-7)
    assert numeric["gamma"] == pytest.approx(analytic["gamma"], abs=1e-7)
    assert numeric["vega"] == pytest.approx(analytic["vega"], abs=1e-5)
    assert numeric["theta"] == pytest.approx(analytic["theta"], abs=1e-5)
    assert numeric["rho"] == pytest.approx(analytic["rho"], abs=1e-5)


@pytest.mark.parametrize("S, T, kind", GRID)
def test_default_bumps_are_accurate_to_about_one_percent(S, T, kind):
    # The default bumps are sized for a 500-step tree, so the tolerance is looser:
    # 1% of the Greek plus a small absolute floor for Greeks that are near zero.
    analytic = bs_greeks(S, 100.0, T, R, SIGMA, Q, kind)
    numeric = fd_greeks(bs_price, S, 100.0, T, R, SIGMA, Q, kind)
    floors = {"delta": 3e-3, "gamma": 1e-4, "vega": 1e-2, "theta": 1e-2, "rho": 1e-3}
    for name in GREEK_NAMES:
        assert numeric[name] == pytest.approx(analytic[name], rel=1e-2, abs=floors[name]), name


def test_central_difference_error_is_second_order():
    # Halving the bump should divide the truncation error by about 4.
    args = (100.0, 100.0, 1.0, R, SIGMA, Q, "call")
    exact = bs_delta(*args)
    error_h = abs(fd_greeks(bs_price, *args, rel_bump_S=4e-2)["delta"] - exact)
    error_half_h = abs(fd_greeks(bs_price, *args, rel_bump_S=2e-2)["delta"] - exact)
    assert error_h / error_half_h == pytest.approx(4.0, rel=0.05)


def test_reference_greeks():
    # S=K=100, T=1, r=5%, sigma=20%, q=0: d1 = 0.35, d2 = 0.15.
    call = bs_greeks(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, "call")
    put = bs_greeks(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, "put")
    assert call["delta"] == pytest.approx(0.6368, abs=5e-5)
    assert put["delta"] == pytest.approx(-0.3632, abs=5e-5)
    assert call["gamma"] == pytest.approx(0.01876, abs=5e-6)
    assert call["vega"] == pytest.approx(37.524, abs=5e-4)
    assert call["theta"] == pytest.approx(-6.4140, abs=5e-5)
    assert put["theta"] == pytest.approx(-1.6579, abs=5e-5)
    assert call["rho"] == pytest.approx(53.232, abs=5e-4)
    assert put["rho"] == pytest.approx(-41.890, abs=5e-4)


def test_bs_greeks_agrees_with_individual_functions():
    args = (95.0, 100.0, 0.75, R, SIGMA, Q, "put")
    greeks = bs_greeks(*args)
    assert greeks == {
        "delta": bs_delta(*args), "gamma": bs_gamma(*args), "vega": bs_vega(*args),
        "theta": bs_theta(*args), "rho": bs_rho(*args),
    }


# ---------------------------------------------------------------- call/put relations

@pytest.mark.parametrize("S, T", [(S, T) for S, T, kind in GRID if kind == "call"])
def test_call_put_greek_relations(S, T):
    """Differentiating put-call parity C - P = S e^{-qT} - K e^{-rT}."""
    K = 100.0
    call = bs_greeks(S, K, T, R, SIGMA, Q, "call")
    put = bs_greeks(S, K, T, R, SIGMA, Q, "put")
    assert call["gamma"] == put["gamma"]
    assert call["vega"] == put["vega"]
    assert call["delta"] - put["delta"] == pytest.approx(math.exp(-Q * T), abs=1e-12)
    assert call["rho"] - put["rho"] == pytest.approx(K * T * math.exp(-R * T), abs=1e-10)
    # theta = -d/dT, so the difference is q S e^{-qT} - r K e^{-rT}.
    expected_theta_gap = Q * S * math.exp(-Q * T) - R * K * math.exp(-R * T)
    assert call["theta"] - put["theta"] == pytest.approx(expected_theta_gap, abs=1e-10)


# ---------------------------------------------------------------- Black-Scholes PDE

@pytest.mark.parametrize("S, T, kind", GRID)
@pytest.mark.parametrize("r, q", [(0.03, 0.01), (0.0, 0.0), (-0.01, 0.04)])
def test_black_scholes_pde(S, T, kind, r, q):
    """theta + (r - q) S delta + 0.5 sigma^2 S^2 gamma - r V = 0."""
    value = bs_price(S, 100.0, T, r, SIGMA, q, kind)
    g = bs_greeks(S, 100.0, T, r, SIGMA, q, kind)
    residual = (g["theta"] + (r - q) * S * g["delta"]
                + 0.5 * SIGMA**2 * S**2 * g["gamma"] - r * value)
    assert residual == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------- signs and bounds

def test_signs_and_bounds_of_greeks():
    S = np.linspace(60.0, 160.0, 101)
    call = bs_greeks(S, 100.0, 1.0, R, SIGMA, Q, "call")
    put = bs_greeks(S, 100.0, 1.0, R, SIGMA, Q, "put")
    discount_q = math.exp(-Q * 1.0)
    assert np.all((call["delta"] > 0.0) & (call["delta"] < discount_q))
    assert np.all((put["delta"] < 0.0) & (put["delta"] > -discount_q))
    assert np.all(call["gamma"] > 0.0)
    assert np.all(call["vega"] > 0.0)
    assert np.all(call["rho"] > 0.0)
    assert np.all(put["rho"] < 0.0)
    assert np.all(np.diff(call["delta"]) > 0.0)   # positive gamma


def test_greeks_are_vectorised():
    S = np.array([[90.0], [110.0]])               # shape (2, 1)
    T = np.array([0.25, 1.0, 2.0])                # shape (3,)
    for fn in (bs_delta, bs_gamma, bs_vega, bs_theta, bs_rho):
        values = fn(S, 100.0, T, R, SIGMA, Q, "put")
        assert values.shape == (2, 3)
        assert values[1, 2] == pytest.approx(fn(110.0, 100.0, 2.0, R, SIGMA, Q, "put"), abs=1e-14)


# ---------------------------------------------------------------- fd_greeks mechanics

def test_fd_greeks_forwards_extra_keyword_arguments():
    def scaled_pricer(S, K, T, r, sigma, q=0.0, kind="call", scale=1.0):
        return scale * bs_price(S, K, T, r, sigma, q, kind)

    args = (100.0, 100.0, 1.0, R, SIGMA, Q, "call")
    plain = fd_greeks(bs_price, *args)
    tripled = fd_greeks(scaled_pricer, *args, scale=3.0)
    for name in GREEK_NAMES:
        assert tripled[name] == pytest.approx(3.0 * plain[name], rel=1e-9)


def test_fd_greeks_on_a_pricer_with_known_derivatives():
    # Forward contract value S e^{-qT} - K e^{-rT}: every derivative is known exactly.
    def forward_value(S, K, T, r, sigma, q=0.0, kind="call"):
        return S * math.exp(-q * T) - K * math.exp(-r * T)

    S, K, T, r, q = 100.0, 90.0, 2.0, 0.04, 0.01
    g = fd_greeks(forward_value, S, K, T, r, 0.2, q)
    assert g["delta"] == pytest.approx(math.exp(-q * T), abs=1e-9)
    assert g["gamma"] == pytest.approx(0.0, abs=1e-9)
    assert g["vega"] == 0.0
    assert g["theta"] == pytest.approx(q * S * math.exp(-q * T) - r * K * math.exp(-r * T), abs=1e-6)
    assert g["rho"] == pytest.approx(K * T * math.exp(-r * T), abs=1e-5)


def test_fd_theta_uses_one_sided_difference_at_expiry():
    # At T = 0 a central difference would need a negative maturity.
    g = fd_greeks(bs_price, 120.0, 100.0, 0.0, 0.05, 0.2, 0.0, "call", rel_bump_T=1e-6)
    assert g["theta"] == pytest.approx(-0.05 * 100.0, abs=1e-4)   # deep ITM: theta -> -r K
    assert g["delta"] == pytest.approx(1.0, abs=1e-12)


def test_fd_vega_uses_one_sided_difference_at_zero_volatility():
    g = fd_greeks(bs_price, 120.0, 100.0, 1.0, 0.05, 0.0, 0.0, "call")
    assert all(math.isfinite(v) for v in g.values())
    assert g["vega"] == pytest.approx(0.0, abs=1e-12)             # deep ITM, vol 0 -> 1%


def test_fd_greeks_returns_python_floats():
    g = fd_greeks(bs_price, 100.0, 100.0, 1.0, R, SIGMA)
    assert all(type(v) is float for v in g.values())


@pytest.mark.parametrize("bad", [
    dict(kind="binary"),
    dict(rel_bump_S=0.0),
    dict(rel_bump_S=1.0),
    dict(rel_bump_sigma=-1e-2),
    dict(rel_bump_T=2.0),
    dict(bump_r=0.0),
])
def test_fd_greeks_rejects_invalid_arguments(bad):
    with pytest.raises(ValueError):
        fd_greeks(bs_price, 100.0, 100.0, 1.0, R, SIGMA, **bad)


def test_fd_greeks_rejects_invalid_market_inputs():
    with pytest.raises(ValueError):
        fd_greeks(bs_price, -100.0, 100.0, 1.0, R, SIGMA)
    with pytest.raises(ValueError):
        fd_greeks(bs_price, 100.0, 100.0, -1.0, R, SIGMA)
    with pytest.raises(ValueError):
        fd_greeks(bs_price, 100.0, 100.0, 1.0, R, -SIGMA)
