"""Cross-checks: three independent pricers must agree on European options."""

import itertools

import pytest

from volkit.pricing.binomial import crr_price
from volkit.pricing.black_scholes import bs_greeks, bs_price
from volkit.pricing.greeks import fd_greeks
from volkit.pricing.monte_carlo import mc_price

S, T, R = 100.0, 0.75, 0.04
GRID = list(itertools.product([80.0, 100.0, 125.0], [0.15, 0.35], [0.0, 0.02], ["call", "put"]))


@pytest.mark.parametrize("K,sigma,q,kind", GRID)
def test_tree_matches_black_scholes(K, sigma, q, kind):
    """CRR error is O(1/steps); averaging n and n+1 cancels the odd/even flip."""
    bs = bs_price(S, K, T, R, sigma, q, kind)
    tree = 0.5 * (
        crr_price(S, K, T, R, sigma, q, kind, steps=1000)
        + crr_price(S, K, T, R, sigma, q, kind, steps=1001)
    )
    assert tree == pytest.approx(bs, abs=2e-3)


@pytest.mark.parametrize("K,sigma,q,kind", GRID)
def test_monte_carlo_matches_black_scholes(K, sigma, q, kind):
    """Black-Scholes must sit inside a 4-standard-error band of the MC estimate."""
    bs = bs_price(S, K, T, R, sigma, q, kind)
    res = mc_price(S, K, T, R, sigma, q, kind, n_paths=400_000, seed=7)
    assert abs(res.price - bs) < 4 * res.stderr
    assert res.stderr < 0.06  # the band is tight enough to be a real test


@pytest.mark.parametrize("K,sigma,q,kind", GRID)
def test_tree_and_monte_carlo_agree(K, sigma, q, kind):
    tree = crr_price(S, K, T, R, sigma, q, kind, steps=1000)
    res = mc_price(S, K, T, R, sigma, q, kind, n_paths=400_000, seed=11)
    assert abs(res.price - tree) < 4 * res.stderr + 5e-3


@pytest.mark.parametrize("kind", ["call", "put"])
def test_tree_greeks_match_analytic(kind):
    """Finite-difference Greeks of the European tree approximate the closed form."""
    args = (S, 100.0, T, R, 0.25, 0.01, kind)
    exact = bs_greeks(*args)
    approx = fd_greeks(crr_price, *args, steps=1000)
    assert approx["delta"] == pytest.approx(exact["delta"], abs=5e-3)
    assert approx["gamma"] == pytest.approx(exact["gamma"], rel=0.05)
    assert approx["vega"] == pytest.approx(exact["vega"], rel=0.02)
    assert approx["rho"] == pytest.approx(exact["rho"], rel=0.02)
