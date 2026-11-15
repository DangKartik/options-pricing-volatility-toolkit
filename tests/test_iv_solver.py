"""Tests for volkit.iv.solver (European Newton/Brent and American tree inversion)."""

import math

import numpy as np
import pandas as pd
import pytest

from volkit.iv.solver import (
    _implied_vol_info,
    american_sigma_floor,
    european_bounds,
    implied_vol,
    implied_vol_american,
    implied_vol_array,
    implied_vol_frame,
)
from volkit.pricing.binomial import crr_price
from volkit.pricing.black_scholes import bs_price, bs_vega

S, R, Q = 100.0, 0.04, 0.01
STRIKES = [20, 50, 80, 95, 100, 105, 120, 200, 500]
MATURITIES = [1 / 365, 7 / 365, 0.1, 0.5, 1.0, 3.0, 10.0]
SIGMAS = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("K", STRIKES)
@pytest.mark.parametrize("T", MATURITIES)
def test_round_trip_grid(kind, K, T):
    """Where vega is not negligible, sigma is recovered to 1e-7; elsewhere the
    answer is either nan (not identifiable) or reproduces the price."""
    for sigma in SIGMAS:
        price = bs_price(S, K, T, R, sigma, Q, kind)
        vega = bs_vega(S, K, T, R, sigma, Q, kind)
        iv = implied_vol(price, S, K, T, R, Q, kind)
        if vega > 1e-6:
            assert iv == pytest.approx(sigma, abs=1e-7), (kind, K, T, sigma)
        elif not math.isnan(iv):
            assert bs_price(S, K, T, R, iv, Q, kind) == pytest.approx(price, abs=1e-9)


def test_atm_typical_case_uses_newton():
    price = bs_price(S, 100, 0.5, R, 0.2, Q, "call")
    iv, path = _implied_vol_info(price, S, 100, 0.5, R, Q, "call")
    assert path == "newton"
    assert iv == pytest.approx(0.2, abs=1e-10)


def test_fallback_to_brent_is_exercised():
    """Deep-OTM, very short-dated: the Manaster-Koehler start (sigma0 ~ 1.0)
    is far above the root and a Newton step leaves the bracket."""
    hits = 0
    for kind, K, T, sigma in [("call", 150, 0.02, 0.3), ("put", 50, 0.1, 1.0)]:
        price = bs_price(S, K, T, R, sigma, Q, kind)
        iv, path = _implied_vol_info(price, S, K, T, R, Q, kind)
        assert iv == pytest.approx(sigma, abs=1e-7)
        hits += path == "brent"
    assert hits >= 1


def test_fallback_on_max_iter():
    """With max_iter=1 Newton cannot converge, so the answer must come from brent."""
    price = bs_price(S, 120, 1.0, R, 0.3, Q, "call")
    iv, path = _implied_vol_info(price, S, 120, 1.0, R, Q, "call", max_iter=1)
    assert path == "brent"
    assert iv == pytest.approx(0.3, abs=1e-7)


def test_tiny_vega_deep_otm_still_recovered():
    """Price ~1e-12 (vega ~1e-10): Newton bails out, bracketing still finds sigma."""
    K, T, sigma = 200.0, 0.1, 0.2
    price = bs_price(S, K, T, R, sigma, Q, "call")
    assert 0 < price < 1e-10
    iv, path = _implied_vol_info(price, S, K, T, R, Q, "call")
    assert path == "brent"
    assert iv == pytest.approx(sigma, rel=1e-4)


def test_time_value_lost_to_rounding_is_nan():
    """Deep ITM 1-day call: price equals the zero-vol value in floating point."""
    price = bs_price(S, 20, 1 / 365, R, 0.2, Q, "call")
    lower, _ = european_bounds(S, 20, 1 / 365, R, Q, "call")
    assert price == lower
    assert math.isnan(implied_vol(price, S, 20, 1 / 365, R, Q, "call"))


@pytest.mark.parametrize("kind", ["call", "put"])
def test_nan_below_intrinsic(kind):
    K, T = (80.0, 1.0) if kind == "call" else (120.0, 1.0)
    lower, _ = european_bounds(S, K, T, R, Q, kind)
    assert math.isnan(implied_vol(lower - 0.01, S, K, T, R, Q, kind))


@pytest.mark.parametrize("kind", ["call", "put"])
def test_nan_at_or_above_upper_bound(kind):
    _, upper = european_bounds(S, 100, 1.0, R, Q, kind)
    assert math.isnan(implied_vol(upper, S, 100, 1.0, R, Q, kind))
    assert math.isnan(implied_vol(upper + 1, S, 100, 1.0, R, Q, kind))


def test_nan_for_expired_and_nonfinite():
    assert math.isnan(implied_vol(5.0, S, 100, 0.0, R, Q, "call"))
    assert math.isnan(implied_vol(5.0, S, 100, -0.1, R, Q, "call"))
    assert math.isnan(implied_vol(math.nan, S, 100, 1.0, R, Q, "call"))


def test_nan_when_vol_above_bracket():
    price = bs_price(S, 100, 1.0, R, 6.0, Q, "call")
    assert math.isnan(implied_vol(price, S, 100, 1.0, R, Q, "call"))


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        implied_vol(5.0, S, 100, 1.0, R, Q, "straddle")
    with pytest.raises(ValueError):
        implied_vol(5.0, -1.0, 100, 1.0, R, Q, "call")
    with pytest.raises(ValueError):
        implied_vol_american(5.0, S, 0.0, 1.0, R, Q, "put")


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
@pytest.mark.parametrize("T", [0.1, 1.0])
@pytest.mark.parametrize("sigma", [0.1, 0.3, 0.8])
def test_american_round_trip(kind, K, T, sigma):
    price = crr_price(S, K, T, R, sigma, Q, kind, steps=200, american=True)
    iv = implied_vol_american(price, S, K, T, R, Q, kind, steps=200)
    floor = american_sigma_floor(T, R, Q, 200)
    if price - crr_price(S, K, T, R, floor, Q, kind, steps=200, american=True) > 1e-4:
        assert iv == pytest.approx(sigma, abs=1e-5)
    elif not math.isnan(iv):  # price within 1e-4 of the floor value: only check repricing
        assert crr_price(S, K, T, R, iv, Q, kind, steps=200, american=True) == pytest.approx(
            price, abs=1e-6)


def test_american_put_iv_below_european_iv():
    """The early-exercise premium is in the American price; a European inversion
    attributes it to volatility and so over-states sigma for an ITM put."""
    price = crr_price(S, 120, 1.0, 0.05, 0.2, 0.0, "put", steps=200, american=True)
    iv_am = implied_vol_american(price, S, 120, 1.0, 0.05, 0.0, "put", steps=200)
    iv_eu = implied_vol(price, S, 120, 1.0, 0.05, 0.0, "put")
    assert iv_am == pytest.approx(0.2, abs=1e-5)
    assert iv_eu > iv_am + 0.01


def test_american_nan_below_exercise_value():
    assert math.isnan(implied_vol_american(19.0, S, 120, 1.0, R, Q, "put"))
    assert math.isnan(implied_vol_american(100.0, S, 120, 1.0, R, Q, "call"))
    assert math.isnan(implied_vol_american(5.0, S, 100, 0.0, R, Q, "call"))


def test_array_and_frame_wrappers():
    K = np.array([90.0, 100.0, 110.0])
    kinds = np.array(["put", "call", "call"], dtype=object)
    sig = np.array([0.25, 0.2, 0.18])
    prices = np.array([bs_price(S, k, 0.5, R, s, Q, kd) for k, s, kd in zip(K, sig, kinds)])
    out = implied_vol_array(prices, S, K, 0.5, R, Q, kinds)
    np.testing.assert_allclose(out, sig, atol=1e-8)

    df = pd.DataFrame({"mid": prices, "strike": K, "T": 0.5, "kind": kinds,
                       "spot": S, "r": R, "q": Q})
    np.testing.assert_allclose(implied_vol_frame(df).to_numpy(), sig, atol=1e-8)
    np.testing.assert_allclose(implied_vol_frame(df, S=S, r=R, q=Q).to_numpy(), sig, atol=1e-8)
    with pytest.raises(ValueError):
        implied_vol_frame(df.drop(columns="T"))
