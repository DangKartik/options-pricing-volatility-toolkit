"""Tests for volkit.iv.surface on synthetic chains generated from known vol functions."""

import math

import numpy as np
import pandas as pd
import pytest

from volkit.data.chain import CHAIN_COLUMNS, clean_chain
from volkit.iv.surface import (
    build_surface,
    butterfly_violations,
    calendar_violations,
    european_vs_american,
    flag_outliers,
    forward_from_parity,
    otm_quotes,
    synthetic_chain,
    valuation_time,
)
from volkit.pricing.black_scholes import bs_price

S, R, Q = 100.0, 0.04, 0.015
SNAP = "2026-03-02 15:00:00+00:00"          # a Monday, 10:00 New York
DAYS = [14, 45, 120, 365]
STRIKES = np.arange(60.0, 141.0, 2.5)


def flat(k, T):
    return 0.20


def skew(k, T):
    return 0.18 - 0.20 * k + 0.40 * k * k + 0.02 * T


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Run the real cleaning step with permissive thresholds (keeps every priced quote)."""
    return clean_chain(raw, S, min_bid=0.0, max_rel_spread=1.0, min_dte=1.0, max_dte=1000.0,
                       moneyness_range=(-1.0, 1.0))


@pytest.fixture(scope="module")
def flat_surface():
    raw = synthetic_chain(S, R, Q, flat, DAYS, STRIKES, SNAP)
    return build_surface(prepare(raw), S, R)


@pytest.fixture(scope="module")
def skew_surface():
    raw = synthetic_chain(S, R, Q, skew, DAYS, STRIKES, SNAP)
    return build_surface(prepare(raw), S, R)


def test_synthetic_chain_format():
    raw = synthetic_chain(S, R, Q, flat, [30], np.array([90.0, 100.0]), SNAP)
    assert list(raw.columns) == CHAIN_COLUMNS
    assert len(raw) == 4
    assert set(raw["kind"]) == {"call", "put"}


def test_valuation_time_closed_vs_regular():
    raw = synthetic_chain(S, R, Q, flat, [30], np.array([100.0]), SNAP)
    assert valuation_time(raw) == pd.Timestamp(SNAP)
    closed = raw.copy()
    closed["market_state"] = "CLOSED"
    closed["snapshot_time"] = pd.Timestamp("2026-03-08 07:00:00+00:00")       # Sunday
    closed["last_trade_time"] = pd.Timestamp("2026-03-06 20:59:00+00:00")     # Friday
    assert valuation_time(closed) == pd.Timestamp("2026-03-06 21:00:00+00:00")  # 16:00 NY (EST)


def test_forward_and_discount_factor_recovered(flat_surface):
    sl = flat_surface.slices
    assert len(sl) == len(DAYS)
    F_true = S * np.exp((R - Q) * sl["T"])
    np.testing.assert_allclose(sl["F"], F_true, rtol=1e-10)
    np.testing.assert_allclose(sl["DF"], np.exp(-R * sl["T"]), rtol=1e-14)
    np.testing.assert_allclose(sl["q"], Q, atol=1e-9)
    # the regression estimate needs no rate input and must also be exact here
    np.testing.assert_allclose(sl["DF_reg"], np.exp(-R * sl["T"]), rtol=1e-9)
    np.testing.assert_allclose(sl["F_reg"], F_true, rtol=1e-9)


def test_flat_vol_recovered(flat_surface):
    pts = flat_surface.points
    assert len(pts) > 100
    np.testing.assert_allclose(pts["iv"], 0.20, atol=1e-7)
    assert flat_surface.outliers.empty


def test_skew_recovered_at_every_quote(skew_surface):
    pts = skew_surface.points
    expected = [skew(k, T) for k, T in zip(pts["k"], pts["T"])]
    np.testing.assert_allclose(pts["iv"], expected, atol=1e-7)
    assert skew_surface.outliers.empty            # a smooth smile must not be "cleaned"


def test_only_otm_quotes_used(skew_surface):
    pts = skew_surface.points
    assert (pts.loc[pts["kind"] == "put", "k"] < 0).all()
    assert (pts.loc[pts["kind"] == "call", "k"] >= 0).all()


def test_otm_quotes_selection():
    df = pd.DataFrame({"kind": ["put", "put", "call", "call"], "strike": [90.0, 110.0, 90.0, 110.0]})
    out = otm_quotes(df, 100.0)
    assert list(zip(out["kind"], out["strike"])) == [("put", 90.0), ("call", 110.0)]


def test_interpolation_flat_vol_everywhere(flat_surface):
    T = np.array([15, 30, 80, 200, 364]) / 365.0
    k = np.array([-0.1, 0.0, 0.05, 0.1, -0.2])
    np.testing.assert_allclose(flat_surface.implied_vol(k, T), 0.20, atol=1e-7)
    np.testing.assert_allclose(flat_surface.total_variance(0.0, T), 0.04 * T, atol=1e-9)


def test_interpolation_linear_in_total_variance(skew_surface):
    sl = skew_surface.slices
    T0, T1 = sl["T"].iloc[1], sl["T"].iloc[2]
    k = 0.02
    w0 = skew_surface.total_variance(k, T0)
    w1 = skew_surface.total_variance(k, T1)
    Tm = 0.25 * T0 + 0.75 * T1
    assert skew_surface.total_variance(k, Tm) == pytest.approx(0.25 * w0 + 0.75 * w1, rel=1e-12)
    # on a slice the query reproduces the quote-level value (linear in k between quotes)
    p = skew_surface.points[skew_surface.points["expiry"] == sl["expiry"].iloc[1]]
    row = p.iloc[len(p) // 2]
    assert skew_surface.implied_vol(row["k"], T0) == pytest.approx(row["iv"], abs=1e-12)


def test_no_extrapolation(skew_surface):
    sl = skew_surface.slices
    assert np.isnan(skew_surface.implied_vol(0.0, sl["T"].min() / 2))
    assert np.isnan(skew_surface.implied_vol(0.0, sl["T"].max() * 2))
    assert np.isnan(skew_surface.implied_vol(sl["k_max"].min() + 0.5, sl["T"].iloc[0]))


def test_atm_term_structure(skew_surface):
    atm = skew_surface.atm_term_structure()
    np.testing.assert_allclose(atm["atm_iv"], 0.18 + 0.02 * atm["T"], atol=2e-4)


def test_clean_surface_has_no_arbitrage_flags(skew_surface):
    assert calendar_violations(skew_surface)["n_violations"].sum() == 0
    bf = butterfly_violations(skew_surface, tol=1e-9)
    assert bf["n_convexity_violations"].sum() == 0
    assert bf["n_monotonicity_violations"].sum() == 0


def test_calendar_check_flags_inverted_term_structure():
    """Vol halves from the 45d to the 120d expiry: total variance falls."""
    def inverted(k, T):
        return 0.40 if T < 0.2 else 0.12

    raw = synthetic_chain(S, R, Q, inverted, DAYS, STRIKES, SNAP)
    cal = calendar_violations(build_surface(prepare(raw), S, R))
    assert cal["n_violations"].sum() > 0
    assert (cal["n_violations"] > 0).sum() == 1       # only the 45d -> 120d pair
    assert cal["n_violations"].iloc[1] > 0


def test_butterfly_check_and_outlier_rule_flag_a_bad_quote():
    """Bump one OTM put by $1 (a stale quote): convexity breaks around it."""
    raw = synthetic_chain(S, R, Q, skew, [45], STRIKES, SNAP)
    bad = (raw["kind"] == "put") & (raw["strike"] == 90.0)
    raw.loc[bad, ["bid", "ask"]] += 1.0
    kept = build_surface(prepare(raw), S, R, drop_outliers=False)
    bf = butterfly_violations(kept, tol=1e-9)
    assert bf["n_convexity_violations"].sum() >= 1
    # with the outlier rule on, exactly that quote is removed and convexity is restored
    cleaned = build_surface(prepare(raw), S, R, drop_outliers=True)
    assert list(cleaned.outliers["strike"]) == [90.0]
    assert butterfly_violations(cleaned, tol=1e-9)["n_convexity_violations"].sum() == 0


def test_flag_outliers_basic():
    k = np.linspace(-0.2, 0.2, 21)
    iv = 0.2 - 0.3 * k
    width = np.full_like(k, 0.002)
    assert not flag_outliers(k, iv, width).any()
    iv2 = iv.copy()
    iv2[[7, 9]] += 0.04                    # two nearby stale quotes
    assert list(np.flatnonzero(flag_outliers(k, iv2, width))) == [7, 9]
    with pytest.raises(ValueError):
        flag_outliers(k[::-1], iv, width)


def test_expiry_dropped_when_too_few_pairs():
    raw = synthetic_chain(S, R, Q, flat, [30, 90], STRIKES, SNAP)
    first = raw["expiry"] == raw["expiry"].min()
    raw = raw[~(first & (raw["kind"] == "call") & (raw["strike"] > 61.0))]   # 30d: 1 pair left
    surf = build_surface(prepare(raw), S, R)
    assert len(surf.slices) == 1
    assert len(surf.dropped) == 1 and "pairs" in surf.dropped[0]["reason"]


def test_forward_from_parity_rejects_bad_input():
    with pytest.raises(ValueError):
        forward_from_parity(np.array([100.0]), np.array([5.0]), np.array([4.0]), 0.0, R, S)


def test_american_parity_forward_is_biased_low():
    """With American quotes the put's early-exercise premium exceeds the call's
    (r > q), so C - P is too small and the parity forward comes out low."""
    strikes = np.arange(90.0, 111.0, 1.0)
    raw = synthetic_chain(S, 0.05, 0.01, flat, [365], strikes, SNAP, american=True, steps=200)
    surf = build_surface(prepare(raw), S, 0.05, bid_ask_iv=False, drop_outliers=False)
    s = surf.slices.iloc[0]
    F_true = S * math.exp((0.05 - 0.01) * s["T"])
    assert s["F"] < F_true - 0.1
    assert s["q"] > 0.01


def test_european_vs_american_on_european_quotes(flat_surface):
    """Inverting European prices with an American tree can only lower the IV
    (American price >= European); for OTM calls with small q the gap is ~0."""
    exp = flat_surface.slices["expiry"].iloc[1]
    eva = european_vs_american(flat_surface, [exp], per_expiry=6, steps=200)
    assert len(eva) == 6
    assert (eva["diff_eep"] > -1e-4).all()
    calls = eva[eva["kind"] == "call"]
    assert (calls["diff_eep"].abs() < 2e-3).all()


def test_build_surface_validates_input():
    with pytest.raises(ValueError):
        build_surface(pd.DataFrame({"strike": [100.0]}), S, R)
    raw = synthetic_chain(S, R, Q, flat, [30], STRIKES, SNAP)
    with pytest.raises(ValueError):
        build_surface(prepare(raw), -1.0, R)


def test_monotonicity_check_flags_too_steep_put_wing():
    """A smile that rises too fast into the put wing (sigma 0.47 at k = -0.5 on a
    1y expiry) makes call prices fall faster than DF per unit strike there:
    dC/dK < -DF is a static arbitrage (a put spread worth more than its payoff)."""
    def steep(k, T):
        return 0.18 - 0.30 * k + 0.50 * k * k + 0.02 * T

    raw = synthetic_chain(S, R, Q, steep, [365], STRIKES, SNAP)
    bf = butterfly_violations(build_surface(prepare(raw), S, R), tol=1e-9)
    assert bf["n_monotonicity_violations"].sum() > 0
