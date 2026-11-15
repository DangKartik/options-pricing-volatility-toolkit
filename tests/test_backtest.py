"""Validate the delta-hedging engine against Black-Scholes theory on simulated paths.

The engine is only trusted on real data after it reproduces, on GBM paths with
fixed seeds: zero expected P&L when hedging at the true vol, a 1/sqrt(N)
hedging error, E[P&L] = e^{rT} (price at implied - price at realised), drift
independence, and exact cash/cost accounting in hand-computable cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from volkit.backtest.costs import BASE_COSTS, ZERO_COSTS, CostModel
from volkit.backtest.delta_hedge import Leg, delta_hedge_pnl, single_option, straddle
from volkit.backtest import metrics
from volkit.pricing.black_scholes import bs_delta, bs_price, bs_vega
from volkit.pricing.monte_carlo import mc_paths

S0, K, T, R, Q = 100.0, 100.0, 0.25, 0.03, 0.01
N_STEPS = 64
TIMES = np.linspace(0.0, T, N_STEPS + 1)


def straddle_price(sigma: float) -> float:
    return bs_price(S0, K, T, R, sigma, Q, "call") + bs_price(S0, K, T, R, sigma, Q, "put")


def gbm_real_world(mu: float, sigma: float, n_paths: int, seed: int) -> np.ndarray:
    """GBM under a real-world drift: S_{t+dt} = S_t exp((mu - q - sigma^2/2) dt + sigma sqrt(dt) Z).

    mu is the total expected return; the price drifts at mu - q because the
    stock pays the dividend yield q. Same seed -> same normals for every mu.
    """
    rng = np.random.default_rng(seed)
    dt = T / N_STEPS
    Z = rng.standard_normal((n_paths, N_STEPS))
    inc = (mu - Q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    out = np.zeros((n_paths, N_STEPS + 1))
    out[:, 1:] = np.cumsum(inc, axis=1)
    return S0 * np.exp(out)


def mean_and_se(x: np.ndarray) -> tuple[float, float]:
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(x.size))


# --------------------------------------------------------------------------
# theory on simulated paths
# --------------------------------------------------------------------------

def test_hedging_at_true_vol_has_zero_mean():
    sigma = 0.20
    paths = mc_paths(S0, T, R, sigma, Q, n_steps=N_STEPS, n_paths=20_000, seed=1)
    res = delta_hedge_pnl(paths, TIMES, sigma, R, Q, strike=K)
    m, se = mean_and_se(res.total)
    assert abs(m) < 3 * se
    # and the hedging error is small relative to the premium
    assert res.total.std() < 0.15 * straddle_price(sigma)


def test_hedging_error_shrinks_like_one_over_sqrt_rebalances():
    sigma = 0.20
    paths = mc_paths(S0, T, R, sigma, Q, n_steps=N_STEPS, n_paths=20_000, seed=2)
    daily = delta_hedge_pnl(paths, TIMES, sigma, R, Q, strike=K, hedge_every=1)
    every4 = delta_hedge_pnl(paths, TIMES, sigma, R, Q, strike=K, hedge_every=4)
    ratio = every4.total.std() / daily.total.std()
    assert 1.8 < ratio < 2.2          # sqrt(64 / 16) = 2
    m, se = mean_and_se(every4.total)
    assert abs(m) < 3 * se            # still unbiased


@pytest.mark.parametrize("sigma_i, sigma_r", [(0.25, 0.15), (0.20, 0.30)])
def test_mean_pnl_equals_bs_price_difference(sigma_i, sigma_r):
    # Sell at sigma_i, path at sigma_r, hedge at the implied vol (the default).
    paths = mc_paths(S0, T, R, sigma_r, Q, n_steps=N_STEPS, n_paths=40_000, seed=3)
    res = delta_hedge_pnl(paths, TIMES, sigma_i, R, Q, strike=K)
    expected = np.exp(R * T) * (straddle_price(sigma_i) - straddle_price(sigma_r))
    m, se = mean_and_se(res.total)
    assert abs(m - expected) < 3.5 * se
    assert np.sign(m) == np.sign(sigma_i - sigma_r)
    assert abs(m) > 10 * se


def test_hedging_at_realised_vol_locks_in_price_difference():
    # With the hedge delta at the true vol the P&L is (nearly) deterministic.
    sigma_i, sigma_r = 0.25, 0.15
    paths = mc_paths(S0, T, R, sigma_r, Q, n_steps=N_STEPS, n_paths=20_000, seed=4)
    res = delta_hedge_pnl(paths, TIMES, sigma_i, R, Q, strike=K, hedge_vol=sigma_r)
    expected = np.exp(R * T) * (straddle_price(sigma_i) - straddle_price(sigma_r))
    assert abs(res.total.mean() - expected) < 3 * res.total.std() / np.sqrt(20_000)
    assert res.total.std() < 0.25 * expected


def test_pnl_does_not_depend_on_drift():
    # Continuous-time theory: hedged at the true vol the P&L is zero path by path,
    # whatever the drift. With discrete hedging a drift-dependent bias of order dt
    # remains (E[(dS/S)^2] = sigma^2 dt + (mu dt)^2); it must be tiny relative to the
    # premium, shrink ~linearly in the rebalance interval, and be dwarfed by the
    # effect of the same drift on an UNHEDGED short straddle.
    sigma = 0.20
    results, unhedged = {}, {}
    for mu in (-0.30, R, 0.40):
        paths = gbm_real_world(mu, sigma, n_paths=20_000, seed=5)
        results[mu] = delta_hedge_pnl(paths, TIMES, sigma, R, Q, strike=K).total
        unhedged[mu] = straddle_price(sigma) * np.exp(R * T) - np.abs(paths[:, -1] - K)
        if mu != R:
            coarse = delta_hedge_pnl(paths, TIMES, sigma, R, Q, strike=K, hedge_every=4).total
            results[(mu, "coarse")] = coarse
    base = results[R]
    coarse_base = delta_hedge_pnl(gbm_real_world(R, sigma, 20_000, seed=5), TIMES, sigma, R, Q,
                                  strike=K, hedge_every=4).total
    for mu in (-0.30, 0.40):
        bias = (results[mu] - base).mean()                 # same normals -> paired comparison
        bias_coarse = (results[(mu, "coarse")] - coarse_base).mean()
        assert abs(bias) < 0.01 * straddle_price(sigma)
        assert abs(bias) < 0.1 * base.std()
        assert abs(bias_coarse) > 2.5 * abs(bias)          # O(dt): 4x interval -> ~4x bias
        assert abs(unhedged[mu].mean() - unhedged[R].mean()) > 20 * abs(bias)


def test_single_call_and_put_parity_of_hedged_pnl():
    # Hedged call and hedged put at the same strike differ by a forward, which the
    # hedge neutralises exactly -> identical P&L path by path (gross of costs).
    sigma = 0.2
    paths = mc_paths(S0, T, R, sigma, Q, n_steps=N_STEPS, n_paths=2_000, seed=6)
    call = delta_hedge_pnl(paths, TIMES, 0.22, R, Q, legs=single_option(K, "call"))
    put = delta_hedge_pnl(paths, TIMES, 0.22, R, Q, legs=single_option(K, "put"))
    np.testing.assert_allclose(call.total, put.total, atol=1e-9)
    both = delta_hedge_pnl(paths, TIMES, 0.22, R, Q, strike=K)
    np.testing.assert_allclose(both.total, call.total + put.total, atol=1e-9)


def test_entry_vol_hedge_differs_from_current_vol_hedge():
    paths = mc_paths(S0, T, R, 0.2, Q, n_steps=N_STEPS, n_paths=50, seed=7)
    iv = np.linspace(0.2, 0.3, N_STEPS + 1)
    cur = delta_hedge_pnl(paths, TIMES, iv, R, Q, strike=K, hedge_vol="current")
    ent = delta_hedge_pnl(paths, TIMES, iv, R, Q, strike=K, hedge_vol="entry")
    fixed = delta_hedge_pnl(paths, TIMES, iv, R, Q, strike=K, hedge_vol=0.2)
    np.testing.assert_allclose(ent.total, fixed.total)
    np.testing.assert_allclose(cur.premium, ent.premium)
    assert not np.allclose(cur.total, ent.total)


# --------------------------------------------------------------------------
# exact accounting
# --------------------------------------------------------------------------

def test_components_sum_to_total():
    paths = mc_paths(S0, T, R, 0.2, Q, n_steps=N_STEPS, n_paths=500, seed=8)
    res = delta_hedge_pnl(paths, TIMES, 0.22, R, Q, strike=K, costs=BASE_COSTS, hedge_every=3)
    parts = (res.premium + res.payoff + res.hedge_pnl + res.dividends + res.financing
             - res.option_cost - res.hedge_cost)
    np.testing.assert_allclose(parts, res.total, rtol=0, atol=1e-10)
    np.testing.assert_allclose(res.gross - res.costs, res.total)
    assert res.n_rebalances == len(range(0, N_STEPS, 3)) + 1


def test_never_moving_path_keeps_premium():
    # Spot frozen, r = q = 0, ATM: no hedge P&L, no payoff, P&L = premium exactly.
    path = np.full(N_STEPS + 1, S0)
    res = delta_hedge_pnl(path, TIMES, 0.2, 0.0, 0.0, strike=S0)
    prem = bs_price(S0, S0, T, 0.0, 0.2, 0.0, "call") + bs_price(S0, S0, T, 0.0, 0.2, 0.0, "put")
    assert res.total == pytest.approx(prem, abs=1e-12)
    assert res.hedge_pnl == pytest.approx(0.0, abs=1e-12)
    assert res.payoff == 0.0 and res.financing == 0.0


def test_zero_vol_deterministic_path_is_exactly_zero_pnl():
    # sigma = 0 and the stock grows at r - q: the option is a forward, the hedge
    # replicates it exactly, P&L = 0 to rounding (checks carry/dividend accounting).
    r, q = 0.05, 0.02
    path = S0 * np.exp((r - q) * TIMES)
    for k in (90.0, 100.0, 110.0):
        for kind in ("call", "put"):
            res = delta_hedge_pnl(path, TIMES, 1e-12, r, q, legs=single_option(k, kind))
            assert res.total == pytest.approx(0.0, abs=1e-9)


def test_costs_hand_computed():
    # Short one call, two 10-day steps, r = q = 0 so costs carry no interest.
    times = np.array([0.0, 10 / 365, 20 / 365])
    path = np.array([100.0, 103.0, 98.0])
    sig = 0.25
    cm = CostModel(option_half_spread_vol=0.01, hedge_cost_bps=2.0)
    res = delta_hedge_pnl(path, times, sig, 0.0, 0.0, legs=single_option(100.0, "call"), costs=cm)
    free = delta_hedge_pnl(path, times, sig, 0.0, 0.0, legs=single_option(100.0, "call"))
    Tm = times[-1]
    h0 = bs_delta(100.0, 100.0, Tm, 0.0, sig, 0.0, "call")
    h1 = bs_delta(103.0, 100.0, Tm - times[1], 0.0, sig, 0.0, "call")
    vega = bs_vega(100.0, 100.0, Tm, 0.0, sig, 0.0, "call")
    expected_opt = vega * 0.01
    expected_hedge = 2e-4 * (abs(h0) * 100.0 + abs(h1 - h0) * 103.0 + abs(h1) * 98.0)
    assert res.option_cost == pytest.approx(expected_opt, rel=1e-12)
    assert res.hedge_cost == pytest.approx(expected_hedge, rel=1e-12)
    assert free.total - res.total == pytest.approx(expected_opt + expected_hedge, rel=1e-12)
    # hedge P&L by hand
    assert res.hedge_pnl == pytest.approx(h0 * 3.0 + h1 * (-5.0), rel=1e-12)
    prem = bs_price(100.0, 100.0, Tm, 0.0, sig, 0.0, "call")
    assert res.total == pytest.approx(prem - 0.0 + h0 * 3 - h1 * 5 - expected_opt - expected_hedge)


def test_costs_accrue_interest_to_expiry():
    times = np.array([0.0, 0.5, 1.0])
    path = np.array([100.0, 100.0, 100.0])
    r = 0.04
    cm = CostModel(option_half_spread_vol=0.01, hedge_cost_bps=0.0)
    res = delta_hedge_pnl(path, times, 0.2, r, 0.0, legs=single_option(100.0, "call"), costs=cm)
    vega = bs_vega(100.0, 100.0, 1.0, r, 0.2, 0.0, "call")
    assert res.option_cost == pytest.approx(vega * 0.01 * np.exp(r * 1.0), rel=1e-12)


def test_long_position_mirrors_short():
    paths = mc_paths(S0, T, R, 0.2, Q, n_steps=N_STEPS, n_paths=200, seed=9)
    short = delta_hedge_pnl(paths, TIMES, 0.22, R, Q, strike=K)
    long = delta_hedge_pnl(paths, TIMES, 0.22, R, Q, legs=straddle(K, quantity=+1.0))
    np.testing.assert_allclose(long.gross, -short.gross, atol=1e-10)


def test_engine_input_validation():
    path = np.full(5, 100.0)
    times = np.linspace(0, 0.1, 5)
    with pytest.raises(ValueError):
        delta_hedge_pnl(path, times[::-1], 0.2, 0.0, strike=100.0)
    with pytest.raises(ValueError):
        delta_hedge_pnl(path, times, -0.2, 0.0, strike=100.0)
    with pytest.raises(ValueError):
        delta_hedge_pnl(path, times, 0.2, 0.0, strike=100.0, hedge_every=0)
    with pytest.raises(ValueError):
        delta_hedge_pnl(path, times, 0.2, 0.0, strike=100.0, hedge_vol="later")
    with pytest.raises(ValueError):
        delta_hedge_pnl(path, times, 0.2, 0.0)
    with pytest.raises(ValueError):
        Leg(100.0, "straddle")
    with pytest.raises(ValueError):
        CostModel(option_half_spread_vol=-1.0)


def test_cost_model_scaling():
    double = BASE_COSTS.scaled(2.0)
    assert double.hedge_cost_bps == 2 * BASE_COSTS.hedge_cost_bps
    assert BASE_COSTS.scaled(0.0) == ZERO_COSTS
    assert ZERO_COSTS.hedge_cost(5.0, 100.0) == 0.0
    assert BASE_COSTS.hedge_cost(-2.0, 1000.0) == pytest.approx(2 * 1000 * 1e-4)


# --------------------------------------------------------------------------
# metrics on hand-computable inputs
# --------------------------------------------------------------------------

def test_realised_vol_by_hand():
    p = np.array([100.0, 110.0, 99.0])
    r1, r2 = np.log(1.1), np.log(0.9)
    assert metrics.realised_vol(p) == pytest.approx(np.sqrt(252 * (r1**2 + r2**2) / 2))
    assert metrics.realised_vol(p, demean=True) == pytest.approx(
        np.sqrt(252) * np.std([r1, r2], ddof=1))
    assert metrics.realised_vol(np.full(10, 50.0)) == 0.0


def test_realised_vol_recovers_simulated_vol():
    paths = mc_paths(100.0, 1.0, 0.0, 0.3, n_steps=252, n_paths=1, seed=10)
    assert metrics.realised_vol(paths[0]) == pytest.approx(0.3, rel=0.15)


def test_summary_statistics_by_hand():
    x = np.array([1.0, -2.0, 3.0, -5.0, 1.0])
    assert metrics.max_drawdown(x) == pytest.approx(5.0)     # cum 1,-1,2,-3,-2
    assert metrics.max_drawdown([-1.0, -1.0]) == pytest.approx(2.0)
    assert metrics.max_drawdown([1.0, 2.0]) == 0.0
    assert metrics.hit_rate(x) == pytest.approx(0.6)
    mean, sd = x.mean(), x.std(ddof=1)
    assert metrics.t_stat(x) == pytest.approx(mean / (sd / np.sqrt(5)))
    assert metrics.sharpe(x, 12) == pytest.approx(mean / sd * np.sqrt(12))
    assert metrics.skewness([1.0, 2.0, 3.0]) == pytest.approx(0.0)
    d = np.array([0.0, 0.0, 3.0]) - 1.0
    assert metrics.skewness([0.0, 0.0, 3.0]) == pytest.approx(np.mean(d**3) / np.mean(d**2)**1.5)
    s = metrics.summarize(x, n_boot=500)
    assert s["n"] == 5 and s["worst"] == -5.0 and s["median"] == 1.0


def test_bootstrap_ci():
    lo, hi = metrics.bootstrap_ci(np.full(20, 0.7))
    assert lo == pytest.approx(0.7) and hi == pytest.approx(0.7)
    rng = np.random.default_rng(0)
    x = rng.normal(1.0, 2.0, 400)
    lo, hi = metrics.bootstrap_ci(x, seed=1)
    assert lo < x.mean() < hi
    se = x.std(ddof=1) / 20
    assert (hi - lo) == pytest.approx(2 * 1.96 * se, rel=0.15)
    assert metrics.bootstrap_ci(x, seed=1) == (lo, hi)      # deterministic


def test_metrics_reject_bad_input():
    with pytest.raises(ValueError):
        metrics.realised_vol([100.0])
    with pytest.raises(ValueError):
        metrics.summarize([])
    with pytest.raises(ValueError):
        metrics.t_stat([1.0, np.nan])
