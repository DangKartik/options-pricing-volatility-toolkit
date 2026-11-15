"""Self-financing P&L of a delta-hedged European option position held to expiry.

One trade = a static book of European legs (default: short one call + one put,
a straddle) opened at the close of day 0 and cash settled at the close of day n.
Inputs are the closes S_0..S_n on a time grid t_0 = 0 < ... < t_n = T (years,
calendar ACT/365 per CONVENTIONS.md). The book is hedged with h_i shares chosen
at the close of day i and held over (t_i, t_{i+1}].

Accounting (exact, no marking of the options in between):
    cash X_0 = premium - h_0 S_0
    over (t_i, t_{i+1}]:   X += X (e^{r_i dt} - 1)               financing
                           X += h_i S_{i+1} (e^{q_i dt} - 1)     dividends on the hedge
    rebalance at i+1 < n:  X -= (h_{i+1} - h_i) S_{i+1}
    expiry:                X += h_{n-1} S_n + payoff             unwind + cash settlement
Summation by parts gives the exact identity
    total = premium + payoff + hedge_pnl + dividends + financing - costs,
    hedge_pnl = sum_i h_i (S_{i+1} - S_i).
Transaction costs are booked in a separate account that also accrues at r, so
``costs`` is their value at expiry and the gross (cost-free) P&L is
``total + costs`` exactly: costs do not change any hedge ratio.

All money amounts are in expiry-date currency per one unit of the underlying.
Dividends use the end-of-interval convention h_i S_{i+1} (e^{q dt} - 1), which
makes the discounted hedge gains an exact martingale when the stock drifts at
r - q, so E[total] = e^{rT} (premium - risk-neutral value) with no discretisation bias.

The engine broadcasts over leading axes: ``spot`` may be (n+1,) for one real
trade or (n_paths, n+1) for Monte Carlo validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from volkit.backtest.costs import ZERO_COSTS, CostModel
from volkit.pricing.black_scholes import bs_delta, bs_price, bs_vega

__all__ = ["Leg", "straddle", "single_option", "HedgeResult", "delta_hedge_pnl"]


@dataclass(frozen=True)
class Leg:
    """One European option: strike, kind ('call'/'put'), signed quantity (-1 = short one)."""

    strike: float
    kind: str
    quantity: float = -1.0

    def __post_init__(self) -> None:
        if self.kind not in ("call", "put"):
            raise ValueError(f"kind must be 'call' or 'put', got {self.kind!r}")
        if not self.strike > 0:
            raise ValueError("strike must be positive")


def straddle(strike: float, quantity: float = -1.0) -> tuple[Leg, Leg]:
    """Call + put at the same strike, each with ``quantity`` (default short one of each)."""
    return (Leg(strike, "call", quantity), Leg(strike, "put", quantity))


def single_option(strike: float, kind: str, quantity: float = -1.0) -> tuple[Leg]:
    """A one-leg book (default short one option)."""
    return (Leg(strike, kind, quantity),)


@dataclass(frozen=True)
class HedgeResult:
    """P&L components of one trade (or one per path), all valued at expiry except ``premium``.

    total = premium + payoff + hedge_pnl + dividends + financing - costs.
    Signs are cash-flow signs for the holder: a short book has premium > 0, payoff <= 0.
    """

    premium: np.ndarray        # cash received at entry for the book at the entry vol (gross of spread)
    payoff: np.ndarray         # cash received at expiry settlement (<= 0 for a short book)
    hedge_pnl: np.ndarray      # sum_i h_i (S_{i+1} - S_i), price gains of the hedge
    dividends: np.ndarray      # dividends earned on the hedge shares
    financing: np.ndarray      # interest on the cash account (premium carry + hedge funding)
    option_cost: np.ndarray    # option half-spread paid at entry, carried to expiry at r
    hedge_cost: np.ndarray     # hedge commissions/slippage, each carried to expiry at r
    entry_vega: np.ndarray     # |book vega| at entry (per 1.00 of vol), for vol-point scaling
    n_rebalances: int          # hedge trades including the initial hedge and the final unwind

    @property
    def costs(self) -> np.ndarray:
        """Total transaction costs at expiry value (>= 0)."""
        return self.option_cost + self.hedge_cost

    @property
    def gross(self) -> np.ndarray:
        """P&L before transaction costs."""
        return self.premium + self.payoff + self.hedge_pnl + self.dividends + self.financing

    @property
    def total(self) -> np.ndarray:
        """P&L after transaction costs (expiry value)."""
        return self.gross - self.costs


def _as_path(x: ArrayLike, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Broadcast a scalar / per-day array to the spot shape, or raise ValueError."""
    arr = np.asarray(x, dtype=float)
    try:
        return np.broadcast_to(arr, shape)
    except ValueError as exc:
        raise ValueError(f"{name} with shape {arr.shape} does not broadcast to spot {shape}") from exc


def delta_hedge_pnl(
    spot: ArrayLike,
    times: ArrayLike,
    iv: ArrayLike,
    r: ArrayLike,
    q: ArrayLike = 0.0,
    legs: tuple[Leg, ...] | None = None,
    strike: float | None = None,
    hedge_every: int = 1,
    hedge_vol: str | ArrayLike = "current",
    costs: CostModel = ZERO_COSTS,
) -> HedgeResult:
    """P&L of a static option book sold at iv[0] and delta hedged until cash settlement.

    spot:   closes S_0..S_n, shape (..., n+1); S_n is the settlement close.
    times:  t_0 = 0 < ... < t_n = T in years, shape (n+1,); time to expiry is T - t_i.
    iv:     implied vol on each day (broadcast to spot). The premium uses iv[..., 0];
            iv[..., n] is never used.
    r, q:   rate and dividend yield per day (scalar or broadcast to spot); r_i, q_i
            apply over (t_i, t_{i+1}], i.e. only values known at the start of the interval.
    legs:   the book; default ``straddle(strike)`` (short one call + one put).
    hedge_every: rebalance at the close of days 0, k, 2k, ... (< n); always unwind at n.
    hedge_vol: vol in the hedge delta: "current" = iv on that day (default),
            "entry" = iv[..., 0] for the whole life, or an explicit array/scalar.
    costs:  a CostModel; option half-spread at entry, hedge cost on every trade.
    """
    S = np.asarray(spot, dtype=float)
    t = np.asarray(times, dtype=float)
    if S.ndim < 1 or S.shape[-1] < 2:
        raise ValueError("spot must have at least two closes (entry and settlement)")
    if t.shape != (S.shape[-1],):
        raise ValueError(f"times must have shape ({S.shape[-1]},), got {t.shape}")
    if t[0] != 0.0 or np.any(np.diff(t) <= 0.0):
        raise ValueError("times must start at 0 and be strictly increasing")
    if np.any(S <= 0.0) or not np.all(np.isfinite(S)):
        raise ValueError("spot must be finite and strictly positive")
    if not (isinstance(hedge_every, (int, np.integer)) and hedge_every >= 1):
        raise ValueError("hedge_every must be a positive integer")
    if legs is None:
        if strike is None:
            raise ValueError("give either legs or strike")
        legs = straddle(strike)
    elif strike is not None:
        raise ValueError("give legs or strike, not both")
    if len(legs) == 0:
        raise ValueError("legs must not be empty")

    shape = S.shape
    n = shape[-1] - 1
    T = t[-1]
    tau = T - t                                   # time to expiry on each day
    dt = np.diff(t)                               # (n,)
    sig = _as_path(iv, shape, "iv")
    rr = _as_path(r, shape, "r")
    qq = _as_path(q, shape, "q")
    if np.any(sig[..., :n] <= 0.0) or not np.all(np.isfinite(sig[..., :n])):
        raise ValueError("implied vol must be finite and positive on every day before expiry")

    if isinstance(hedge_vol, str):
        if hedge_vol == "current":
            hvol = sig
        elif hedge_vol == "entry":
            hvol = np.broadcast_to(sig[..., :1], shape)
        else:
            raise ValueError(f"hedge_vol must be 'current', 'entry' or an array, got {hedge_vol!r}")
    else:
        hvol = _as_path(hedge_vol, shape, "hedge_vol")
        if np.any(hvol[..., :n] <= 0.0):
            raise ValueError("hedge_vol must be positive before expiry")

    # ---- entry: premium, vega, option spread cost (at the entry implied vol) ----
    S0, sig0, r0, q0 = S[..., 0], sig[..., 0], rr[..., 0], qq[..., 0]
    premium = np.zeros(shape[:-1])
    vega0 = np.zeros(shape[:-1])
    for leg in legs:
        premium -= leg.quantity * bs_price(S0, leg.strike, T, r0, sig0, q0, leg.kind)
        vega0 += leg.quantity * bs_vega(S0, leg.strike, T, r0, sig0, q0, leg.kind)
    entry_vega = np.abs(vega0)

    # ---- hedge ratios on days 0..n-1 (vectorised; tau > 0 there) ----
    book_delta = np.zeros(shape[:-1] + (n,))
    for leg in legs:
        book_delta += leg.quantity * bs_delta(S[..., :n], leg.strike, tau[:n], rr[..., :n],
                                              hvol[..., :n], qq[..., :n], leg.kind)
    target = -book_delta
    # Hold the last chosen hedge between rebalance days.
    rebalance_days = np.arange(0, n, hedge_every)
    last_rebalance = rebalance_days[np.searchsorted(rebalance_days, np.arange(n), side="right") - 1]
    h = target[..., last_rebalance]                       # (..., n): shares held over (t_i, t_{i+1}]

    # ---- cash account and cost accounts, stepped through time ----
    growth = np.exp(rr[..., :n] * dt)                     # cash growth over each interval
    div_factor = np.exp(qq[..., :n] * dt) - 1.0
    cash = premium - h[..., 0] * S0
    opt_cost = costs.option_entry_cost(vega0) * np.ones(shape[:-1])
    hed_cost = costs.hedge_cost(h[..., 0], S0) * np.ones(shape[:-1])
    hedge_pnl = np.zeros(shape[:-1])
    dividends = np.zeros(shape[:-1])
    financing = np.zeros(shape[:-1])
    for i in range(n):
        g = growth[..., i]
        S_next = S[..., i + 1]
        interest = cash * (g - 1.0)
        div = h[..., i] * S_next * div_factor[..., i]
        financing += interest
        dividends += div
        hedge_pnl += h[..., i] * (S_next - S[..., i])
        cash = cash + interest + div
        opt_cost = opt_cost * g
        hed_cost = hed_cost * g
        h_new = h[..., i + 1] if i + 1 < n else np.zeros(shape[:-1])   # unwind at expiry
        traded = h_new - h[..., i]
        cash = cash - traded * S_next
        hed_cost = hed_cost + costs.hedge_cost(traded, S_next)

    payoff = np.zeros(shape[:-1])
    S_T = S[..., n]
    for leg in legs:
        intrinsic = np.maximum(S_T - leg.strike, 0.0) if leg.kind == "call" \
            else np.maximum(leg.strike - S_T, 0.0)
        payoff += leg.quantity * intrinsic
    cash = cash + payoff

    result = HedgeResult(premium=premium, payoff=payoff, hedge_pnl=hedge_pnl,
                         dividends=dividends, financing=financing, option_cost=opt_cost,
                         hedge_cost=hed_cost, entry_vega=entry_vega,
                         n_rebalances=int(len(rebalance_days) + 1))
    # The component identity is exact by construction; check it against the
    # independently stepped cash account to catch any accounting regression.
    if not np.allclose(cash, result.gross, rtol=1e-10, atol=1e-9 * float(np.max(S))):
        raise AssertionError("cash account does not reconcile with P&L components")
    return result
