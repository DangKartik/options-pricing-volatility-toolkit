"""Transaction-cost model for a delta-hedged option position.

EVERY NUMBER HERE IS AN ASSUMPTION. There is no historical option-quote
database behind this project, so neither the option bid-ask nor the hedge
slippage is observed; both are set to round, defensible values and the
backtest reports results at zero, base and double costs.

Option leg:  cost = |book vega| * option_half_spread_vol, paid once at entry
             (selling at the bid = mid minus half the spread, quoted in vol
             points because index option spreads are roughly constant in vol).
             There is NO exit cost: positions are held to expiry and cash
             settled at the index close, so the spread is never crossed twice.
Hedge leg:   cost = |shares traded| * S * hedge_cost_bps / 1e4 on every
             rebalance, including the initial hedge and the final unwind.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike

# ASSUMPTION: ATM 1-month SPX options trade roughly 0.5-1.0 vol points wide;
# half of that, at the conservative end, is 0.5 vol points (0.005 in decimal vol).
DEFAULT_OPTION_HALF_SPREAD_VOL = 0.005

# ASSUMPTION: hedging with E-mini S&P futures costs about half a tick
# (0.25 index points on a ~2000-7000 index = 0.2-0.6 bp) plus fees: ~1 bp of notional.
DEFAULT_HEDGE_COST_BPS = 1.0


@dataclass(frozen=True)
class CostModel:
    """Option half-spread (decimal vol, per unit |vega|) and hedge cost (bp of notional)."""

    option_half_spread_vol: float = DEFAULT_OPTION_HALF_SPREAD_VOL
    hedge_cost_bps: float = DEFAULT_HEDGE_COST_BPS

    def __post_init__(self) -> None:
        if self.option_half_spread_vol < 0 or self.hedge_cost_bps < 0:
            raise ValueError("cost parameters must be non-negative")

    def option_entry_cost(self, book_vega: ArrayLike) -> np.ndarray:
        """Entry cost = |book vega| * option_half_spread_vol (premium lost by selling at the bid)."""
        return np.abs(np.asarray(book_vega, dtype=float)) * self.option_half_spread_vol

    def hedge_cost(self, shares_traded: ArrayLike, spot: ArrayLike) -> np.ndarray:
        """Hedge cost = |shares traded| * S * hedge_cost_bps / 1e4."""
        traded = np.abs(np.asarray(shares_traded, dtype=float))
        return traded * np.asarray(spot, dtype=float) * self.hedge_cost_bps * 1e-4

    def scaled(self, factor: float) -> "CostModel":
        """Both cost parameters multiplied by ``factor`` (0 = frictionless, 2 = double)."""
        if factor < 0:
            raise ValueError("factor must be non-negative")
        return replace(self,
                       option_half_spread_vol=self.option_half_spread_vol * factor,
                       hedge_cost_bps=self.hedge_cost_bps * factor)


ZERO_COSTS = CostModel(option_half_spread_vol=0.0, hedge_cost_bps=0.0)
BASE_COSTS = CostModel()
