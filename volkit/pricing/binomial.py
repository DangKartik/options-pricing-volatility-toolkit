"""Cox-Ross-Rubinstein binomial tree for European and American options."""

from __future__ import annotations

import numpy as np


def _intrinsic(S: np.ndarray | float, K: float, kind: str) -> np.ndarray | float:
    """Payoff from exercising now: max(S - K, 0) for a call, max(K - S, 0) for a put."""
    if kind == "call":
        return np.maximum(S - K, 0.0)
    return np.maximum(K - S, 0.0)


def crr_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    kind: str = "call",
    steps: int = 500,
    american: bool = False,
) -> float:
    """Price an option on a Cox-Ross-Rubinstein tree.

    u = exp(sigma sqrt(dt)), d = 1/u, p = (exp((r-q) dt) - d) / (u - d);
    backward induction V = exp(-r dt) (p V_up + (1-p) V_down), with
    V = max(V, intrinsic) at every node when american=True.
    """
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if S < 0 or K < 0 or T < 0 or sigma < 0:
        raise ValueError("S, K, T and sigma must be non-negative")

    if T == 0:
        return float(_intrinsic(S, K, kind))
    if sigma == 0:
        # u = d = 1, so the tree has no branching and p is 0/0.
        raise ValueError("sigma must be positive to build a tree when T > 0")

    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 <= p <= 1.0:
        # Happens when |r - q| sqrt(dt) > sigma: the forward lies outside [d, u].
        raise ValueError(
            f"risk-neutral probability p={p:.4f} is outside [0, 1]; "
            "the tree is not arbitrage-free for these inputs, increase steps"
        )
    discount = np.exp(-r * dt)

    # Node j at time step i has had j up-moves and i - j down-moves,
    # so its spot is S u^j d^(i-j) = S u^(2j - i).
    j = np.arange(steps + 1)
    values = _intrinsic(S * u ** (2 * j - steps), K, kind)

    for i in range(steps - 1, -1, -1):
        # values[1:] are the up-children, values[:-1] the down-children.
        values = discount * (p * values[1:] + (1.0 - p) * values[:-1])
        if american:
            spots = S * u ** (2 * np.arange(i + 1) - i)
            values = np.maximum(values, _intrinsic(spots, K, kind))

    return float(values[0])
