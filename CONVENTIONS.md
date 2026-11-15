# volkit conventions (binding for every module)

## Symbols and units
- `S` spot, `K` strike, `T` time to expiry in **years**, `r` continuously
  compounded risk-free rate, `q` continuous dividend yield, `sigma` annualised
  volatility. All rates and vols are decimals (0.05, 0.20), never percent.
- `kind` is the string `"call"` or `"put"`. Anything else raises `ValueError`.
- Argument order is always `(S, K, T, r, sigma, q=0.0, kind="call")`.
- Trading-day year = 252 for realised vol; calendar-day year = 365 for `T`.

## Greeks units (raw mathematical derivatives, no rescaling)
- `delta` = dV/dS, `gamma` = d2V/dS2, `vega` = dV/dsigma (per 1.00 of vol),
  `theta` = dV/dt = -dV/dT (per year, usually negative for long options),
  `rho` = dV/dr (per 1.00 of rate).

## Public interfaces
```
volkit/pricing/black_scholes.py
    bs_price(S, K, T, r, sigma, q=0.0, kind="call") -> float | ndarray   (vectorised)
    bs_delta, bs_gamma, bs_vega, bs_theta, bs_rho      same signature
    bs_greeks(...) -> dict with keys delta, gamma, vega, theta, rho
volkit/pricing/greeks.py
    fd_greeks(price_fn, S, K, T, r, sigma, q=0.0, kind="call", **kw) -> dict (same keys)
        central finite differences around any pricer with the standard signature
volkit/pricing/binomial.py
    crr_price(S, K, T, r, sigma, q=0.0, kind="call", steps=500, american=False) -> float
volkit/pricing/monte_carlo.py
    mc_price(S, K, T, r, sigma, q=0.0, kind="call", n_paths=200_000,
             antithetic=True, seed=None) -> MCResult(price, stderr, n_paths)
volkit/iv/solver.py
    implied_vol(price, S, K, T, r, q=0.0, kind="call", tol=1e-8, max_iter=100) -> float
        returns nan when price violates no-arbitrage bounds
volkit/data/
    market.py     load_spx(), load_vix(), load_rates()  -> pandas Series indexed by date
    chain.py      fetch_chain(ticker) -> DataFrame; clean_chain(df, spot, ...) -> DataFrame
```

## Code rules
- Python 3.14, numpy/scipy/pandas/matplotlib only (plus yfinance in data/).
- Type hints and a short docstring on every public function. The docstring
  states the formula or method in one or two lines; no essays.
- No silent failure: invalid inputs raise `ValueError`.
- Tests use pytest, live in `tests/test_<module>.py`, must be deterministic
  (fixed seeds) and must not touch the network.
- Run tests with `.venv/bin/python -m pytest -q` from the repo root.
- Do not commit. Do not edit files outside your assigned ones.
