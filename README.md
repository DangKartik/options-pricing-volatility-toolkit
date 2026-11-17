# volkit

**Options pricing, implied-volatility surfaces, and a delta-hedged short-volatility backtest, written up as a research note.**

![license](https://img.shields.io/badge/license-MIT-blue) ![python](https://img.shields.io/badge/python-3.11%2B-blue) ![tests](https://img.shields.io/badge/tests-1496%20passing-brightgreen)

**Question.** Is S&P 500 implied volatility systematically higher than the volatility that follows, and would a delta-hedged short straddle have earned that gap after costs?

<table>
<tr>
<td width="50%"><img src="figures/smile_spx.png" alt="SPX implied-volatility smiles, 16 Nov 2026"></td>
<td width="50%"><img src="figures/cumulative_pnl.png" alt="Cumulative P&L of the delta-hedged short straddle"></td>
</tr>
<tr>
<td align="center"><sub>SPX smiles from the 16 Nov 2026 close</sub></td>
<td align="center"><sub>Hedged short straddle P&amp;L (synthetic prices; shaded = out of sample)</sub></td>
</tr>
</table>

## Key results

| | Result |
|---|---|
| **Is implied vol above realised vol?** | Yes. Over 272 months (Mar 2004 to Oct 2026), VIX exceeded the next 30 days' realised S&P 500 vol by **3.66 vol points** on average, in **83.5%** of months (t = 7.56). |
| **Does a hedged short straddle capture it?** | Not reliably. Out of sample (2019 on), after assumed costs: **+27.4 bp per month, t = 1.79, Sharpe 0.64, 95% CI [-0.07, 1.69]**. That is not statistically significant. |
| **What does it depend on?** | The gap between VIX and at-the-money implied vol (the "haircut"), which I cannot observe historically. Net break-even is **3.2 to 3.3 vol points**; on the two dates with usable quotes the gap was **4.1** (SPY, 13 Nov) and **5.0** (SPX, 16 Nov), above break-even. |
| **Do the pricers agree?** | Black-Scholes, a binomial tree and Monte Carlo agree within Monte Carlo error, and analytic and finite-difference Greeks match. 1,496 tests. |

**Read this first.** There are no historical option quotes in this project. The backtest prices straddles synthetically (Black-Scholes at VIX minus an assumed haircut, with assumed costs), so it tests whether VIX exceeds realised vol, not whether the strategy made money on real quotes. A conditional-entry variant looks better on paper, but its improvement is within noise (section 5). All parameters were frozen on 2004-2018 data before the single out-of-sample run.

**Contents:** [1. Repo and reproduction](#1-whats-in-the-repo) · [2. Pricing library](#2-pricing-library) · [3. Implied vol and surfaces](#3-implied-volatility-and-volatility-surfaces) · [4. Backtest](#4-backtest) · [5. What did not work](#5-what-did-not-work) · [6. Limitations](#6-limitations) · [7. Next steps](#7-what-i-would-do-next-with-real-data)

---

## 1. What's in the repo

| Path | What it does |
|---|---|
| `volkit/pricing/black_scholes.py` | Black-Scholes-Merton prices and analytic Greeks, with dividend yield |
| `volkit/pricing/binomial.py` | Cox-Ross-Rubinstein tree, European and American exercise |
| `volkit/pricing/monte_carlo.py` | Monte Carlo with antithetic variates and a standard error |
| `volkit/pricing/greeks.py` | Central finite-difference Greeks around any pricer |
| `volkit/iv/solver.py` | Implied vol: Newton with analytic vega, Brent fallback, American tree inversion |
| `volkit/iv/surface.py` | Parity forward, OTM quote selection, total-variance surface, arbitrage checks |
| `volkit/data/market.py`, `chain.py` | S&P 500, VIX family and T-bill series; option-chain download and cleaning |
| `volkit/backtest/` | Exact self-financing hedging P&L, cost model, metrics (Sharpe, bootstrap CI, drawdown) |
| `scripts/` | `download_data.py`, `build_surface.py`, `run_backtest.py` |
| `tests/` | One pytest file per module plus cross-pricer agreement tests |
| `results/`, `figures/` | My outputs, quoted and embedded below |
| `data/raw/` | **Not in the repo** (git-ignored); created by `scripts/download_data.py` |

Conventions (decimal rates and vols, `T` in years ACT/365) are in [CONVENTIONS.md](CONVENTIONS.md).

### Install and run

Developed with Python 3.14 (`pyproject.toml` allows 3.11+, untested).

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q               # 1496 passed; synthetic data, no download
.venv/bin/python scripts/download_data.py   # market data into data/raw/ (needs network)
```

Then run the analysis, giving `--snapshot` the chain file the download printed:

```bash
.venv/bin/python scripts/build_surface.py --snapshot data/raw/spy_chain_<date>.csv
.venv/bin/python scripts/build_surface.py --snapshot data/raw/spx_chain_<date>.csv --skip-bias-study
.venv/bin/python scripts/run_backtest.py    # Parts A and B -> results/backtest_*, figures
```

### About the data

The raw market data is **not included**: it comes from Yahoo Finance (option chains, S&P 500), CBOE (VIX family) and FRED (3-month T-bill), and I do not redistribute third-party data. Yahoo only serves the current chain, so **a fresh download gives different numbers from the ones in this README**, which come from snapshots taken on 15 November 2026 (SPY, plus a weekend SPX chain) and 17 November 2026 (SPX, carrying 16 November's close). The history series will also have extra days. If `data/raw/` is empty, both scripts stop with a message telling you to run the download script first. Running them overwrites the committed `results/` and `figures/` with the new numbers.

## 2. Pricing library

Black-Scholes-Merton, a CRR tree (European and American) and Monte Carlo, with analytic and finite-difference Greeks. For S = K = 100, T = 1, r = 5%, sigma = 20%, q = 0:

| | Black-Scholes | CRR, 2000 steps | MC, 200k paths antithetic, seed 1 | American, CRR 2000 |
|---|---|---|---|---|
| Call | 10.450584 | 10.449584 | 10.4122 (s.e. 0.0232) | 10.449584 |
| Put | 5.573526 | 5.572526 | 5.5488 (s.e. 0.0148) | 6.089990 |

The MC estimates are 1.66 and 1.67 standard errors below Black-Scholes (same paths, so not independent). With q = 0 the American call equals the European one.

- **CRR error is about 2/steps, with a sign flip between even and odd steps.** Error × steps is -2.00 (even) and +1.75 (odd) from 100 to 2001 steps.
- **Antithetic variance reduction depends on moneyness.** At 200,000 paths, variance fell 8.7x for an ITM call (K = 80), 2.0x ATM and 1.2x OTM (K = 120), where most paths pay zero.
- **The finite-difference spot bump is vol-scaled, h = 0.2 · sigma · sqrt(T) · S.** A 1% bump is smaller than the tree's node spacing (1.8% in log price at 500 steps), so the second difference mostly measures whether a node crossed the strike: true gamma is 0.01876, a 1% bump gave 0.0335 (500 steps) and 0.0071 (501), the vol-scaled bump 0.0194 and 0.0185.

| Same call | delta | gamma | vega | theta (per year) | rho |
|---|---|---|---|---|---|
| Analytic BSM | 0.63683 | 0.01876 | 37.524 | -6.414 | 53.232 |
| FD on BSM (default bumps) | 0.63546 | 0.01873 | 37.524 | -6.414 | 53.232 |
| FD on CRR, 2000 steps | 0.63546 | 0.01895 | 37.519 | -6.414 | 53.232 |

The delta gap is truncation error from the large default bump; a 0.01% bump recovers 0.63683.

## 3. Implied volatility and volatility surfaces

**Solver.** Newton-Raphson with analytic vega inside a shrinking bracket, falling back to Brent if a step leaves the bracket, vega is tiny or it fails to converge. Prices outside no-arbitrage bounds return `nan`.

**Data.** A SPY chain from Yahoo Finance, downloaded on Sunday 2026-11-15, so the quotes are the Friday 2026-11-13 close; `T` is measured from Friday 16:00 New York (measuring from the download time would raise the 10-day ATM IV from 14.79% to 15.98%). The SPX chain downloaded at the same time was unusable: only 5% of its 20,894 rows had a bid, and one expiry survived cleaning. A second SPX chain, downloaded on Tuesday 2026-11-17 at 08:59 UTC (about 4am New York, before the open), carried Monday 2026-11-16's closing quotes with bids on 94% of 20,481 rows, and is the basis of the SPX surface below (`T` measured from Monday 16:00 New York).

**Method.**

1. **Forward** from put-call parity, C - P = DF (F - K), with DF = exp(-rT) from the 3-month T-bill (4.22%, one flat rate). Regressing C - P on K would also give DF, but early exercise biases it: its implied rates run from -35% to +24%, and on a synthetic American chain with a true rate of 4.22% it gave -11.9% at 14 days.
2. **IVs** from out-of-the-money mids only (puts below F, calls at or above F), inverted with Black-Scholes. Re-inverting 120 quotes with an American tree, the mean gap per moneyness bucket was at most 0.15 vol points up to 126 days. It grows with maturity and peaks for near-ATM long-dated puts: 2.37 vol points for a December 2028 put (19.39% vs 17.02%), stable to under 0.01 vol points between 200 and 800 tree steps.
3. **Interpolation** in total variance w = sigma² T against k = ln(K/F), linear in k and in T. This adds no calendar arbitrage of its own. No smoothing or parametric fit.
4. **Outliers.** An isolated-quote rule removed 24 quotes, 23 in the March, June and September 2027 quarter-end expiries.

| Cleaning step (first filter failed) | Rows |
|---|---|
| Raw snapshot | 10,354 |
| bid < $0.05 | -1,167 |
| crossed (bid > ask) | -6 |
| relative spread > 25% | -217 |
| fewer than 7 days to expiry | -711 |
| ln(K/S) outside (-0.40, 0.30) | -1,515 |
| **Clean rows** | **6,738** |
| OTM quotes in the surface, after outlier rule | 3,661 in 28 expiries |

**Results** (`results/surface_summary.json`):

- 30-day ATM IV 15.22% vs VIX 19.30%, a 4.08-point gap. Expected: VIX is a variance-swap-style index over all strikes and includes the skew.
- 30-day 25-delta risk reversal 5.99 vol points (put 18.81%, call 12.82%).
- ATM term structure: 15.19% (9d), 15.22% (30d), 15.49% (93d), 17.88% (365d).
- Calendar violations: 0 of 1,107 grid points.
- Butterfly (convexity) violations: 613 of 3,605 strike triples at $0 tolerance, of which 552 are a cent or less and 426 are in expiries of 105 days or less. At 1 cent, 61 remain; 45 of those are in the three quarter-end expiries and January 2029 (largest deficit $4.98).
- Yahoo's IV column differs from mine by a median absolute 1.01 vol points.

![SPY smiles](figures/smile_spy.png)
*Smiles for five expiries, 7 to 763 days (line = mid IV, band = bid-ask IV); the short expiries are much steeper in k.*

![SPY ATM term structure](figures/term_structure_spy.png)
*ATM IV by expiry; VIX9D, VIX and VIX3M sit 2.6, 4.1 and 5.1 vol points above ATM IV at their maturities.*

![SPY total variance](figures/total_variance_spy.png)
*Total variance by expiry; curves do not cross. The small kinks visible in two curves are in quarter-end expiries (138 and 321 days).*

### SPX surface (16 Nov 2026 close)

Same pipeline, run on the SPX snapshot (`results/surface_summary.json`, key `SPX`). SPX options are European and cash-settled, so Black-Scholes inversion is exact and no early-exercise correction applies. (The summary file also contains a European-vs-American table for SPX; it is a by-product of the script and is not meaningful for these options.) 16 November was a stress day: the S&P 500 fell 1.9% from Friday's close (7,561.67 to 7,416.59) and VIX rose from 19.30 to 22.91.

- **Cleaning:** 20,481 rows in; removed 1,270 (bid < $0.05), 111 (crossed), 1,156 (spread > 25%), 1,567 (< 7 days), 120 (> 800 days), 1,452 (moneyness); 14,805 clean rows. The surface has 46 expiries and 8,698 OTM quotes after the outlier rule removed 60 (54 of them in the March, June and September 2027 quarter-end expiries).
- **30-day ATM IV 17.96% vs VIX 22.91%**, a 4.95-point gap, again with VIX above ATM IV (VIX9D 23.15 and VIX3M 23.02).
- **30-day 25-delta risk reversal 7.15 vol points** (put 22.42%, call 15.27%).
- **ATM term structure is inverted:** 20.44% (9d), 17.96% (30d), 17.02% (93d), 17.79% (365d), as expected after a sell-off.
- **Parity works as expected on a European chain.** The regression discount rate for 95-to-213-day expiries is 4.5-5.1% against the 4.22% T-bill (noisy at short expiries), unlike SPY, where early exercise pushed it negative. The implied dividend yield at short expiries is still not meaningful (14% at 7 days): forward × DF implies a spot of about 7,390-7,397 against the reported 7,416.59, the same spot-versus-option-close timing mismatch seen on SPY.
- **Arbitrage checks:** calendar 7 violations out of 1,845 grid points (one pair of expiries, 22 and 24 December, largest shortfall 0.0001 in total variance). Butterfly 3,124 of 8,606 strike triples at $0 tolerance and 2,727 at 1 cent, so this is far worse than SPY. SPX mids move in 5-10 cent steps, so the 1-cent tolerance is below one tick; the largest deficits I checked in the short expiries were 50-83 cents. I have not tried to repair them.
- **Data quality:** 111 crossed quotes; only 12.6% of contracts that traded in the last 30 minutes had a last price inside the bid-ask, so mids are used throughout.

![SPX smiles](figures/smile_spx.png)
*SPX smiles for five expiries, 7 to 760 days; the long-dated curve has visible kinks in the far put wing.*

![SPX ATM term structure](figures/term_structure_spx.png)
*ATM IV by expiry, with VIX9D, VIX and VIX3M. The curve is inverted at the short end.*

## 4. Backtest

**Part A** uses no trading assumptions: VIX at each month's first close vs zero-mean realised S&P 500 vol over the next 30 calendar days.

| Period | n | Mean IV − RV (vol pts) | Median | % IV > RV | t | 95% CI of mean |
|---|---|---|---|---|---|---|
| All 2004-03 to 2026-10 | 272 | 3.66 | 4.36 | 83.5% | 7.56 | [2.69, 4.54] |
| 2004-2007 | 46 | 3.00 | 3.99 | 76.1% | 7.21 | [2.17, 3.81] |
| 2008-2009 | 24 | 2.35 | 5.89 | 75.0% | 0.76 | [-4.00, 7.86] |
| 2010-2014 | 60 | 4.90 | 5.21 | 91.7% | 6.32 | [3.31, 6.32] |
| 2015-2019 | 60 | 3.17 | 4.57 | 83.3% | 4.32 | [1.67, 4.54] |
| 2020-2022 | 36 | 3.89 | 4.32 | 77.8% | 1.78 | [-0.83, 7.55] |
| 2023-2026 | 46 | 3.87 | 3.97 | 89.1% | 4.75 | [2.13, 5.26] |
| In-sample 2004-2018 | 178 | 3.48 | 4.38 | 82.6% | 6.28 | [2.36, 4.55] |
| Out-of-sample 2019- | 94 | 4.01 | 4.34 | 85.1% | 4.30 | [2.01, 5.67] |

In variance units (what a variance swap pays) the evidence is much weaker: VIX² − RV² has t = 1.86 overall and 0.89 out of sample, largely because a few crash months are very negative.

![VIX vs realised](figures/iv_vs_rv.png)
*VIX and the following 30-day realised vol, both at the start date; shading = out-of-sample. The worst months were March 2020, autumn 2008, April 2025 and August 2011.*

**Part B design.** At each month's first close, sell one straddle struck at the forward, K = S₀ exp((r − q)T), expiring on the trading day nearest entry + 30 days (capped at the next entry, so no overlap). Options are priced by Black-Scholes at VIX − haircut, delta-hedged at the close every 2 days at that day's VIX − haircut, and cash settled. Assumptions, not measurements: haircut 2 vol points; option half-spread 0.5 vol points × vega, paid at entry; hedge cost 1 bp of notional traded; q = 1.8%. P&L is in bp of entry spot, and `delta_hedge.py` decomposes it exactly into premium, payoff, hedge P&L, dividends, financing and costs.

**Protocol.** Everything was chosen on in-sample entries (2004-03 to 2018-12). Haircut, costs, hedge vol and q were fixed a priori. Tuned on in-sample net Sharpe: hedge frequency from {1, 2, 5} days (2 won: 0.73 vs 0.51 and 0.30), and an optional entry filter (sell only if VIX minus trailing 21-day RV exceeds a threshold) from seven thresholds 0 to 0.06 (0.05 won). These go into [`results/frozen_params.json`](results/frozen_params.json); the out-of-sample step reads only that file and runs the 2019+ trades once. OOS sensitivities are reported, never used to choose. Re-running both scripts on the same cached data, in a copy of the repo, gave identical results apart from the frozen file's timestamp. One weakness: `run_backtest.py` re-runs the in-sample calibration and rewrites `frozen_params.json` on every run, so the file's timestamp does not prove the order of events. The git history of the file, and the fact that its values never change between runs, are the better evidence.

**Part B results** (bp of notional per trade; Sharpe annualised with sqrt(12); bootstrap CIs):

| Variant | n | Mean (bp) | t | Sharpe [95% CI] | Worst trade (bp, entry) | Max DD (bp) |
|---|---|---|---|---|---|---|
| IS always-sell, gross | 178 | 41.4 | 4.23 | 1.10 [0.55, 1.78] | -574 (2008-10-01) | 828 |
| IS always-sell, net | 178 | 27.5 | 2.80 | 0.73 [0.21, 1.35] | -587 (2008-10-01) | 869 |
| IS conditional, gross | 62 | 59.3 | 4.51 | 1.98 [1.01, 3.24] | -263 (2018-10-01) | 271 |
| IS conditional, net | 62 | 45.4 | 3.44 | 1.52 [0.59, 2.67] | -277 (2018-10-01) | 312 |
| **OOS always-sell, gross** | 94 | 41.4 | 2.71 | 0.97 [0.20, 2.13] | -778 (2020-03-02) | 1,272 |
| **OOS always-sell, net** | 94 | **27.4** | **1.79** | **0.64 [-0.07, 1.69]** | -791 (2020-03-02) | 1,315 |
| OOS conditional, gross | 40 | 66.8 | 3.41 | 1.87 [0.60, 4.06] | -410 (2020-02-03) | 410 |
| OOS conditional, net | 40 | 52.8 | 2.69 | 1.47 [0.29, 3.47] | -425 (2020-02-03) | 446 |

The conditional Sharpes annualise with sqrt(12) although the rule trades in only 40 of 94 OOS months, which flatters them. Counting skipped months as zero, its OOS net Sharpe is 0.92 vs 0.64 for always-sell, and it earns 22.5 bp per month vs 27.4. Costs average 11.3 bp per trade (option spread) plus 2.6 bp (hedging).

**Sensitivities** (net mean bp per trade, Sharpe in brackets; one input moved from the frozen setting):

| Change | In-sample | Out-of-sample |
|---|---|---|
| haircut 0 / 1 / **2** / 3 / 4 vol pts | 72.7 / 50.1 / **27.5** / 4.8 / -17.9 | 70.8 / 49.0 / **27.4** / 5.8 / -15.5 |
| break-even haircut, net (gross) | 3.21 (3.83) | 3.27 (3.93) |
| costs zero / **base** / double | 41.4 / **27.5** / 13.5 (1.10 / 0.73 / 0.36) | 41.4 / **27.4** / 13.4 (0.97 / 0.64 / 0.31) |
| hedge every 1 / **2** / 5 days | 18.7 / **27.5** / 15.0 (0.51 / 0.73 / 0.30) | 24.0 / **27.4** / 33.4 (0.57 / 0.64 / 0.65) |
| hedge vol **current** / entry | **27.5** / 23.8 | **27.4** / 23.7 |
| q 0 / **1.8%** / 3% | 29.1 / **27.5** / 26.0 | 28.9 / **27.4** / 26.3 |

Each vol point of haircut moves the mean by about 22 bp per trade, against a net mean of 27 bp. Costs matter next; q and hedge vol barely matter.

**Attribution.** Gross P&L per unit of entry vega correlates 0.79 (IS) and 0.85 (OOS) with pricing vol minus realised vol over the trade, with slope 0.62 in both. The P&L mostly tracks implied minus realised; the slope below 1 is consistent with path dependence and discrete hedging, which I have not separated.

![Cumulative P&L](figures/cumulative_pnl.png)
*Cumulative P&L, one straddle per month: gross, net, and the conditional rule (net). The 2008 and 2020 drawdowns show in all three.*

![Haircut sensitivity](figures/haircut_sensitivity.png)
*Mean P&L per trade vs haircut; net break-even 3.21 (IS) and 3.27 (OOS). Dashed: the 4.1-point VIX − ATM gap on the one snapshot date.*

![P&L vs implied minus realised](figures/pnl_vs_vrp_scatter.png)
*Per-trade gross P&L (vol points) vs pricing vol minus realised vol; the far bottom-left points are October 2008 (IS) and March 2020 (OOS).*

## 5. What did not work

- **The first SPX snapshot was unusable** (taken on a weekend, bids on 5% of rows). The SPX surface above comes from a second download. That one was taken before the open, not during market hours, so it carries Monday's closing quotes, and one day of quotes is all I have.
- **Residual butterfly arbitrage.** 61 violations at 1 cent, mostly in quarter-end and January 2029 expiries; counted, not repaired.
- **Early-exercise bias measured, not corrected.** The long-dated SPY put wing is inverted as European, overstating IV by up to 2.37 vol points at 2 years.
- **Short-dated implied dividend yields are meaningless.** Parity q at 7 days is 12.7%. Forward × DF for expiries up to 30 days implies a spot near 752.4 vs the reported 754.37, a 0.26% timing mismatch between spot and option closes that a small T blows up.
- **The conditional rule's improvement is within noise.** The in-sample grid is flat (Sharpe 0.72 to 0.85 for thresholds 0 to 0.05). The OOS months it skipped averaged +8.5 bp (t = 0.39, computed from `backtest_trades.csv`). Its edge comes mostly from skipping March 2020 (-791 bp), while it traded February 2020 (-425 bp); dropping March 2020 post hoc, always-sell has the higher OOS Sharpe (1.03 vs 0.93).
- **The in-sample hedge frequency was not best out of sample.** 2 days won in-sample (paired t vs daily only 1.72); out of sample 5 days was slightly better (Sharpe 0.65 vs 0.64, mean 33.4 vs 27.4 bp).
- **One bug found after seeing OOS results.** The first in-sample trade had no entry signal (fewer than 21 prior returns). The in-sample period now starts in March 2004 and a missing signal raises an error. OOS results were unchanged and no parameter was retuned, but the fix came after the OOS run.

## 6. Limitations

- Synthetic option prices. VIX is not ATM IV; the gap is unknown historically and likely time-varying, and my two observations (about 4.1 points on SPY on a calm day, about 5.0 on SPX on a stress day) are above break-even. Two dates cannot say how the gap behaves over time. VIX also has constant 30-day maturity while the option ages.
- Costs are assumed. Hedging is close-only; no intraday hedging, margin or capital limits; P&L is per straddle.
- Small samples: 178 IS and 94 OOS trades.
- Constant q = 1.8%; one flat T-bill rate for all maturities.
- Mild data snooping: 3 hedge frequencies × 7 thresholds tried in-sample.
- Yahoo data is noisy (only 43% of contracts traded in the last 30 minutes had a last inside the bid-ask); the surface is one snapshot with no SVI fit.

## 7. What I would do next with real data

- Historical SPX quotes (OptionMetrics or CBOE end-of-day) to replace synthetic pricing and measure the VIX − ATM gap month by month.
- SPX surfaces from snapshots taken during market hours, on many dates, to see how the VIX − ATM gap and the arbitrage violations vary (the one SPX surface here is a single post-close snapshot).
- Per-expiry SVI fits with no-butterfly and no-calendar constraints.
- SPY forwards and dividends estimated consistently with American exercise, jointly with the tree inversion.
