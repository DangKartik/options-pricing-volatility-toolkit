"""Volatility risk premium on the S&P 500 and a delta-hedged short straddle.

Run:  .venv/bin/python scripts/run_backtest.py               (full study, IS then OOS once)
      .venv/bin/python scripts/run_backtest.py --stage in-sample   (Part A + IS only)

DATA LIMITATION THAT DEFINES THIS STUDY
---------------------------------------
There is no historical option-quote database. Option prices are SYNTHETIC:
Black-Scholes at sigma = VIX_close - iv_haircut. VIX is a model-free 30-day
variance-swap-style measure over ALL strikes, so it contains the put skew and
sits ABOVE at-the-money implied vol; pricing an ATM straddle at VIX overstates
the premium collected. ``iv_haircut`` (vol points subtracted from VIX) and the
break-even haircut make that explicit. There is no real bid-ask either: costs
are assumptions (volkit/backtest/costs.py). Part A (VIX vs realised variance)
needs none of these assumptions; Part B (the strategy) needs all of them.

PROTOCOL
--------
In-sample (IS)      = entries 2004-03 .. 2018-12 (March 2004 is the first month whose
                      entry close has 21 prior returns for the entry signal; SPX data
                      start 2004-01-02 and 2004-02-02 is only the 21st close).
Out-of-sample (OOS) = entries 2019-01 .. last month whose trade has settled.
``calibrate_in_sample`` sees only IS trades, makes every free choice and writes
results/frozen_params.json. ``evaluate_out_of_sample`` reads ONLY that file and
runs the OOS trades once. Nothing OOS is computed before the file exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from volkit.backtest import metrics  # noqa: E402
from volkit.backtest.costs import BASE_COSTS, CostModel  # noqa: E402
from volkit.backtest.delta_hedge import delta_hedge_pnl, straddle  # noqa: E402
from volkit.data.market import (  # noqa: E402
    RATES_FILE, SPX_FILE, VIX_FILE, align_to_calendar, load_rates, load_spx, load_vix,
    require_cached,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FROZEN_PATH = RESULTS / "frozen_params.json"

TENOR_DAYS = 30                       # calendar days, the VIX horizon
RV_LOOKBACK = 21                      # trading-day returns in the entry signal
IS_START, IS_END = pd.Timestamp("2004-03-01"), pd.Timestamp("2018-12-31")
OOS_START = pd.Timestamp("2019-01-01")
TRADES_PER_YEAR = 12.0
DIVIDEND_YIELD = 0.018                # ASSUMPTION: approx. average S&P 500 yield 2004-2026
HEADLINE_HAIRCUT = 0.02               # 2 vol points, fixed a priori (see calibrate_in_sample)
HAIRCUT_GRID = [0.0, 0.01, 0.02, 0.03, 0.04]
HEDGE_EVERY_GRID = [1, 2, 5]
THRESHOLD_GRID = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]   # VIX - RV21, decimal vol
MIN_IS_TRADES_CONDITIONAL = 60        # a filter must keep >= 1/3 of the 178 IS months
Q_GRID = [0.0, DIVIDEND_YIELD, 0.03]
BOOT_SEED = 12345
DPI = 160


# ==========================================================================
# data and trade calendar
# ==========================================================================

@dataclass(frozen=True)
class Market:
    dates: pd.DatetimeIndex
    spx: np.ndarray
    vix: np.ndarray           # decimal vol
    rate: np.ndarray          # continuously compounded decimal
    rv21: np.ndarray          # trailing realised vol of the RV_LOOKBACK returns ending at each close


def load_market() -> Market:
    """SPX closes with VIX and the T-bill rate on the SPX calendar (no forward-filling of VIX)."""
    spx = load_spx()
    vix = load_vix().reindex(spx.index)
    if vix.isna().any():
        raise ValueError("VIX missing on SPX trading days")
    rate = align_to_calendar(load_rates(), spx.index)      # bond holidays: prior quote
    logret = np.log(spx).diff()
    # rv21 at close t uses returns ending at t (closes t-21..t): known at the entry close.
    rv21 = np.sqrt(252.0 * (logret**2).rolling(RV_LOOKBACK).mean())
    return Market(pd.DatetimeIndex(spx.index), spx.to_numpy(float), vix.to_numpy(float),
                  rate.to_numpy(float), rv21.to_numpy(float))


@dataclass(frozen=True)
class TradeSpec:
    entry: int                # index of the entry close
    expiry: int               # index of the settlement close
    capped: bool              # expiry pulled in to the next month's entry (no overlap)


def build_trade_calendar(dates: pd.DatetimeIndex) -> list[TradeSpec]:
    """One trade per calendar month: enter at the first trading day's close; expiry = the
    trading day nearest entry + 30 calendar days (ties -> earlier), capped at the next
    month's entry so trades never overlap. Trades whose target date lies beyond the last
    close are incomplete and dropped. Uses only the exchange calendar, never prices."""
    month_key = dates.year * 12 + dates.month
    first_idx = np.flatnonzero(np.r_[True, month_key[1:] != month_key[:-1]])
    specs = []
    for k, e in enumerate(first_idx):
        target = dates[e] + pd.Timedelta(days=TENOR_DAYS)
        if target > dates[-1]:
            break
        j = int(np.searchsorted(dates, target))           # first close >= target
        if dates[j] != target and (dates[j] - target) >= (target - dates[j - 1]):
            j -= 1                                        # nearest, ties -> earlier
        capped = False
        if k + 1 < len(first_idx) and j > first_idx[k + 1]:
            j, capped = int(first_idx[k + 1]), True
        specs.append(TradeSpec(int(e), j, capped))
    return specs


def split_specs(specs: list[TradeSpec], dates: pd.DatetimeIndex) -> tuple[list, list]:
    """IS = entries in [IS_START, IS_END]; OOS = entries from OOS_START."""
    is_specs = [s for s in specs if IS_START <= dates[s.entry] <= IS_END]
    oos_specs = [s for s in specs if dates[s.entry] >= OOS_START]
    return is_specs, oos_specs


# ==========================================================================
# Part A: raw variance risk premium, no trading assumptions
# ==========================================================================

def forward_rv(mkt: Market, e: int) -> float:
    """Zero-mean realised vol of the SPX returns over (t_e, t_e + 30 calendar days]."""
    end = int(np.searchsorted(mkt.dates, mkt.dates[e] + pd.Timedelta(days=TENOR_DAYS),
                              side="right"))              # closes strictly after +30d excluded
    return metrics.realised_vol(mkt.spx[e:end])


def part_a(mkt: Market, specs: list[TradeSpec]) -> tuple[dict, pd.DataFrame]:
    """VIX at each monthly entry vs realised vol over the next 30 calendar days."""
    rows = []
    for s in specs:
        rv = forward_rv(mkt, s.entry)
        rows.append({"date": mkt.dates[s.entry], "vix": mkt.vix[s.entry], "rv_fwd": rv})
    df = pd.DataFrame(rows)
    df["iv_minus_rv_volpts"] = 100 * (df.vix - df.rv_fwd)
    df["var_premium"] = df.vix**2 - df.rv_fwd**2

    def block(d: pd.DataFrame) -> dict:
        x = d.iv_minus_rv_volpts.to_numpy()
        return {"n": int(len(x)), "mean_volpts": float(x.mean()),
                "median_volpts": float(np.median(x)), "frac_iv_gt_rv": float(np.mean(x > 0)),
                "t_stat": metrics.t_stat(x),
                "mean_ci95": metrics.bootstrap_ci(x, seed=BOOT_SEED),
                "mean_var_premium": float(d.var_premium.mean()),
                "t_stat_var_premium": metrics.t_stat(d.var_premium.to_numpy()),
                "mean_vix": float(d.vix.mean()), "mean_rv": float(d.rv_fwd.mean())}

    out = {"all": block(df)}
    periods = {"2004-2007": ("2004", "2007"), "2008-2009": ("2008", "2009"),
               "2010-2014": ("2010", "2014"), "2015-2019": ("2015", "2019"),
               "2020-2022": ("2020", "2022"), "2023-2026": ("2023", "2026"),
               "IS 2004-2018": ("2004", "2018"), "OOS 2019-": ("2019", "2026")}
    for name, (a, b) in periods.items():
        sub = df[(df.date >= a) & (df.date < str(int(b) + 1))]
        if len(sub) >= 2:
            out[name] = block(sub)
    return out, df


# ==========================================================================
# Part B: the strategy
# ==========================================================================

@dataclass(frozen=True)
class Config:
    haircut: float = HEADLINE_HAIRCUT
    hedge_every: int = 1
    hedge_vol: str = "current"
    costs: CostModel = BASE_COSTS
    q: float = DIVIDEND_YIELD


def run_trade(mkt: Market, s: TradeSpec, cfg: Config) -> dict:
    """Sell one ATM-forward straddle at VIX - haircut, delta hedge, settle at the expiry close."""
    e, x = s.entry, s.expiry
    S = mkt.spx[e:x + 1]
    times = (mkt.dates[e:x + 1] - mkt.dates[e]).days.to_numpy(float) / 365.0
    T = times[-1]
    sigma = mkt.vix[e:x + 1] - cfg.haircut                # that day's VIX minus haircut
    if np.any(sigma[:-1] <= 0.005):
        raise ValueError(f"haircut {cfg.haircut} leaves no vol on {mkt.dates[e].date()}")
    r = mkt.rate[e:x + 1]
    # ATM = forward: K = S0 exp((r - q) T). Call and put then have equal value (parity),
    # the straddle is ~delta neutral at entry, and it is how vol desks define ATM.
    K = float(S[0] * np.exp((r[0] - cfg.q) * T))
    res = delta_hedge_pnl(S, times, sigma, r, cfg.q, legs=straddle(K),
                          hedge_every=cfg.hedge_every, hedge_vol=cfg.hedge_vol, costs=cfg.costs)
    S0 = S[0]
    vega = float(res.entry_vega)
    rv_life = metrics.realised_vol(S)                     # ex post, attribution only
    return {
        "entry_date": mkt.dates[e].date(), "expiry_date": mkt.dates[x].date(),
        "calendar_days": int(round(T * 365)), "trading_days": int(x - e),
        "expiry_capped": s.capped,
        "S0": S0, "ST": S[-1], "K": K, "T": T, "r0": r[0], "vix0": mkt.vix[e],
        "sigma0": sigma[0], "rv21_entry": mkt.rv21[e], "signal": mkt.vix[e] - mkt.rv21[e],
        "premium": float(res.premium), "payoff_paid": float(-res.payoff),
        "hedge_pnl": float(res.hedge_pnl), "dividends": float(res.dividends),
        "financing": float(res.financing), "option_cost": float(res.option_cost),
        "hedge_cost": float(res.hedge_cost), "gross": float(res.gross), "net": float(res.total),
        "entry_vega": vega,
        "gross_bps": 1e4 * float(res.gross) / S0, "net_bps": 1e4 * float(res.total) / S0,
        "gross_volpts": 100 * float(res.gross) / vega, "net_volpts": 100 * float(res.total) / vega,
        "rv_life": rv_life, "iv_minus_rv_life_volpts": 100 * (sigma[0] - rv_life),
    }


def run_trades(mkt: Market, specs: list[TradeSpec], cfg: Config) -> pd.DataFrame:
    return pd.DataFrame([run_trade(mkt, s, cfg) for s in specs])


def summarize_variant(df: pd.DataFrame, traded: np.ndarray | None = None) -> dict:
    """Gross and net stats in bps of entry notional (+ vol points means)."""
    sel = df if traded is None else df[traded]
    if len(sel) < 2:
        return {"gross": {"n": int(len(sel))}, "net": {"n": int(len(sel))}}
    out = {}
    for kind in ("gross", "net"):
        st = metrics.summarize(sel[f"{kind}_bps"].to_numpy(), TRADES_PER_YEAR, seed=BOOT_SEED)
        worst = sel.loc[sel[f"{kind}_bps"].idxmin()]
        st["worst_entry_date"] = str(worst.entry_date)
        st["mean_volpts"] = float(sel[f"{kind}_volpts"].mean())
        st["median_volpts"] = float(sel[f"{kind}_volpts"].median())
        if traded is not None:
            # strategy-level monthly series: flat (0) in months the filter skips
            monthly = np.where(traded, df[f"{kind}_bps"].to_numpy(), 0.0)
            st["months"] = int(len(monthly))
            st["sharpe_ann_incl_flat_months"] = metrics.sharpe(monthly, TRADES_PER_YEAR)
            st["mean_per_month_incl_flat"] = float(monthly.mean())
            st["max_drawdown_incl_flat"] = metrics.max_drawdown(monthly)
        out[kind] = st
    return out


def break_even_haircut(mkt: Market, specs: list[TradeSpec], cfg: Config, kind: str) -> float:
    """Haircut h* (decimal vol) at which the mean per-trade P&L (bps) is zero."""
    def mean_pnl(h: float) -> float:
        c = Config(h, cfg.hedge_every, cfg.hedge_vol, cfg.costs, cfg.q)
        return float(run_trades(mkt, specs, c)[f"{kind}_bps"].mean())
    lo = -0.05
    hi = min(0.08, float(min(mkt.vix[s.entry:s.expiry].min() for s in specs)) - 0.006)
    f_lo, f_hi = mean_pnl(lo), mean_pnl(hi)
    if f_lo * f_hi > 0:
        return float("nan")
    return float(brentq(mean_pnl, lo, hi, xtol=1e-5))


def sensitivities(mkt: Market, specs: list[TradeSpec], base: Config) -> dict:
    """Mean/Sharpe (bps) when one choice at a time is moved away from ``base``."""
    def stats(cfg: Config) -> dict:
        d = run_trades(mkt, specs, cfg)
        return {k: {"mean_bps": float(d[f"{k}_bps"].mean()),
                    "mean_volpts": float(d[f"{k}_volpts"].mean()),
                    "sharpe_ann": metrics.sharpe(d[f"{k}_bps"].to_numpy(), TRADES_PER_YEAR),
                    "std_bps": float(d[f"{k}_bps"].std(ddof=1))} for k in ("gross", "net")}

    def mod(**kw) -> Config:
        d = dict(haircut=base.haircut, hedge_every=base.hedge_every, hedge_vol=base.hedge_vol,
                 costs=base.costs, q=base.q)
        d.update(kw)
        return Config(**d)

    return {
        "haircut": {f"{100 * h:.0f}": stats(mod(haircut=h)) for h in HAIRCUT_GRID},
        "break_even_haircut_volpts": {k: 100 * break_even_haircut(mkt, specs, base, k)
                                      for k in ("gross", "net")},
        "costs": {name: stats(mod(costs=base.costs.scaled(f)))
                  for name, f in (("zero", 0.0), ("base", 1.0), ("double", 2.0))},
        "hedge_every": {str(n): stats(mod(hedge_every=n)) for n in HEDGE_EVERY_GRID},
        "hedge_vol": {v: stats(mod(hedge_vol=v)) for v in ("current", "entry")},
        "dividend_yield": {f"{q:.3f}": stats(mod(q=q)) for q in Q_GRID},
    }


def signal_mask(df: pd.DataFrame, threshold: float | None) -> np.ndarray:
    """Sell only if VIX - RV21 at the entry close exceeds threshold (None = always sell)."""
    if df.signal.isna().any():
        raise ValueError("entry signal undefined (fewer than 21 prior returns) for some trade")
    if threshold is None:
        return np.ones(len(df), dtype=bool)
    return (df.signal > threshold).to_numpy()


# ----------------------------- in-sample ----------------------------------

def calibrate_in_sample(mkt: Market, is_specs: list[TradeSpec]) -> dict:
    """Make every free choice on IS trades only and write results/frozen_params.json.

    a priori (not tuned on P&L): headline haircut = 2 vol points (typical VIX - ATM 30d IV
        gap; tuning it on P&L would optimise a data-mismatch correction), costs = BASE_COSTS,
        hedge vol = that day's implied (what a market maker marks at), q = 1.8%.
    data-driven on IS: hedge frequency = max IS net Sharpe over {1, 2, 5} days (always-sell);
        entry threshold on VIX - RV21 = max IS net strategy Sharpe (flat months count as 0)
        over THRESHOLD_GRID, keeping >= MIN_IS_TRADES_CONDITIONAL trades.
    """
    for s in is_specs:   # hard guard: nothing in calibration may start after IS_END
        assert mkt.dates[s.entry] <= IS_END
    cfg0 = Config()
    freq_sharpe, freq_net = {}, {}
    for n in HEDGE_EVERY_GRID:
        d = run_trades(mkt, is_specs, Config(hedge_every=n))
        freq_net[n] = d.net_bps.to_numpy()
        freq_sharpe[n] = metrics.sharpe(freq_net[n], TRADES_PER_YEAR)
    hedge_every = max(freq_sharpe, key=freq_sharpe.get)
    # diagnostic: is the chosen frequency's edge over daily distinguishable from noise?
    paired_t = {str(n): metrics.t_stat(freq_net[n] - freq_net[1]) for n in HEDGE_EVERY_GRID if n != 1}
    cfg = Config(cfg0.haircut, hedge_every, cfg0.hedge_vol, cfg0.costs, cfg0.q)

    d = run_trades(mkt, is_specs, cfg)
    thr_table = {}
    for thr in THRESHOLD_GRID:
        m = signal_mask(d, thr)
        monthly = np.where(m, d.net_bps.to_numpy(), 0.0)
        thr_table[thr] = {"n_trades": int(m.sum()),
                          "sharpe_incl_flat": metrics.sharpe(monthly, TRADES_PER_YEAR),
                          "mean_per_month_bps": float(monthly.mean())}
    eligible = {t: v for t, v in thr_table.items() if v["n_trades"] >= MIN_IS_TRADES_CONDITIONAL}
    threshold = max(eligible, key=lambda t: eligible[t]["sharpe_incl_flat"])

    frozen = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "Chosen on in-sample entries only; written before any out-of-sample trade is run.",
        "in_sample": [str(IS_START.date()), str(IS_END.date())],
        "out_of_sample_start": str(OOS_START.date()),
        "iv_haircut": cfg.haircut,
        "hedge_every": cfg.hedge_every,
        "hedge_vol": cfg.hedge_vol,
        "option_half_spread_vol": cfg.costs.option_half_spread_vol,
        "hedge_cost_bps": cfg.costs.hedge_cost_bps,
        "dividend_yield": cfg.q,
        "strike_rule": "ATM forward K = S0 exp((r - q) T)",
        "tenor": "nearest trading day to entry + 30 calendar days, capped at next entry",
        "entry_rule_threshold": threshold,
        "entry_rule": "sell only if VIX_close - RV21_close > threshold (RV21: 21 returns ending at entry)",
        "selection": {
            "hedge_every_is_net_sharpe": {str(k): v for k, v in freq_sharpe.items()},
            "hedge_every_paired_t_vs_daily_net_bps": paired_t,
            "threshold_is_table": {f"{k:.2f}": v for k, v in thr_table.items()},
            "a_priori": ["iv_haircut", "costs", "hedge_vol", "dividend_yield"],
        },
    }
    RESULTS.mkdir(exist_ok=True)
    FROZEN_PATH.write_text(json.dumps(frozen, indent=2))
    return frozen


def config_from_frozen(fz: dict) -> Config:
    return Config(haircut=fz["iv_haircut"], hedge_every=fz["hedge_every"],
                  hedge_vol=fz["hedge_vol"],
                  costs=CostModel(fz["option_half_spread_vol"], fz["hedge_cost_bps"]),
                  q=fz["dividend_yield"])


def evaluate_period(mkt: Market, specs: list[TradeSpec], fz: dict, label: str) -> tuple[dict, pd.DataFrame]:
    """Headline results for one period with the frozen parameters."""
    cfg = config_from_frozen(fz)
    d = run_trades(mkt, specs, cfg)
    d.insert(0, "period", label)
    d["traded_conditional"] = signal_mask(d, fz["entry_rule_threshold"])
    corr = float(np.corrcoef(d.iv_minus_rv_life_volpts, d.gross_volpts)[0, 1])
    slope = float(np.polyfit(d.iv_minus_rv_life_volpts, d.gross_volpts, 1)[0])
    out = {
        "n_months": int(len(d)),
        "first_entry": str(d.entry_date.iloc[0]), "last_entry": str(d.entry_date.iloc[-1]),
        "n_expiry_capped": int(d.expiry_capped.sum()),
        "always_sell": summarize_variant(d),
        "conditional": summarize_variant(d, d.traded_conditional.to_numpy()),
        "attribution": {"corr_gross_volpts_vs_iv_minus_rv": corr,
                        "slope_gross_volpts_on_iv_minus_rv": slope},
        "worst5_net": d.nsmallest(5, "net_bps")[["entry_date", "net_bps", "net_volpts", "vix0",
                                                 "rv_life"]].astype(str).to_dict("records"),
        "mean_components_bps": {c: float((1e4 * d[c] / d.S0).mean()) for c in
                                ("premium", "payoff_paid", "hedge_pnl", "dividends",
                                 "financing", "option_cost", "hedge_cost")},
    }
    return out, d


# ----------------------------- out-of-sample ------------------------------

def evaluate_out_of_sample(mkt: Market, oos_specs: list[TradeSpec],
                           frozen_path: Path = FROZEN_PATH) -> tuple[dict, pd.DataFrame, dict]:
    """The single OOS run: reads ONLY the frozen parameter file, makes no choices."""
    if not frozen_path.exists():
        raise RuntimeError("frozen_params.json missing: run the in-sample calibration first")
    fz = json.loads(frozen_path.read_text())
    summary, trades = evaluate_period(mkt, oos_specs, fz, "OOS")
    sens = sensitivities(mkt, oos_specs, config_from_frozen(fz))   # reported, never used to choose
    return summary, trades, sens


# ==========================================================================
# post-hoc diagnostic: how far is VIX above ATM implied vol on the one date we have quotes?
# ==========================================================================

def snapshot_vix_atm_gap(mkt: Market) -> dict:
    """VIX minus 30-day ATM implied vol from the saved SPY chain snapshot (2026-11-13 close).

    POST-HOC and NOT USED FOR ANY CHOICE (it is one date, inside the OOS period). SPY options
    are American and pay discrete dividends, so call and put IVs disagree; their average
    cancels a mis-specified forward to first order. A local brentq inversion is used on
    purpose (volkit/iv is out of scope for this module).
    """
    from volkit.pricing.black_scholes import bs_price
    path = ROOT / "data" / "raw" / "spy_chain_20261115.csv"
    if not path.exists():
        return {"available": False}
    ch = pd.read_csv(path)
    quote_date = mkt.dates[-1]
    S = float(ch.spot.iloc[0])
    ch["days"] = (pd.to_datetime(ch.expiry) - quote_date).dt.days
    ch = ch[(ch.bid > 0) & (ch.ask > ch.bid)]
    r, q = float(mkt.rate[-1]), 0.012           # SPY yield ~1.2%: assumption for this check only
    rows = []
    for days in sorted(ch.days.unique()):
        if not 20 <= days <= 45:
            continue
        T = days / 365.0
        ivs = []
        for kind in ("call", "put"):
            sub = ch[(ch.days == days) & (ch.kind == kind)]
            row = sub.iloc[int(np.argmin(np.abs(sub.strike - S)))]
            mid = 0.5 * (row.bid + row.ask)
            ivs.append(brentq(lambda v: bs_price(S, row.strike, T, r, v, q, kind) - mid, 1e-4, 3.0))
        rows.append((days, float(np.mean(ivs)), ivs[0], ivs[1]))
    d = np.array([x[0] for x in rows], float)
    iv = np.array([x[1] for x in rows])
    atm30 = float(np.interp(30.0, d, iv))
    return {"available": True, "quote_date": str(quote_date.date()), "vix": float(mkt.vix[-1]),
            "atm_iv_30d_interp": atm30, "gap_volpts": 100 * (float(mkt.vix[-1]) - atm30),
            "by_expiry_days_avg_call_put": {str(int(a)): b for a, b, _, _ in rows}}


# ==========================================================================
# figures
# ==========================================================================

def _shade_oos(ax, end: pd.Timestamp) -> None:
    """Grey band over the out-of-sample period (only if the plot reaches it)."""
    if end > OOS_START:
        ax.axvspan(OOS_START, end, color="0.85", alpha=0.6, lw=0, label="out-of-sample (2019-)")
    ax.set_xlim(pd.Timestamp("2003-10-01"), end + pd.Timedelta(days=90))


def fig_iv_vs_rv(mkt: Market, path: Path, until: pd.Timestamp | None = None) -> None:
    if until is not None:
        cut = int(np.searchsorted(mkt.dates, until + pd.Timedelta(days=TENOR_DAYS + 5)))
        mkt = Market(mkt.dates[:cut], mkt.spx[:cut], mkt.vix[:cut], mkt.rate[:cut], mkt.rv21[:cut])
    n = len(mkt.dates)
    rv = np.full(n, np.nan)
    last_ok = mkt.dates[-1] - pd.Timedelta(days=TENOR_DAYS)
    for e in range(n - 1):
        if mkt.dates[e] <= last_ok:
            rv[e] = forward_rv(mkt, e)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    _shade_oos(ax, mkt.dates[-1])
    ax.plot(mkt.dates, 100 * mkt.vix, lw=0.8, color="tab:red", label="VIX close (implied)")
    ax.plot(mkt.dates, 100 * rv, lw=0.8, color="tab:blue", alpha=0.8,
            label="SPX realised vol over the NEXT 30 calendar days")
    ax.set_yscale("log")
    ax.set_yticks([5, 10, 20, 40, 80, 120])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylabel("annualised volatility (vol points, log scale)")
    ax.set_xlabel("date (implied vol and the realised vol that followed, both plotted at the start date)")
    ax.set_title("S&P 500: VIX vs subsequently realised volatility")
    ax.legend(loc="lower left", fontsize=8, ncol=3, framealpha=0.9)
    ax.set_ylim(3, 130)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_vrp_hist(a_df: pd.DataFrame, path: Path) -> None:
    x = a_df.iv_minus_rv_volpts
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(x, bins=np.arange(np.floor(x.min()) - 1, np.ceil(x.max()) + 2, 1.0),
            color="tab:gray", edgecolor="white")
    ax.axvline(x.mean(), color="tab:red", lw=1.5, label=f"mean {x.mean():.2f}")
    ax.axvline(x.median(), color="tab:blue", lw=1.5, ls="--", label=f"median {x.median():.2f}")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("VIX at month start minus realised vol over next 30 days (vol points)")
    ax.set_ylabel("number of months")
    ax.set_title(f"Implied minus subsequent realised vol, monthly, n = {len(x)}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_cumulative(trades: pd.DataFrame, path: Path) -> None:
    d = trades.sort_values("entry_date")
    dates = pd.to_datetime(d.expiry_date)
    cond_net = np.where(d.traded_conditional, d.net_bps, 0.0)
    fig, ax = plt.subplots(figsize=(11, 5))
    _shade_oos(ax, dates.max())
    ax.plot(dates, d.gross_bps.cumsum(), color="tab:green", lw=1.2, label="always sell, gross")
    ax.plot(dates, d.net_bps.cumsum(), color="tab:blue", lw=1.4, label="always sell, net of costs")
    ax.plot(dates, np.cumsum(cond_net), color="tab:orange", lw=1.2,
            label="conditional entry, net of costs")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("settlement date")
    ax.set_ylabel("cumulative P&L (bp of entry notional, 1 straddle per month)")
    ax.set_title("Delta-hedged short 30-day ATM straddle on synthetic VIX-based prices")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_scatter(trades: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for label, color in (("IS", "tab:blue"), ("OOS", "tab:red")):
        d = trades[trades.period == label]
        if d.empty:
            continue
        c = np.corrcoef(d.iv_minus_rv_life_volpts, d.gross_volpts)[0, 1]
        ax.scatter(d.iv_minus_rv_life_volpts, d.gross_volpts, s=12, alpha=0.6, color=color,
                   label=f"{label} (n={len(d)}, corr={c:.2f})")
    lim = [min(trades.iv_minus_rv_life_volpts.min(), trades.gross_volpts.min()) - 2,
           max(trades.iv_minus_rv_life_volpts.max(), trades.gross_volpts.max()) + 2]
    ax.plot(lim, lim, color="k", lw=0.8, ls="--", label="y = x")
    ax.axhline(0, color="0.5", lw=0.5)
    ax.axvline(0, color="0.5", lw=0.5)
    ax.set_xlabel("pricing vol at entry minus realised vol over trade life (vol points)")
    ax.set_ylabel("trade P&L gross of costs (vol points = P&L / entry vega)")
    ax.set_title("P&L attribution: hedged short straddle vs implied - realised")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_haircut(sens: dict[str, dict], path: Path, snapshot_gap: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if snapshot_gap is not None and np.isfinite(snapshot_gap):
        ax.axvline(snapshot_gap, color="0.4", ls="--", lw=1,
                   label=f"VIX - ATM IV on 2026-11-13 snapshot ({snapshot_gap:.1f}, one date)")
    colors = {"IS": "tab:blue", "OOS": "tab:red"}
    for label, s in sens.items():
        hs = [float(h) for h in s["haircut"]]
        for kind, ls in (("gross", ":"), ("net", "-")):
            ax.plot(hs, [s["haircut"][h][kind]["mean_bps"] for h in s["haircut"]], ls=ls,
                    marker="o", color=colors[label], label=f"{label} {kind}")
        be = s["break_even_haircut_volpts"]["net"]
        if np.isfinite(be):
            ax.axvline(be, color=colors[label], lw=0.8, alpha=0.6)
            ax.annotate(f"{label} net break-even\n{be:.2f} vol pts", (be, 0),
                        textcoords="offset points", xytext=(4, 18 if label == "IS" else -38),
                        fontsize=8, color=colors[label],
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("iv_haircut: vol points subtracted from VIX before pricing and hedging")
    ax.set_ylabel("mean P&L per trade (bp of entry notional)")
    ax.set_title("Sensitivity to the VIX -> ATM implied vol haircut")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


# ==========================================================================
# main
# ==========================================================================

def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main(stage: str) -> None:
    require_cached(SPX_FILE, VIX_FILE, RATES_FILE)   # clear error if data/raw is empty
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    mkt = load_market()
    specs = build_trade_calendar(mkt.dates)
    specs = [s for s in specs if mkt.dates[s.entry] >= IS_START]
    is_specs, oos_specs = split_specs(specs, mkt.dates)

    # Part A is descriptive and uses no free parameter, but in the in-sample stage it is
    # still restricted to IS dates so that nothing after 2018 is looked at before freezing.
    a_specs = specs if stage == "all" else is_specs
    a_summary, a_df = part_a(mkt, a_specs)
    fig_iv_vs_rv(mkt, FIGURES / "iv_vs_rv.png", None if stage == "all" else IS_END)
    fig_vrp_hist(a_df, FIGURES / "vrp_hist.png")

    # ---- in-sample: choose, freeze ----
    frozen = calibrate_in_sample(mkt, is_specs)
    is_summary, is_trades = evaluate_period(mkt, is_specs, frozen, "IS")
    is_sens = sensitivities(mkt, is_specs, config_from_frozen(frozen))
    summary = {"data_limitation": "Synthetic option prices: Black-Scholes at VIX - iv_haircut; "
                                  "VIX > ATM IV (skew); costs assumed.",
               "units": "P&L in bp of entry spot notional per straddle unless *_volpts "
                        "(P&L / entry vega); Sharpe annualised sqrt(12) from monthly trades.",
               "part_a": a_summary, "frozen_params": frozen,
               "in_sample": is_summary, "sensitivities": {"IS": is_sens}}
    trades = is_trades
    if stage == "all":
        # ---- out-of-sample: exactly one run with the frozen file ----
        oos_summary, oos_trades, oos_sens = evaluate_out_of_sample(mkt, oos_specs)
        summary["out_of_sample"] = oos_summary
        summary["sensitivities"]["OOS (reported only, not used for any choice)"] = oos_sens
        trades = pd.concat([is_trades, oos_trades], ignore_index=True)
        sens_fig = {"IS": is_sens, "OOS": oos_sens}
        snap = snapshot_vix_atm_gap(mkt)
        summary["post_hoc_vix_minus_atm_iv_snapshot"] = snap
    else:
        sens_fig = {"IS": is_sens}
        snap = {"available": False}
    trades.to_csv(RESULTS / "backtest_trades.csv", index=False)
    (RESULTS / "backtest_summary.json").write_text(json.dumps(_jsonable(summary), indent=2))
    fig_cumulative(trades, FIGURES / "cumulative_pnl.png")
    fig_scatter(trades, FIGURES / "pnl_vs_vrp_scatter.png")
    fig_haircut(sens_fig, FIGURES / "haircut_sensitivity.png",
                snap.get("gap_volpts") if snap.get("available") else None)
    print(json.dumps(_jsonable(summary), indent=1)[:200])
    print("wrote", RESULTS / "backtest_summary.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stage", choices=["in-sample", "all"], default="all")
    main(p.parse_args().stage)
