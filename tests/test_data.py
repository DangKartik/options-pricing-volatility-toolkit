"""Offline tests for volkit.data (no network: synthetic frames and temp CSVs)."""

import numpy as np
import pandas as pd
import pytest

from volkit.data import chain, market

SNAP = pd.Timestamp("2026-01-05 15:00:00", tz="UTC")  # Monday 10:00 New York (EST)
SPOT = 100.0


def make_chain(rows):
    """rows: list of dicts overriding a valid default contract."""
    base = {
        "ticker": "TEST", "snapshot_time": SNAP, "spot": SPOT, "market_state": "REGULAR",
        "contract_symbol": "X", "expiry": pd.Timestamp("2026-02-04"), "kind": "call",
        "strike": 100.0, "bid": 2.0, "ask": 2.2, "last": 2.1, "volume": 10.0,
        "open_interest": 100.0, "last_trade_time": SNAP, "yf_iv": 0.2,
    }
    return pd.DataFrame([{**base, **r} for r in rows])[chain.CHAIN_COLUMNS]


# ---------------------------------------------------------------- clean_chain

def test_derived_columns():
    out = chain.clean_chain(make_chain([{"strike": 110.0, "bid": 1.0, "ask": 1.2}]), SPOT)
    row = out.iloc[0]
    assert row["mid"] == pytest.approx(1.1)
    assert row["rel_spread"] == pytest.approx(0.2 / 1.1)
    assert row["log_moneyness"] == pytest.approx(np.log(1.1))
    # expiry 2026-02-04 16:00 EST = 21:00 UTC; snapshot 2026-01-05 15:00 UTC -> 30 days + 6h
    assert row["dte"] == pytest.approx(30.25)
    assert row["T"] == pytest.approx(30.25 / 365.0)


def test_time_to_expiry_uses_new_york_dst():
    # July: 16:00 EDT = 20:00 UTC
    T = chain.time_to_expiry(pd.Series([pd.Timestamp("2026-07-10")]),
                             pd.Series([pd.Timestamp("2026-07-09 20:00", tz="UTC")]))
    assert T.iloc[0] == pytest.approx(1.0 / 365.0)


def test_each_filter_removes_its_row_and_counts_it():
    df = make_chain([
        {},                                          # good
        {"bid": np.nan},                             # missing_quote
        {"bid": 0.0, "ask": 0.0},                    # low_bid (zero quote, closed market)
        {"bid": 0.01, "ask": 0.02},                  # low_bid (below min_bid)
        {"bid": 2.0, "ask": 1.9},                    # crossed
        {"bid": 1.0, "ask": 2.0},                    # wide_spread (0.667)
        {"expiry": pd.Timestamp("2026-01-07")},      # dte_too_short (~2.25 d)
        {"expiry": pd.Timestamp("2028-01-21")},      # dte_too_long
        {"strike": 200.0},                           # moneyness (ln 2)
        {"kind": "put", "strike": 90.0},             # good
    ])
    out = chain.clean_chain(df, SPOT)
    counts = out.attrs["filter_counts"]
    assert counts == {
        "input": 10, "missing_quote": 1, "low_bid": 2, "crossed": 1, "wide_spread": 1,
        "dte_too_short": 1, "dte_too_long": 1, "moneyness": 1, "output": 2,
    }
    assert len(out) == 2
    assert sorted(out["strike"]) == [90.0, 100.0]
    assert out[["mid", "T", "log_moneyness"]].notna().all().all()
    report = chain.filter_report(out)
    assert report["removed"].sum() == 8
    assert report["remaining"].iloc[-1] == 2


def test_row_failing_several_filters_counted_once_under_first():
    df = make_chain([{"bid": 0.0, "ask": 0.0, "strike": 500.0,
                      "expiry": pd.Timestamp("2026-01-06")}])
    counts = chain.clean_chain(df, SPOT).attrs["filter_counts"]
    assert counts["low_bid"] == 1
    assert counts["dte_too_short"] == 0 and counts["moneyness"] == 0
    assert counts["output"] == 0


def test_filter_parameters_are_respected():
    df = make_chain([{"bid": 1.0, "ask": 2.0}, {"strike": 200.0},
                     {"expiry": pd.Timestamp("2026-01-07")}])
    out = chain.clean_chain(df, SPOT, max_rel_spread=1.0, min_dte=1.0,
                            moneyness_range=(-1.0, 1.0))
    assert len(out) == 3


def test_all_zero_bids_gives_empty_frame_not_error():
    out = chain.clean_chain(make_chain([{"bid": 0.0, "ask": 0.0}] * 3), SPOT)
    assert out.empty and out.attrs["filter_counts"]["low_bid"] == 3


def test_clean_chain_does_not_mutate_input():
    df = make_chain([{}])
    before = df.copy()
    chain.clean_chain(df, SPOT)
    pd.testing.assert_frame_equal(df, before)


@pytest.mark.parametrize("kwargs", [
    {"spot": 0.0}, {"spot": float("nan")}, {"spot": SPOT, "min_dte": 10, "max_dte": 5},
    {"spot": SPOT, "moneyness_range": (0.1, -0.1)}, {"spot": SPOT, "max_rel_spread": 0.0},
])
def test_invalid_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        chain.clean_chain(make_chain([{}]), **kwargs)


def test_invalid_kind_and_missing_column_raise():
    with pytest.raises(ValueError):
        chain.clean_chain(make_chain([{"kind": "C"}]), SPOT)
    with pytest.raises(ValueError):
        chain.clean_chain(make_chain([{}]).drop(columns="bid"), SPOT)


def test_save_load_roundtrip(tmp_path):
    df = make_chain([{}, {"kind": "put", "strike": 95.0, "last_trade_time": pd.NaT}])
    path = chain.save_chain(df, tmp_path / "sub" / "chain.csv")
    back = chain.load_chain(path)
    assert list(back.columns) == chain.CHAIN_COLUMNS
    assert back["snapshot_time"].iloc[0] == SNAP
    assert back["expiry"].iloc[0] == pd.Timestamp("2026-02-04")
    assert back["kind"].tolist() == ["call", "put"]
    a = chain.clean_chain(df, SPOT)
    b = chain.clean_chain(back, SPOT)
    np.testing.assert_allclose(a["T"], b["T"])
    np.testing.assert_allclose(a["mid"], b["mid"])


# -------------------------------------------------------------------- loaders

def write_csv(path, header, rows):
    path.write_text(header + "\n" + "\n".join(rows) + "\n")


def check_series(s, n):
    assert isinstance(s, pd.Series)
    assert isinstance(s.index, pd.DatetimeIndex)
    assert len(s) == n
    assert s.index.is_monotonic_increasing and s.index.is_unique
    assert not s.isna().any()
    assert s.dtype == float


def test_load_spx_from_temp_cache(tmp_path):
    # unsorted, with a duplicated date: loader must sort and keep the last duplicate
    write_csv(tmp_path / market.SPX_FILE, "date,gspc_close",
              ["2024-01-03,4700.0", "2024-01-02,4740.0", "2024-01-04,4690.0", "2024-01-04,4688.5"])
    s = market.load_spx(cache_dir=tmp_path)
    check_series(s, 3)
    assert s.loc["2024-01-04"] == 4688.5
    assert s.name == "spx"


def test_load_vix_returns_decimal(tmp_path):
    write_csv(tmp_path / market.VIX_FILE, "date,cboe_vix_close",
              ["2024-01-02,18.5", "2024-01-03,20.0"])
    s = market.load_vix(cache_dir=tmp_path)
    check_series(s, 2)
    assert s.iloc[0] == pytest.approx(0.185)
    assert s.iloc[1] == pytest.approx(0.20)


def test_load_rates_converts_discount_to_continuous_and_drops_gaps(tmp_path):
    write_csv(tmp_path / market.RATES_FILE, "date,fred_dtb3_discount_pct",
              ["2024-01-02,5.20", "2024-01-03,", "2024-01-04,5.22"])
    s = market.load_rates(cache_dir=tmp_path)
    check_series(s, 2)
    expected = -np.log(1 - 0.052 * 91 / 360) * 365 / 91
    assert s.iloc[0] == pytest.approx(expected)
    assert 0.052 < s.iloc[0] < 0.056  # cc ACT/365 sits above the discount quote


def test_discount_to_continuous_scalar_and_bounds():
    assert market.discount_to_continuous(0.0) == 0.0
    # reprices the bill: exp(-r * 91/365) equals the discount price
    r = market.discount_to_continuous(4.0)
    assert np.exp(-r * 91 / 365) == pytest.approx(1 - 0.04 * 91 / 360)
    with pytest.raises(ValueError):
        market.discount_to_continuous(500.0)


def test_cache_dir_from_env(tmp_path, monkeypatch):
    write_csv(tmp_path / market.SPX_FILE, "date,gspc_close", ["2024-01-02,4740.0"])
    monkeypatch.setenv(market.ENV_CACHE_DIR, str(tmp_path))
    assert market.cache_dir_path() == tmp_path
    check_series(market.load_spx(), 1)
    monkeypatch.delenv(market.ENV_CACHE_DIR)
    assert market.cache_dir_path() == market.REPO_ROOT / "data" / "raw"


def test_missing_cache_triggers_download_and_writes_cache(tmp_path, monkeypatch):
    calls = []

    def fake_yf_close(ticker, start=market.START_DATE):
        calls.append(ticker)
        idx = pd.to_datetime(["2024-01-03", "2024-01-02"])
        return pd.Series([2.0, 1.0], index=idx)

    monkeypatch.setattr(market, "_yf_close", fake_yf_close)
    s = market.load_spx(cache_dir=tmp_path)
    assert calls == ["^GSPC"] and (tmp_path / market.SPX_FILE).exists()
    assert s.tolist() == [1.0, 2.0]
    market.load_spx(cache_dir=tmp_path)            # cache hit: no new download
    assert calls == ["^GSPC"]
    market.load_spx(refresh=True, cache_dir=tmp_path)
    assert calls == ["^GSPC", "^GSPC"]


def test_vix_falls_back_to_yfinance_when_cboe_fails(tmp_path, monkeypatch):
    def boom(name):
        raise OSError("blocked")

    monkeypatch.setattr(market, "_cboe_close", boom)
    monkeypatch.setattr(market, "_yf_close", lambda t, start=market.START_DATE: pd.Series(
        [15.0], index=pd.to_datetime(["2024-01-02"])))
    s = market.load_vix(cache_dir=tmp_path)
    assert s.iloc[0] == pytest.approx(0.15)
    header = (tmp_path / market.VIX_FILE).read_text().splitlines()[0]
    assert header == "date,yf_vix_close"


def test_all_sources_failing_raises(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise OSError("blocked")

    monkeypatch.setattr(market, "_cboe_close", boom)
    monkeypatch.setattr(market, "_yf_close", boom)
    with pytest.raises(RuntimeError):
        market.load_vix(cache_dir=tmp_path)
    assert not (tmp_path / market.VIX_FILE).exists()


def test_load_vix_family_shape(tmp_path):
    write_csv(tmp_path / market.VIX_FAMILY_FILE, "date,vix9d,vix,vix3m",
              ["2009-09-18,,25.0,26.5", "2011-01-04,16.0,17.0,19.0"])
    fam = market.load_vix_family(cache_dir=tmp_path)
    assert list(fam.columns) == ["vix9d", "vix", "vix3m"]
    assert np.isnan(fam["vix9d"].iloc[0])
    assert fam["vix3m"].iloc[1] == pytest.approx(0.19)


def test_align_to_calendar_forward_fills_holidays():
    rates = pd.Series([0.01, 0.02], index=pd.to_datetime(["2024-01-02", "2024-01-04"]))
    cal = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    out = market.align_to_calendar(rates, cal)
    assert out.tolist() == [0.01, 0.01, 0.02, 0.02]
    assert out.index.equals(pd.DatetimeIndex(cal))
    with pytest.raises(ValueError):
        market.align_to_calendar(rates, pd.to_datetime(["2023-12-29"]))
