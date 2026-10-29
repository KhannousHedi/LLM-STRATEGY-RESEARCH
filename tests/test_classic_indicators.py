import math
import re
import statistics

import pandas as pd
import pytest

from src.features import build_features, load_feature_set

CLASSIC_COLUMNS = (
    "r1",
    "r5",
    "r20",
    "ma50_gap",
    "ma200_gap",
    "high52_gap",
    "rsi14",
    "vol20",
    "volume_ratio",
    "open_gap",
    "day_range",
    "close_location",
)

WINDOWED = {
    "r1": "r1_window",
    "r5": "r5_window",
    "r20": "r20_window",
    "ma50_gap": "ma50_window",
    "ma200_gap": "ma200_window",
    "high52_gap": "high52_window",
    "rsi14": "rsi14_window",
    "vol20": "vol20_window",
    "volume_ratio": "volume_window",
}

_SESSION_COUNT = re.compile(r"\b\d+\s+sessions?\b|\b\d+-session\b")


def _bars(close, high=None, low=None, open_=None, volume=None):
    count = len(close)
    index = pd.date_range("2020-01-01", periods=count, freq="D", tz="UTC")
    if high is None:
        high = list(close)
    if low is None:
        low = list(close)
    if open_ is None:
        open_ = list(close)
    if volume is None:
        volume = [1.0] * count
    return pd.DataFrame(
        {
            "open": list(open_),
            "high": list(high),
            "low": list(low),
            "close": list(close),
            "volume": list(volume),
        },
        index=index,
    )


def _classic(bars, **windows):
    return load_feature_set("classic_indicators").compute(bars, **windows)


def test_classic_indicators_columns_and_descriptions():
    feature_set = load_feature_set("classic_indicators")
    assert feature_set.columns == CLASSIC_COLUMNS
    assert feature_set.descriptions == {
        "r1": "close-to-close return over the r1_window sessions",
        "r5": "close-to-close return over the r5_window sessions",
        "r20": "close-to-close return over the r20_window sessions",
        "ma50_gap": "close divided by its simple moving average of ma50_window sessions, minus 1",
        "ma200_gap": "close divided by its simple moving average of ma200_window sessions, minus 1",
        "high52_gap": "close divided by the highest high of the last high52_window sessions, minus 1",
        "rsi14": "Wilder's RSI over rsi14_window sessions, from 0 to 100",
        "vol20": "sample standard deviation of vol20_window one-session returns, annualized with the square root of 252",
        "volume_ratio": "volume divided by the mean volume of the last volume_window sessions",
        "open_gap": "open divided by the previous close, minus 1",
        "day_range": "high minus low, divided by the close",
        "close_location": "where the close sits between the low and the high; 0.5 when high equals low",
    }
    for column, parameter in WINDOWED.items():
        text = feature_set.descriptions[column]
        assert parameter in text
        assert _SESSION_COUNT.search(text) is None
    for column in ("open_gap", "day_range", "close_location"):
        assert "_window" not in feature_set.descriptions[column]


def test_classic_indicator_returns():
    frame = _classic(_bars([100.0, 110.0, 90.0, 99.0]), r1_window=1, r5_window=2, r20_window=3)
    assert math.isnan(frame["r1"].iloc[0])
    assert frame["r1"].iloc[1] == pytest.approx(0.1)
    assert frame["r1"].iloc[3] == pytest.approx(0.1)
    assert math.isnan(frame["r5"].iloc[1])
    assert frame["r5"].iloc[2] == pytest.approx(-0.1)
    assert frame["r5"].iloc[3] == pytest.approx(-0.1)
    assert math.isnan(frame["r20"].iloc[2])
    assert frame["r20"].iloc[3] == pytest.approx(-0.01)


def test_moving_average_gaps():
    frame = _classic(_bars([10.0, 30.0, 20.0]), ma50_window=2, ma200_window=3)
    assert math.isnan(frame["ma50_gap"].iloc[0])
    assert math.isnan(frame["ma200_gap"].iloc[1])
    assert frame["ma50_gap"].iloc[1] == pytest.approx(0.5)
    assert frame["ma50_gap"].iloc[2] == pytest.approx(-0.2)
    assert frame["ma200_gap"].iloc[2] == pytest.approx(0.0)
    assert list(frame.columns) == list(CLASSIC_COLUMNS)


def test_high52_gap_is_zero_or_below():
    frame = _classic(_bars([10.0, 15.0, 12.0], high=[10.0, 15.0, 12.0]), high52_window=2)
    assert math.isnan(frame["high52_gap"].iloc[0])
    assert frame["high52_gap"].iloc[1] == pytest.approx(0.0)
    assert frame["high52_gap"].iloc[2] == pytest.approx(-0.2)


def test_wilder_rsi_with_a_window_of_three():
    gains = _classic(_bars([10.0, 11.0, 12.0, 13.0]), rsi14_window=3)
    assert math.isnan(gains["rsi14"].iloc[2])
    assert gains["rsi14"].iloc[3] == pytest.approx(100.0)

    losses = _classic(_bars([13.0, 12.0, 11.0, 10.0]), rsi14_window=3)
    assert losses["rsi14"].iloc[3] == pytest.approx(0.0)

    flat = _classic(_bars([10.0, 10.0, 10.0, 10.0]), rsi14_window=3)
    assert flat["rsi14"].iloc[3] == pytest.approx(50.0)

    step = _classic(_bars([10.0, 11.0, 12.0, 13.0, 12.0]), rsi14_window=3)
    assert step["rsi14"].iloc[3] == pytest.approx(100.0)
    assert step["rsi14"].iloc[4] == pytest.approx(200 / 3)


def test_vol20_matches_sample_standard_deviation():
    close = [100.0, 110.0, 121.0, 100.0]
    frame = _classic(_bars(close), vol20_window=3)
    returns = [110.0 / 100.0 - 1.0, 121.0 / 110.0 - 1.0, 100.0 / 121.0 - 1.0]
    assert math.isnan(frame["vol20"].iloc[2])
    assert frame["vol20"].iloc[3] == pytest.approx(statistics.stdev(returns) * math.sqrt(252))


def test_volume_ratio_open_gap_and_day_range():
    frame = _classic(
        _bars(
            [100.0, 100.0, 110.0],
            high=[110.0, 105.0, 120.0],
            low=[90.0, 95.0, 100.0],
            open_=[100.0, 105.0, 90.0],
            volume=[10.0, 30.0, 60.0],
        ),
        volume_window=2,
    )
    assert math.isnan(frame["volume_ratio"].iloc[0])
    assert frame["volume_ratio"].iloc[1] == pytest.approx(1.5)
    assert frame["volume_ratio"].iloc[2] == pytest.approx(4 / 3)
    assert math.isnan(frame["open_gap"].iloc[0])
    assert frame["open_gap"].iloc[1] == pytest.approx(0.05)
    assert frame["open_gap"].iloc[2] == pytest.approx(-0.1)
    assert frame["day_range"].iloc[0] == pytest.approx(0.2)
    assert frame["day_range"].iloc[1] == pytest.approx(0.1)
    assert frame["day_range"].iloc[2] == pytest.approx(2 / 11)


def test_close_location_is_half_when_high_equals_low():
    frame = _classic(
        _bars(
            [100.0, 108.0, 100.0],
            high=[110.0, 110.0, 100.0],
            low=[90.0, 100.0, 100.0],
        )
    )
    assert frame["close_location"].iloc[0] == pytest.approx(0.5)
    assert frame["close_location"].iloc[1] == pytest.approx(0.8)
    assert frame["close_location"].iloc[2] == pytest.approx(0.5)


def test_early_rows_stay_nan_until_the_default_window_is_full():
    frame = _classic(_bars([float(i + 1) for i in range(252)]))
    assert math.isnan(frame["high52_gap"].iloc[250])
    assert frame["high52_gap"].iloc[251] == pytest.approx(0.0)
    assert math.isnan(frame["r1"].iloc[0]) and pd.notna(frame["r1"].iloc[1])
    assert math.isnan(frame["r5"].iloc[4]) and pd.notna(frame["r5"].iloc[5])
    assert math.isnan(frame["r20"].iloc[19]) and pd.notna(frame["r20"].iloc[20])
    assert math.isnan(frame["ma50_gap"].iloc[48]) and pd.notna(frame["ma50_gap"].iloc[49])
    assert math.isnan(frame["ma200_gap"].iloc[198]) and pd.notna(frame["ma200_gap"].iloc[199])
    assert math.isnan(frame["rsi14"].iloc[13]) and pd.notna(frame["rsi14"].iloc[14])
    assert math.isnan(frame["vol20"].iloc[19]) and pd.notna(frame["vol20"].iloc[20])
    assert math.isnan(frame["volume_ratio"].iloc[18]) and pd.notna(frame["volume_ratio"].iloc[19])
    assert math.isnan(frame["open_gap"].iloc[0]) and pd.notna(frame["open_gap"].iloc[1])
    assert pd.notna(frame["day_range"].iloc[0])
    assert frame["close_location"].iloc[0] == pytest.approx(0.5)


def test_classic_indicator_window_checks():
    bars = _bars([1.0, 2.0, 3.0])
    with pytest.raises(TypeError):
        _classic(bars, nope_window=3)
    with pytest.raises(ValueError, match="r1_window"):
        _classic(bars, r1_window=0)
    with pytest.raises(ValueError, match="vol20_window"):
        _classic(bars, vol20_window=1)


def test_classic_indicators_builds_without_keywords():
    frame = build_features({"SPY": _bars([10.0, 11.0, 12.0])}, "classic_indicators")
    rows = frame.xs("SPY", level="symbol")
    assert list(rows.columns) == list(CLASSIC_COLUMNS)
    assert len(rows) == 3
