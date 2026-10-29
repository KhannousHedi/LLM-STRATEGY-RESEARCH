"""Standard daily indicators.

Windows are keyword arguments with textbook defaults: ``r1_window`` 1,
``r5_window`` 5, ``r20_window`` 20, ``ma50_window`` 50, ``ma200_window``
200, ``high52_window`` 252, ``rsi14_window`` 14, ``vol20_window`` 20, and
``volume_window`` 20. Each windowed description names that parameter.
``open_gap``, ``day_range``, and ``close_location`` have no window.
Row t uses only bars at or before t. A window that is not full yet is
NaN. ``vol20_window`` is at least 2. The other windows are at least 1.
``vol20`` annualizes with the square root of 252. That 252 is not a
parameter.
"""

import math

import pandas as pd

from src.features import FeatureSet


def _require_window(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _rsi_value(avg_gain, avg_loss):
    if avg_gain == 0.0 and avg_loss == 0.0:
        return 50.0
    if avg_loss == 0.0 and avg_gain > 0.0:
        return 100.0
    if avg_gain == 0.0 and avg_loss > 0.0:
        return 0.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def _wilder_rsi(close, window):
    delta = close.diff()
    gains = delta.clip(lower=0.0).to_numpy(dtype="float64")
    losses = (-delta).clip(lower=0.0).to_numpy(dtype="float64")
    values = [float("nan")] * len(close)
    if len(close) <= window:
        return pd.Series(values, index=close.index, dtype="float64")
    avg_gain = float(gains[1 : window + 1].mean())
    avg_loss = float(losses[1 : window + 1].mean())
    values[window] = _rsi_value(avg_gain, avg_loss)
    for index in range(window + 1, len(close)):
        avg_gain = (avg_gain * (window - 1) + float(gains[index])) / window
        avg_loss = (avg_loss * (window - 1) + float(losses[index])) / window
        values[index] = _rsi_value(avg_gain, avg_loss)
    return pd.Series(values, index=close.index, dtype="float64")


class ClassicIndicators(FeatureSet):
    columns = (
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
    descriptions = {
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

    def compute(
        self,
        bars,
        r1_window=1,
        r5_window=5,
        r20_window=20,
        ma50_window=50,
        ma200_window=200,
        high52_window=252,
        rsi14_window=14,
        vol20_window=20,
        volume_window=20,
    ):
        _require_window("r1_window", r1_window, 1)
        _require_window("r5_window", r5_window, 1)
        _require_window("r20_window", r20_window, 1)
        _require_window("ma50_window", ma50_window, 1)
        _require_window("ma200_window", ma200_window, 1)
        _require_window("high52_window", high52_window, 1)
        _require_window("rsi14_window", rsi14_window, 1)
        _require_window("vol20_window", vol20_window, 2)
        _require_window("volume_window", volume_window, 1)
        close = bars["close"]
        high = bars["high"]
        low = bars["low"]
        span = high - low
        one_session = close.pct_change(1)
        values = {
            "r1": close.pct_change(r1_window),
            "r5": close.pct_change(r5_window),
            "r20": close.pct_change(r20_window),
            "ma50_gap": close / close.rolling(ma50_window, min_periods=ma50_window).mean() - 1,
            "ma200_gap": close / close.rolling(ma200_window, min_periods=ma200_window).mean() - 1,
            "high52_gap": close / high.rolling(high52_window, min_periods=high52_window).max() - 1,
            "rsi14": _wilder_rsi(close, rsi14_window),
            "vol20": one_session.rolling(vol20_window, min_periods=vol20_window).std(ddof=1) * math.sqrt(252),
            "volume_ratio": bars["volume"] / bars["volume"].rolling(volume_window, min_periods=volume_window).mean(),
            "open_gap": bars["open"] / close.shift(1) - 1,
            "day_range": span / close,
            "close_location": ((close - low) / span).where(span != 0, 0.5),
        }
        return pd.DataFrame({name: values[name] for name in self.columns}, index=bars.index)
