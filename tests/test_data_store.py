from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.data import load_prices
from src.download import download_bars

SYMBOL = "TEST"
BARS_PER_DAY = 78


def yahoo_bars(days):
    """Frame shaped like yfinance 5-minute history: New York timestamps, capitalized columns."""
    frames = []
    for day in days:
        index = pd.date_range(f"{day} 09:30", periods=BARS_PER_DAY, freq="5min", tz="America/New_York")
        base = 100 + day.toordinal() % 7 + np.arange(BARS_PER_DAY) * 0.01
        frames.append(pd.DataFrame(
            {
                "Open": base,
                "High": base + 0.05,
                "Low": base - 0.05,
                "Close": base + 0.01,
                "Adj Close": base + 0.01,
                "Volume": np.arange(BARS_PER_DAY) * 10 + 1000,
            },
            index=index.rename("Datetime"),
        ))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames)


def as_store(raw):
    bars = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    bars.index = bars.index.tz_convert("UTC").rename("timestamp")
    return bars


def store_day(root, day):
    bars = as_store(yahoo_bars([day]))
    path = root / SYMBOL / f"{day.isoformat()}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out = bars.copy()
    out.index = pd.Index([ts.isoformat() for ts in out.index], name="timestamp")
    out.to_csv(path)
    return bars


def test_download_and_load_return_the_bars(tmp_path):
    symbol = "SPY"
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=6)
    download_bars([symbol], start, end, root=tmp_path)

    files = sorted((tmp_path / symbol).glob("*.csv"))
    assert files
    saved_rows = 0
    for path in files:
        day = date.fromisoformat(path.stem)
        assert start <= day <= end and day.weekday() < 5
        lines = path.read_text().splitlines()
        assert lines[0] == "timestamp,open,high,low,close,volume"
        for line in lines[1:]:
            stamp = line.split(",", 1)[0]
            assert stamp[:10] == day.isoformat()
            assert stamp.endswith("+00:00")
        saved_rows += len(lines) - 1

    loaded = load_prices([symbol], start, end, "5min", root=tmp_path)[symbol]
    assert len(loaded) == saved_rows


def test_download_chunks_the_range(tmp_path):
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=90)
    assert (end - start).days > 60
    download_bars(["SPY"], start, end, root=tmp_path)


def test_load_prices_warns_on_partial_range(tmp_path):
    start, end = date(2024, 3, 4), date(2024, 3, 6)
    for day in [date(2024, 3, 4), date(2024, 3, 6), date(2024, 3, 7)]:
        store_day(tmp_path, day)

    with pytest.warns(UserWarning) as record:
        loaded = load_prices([SYMBOL], start, end, "5min", root=tmp_path)[SYMBOL]

    messages = [str(w.message) for w in record]
    assert any(SYMBOL in m and "2024-03-05" in m for m in messages)
    assert all(start <= d <= end for d in loaded.index.date)


def test_load_prices_resamples_to_frequency(tmp_path):
    days = [date(2024, 3, 4), date(2024, 3, 5)]
    stored = [store_day(tmp_path, day) for day in days]

    daily = load_prices([SYMBOL], days[0], days[-1], "1D", root=tmp_path)[SYMBOL]

    assert list(daily.index) == [pd.Timestamp(day, tz="UTC") for day in days]
    for (_, row), bars in zip(daily.iterrows(), stored):
        assert row["open"] == bars["open"].iloc[0]
        assert row["high"] == bars["high"].max()
        assert row["low"] == bars["low"].min()
        assert row["close"] == bars["close"].iloc[-1]
        assert row["volume"] == bars["volume"].sum()

