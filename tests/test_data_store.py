from datetime import date

import numpy as np
import pandas as pd
import pytest

from src import download
from src.data import load_prices, weekdays
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


class FakeYahoo:
    def __init__(self, oldest=None):
        self.oldest = oldest
        self.calls = []
        self.served = []

    def __call__(self, symbol, first, last):
        self.calls.append((first, last))
        if (last - first).days + 1 > 60:
            raise ValueError("5m data is limited to 60 days per request")
        days = [d for d in weekdays(first, last) if self.oldest is None or d >= self.oldest]
        self.served.extend(days)
        return yahoo_bars(days)


def stored_dates(root):
    return sorted(date.fromisoformat(p.stem) for p in (root / SYMBOL).glob("*.csv"))


def test_download_and_load_return_the_bars(tmp_path, monkeypatch):
    days = [date(2024, 3, 4), date(2024, 3, 5)]
    monkeypatch.setattr(download, "fetch_yahoo", FakeYahoo())

    download_bars([SYMBOL], days[0], days[-1], root=tmp_path)
    loaded = load_prices([SYMBOL], days[0], days[-1], "5min", root=tmp_path)[SYMBOL]

    assert stored_dates(tmp_path) == days
    expected = as_store(yahoo_bars(days))
    pd.testing.assert_frame_equal(loaded, expected, check_freq=False, check_index_type=False)
    assert loaded.index.equals(expected.index)


def test_download_chunks_the_range(tmp_path, monkeypatch):
    start, end = date(2024, 1, 1), date(2024, 4, 30)
    fake = FakeYahoo(oldest=date(2024, 3, 1))
    monkeypatch.setattr(download, "fetch_yahoo", fake)

    download_bars([SYMBOL], start, end, root=tmp_path)

    assert len(fake.calls) > 1
    assert all((last - first).days + 1 <= 60 for first, last in fake.calls)
    requested = [d for first, last in fake.calls for d in weekdays(first, last)]
    assert requested == weekdays(start, end)
    assert stored_dates(tmp_path) == sorted(fake.served)
    assert fake.served and min(fake.served) == date(2024, 3, 1)


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


def test_second_download_skips_yahoo(tmp_path, monkeypatch):
    start, end = date(2024, 3, 4), date(2024, 3, 10)
    monkeypatch.setattr(download, "fetch_yahoo", FakeYahoo())
    download_bars([SYMBOL], start, end, root=tmp_path)
    assert stored_dates(tmp_path) == weekdays(start, end)

    second = FakeYahoo()
    monkeypatch.setattr(download, "fetch_yahoo", second)
    download_bars([SYMBOL], start, end, root=tmp_path)

    assert second.calls == []
