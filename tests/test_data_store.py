from datetime import date, timedelta

import pandas as pd
import pytest

from src.data import load_prices
from src.download import download_bars

SYMBOL = "TEST"


def store_year(root, symbol, rows):
    """Write known daily sessions into ``{year}.csv``. Each row is date, O, H, L, C, V."""
    by_year = {}
    for row in rows:
        by_year.setdefault(row[0].year, []).append(row)
    for year, items in by_year.items():
        index = pd.DatetimeIndex([pd.Timestamp(item[0], tz="UTC") for item in items], name="timestamp")
        frame = pd.DataFrame(
            [item[1:] for item in items],
            index=index,
            columns=["open", "high", "low", "close", "volume"],
        )
        path = root / symbol / f"{year}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        out = frame.copy()
        out.index = pd.Index([ts.isoformat() for ts in out.index], name="timestamp")
        out.to_csv(path)


def session_dates(path):
    dates = []
    lines = path.read_text().splitlines()
    assert lines[0] == "timestamp,open,high,low,close,volume"
    for line in lines[1:]:
        stamp = line.split(",", 1)[0]
        assert stamp.endswith("+00:00")
        dates.append(date.fromisoformat(stamp[:10]))
    return dates


def test_download_and_load_return_the_bars(tmp_path, monkeypatch):
    symbol = "SPY"
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=6)
    download_bars([symbol], start, end, root=tmp_path)

    files = sorted((tmp_path / symbol).glob("*.csv"))
    assert files
    assert [path.stem for path in files] == [str(year) for year in range(start.year, end.year + 1)]
    saved = []
    for path in files:
        year = int(path.stem)
        dates = session_dates(path)
        assert dates
        assert len(dates) == len(set(dates))
        assert all(day.year == year and day < date.today() for day in dates)
        saved.extend(dates)
    assert any(day < start or day > end for day in saved)

    def fail(*args, **kwargs):
        raise AssertionError("Yahoo was called")

    monkeypatch.setattr("src.download.fetch_yahoo", fail)
    monkeypatch.setattr("yfinance.Ticker.history", fail)
    loaded = load_prices([symbol], start, end, "1D", root=tmp_path)[symbol]
    assert list(loaded.index.date) == [day for day in saved if start <= day <= end]


def test_download_chunks_the_range(tmp_path, monkeypatch):
    start, end = date(2024, 12, 1), date(2025, 1, 31)
    download_bars(["SPY"], start, end, root=tmp_path)

    dates_2024 = session_dates(tmp_path / "SPY" / "2024.csv")
    assert dates_2024[0] == date(2024, 1, 2)
    assert dates_2024[-1] == date(2024, 12, 31)
    assert (tmp_path / "SPY" / "2025.csv").is_file()

    called = []

    def fail(*args, **kwargs):
        called.append(args)
        raise AssertionError("Yahoo was called")

    monkeypatch.setattr("src.download.fetch_yahoo", fail)
    monkeypatch.setattr("yfinance.Ticker.history", fail)
    download_bars(["SPY"], date(2024, 6, 1), date(2024, 6, 30), root=tmp_path)
    load_prices(["SPY"], start, end, "1D", root=tmp_path)
    assert called == []


def test_load_prices_warns_on_partial_range(tmp_path):
    start, end = date(2024, 3, 4), date(2024, 3, 6)
    store_year(tmp_path, SYMBOL, [
        (date(2024, 3, 4), 10, 12, 9, 11, 100),
        (date(2024, 3, 6), 11, 13, 10, 12, 110),
        (date(2024, 3, 7), 12, 14, 11, 13, 120),
    ])

    with pytest.warns(UserWarning) as record:
        loaded = load_prices([SYMBOL], start, end, "1D", root=tmp_path)[SYMBOL]

    messages = [str(w.message) for w in record]
    assert any(SYMBOL in m and "2024-03-05" in m for m in messages)
    assert all(start <= d <= end for d in loaded.index.date)


def test_load_prices_resamples_to_frequency(tmp_path):
    days = [date(2024, 3, 4), date(2024, 3, 5)]
    store_year(tmp_path, SYMBOL, [
        (days[0], 10, 12, 9, 11, 100),
        (days[1], 11, 14, 8, 13, 250),
    ])

    weekly = load_prices([SYMBOL], days[0], days[-1], "W", root=tmp_path)[SYMBOL]

    assert list(weekly.index) == [pd.Timestamp("2024-03-03", tz="UTC")]
    row = weekly.iloc[0]
    assert row["open"] == 10
    assert row["high"] == 14
    assert row["low"] == 8
    assert row["close"] == 13
    assert row["volume"] == 350
