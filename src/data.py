"""Read the daily year store and set it to the frequency a run asks for."""

import warnings
from datetime import timedelta
from pathlib import Path

import pandas as pd
from pandas.tseries.frequencies import to_offset

BARS_DIR = Path(__file__).resolve().parent.parent / "data" / "bars"
COLUMNS = ["open", "high", "low", "close", "volume"]
BASE_FREQ = "1D"
MAX_LISTED_MISSING_WEEKDAYS = 3


def as_date(value):
    return pd.Timestamp(value).date()


def _missing_weekdays_message(symbol, start, end, missing):
    count = len(missing)
    noun = "weekday" if count == 1 else "weekdays"
    shown = missing[:MAX_LISTED_MISSING_WEEKDAYS]
    listed = ", ".join(day.isoformat() for day in shown)
    if count > MAX_LISTED_MISSING_WEEKDAYS:
        listed = f"{listed}, and {count - MAX_LISTED_MISSING_WEEKDAYS} more"
    return f"{symbol}: {start.isoformat()} to {end.isoformat()} is missing {count} {noun}: {listed}"


def weekdays(start, end):
    day, end = as_date(start), as_date(end)
    out = []
    while day <= end:
        if day.weekday() < 5:
            out.append(day)
        day += timedelta(days=1)
    return out


def year_path(root, symbol, year):
    return Path(root) / symbol / f"{int(year)}.csv"


def read_bars(path):
    frame = pd.read_csv(path, index_col="timestamp", float_precision="round_trip")
    frame.index = pd.to_datetime(frame.index, utc=True)
    frame.index.name = "timestamp"
    return frame[COLUMNS]


def load_prices(symbols, start, end, freq, root=BARS_DIR):
    if isinstance(symbols, str):
        symbols = [symbols]
    start, end = as_date(start), as_date(end)
    return {symbol: _load_symbol(symbol, start, end, freq, root) for symbol in symbols}


def _load_symbol(symbol, start, end, freq, root):
    frames = []
    for year in range(start.year, end.year + 1):
        path = year_path(root, symbol, year)
        if path.is_file():
            frames.append(read_bars(path))

    if frames:
        bars = pd.concat(frames).sort_index()
    else:
        bars = pd.DataFrame(columns=COLUMNS, index=pd.DatetimeIndex([], tz="UTC", name="timestamp"))
    dates = bars.index.normalize()
    bars = bars[(dates >= pd.Timestamp(start, tz="UTC")) & (dates <= pd.Timestamp(end, tz="UTC"))]

    covered = set(bars.index.date)
    missing = [day for day in weekdays(start, end) if day not in covered]
    if missing:
        warnings.warn(_missing_weekdays_message(symbol, start, end, missing), stacklevel=3)

    if to_offset(freq) == to_offset(BASE_FREQ):
        return bars
    resampled = bars.resample(freq, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    resampled.index.name = "timestamp"
    return resampled.dropna(subset=["open"])
