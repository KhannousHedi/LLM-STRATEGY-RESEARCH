"""Read the 5-minute day store and set it to the frequency a run asks for."""

import warnings
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from pandas.tseries.frequencies import to_offset

BARS_DIR = Path(__file__).resolve().parent.parent / "data" / "bars"
COLUMNS = ["open", "high", "low", "close", "volume"]
BASE_FREQ = "5min"


def as_date(value):
    return pd.Timestamp(value).date()


def weekdays(start, end):
    day, end = as_date(start), as_date(end)
    out = []
    while day <= end:
        if day.weekday() < 5:
            out.append(day)
        day += timedelta(days=1)
    return out


def day_path(root, symbol, day):
    return Path(root) / symbol / f"{day.isoformat()}.csv"


def read_day(path):
    frame = pd.read_csv(path, index_col="timestamp", float_precision="round_trip")
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame[COLUMNS]


def load_prices(symbols, start, end, freq, root=BARS_DIR):
    if isinstance(symbols, str):
        symbols = [symbols]
    start, end = as_date(start), as_date(end)
    return {symbol: _load_symbol(symbol, start, end, freq, root) for symbol in symbols}


def _load_symbol(symbol, start, end, freq, root):
    days = []
    folder = Path(root) / symbol
    if folder.is_dir():
        for path in sorted(folder.glob("*.csv")):
            try:
                day = date.fromisoformat(path.stem)
            except ValueError:
                continue
            if start <= day <= end:
                days.append(read_day(path))

    if days:
        bars = pd.concat(days).sort_index()
    else:
        bars = pd.DataFrame(columns=COLUMNS, index=pd.DatetimeIndex([], tz="UTC", name="timestamp"))
    bar_dates = bars.index.date
    bars = bars[(bar_dates >= start) & (bar_dates <= end)]

    covered = set(bars.index.date)
    missing = [day for day in weekdays(start, end) if day not in covered]
    if missing:
        listed = ", ".join(day.isoformat() for day in missing)
        warnings.warn(
            f"{symbol}: {start.isoformat()} to {end.isoformat()} is missing weekdays {listed}",
            stacklevel=3,
        )

    if to_offset(freq) == to_offset(BASE_FREQ):
        return bars
    resampled = bars.resample(freq, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return resampled.dropna(subset=["open"])
