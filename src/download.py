"""Fetch 5-minute bars from Yahoo and write one UTC day file per date."""

import logging
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

from src.data import BARS_DIR, COLUMNS, as_date, day_path, weekdays

CHUNK_DAYS = 60

log = logging.getLogger(__name__)


def fetch_yahoo(symbol, first, last):
    return yf.Ticker(symbol).history(
        start=first.isoformat(),
        end=(last + timedelta(days=1)).isoformat(),
        interval="5m",
        prepost=False,
        auto_adjust=False,
        actions=False,
        raise_errors=True,
    )


def download_bars(symbols, start, end, root=BARS_DIR):
    if isinstance(symbols, str):
        symbols = [symbols]
    return {symbol: _download_symbol(symbol, as_date(start), as_date(end), root) for symbol in symbols}


def _download_symbol(symbol, start, end, root):
    # Today's session may still be open; a partial day file would never be refetched.
    today = datetime.now(timezone.utc).date()
    missing = [
        day for day in weekdays(start, end)
        if day < today and not day_path(root, symbol, day).exists()
    ]
    written = []
    for chunk in _chunks(missing):
        try:
            raw = fetch_yahoo(symbol, chunk[0], chunk[-1])
        except Exception as error:
            log.warning("%s: Yahoo rejected %s to %s: %s", symbol, chunk[0], chunk[-1], error)
            continue
        if raw is None or raw.empty:
            continue
        bars = _to_utc(symbol, raw)
        wanted = set(chunk)
        for day, day_bars in bars.groupby(bars.index.date):
            if day not in wanted:
                continue
            _check_day(symbol, day, day_bars)
            _write_day(day_path(root, symbol, day), day_bars)
            written.append(day)
    return written


def _chunks(days):
    chunk = []
    for day in days:
        if chunk and (day - chunk[0]).days >= CHUNK_DAYS:
            yield chunk
            chunk = []
        chunk.append(day)
    if chunk:
        yield chunk


def _to_utc(symbol, raw):
    if not isinstance(raw.index, pd.DatetimeIndex) or raw.index.tz is None:
        raise ValueError(f"{symbol}: Yahoo returned timestamps without a timezone")
    bars = raw.rename(columns=str.lower)[COLUMNS]
    bars.index = bars.index.tz_convert("UTC")
    bars.index.name = "timestamp"
    return bars


def _check_day(symbol, day, bars):
    def fail(reason):
        raise ValueError(f"{symbol} {day.isoformat()}: {reason}")

    index = bars.index
    if (index.second != 0).any() or (index.microsecond != 0).any() or (index.nanosecond != 0).any():
        fail("timestamp has seconds")
    if (index.minute % 5 != 0).any():
        fail("timestamp is not on a 5-minute boundary")
    if not index.is_unique or not index.is_monotonic_increasing:
        fail("timestamps are not unique and increasing")

    open_, high, low, close, volume = (bars[name] for name in COLUMNS)
    if bars[["open", "high", "low", "close"]].isna().any().any():
        fail("missing open, high, low, or close")
    if not ((high >= low) & (low <= open_) & (open_ <= high) & (low <= close) & (close <= high)).all():
        fail("OHLC values are inconsistent")
    if volume.isna().any() or (volume < 0).any():
        fail("volume is missing or negative")


def _write_day(path, bars):
    path.parent.mkdir(parents=True, exist_ok=True)
    out = bars.copy()
    out.index = pd.Index([ts.isoformat() for ts in out.index], name="timestamp")
    tmp = path.with_suffix(".tmp")
    out.to_csv(tmp)
    os.replace(tmp, path)
