"""Fetch daily bars from Yahoo and write one UTC year file per year."""

import logging
import os
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

from src.data import BARS_DIR, COLUMNS, as_date, year_path

log = logging.getLogger(__name__)


def fetch_yahoo(symbol, first, last):
    yf.config.debug.hide_exceptions = False
    return yf.Ticker(symbol).history(
        start=first.isoformat(),
        end=(last + timedelta(days=1)).isoformat(),
        interval="1d",
        prepost=False,
        auto_adjust=False,
        actions=False,
    )


def download_bars(symbols, start, end, root=BARS_DIR):
    if isinstance(symbols, str):
        symbols = [symbols]
    return {symbol: _download_symbol(symbol, as_date(start), as_date(end), root) for symbol in symbols}


def _download_symbol(symbol, start, end, root):
    today = datetime.now(timezone.utc).date()
    written = []
    for year in range(start.year, end.year + 1):
        bounds = _year_bounds(year, today)
        if bounds is None:
            continue
        path = year_path(root, symbol, year)
        if year < today.year and path.is_file():
            continue
        first, last = bounds
        try:
            raw = fetch_yahoo(symbol, first, last)
        except Exception as error:
            log.warning("%s: Yahoo rejected %s: %s", symbol, year, error)
            continue
        if raw is None or raw.empty:
            continue
        bars = _in_year(_to_utc(symbol, raw), first, last)
        if bars.empty:
            continue
        _check_year(symbol, year, bars)
        _write_year(path, bars)
        written.append(year)
    return written


def _year_bounds(year, today):
    if year > today.year:
        return None
    first = date(year, 1, 1)
    if year < today.year:
        return first, date(year, 12, 31)
    last = today - timedelta(days=1)
    if last < first:
        return None
    return first, last


def _to_utc(symbol, raw):
    if not isinstance(raw.index, pd.DatetimeIndex) or raw.index.tz is None:
        raise ValueError(f"{symbol}: Yahoo returned timestamps without a timezone")
    bars = raw.rename(columns=str.lower)[COLUMNS]
    bars.index = bars.index.tz_convert("UTC")
    bars.index.name = "timestamp"
    return bars.sort_index()


def _in_year(bars, first, last):
    dates = bars.index.normalize()
    return bars[(dates >= pd.Timestamp(first, tz="UTC")) & (dates <= pd.Timestamp(last, tz="UTC"))]


def _check_year(symbol, year, bars):
    def fail(reason):
        raise ValueError(f"{symbol} {year}: {reason}")

    index = bars.index
    if index.tz is None or str(index.tz) != "UTC":
        fail("timestamp is not timezone-aware UTC")
    if (index.year != year).any():
        fail("a timestamp falls outside the year")
    if not index.is_unique or not index.is_monotonic_increasing:
        fail("timestamps are not unique and increasing")
    if len(set(index.date)) != len(index):
        fail("more than one row for a session")

    open_, high, low, close, volume = (bars[name] for name in COLUMNS)
    if bars[["open", "high", "low", "close"]].isna().any().any():
        fail("missing open, high, low, or close")
    if not ((high >= low) & (low <= open_) & (open_ <= high) & (low <= close) & (close <= high)).all():
        fail("OHLC values are inconsistent")
    if volume.isna().any() or (volume < 0).any():
        fail("volume is missing or negative")


def _write_year(path, bars):
    path.parent.mkdir(parents=True, exist_ok=True)
    out = bars.copy()
    out.index = pd.Index([ts.isoformat() for ts in out.index], name="timestamp")
    tmp = path.with_suffix(".tmp")
    out.to_csv(tmp)
    os.replace(tmp, path)
