"""Fill booleans with vectorbt at the session open.

``simulate(close, opens, entries, exits, freq, slippage, transaction_cost)``
calls vectorbt ``Portfolio.from_signals`` with the boolean index as given.
``price`` is that session's open. Nothing is shifted, and the timestamps
are not checked again. Cash is 100,000. Slippage defaults to 0.0005. The
transaction cost defaults to 0.001, a fraction of order value, and is
passed as fees.

The return value keeps that portfolio's attributes. ``stats`` also
includes ``time_in_market`` and ``avg_holding_period``. The average is
null when ``Total Trades`` is 0.
"""

import numpy as np
import pandas as pd
import vectorbt as vbt

INIT_CASH = 100_000
SLIPPAGE = 0.0005
TRANSACTION_COST = 0.001


def _holding_stats(orders, sessions):
    """Return time in the market and the mean holding length.

    ``orders`` has ``Timestamp`` and ``Side``. ``sessions`` is the equity
    index. A buy at index ``i`` and a sell at index ``j`` covers ``[i, j)``.
    A buy with no later sell covers ``[i, last]`` inclusive. The mean is
    ``None`` when there is no holding.
    """
    sessions = pd.DatetimeIndex(pd.to_datetime(sessions, utc=True))
    count = len(sessions)
    if count == 0:
        raise ValueError("equity sessions are empty")
    covered = [False] * count
    lengths = []
    entry = None
    if orders is not None and len(orders) > 0:
        stamps = pd.DatetimeIndex(pd.to_datetime(orders["Timestamp"], utc=True))
        places = sessions.get_indexer(stamps)
        for place, side in zip(places, orders["Side"]):
            if place < 0:
                raise ValueError("order timestamp is not an equity session")
            if side == "Buy":
                if entry is not None:
                    raise ValueError("buy while already in a position")
                entry = int(place)
            elif side == "Sell":
                if entry is None:
                    raise ValueError("sell while flat")
                if place < entry:
                    raise ValueError("sell before its buy")
                for cursor in range(entry, int(place)):
                    covered[cursor] = True
                lengths.append(int(place) - entry)
                entry = None
            else:
                raise ValueError(f"unknown order side {side!r}")
    if entry is not None:
        for cursor in range(entry, count):
            covered[cursor] = True
        lengths.append(count - entry)
    time_in_market = sum(covered) / count
    average = None if not lengths else sum(lengths) / len(lengths)
    return time_in_market, average


class _Stats(pd.Series):
    def to_string(self, *args, **kwargs):
        shown = pd.Series(self)
        if pd.isna(shown["avg_holding_period"]):
            shown["avg_holding_period"] = "null"
        return pd.Series.to_string(shown, *args, **kwargs)


class Portfolio:
    """Vectorbt portfolio whose ``stats`` include the holding fields."""

    def __init__(self, portfolio):
        self._portfolio = portfolio

    def stats(self):
        stats = self._portfolio.stats().copy()
        time_in_market, average = _holding_stats(
            self._portfolio.orders.records_readable,
            self._portfolio.value().index,
        )
        if stats["Total Trades"] == 0:
            average = None
        elif average is None:
            raise ValueError("has trades but no holding period")
        stats["time_in_market"] = time_in_market
        stats["avg_holding_period"] = average
        return _Stats(stats)

    def __getattr__(self, name):
        return getattr(self._portfolio, name)


def simulate(close, opens, entries, exits, freq, slippage=SLIPPAGE, transaction_cost=TRANSACTION_COST):
    portfolio = vbt.Portfolio.from_signals(
        close,
        entries,
        exits,
        price=opens,
        init_cash=INIT_CASH,
        cash_sharing=False,
        direction="longonly",
        size=np.inf,
        slippage=slippage,
        fees=transaction_cost,
        freq=freq,
    )
    return Portfolio(portfolio)
