"""Fill booleans with vectorbt at the session open.

``simulate(close, opens, entries, exits, freq)`` calls
``Portfolio.from_signals`` with the boolean index as given. ``price``
is that session's open. Nothing is shifted, and the timestamps are not
checked again.

``benchmark(close, opens, execute_on, freq)`` is the same call with one
entry on ``execute_on`` and no exits. Cash is 100,000. Slippage is
0.0005. Fees are 0.
"""

import numpy as np
import pandas as pd
import vectorbt as vbt

INIT_CASH = 100_000
SLIPPAGE = 0.0005
FEES = 0.0


def _from_signals(close, opens, entries, exits, freq):
    return vbt.Portfolio.from_signals(
        close,
        entries,
        exits,
        price=opens,
        init_cash=INIT_CASH,
        cash_sharing=False,
        direction="longonly",
        size=np.inf,
        fees=FEES,
        slippage=SLIPPAGE,
        freq=freq,
    )


def simulate(close, opens, entries, exits, freq):
    return _from_signals(close, opens, entries, exits, freq)


def benchmark(close, opens, execute_on, freq):
    entries = pd.Series(False, index=close.index)
    exits = pd.Series(False, index=close.index)
    entries.loc[execute_on] = True
    return _from_signals(close, opens, entries, exits, freq)
