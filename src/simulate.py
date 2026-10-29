"""Fill booleans with vectorbt at the session open.

``simulate(close, opens, entries, exits, freq, slippage, transaction_cost)``
calls ``Portfolio.from_signals`` with the boolean index as given.
``price`` is that session's open. Nothing is shifted, and the timestamps
are not checked again. Cash is 100,000. Slippage defaults to 0.0005. The
transaction cost defaults to 0.001, a fraction of order value, and is
passed as fees.
"""

import numpy as np
import vectorbt as vbt

INIT_CASH = 100_000
SLIPPAGE = 0.0005
TRANSACTION_COST = 0.001


def simulate(close, opens, entries, exits, freq, slippage=SLIPPAGE, transaction_cost=TRANSACTION_COST):
    return vbt.Portfolio.from_signals(
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
