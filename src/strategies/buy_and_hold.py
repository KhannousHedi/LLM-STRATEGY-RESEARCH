"""BUY on the first decision, then HOLD.

The feature set is ``void``. Its row has no feature columns, so the
decision stands on the call sequence alone.
"""

from src.strategies import Strategy


class BuyAndHold(Strategy):
    def __init__(self, parameters):
        super().__init__(parameters)
        self._bought = False

    def decide(self, row, position):
        if not self._bought:
            self._bought = True
            return "BUY"
        return "HOLD"
