"""Buy a low RSI and sell a high RSI, at the fixed levels 30 and 70.

``decide`` uses ``rsi14`` from the ``classic_indicators`` feature set.
The feature set computes that column from the feature parameters, including
``rsi14_window``. The comparison levels are the constants 30 and 70.
"""

from src.strategies import Strategy


class RsiReversion(Strategy):
    feature_set = "classic_indicators"

    def decide(self, row, position):
        rsi = row["rsi14"]
        if position == "out" and rsi < 30:
            return "BUY"
        if position == "in" and rsi > 70:
            return "SELL"
        return "HOLD"
