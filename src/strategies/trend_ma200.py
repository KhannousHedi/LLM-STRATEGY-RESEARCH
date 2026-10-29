"""Be long while the close is above its 200-session average.

``decide`` uses ``ma200_gap`` from the ``classic_indicators`` feature set.
The feature set computes that column from the feature parameters, including
``ma200_window``. The comparison level is the constant 0.
"""

from src.strategies import Strategy


class TrendMa200(Strategy):
    feature_set = "classic_indicators"

    def decide(self, row, position):
        gap = row["ma200_gap"]
        if position == "out" and gap > 0:
            return "BUY"
        if position == "in" and gap < 0:
            return "SELL"
        return "HOLD"
