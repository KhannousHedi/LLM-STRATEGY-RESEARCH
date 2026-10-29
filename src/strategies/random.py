"""BUY with probability ``p_buy`` while out, and SELL with probability ``p_sell`` while in.

The feature set is ``void``. Its row has no feature columns. Each draw
uses ``p_buy`` or ``p_sell``, and ``seed`` fixes the stream.
"""

from random import Random

from src.strategies import Strategy


def _probability(parameters, key):
    if key not in parameters:
        raise ValueError(f"random requires {key}")
    value = parameters[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"random {key} must be an int or float from 0 to 1")
    if not 0 <= value <= 1:
        raise ValueError(f"random {key} must be from 0 to 1")
    return value


def _seed(parameters):
    if "seed" not in parameters:
        raise ValueError("random requires seed")
    value = parameters["seed"]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("random seed must be an int")
    return value


class RandomStrategy(Strategy):
    def __init__(self, parameters):
        super().__init__(parameters)
        self.p_buy = _probability(parameters, "p_buy")
        self.p_sell = _probability(parameters, "p_sell")
        self._rng = Random(_seed(parameters))

    def decide(self, row, position):
        if position == "out":
            return "BUY" if self._rng.random() < self.p_buy else "HOLD"
        return "SELL" if self._rng.random() < self.p_sell else "HOLD"
