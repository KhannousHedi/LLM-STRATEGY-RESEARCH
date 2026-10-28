"""Hold while the price sits above its moving average by ``buy_above``."""

from src.strategies import Strategy


class DistMa(Strategy):
    feature_set = "momentum_volatility_volume"
    feature_parameters = {"ma_window": 20, "vol_window": 20, "volume_window": 20}

    def __init__(self, parameters):
        super().__init__(parameters)
        self.buy_above = parameters.get("buy_above", 0)

    def decide(self, row, position):
        if position == "out" and row["dist_ma"] > self.buy_above:
            return "BUY"
        if position == "in" and row["dist_ma"] < self.buy_above:
            return "SELL"
        return "HOLD"
