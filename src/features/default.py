"""The ``default`` feature set.

``compute`` receives one symbol's bars. ``build_features`` loops over symbols
and stacks the dataframes. ``r1``, ``r5``, and ``r20`` are returns over 1, 5,
and 20 bars. The caller sets ``ma_window``, ``vol_window``, and
``volume_window``, as counts of bars. A window that runs past the first bar
is NaN. ``vol`` stays NaN until ``vol_window`` one-bar returns exist.
``vol_window`` is at least 2. Standard deviation is the sample standard
deviation (``ddof=1``).

The windows are positional and end at t.
"""

import pandas as pd

from src.features import FeatureSet


class Default(FeatureSet):
    columns = ("r1", "r5", "r20", "dist_ma", "vol", "vol_ratio")

    def compute(self, bars, ma_window, vol_window, volume_window):
        for name, value in (("ma_window", ma_window), ("volume_window", volume_window)):
            if value < 1:
                raise ValueError(f"{name} must be at least 1")
        if vol_window < 2:
            raise ValueError("vol_window must be at least 2")
        close = bars["close"]
        volume = bars["volume"]
        r1 = close.pct_change(1)
        mean_close = close.rolling(ma_window, min_periods=ma_window).mean()
        mean_volume = volume.rolling(volume_window, min_periods=volume_window).mean()
        values = {
            "r1": r1,
            "r5": close.pct_change(5),
            "r20": close.pct_change(20),
            "dist_ma": (close / mean_close) - 1,
            "vol": r1.rolling(vol_window, min_periods=vol_window).std(),
            "vol_ratio": volume / mean_volume,
        }
        return pd.DataFrame({name: values[name] for name in self.columns}, index=bars.index)
