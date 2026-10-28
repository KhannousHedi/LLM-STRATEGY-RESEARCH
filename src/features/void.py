"""A feature set with no columns.

``compute`` returns the loaded timestamps and no values. A strategy that
does not name a set gets this one.
"""

import pandas as pd

from src.features import FeatureSet


class Void(FeatureSet):
    columns = ()
    descriptions = {}

    def compute(self, bars):
        return pd.DataFrame(index=bars.index)
