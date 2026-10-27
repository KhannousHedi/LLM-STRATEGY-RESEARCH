"""Build one feature dataframe by looping over symbols.

``build_features(prices, feature_set, **kwargs)`` loads
``src/features/{feature_set}.py``, builds its ``FeatureSet``, and calls
``compute`` once per symbol. Extra keyword arguments are passed on to
``compute``. This package does not define prompts.

A feature module defines one subclass. ``compute`` takes ``bars`` and any
arguments that subclass adds::

    from src.features import FeatureSet

    class Example(FeatureSet):
        columns = ("r1",)

        def compute(self, bars, window):
            # bars is one symbol.
            # Return a dataframe indexed by that symbol's datetimes.
            ...

A subclass inherits ``columns`` and ``compute`` until it replaces them.
This file checks ``columns``, then checks that each returned frame is a
dataframe with those columns and that symbol's datetimes.
"""

import importlib

import pandas as pd


class FeatureSet:
    """Base for one feature computation. Subclasses set ``columns`` and ``compute``."""

    columns = ()

    def compute(self, bars):
        raise NotImplementedError


def load_feature_set(name):
    if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
        raise ValueError(f"no feature set {name!r}")
    module_name = f"{__name__}.{name}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name:
            raise ValueError(f"no feature set {name!r}") from None
        raise
    found = [
        value
        for value in vars(module).values()
        if isinstance(value, type) and issubclass(value, FeatureSet) and value.__module__ == module.__name__
    ]
    if len(found) != 1:
        raise TypeError(f"{module_name} must define one FeatureSet subclass")
    feature_set = found[0]()
    if feature_set.compute.__func__ is FeatureSet.compute:
        raise TypeError(f"{module_name} must implement compute")
    columns = feature_set.columns
    if not isinstance(columns, tuple) or not columns or not all(isinstance(item, str) and item.isidentifier() for item in columns):
        raise TypeError(f"{module_name} must set columns to a non-empty tuple of names")
    if len(set(columns)) != len(columns):
        raise TypeError(f"{module_name} repeats a column name")
    return feature_set


def _check_frame(feature_set, symbol, bars, frame, columns):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{feature_set}: {symbol} must be a dataframe")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{feature_set}: {symbol} index must be datetimes")
    if list(frame.columns) != list(columns):
        raise TypeError(f"{feature_set}: {symbol} columns must be {list(columns)}")
    if not frame.index.equals(bars.index):
        raise TypeError(f"{feature_set}: {symbol} datetimes must match the loaded bars")


def build_features(prices, feature_set, **kwargs):
    feature = load_feature_set(feature_set)
    parts = []
    for symbol, bars in prices.items():
        frame = feature.compute(bars, **kwargs)
        _check_frame(feature_set, symbol, bars, frame, feature.columns)
        parts.append(frame)
    if not parts:
        index = pd.MultiIndex.from_arrays(
            [pd.Index([], dtype="object", name="symbol"), pd.DatetimeIndex([], tz="UTC", name="timestamp")],
        )
        return pd.DataFrame(columns=list(feature.columns), index=index)
    out = pd.concat(parts, keys=list(prices), names=["symbol", "timestamp"])
    if not isinstance(out.index.get_level_values("timestamp"), pd.DatetimeIndex):
        raise TypeError(f"{feature_set}: timestamp level must be datetimes")
    return out
