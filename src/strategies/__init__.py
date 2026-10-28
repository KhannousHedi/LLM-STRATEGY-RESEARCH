"""Load one strategy class by module name.

``load_strategy(name, parameters)`` imports ``src/strategies/{name}.py``,
requires one ``Strategy`` subclass defined in that module, and constructs
it with the parameters dict. A name that is not an identifier, or that
starts with ``_``, is refused. The class does not open a parameters file.

The base reads ``feature_set`` and ``feature_parameters`` from the dict
and keeps the class values when a key is absent. A subclass sets either
only when it needs something other than the base value. ``decide`` takes
one row holding every column of that feature set, in the set's order,
and the position ``in`` or ``out``. ``detail`` returns ``{}``.

This package does not download, resample, call vectorbt, or assign
``execute_on``. It does not read the open position from the portfolio.
"""

import importlib


class Strategy:
    feature_set = "void"
    feature_parameters = {}

    def __init__(self, parameters):
        if "feature_set" in parameters:
            self.feature_set = parameters["feature_set"]
        if "feature_parameters" in parameters:
            self.feature_parameters = parameters["feature_parameters"]
        else:
            self.feature_parameters = dict(self.__class__.feature_parameters)

    def decide(self, row, position):
        raise NotImplementedError

    def detail(self):
        return {}


def load_strategy(name, parameters):
    if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
        raise ValueError(f"no strategy {name!r}")
    module_name = f"{__name__}.{name}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name:
            raise ValueError(f"no strategy {name!r}") from None
        raise
    found = [
        value
        for value in vars(module).values()
        if isinstance(value, type) and issubclass(value, Strategy) and value.__module__ == module.__name__
    ]
    if len(found) != 1:
        raise TypeError(f"{module_name} must define one Strategy subclass")
    strategy = found[0](parameters)
    if type(strategy).decide is Strategy.decide:
        raise TypeError(f"{module_name} must implement decide")
    if not isinstance(strategy.feature_set, str):
        raise TypeError(f"{module_name} must set feature_set to a string")
    if not isinstance(strategy.feature_parameters, dict):
        raise TypeError(f"{module_name} must set feature_parameters to a dict")
    return strategy
