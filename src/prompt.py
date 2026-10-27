"""Load a prompt file and fill one anonymous row.

``load_template(name)`` reads ``prompts/{name}.json``. The default name is
``default``. The file has ``feature_set``, ``parameters``, and ``wording``.
The ordered feature names are that set's ``columns``. The wording's slots
are ``{features#1}`` through ``{features#n}`` for those columns, plus
``{position}``. Parameter values are not slots and are not written into
the model text.

``render(template, row, position)`` checks the row, then fills each slot.
The row's keys are exactly those columns, and every value is a finite
number. Position is ``in`` or ``out``. A row that fails raises, and no
filled text is returned.
"""

import json
import math
import numbers
import re
from pathlib import Path

from src.features import load_feature_set

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

_SLOT = re.compile(r"\{(features#\d+|position|[A-Za-z_][A-Za-z0-9_]*)\}")


class Template:
    """One prompt file plus the feature set's column order."""

    def __init__(self, feature_set, parameters, wording, columns):
        self.feature_set = feature_set
        self.parameters = parameters
        self.wording = wording
        self.columns = columns


def load_template(name="default", root=PROMPTS_DIR):
    if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
        raise ValueError(f"no prompt {name!r}")
    path = Path(root) / f"{name}.json"
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or set(data) != {"feature_set", "parameters", "wording"}:
        raise ValueError(f"{name}: prompt file must contain feature_set, parameters, and wording")
    feature_set = data["feature_set"]
    parameters = data["parameters"]
    wording = data["wording"]
    if not isinstance(feature_set, str):
        raise ValueError(f"{name}: feature_set must be a string")
    if not isinstance(parameters, dict):
        raise ValueError(f"{name}: parameters must be an object")
    if not isinstance(wording, str):
        raise ValueError(f"{name}: wording must be a string")
    columns = tuple(load_feature_set(feature_set).columns)
    _check_wording(name, wording, len(columns))
    return Template(feature_set, parameters, wording, columns)


def _check_wording(name, wording, count):
    expected = [f"features#{index}" for index in range(1, count + 1)] + ["position"]
    found = _SLOT.findall(wording)
    if any(slot not in found for slot in expected) or set(found) - set(expected):
        slots = ", ".join("{" + slot + "}" for slot in expected)
        raise ValueError(f"{name}: wording slots must be {slots}")


def _finite_number(value):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def render(template, row, position):
    columns = tuple(template.columns)
    try:
        keys = tuple(row.keys())
    except AttributeError:
        raise TypeError("row must map feature names to values") from None
    if set(keys) != set(columns) or len(keys) != len(columns):
        raise ValueError(f"row keys must be {', '.join(columns)}")
    values = []
    for name in columns:
        number = _finite_number(row[name])
        if number is None:
            raise ValueError(f"{name} must be a finite number")
        values.append(number)
    if position not in ("in", "out"):
        raise ValueError("position must be in or out")
    text = template.wording
    for index, number in enumerate(values, start=1):
        text = text.replace("{features#" + str(index) + "}", format(number, ".6f"))
    return text.replace("{position}", position)
