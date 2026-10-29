"""Fill a prompt by feature name and ask the model.

The parameters dict is the one dict for this strategy. The base reads
``feature_set`` and ``feature_parameters`` from it; this class also reads
``prompt`` (required) and ``model`` (absent, ``qwen3:1.7b``) by
name. There is no prompt file of its own: a file such as
``params/default.json`` is a parameters file passed with
``--parameters``, and its ``prompt`` value is the text.

A slot is ``{key}`` or ``{key:spec}``, where key is one column of the
feature set, or ``{position}``. ``{key}`` renders at six decimal places.
``{key:spec}`` renders with that Python format spec, and only the spec
string is passed to ``format``. ``{position}`` is ``in`` or ``out`` and
takes no spec. Run `python -m src.cli features {feature_set}` to see
the keys and what each one means. Building the strategy checks every
key it uses against that set and dry-runs every spec, so an unknown key
or a bad spec fails before a run folder is made. A prompt does not have
to use every column. It does not have to use ``{position}`` either.
Ordinary braces in the prompt, including the reply-schema JSON, are
left as they are.
"""

import re

from src.features import load_feature_set
from src.llm import MODEL, decide as ask
from src.strategies import Strategy

_SLOT = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)(?::([^{}]*))?\}")


def _fill(prompt, row, position):
    if position not in ("in", "out"):
        raise ValueError("position must be in or out")

    def replace(match):
        key = match.group(1)
        spec = match.group(2)
        if key == "position":
            if spec is not None:
                raise ValueError(f"llm prompt slot {{position:{spec}}} takes no format spec")
            return position
        if key not in row:
            available = ", ".join(row) or "none"
            raise ValueError(f"prompt uses {{{key}}}, which is not a feature. The row has: {available}")
        if spec is None:
            return format(row[key], ".6f")
        return format(row[key], spec)

    return _SLOT.sub(replace, prompt)


class Llm(Strategy):
    feature_set = "momentum_volatility_volume"
    feature_parameters = {"ma_window": 20, "vol_window": 20, "volume_window": 20}

    def __init__(self, parameters):
        super().__init__(parameters)
        if "prompt" not in parameters:
            raise ValueError(
                "llm requires a prompt: pass \"prompt\" in the parameters "
                "dict, e.g. --parameters params/default.json"
            )
        if not isinstance(parameters["prompt"], str):
            raise ValueError("llm prompt must be a string")
        self.prompt = parameters["prompt"]
        self.model = parameters.get("model", MODEL)
        self.text = None
        columns = load_feature_set(self.feature_set).columns
        matches = list(_SLOT.finditer(self.prompt))
        unknown = sorted(
            {match.group(1) for match in matches if match.group(1) != "position" and match.group(1) not in columns}
        )
        if unknown:
            used = ", ".join("{" + key + "}" for key in unknown)
            available = ", ".join(columns) or "none"
            raise ValueError(
                f"llm prompt uses {used}, which feature_set {self.feature_set!r} does not return. "
                f"It returns: {available}"
            )
        for match in matches:
            key = match.group(1)
            spec = match.group(2)
            if key == "position":
                if spec is not None:
                    raise ValueError(f"llm prompt slot {{position:{spec}}} takes no format spec")
                continue
            if spec is None:
                continue
            try:
                format(0.5, spec)
            except (ValueError, TypeError) as error:
                raise ValueError(f"llm prompt slot {{{key}:{spec}}} is not a valid format spec") from error

    def decide(self, row, position):
        self.text = _fill(self.prompt, row, position)
        return ask(self.text, model=self.model)

    def detail(self):
        return {"text": self.text}
