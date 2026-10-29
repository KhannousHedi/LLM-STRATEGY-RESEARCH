# LLM strategy backtest

One local LLM decides `BUY`, `SELL`, or `HOLD` on a single stock, and vectorbt simulates the result. Daily bars are stored as one year file per symbol. This file is the contract for a run and for adding a feature set or a strategy.

## Invariants

- One rule lives in one file. The next file trusts that result.
- Bars are downloaded once into `data/bars/`. A finished backtest writes only under its own folder in `runs/`.
- The row at time t uses only bars timestamped at or before t. `place` names the next session, and `simulate` fills that session's open.
- The symbol and the calendar date are not inputs, so they cannot reach the model.
- A strategy returns `BUY`, `SELL`, or `HOLD`. It does not download, resample, call vectorbt, or assign `execute_on`. The run loop owns the position.
- A new feature set or strategy is a new module under the existing base class. A change to what `FeatureSet` or `Strategy` guarantees changes that base class and this file.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Data store

Daily bars are stored as one UTC year per file at `data/bars/{SYMBOL}/{YYYY}.csv` (local only, not tracked). One row is one session.

```python
from src.download import download_bars
from src.data import load_prices

download_bars(["SPY"], "2024-01-01", "2024-12-31")
prices = load_prices(["SPY"], "2024-01-01", "2024-12-31", "1D")
```

`download_bars` asks Yahoo for `1d` bars, unadjusted, regular session. A missing year is always fetched from January 1 through December 31, even when the requested range is shorter. A finished year that already has a file is left as it is. The current UTC year is replaced through the last finished session. Today and later dates are not requested. `load_prices` reads the store and does not call Yahoo. It warns about missing weekdays and returns the rows inside the requested range. `1D` is those daily rows. Weekly and monthly resample them: open is the first, high is the max, low is the min, close is the last, and volume is the sum.

## Features

`build_features` loops over symbols and stacks one dataframe. The index levels are `symbol` and `timestamp`, and `timestamp` is datetimes. A strategy names a feature set and passes `feature_parameters` into this call. Several strategies can name the same set and pass different parameters. The set does not list those strategies.

Each set declares `columns`, the names it returns, and `descriptions`, one definition per column. Ask the CLI what a set returns before you use it:

```powershell
.\.venv\Scripts\python.exe -m src.cli features momentum_volatility_volume
```

```text
r1: one-bar close-to-close return
r5: five-bar close-to-close return
r20: twenty-bar close-to-close return
dist_ma: distance of the close from its moving average, as a fraction
vol: sample standard deviation of recent one-bar returns
vol_ratio: last volume divided by its recent average volume
```

```python
from src.features import build_features

features = build_features(
    prices, "momentum_volatility_volume", ma_window=20, vol_window=20, volume_window=20
)
```

`void` returns the bar timestamps and no columns. A strategy that does not name a set gets it. `momentum_volatility_volume` is the set the model decides from. The row at time t uses only bars timestamped at or before t.

## Parameters and prompts

A strategy is built from one parameters dict. There is no separate prompt file type: for `llm`, the prompt text is the `prompt` value in that dict. Parameter files live in `params/`. `params/default.json` is the example for `llm`:

```json
{
  "feature_set": "momentum_volatility_volume",
  "feature_parameters": {"ma_window": 20, "vol_window": 20, "volume_window": 20},
  "model": "qwen3:1.7b",
  "prompt": "Decide ... one-bar return {r1}; ... Position: {position} ..."
}
```

The format is one JSON object. `feature_set` and `feature_parameters` are read by the strategy base. Every other key belongs to the strategy: `llm` reads `prompt` and `model`, `dist_ma` reads `buy_above`. The dict is stored unchanged in the run's `config.json`.

### Writing a prompt

1. Pick a feature set and list its columns: `python -m src.cli features momentum_volatility_volume`.
2. Copy `params/default.json` and edit the `prompt` text.
3. Write a description, then the slot it describes: `one-bar return {r1}`. The slot is replaced by that column's number, at six decimal places. The model never sees the column name.
4. Use only the columns you want. A prompt does not have to use every column, and it does not have to use `{position}`.
5. `{position}` is filled with `in` or `out`.

Building the strategy checks each `{key}` against the set's columns. An unknown key fails before a run folder is made, and the error lists what the set returns. Nothing checks the description in front of a slot, so keep it next to the slot it describes. If you change `feature_parameters`, check the descriptions still match: `ma_window` changes what `dist_ma` measures.

The symbol and the calendar date are not inputs, so they cannot reach the model.

## Strategies

A strategy is a `Strategy` subclass in `src/strategies/{name}.py`, built from one parameters dict. The base reads `feature_set` (default `void`) and `feature_parameters` from that dict, and a subclass reads its own keys. `decide(row, position)` gets every column of that feature set, in the set's order, as finite numbers, plus `in` or `out`, and returns `BUY`, `SELL`, or `HOLD`. The run does not know which strategy it is running.

- `llm` reads `prompt` (required — a run without it raises `ValueError`) and `model` (default `qwen3:1.7b`) from the same parameters dict as `feature_set` and `feature_parameters`. Its class defaults are the `momentum_volatility_volume` set with windows of 20. It fills each `{key}` slot with that column's value and `{position}` with the position. Each decision line keeps the filled `text`.
- `dist_ma` buys while out when `dist_ma` is above `buy_above` (default 0), and sells while in when it is below. Its windows default to 20.

## Model

`decide` sends that text to Ollama: model `qwen3:1.7b` at `http://127.0.0.1:11434`, temperature 0, thinking off. The reply has to be a JSON object with a string `analysis` and a `decision` of `BUY`, `SELL`, or `HOLD`. Ollama writes the keys in alphabetical order, whatever order the schema or the prompt gives, so the model writes its `analysis` before it picks the `decision`. A renamed key must keep the reasoning key sorting first. A prompt's reply line lists them in that order. Anything else is sent once more, then becomes `HOLD`. If Ollama is down, the call exits.

```python
from src.llm import decide

action = decide(text)
```

## Decisions

`place` names the next session in the bar index. `BUY` writes the entry boolean on that session, `SELL` writes the exit boolean, and `HOLD` writes neither. The position is not read. A `SELL` while flat still writes the exit boolean. `signals` stamps the booleans on the session index.

```python
from src.decisions import place, signals

record = place("BUY", sessions[0], sessions)
entries, exits = signals([record], sessions)
```

## Simulation

`simulate` passes those booleans to vectorbt unchanged. The fill price is that session's open. Cash is 100,000, slippage is 0.0005, and fees are 0.

```python
from src.simulate import simulate

pf = simulate(close, opens, entries, exits, "1D")
```

## Runs

A finished backtest writes only under `runs/YYYYMMDD-HHMMSS-` plus six hex characters. The folder holds `config.json`, `decisions.jsonl`, `orders.csv`, `equity.csv`, `summary.txt`, `summary.json`, and `equity.png`. `config.json` records the strategy name and the parameters dict passed to it. A `decisions.jsonl` line has `as_of`, `execute_on`, `action`, `features`, and whatever the strategy's `detail()` adds. `summary.txt` is the vectorbt stats table, including the Sharpe ratio, win rate, and max drawdown, and `summary.json` is the same stats as JSON. A run that raises or exits is removed. Tests write outside `runs/`.

## CLI

```powershell
.\.venv\Scripts\python.exe -m src.cli download SPY --start 2024-01-01 --end 2024-12-31
.\.venv\Scripts\python.exe -m src.cli features momentum_volatility_volume
.\.venv\Scripts\python.exe -m src.cli backtest --symbol SPY --start 2024-01-01 --end 2024-12-31 --parameters params/default.json
.\.venv\Scripts\python.exe -m src.cli backtest --symbol SPY --start 2024-01-01 --end 2024-12-31 --parameters params/default.json --param model=qwen3:4b-instruct
.\.venv\Scripts\python.exe -m src.cli backtest --symbol SPY --start 2024-01-01 --end 2024-12-31 --strategy dist_ma --param buy_above=0
```

`download` takes one or more symbols. `features` takes one feature set name and prints its columns with their definitions. `backtest` takes one `--symbol`, `--start`, `--end`, `--freq` (default `1D`), and `--strategy` (default `llm`). Because the default `llm` strategy requires a `prompt`, a bare `llm` run without parameters fails; pass `--parameters params/default.json` (or `--param prompt=...`). The parameters dict starts from `--parameters PATH`, a JSON object, or `{}`. Each repeated `--param key=value` replaces that key: a value that parses as JSON is that value, so `20` is an integer and an object stays an object, and anything else is a string. It prints the run folder when the backtest finishes. It does not download bars.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Layout

```text
src/            source
src/features/   one feature computation per set; void is timestamps only
src/strategies/ one strategy class per module; the run takes any of them
tests/          tests
params/         example strategy parameter files (tracked)
data/bars/      daily year files (local only)
runs/{id}/      one folder per finished backtest (local only)
```

## Adding a feature set

Create `src/features/{name}.py`. The name is an identifier and does not start with `_`. Define one `FeatureSet` subclass in that module.

```python
import pandas as pd

from src.features import FeatureSet


class Example(FeatureSet):
    columns = ("r1",)
    descriptions = {"r1": "one-bar close-to-close return"}

    def compute(self, bars, window):
        close = bars["close"]
        return pd.DataFrame({"r1": close.pct_change(window)}, index=bars.index)
```

- `columns` is a tuple of unique identifiers, in the order `compute` returns them. No column may be named `position`.
- `descriptions` defines every column in one line. `python -m src.cli features {name}` prints these lines. Say what the number means.
- `compute` takes one symbol's bars and the keyword arguments the set adds. It returns a dataframe indexed by that symbol's datetimes, with exactly `columns`. `build_features` checks this on every call.
- Row t uses only bars at or before t. The next session is `place` in `src/decisions.py`.
- A set with no values sets `columns = ()` and `descriptions = {}` and returns `pd.DataFrame(index=bars.index)`. `void` is that set.

Check it:

```powershell
.\.venv\Scripts\python.exe -m src.cli features example
.\.venv\Scripts\python.exe -m pytest tests/test_features.py
```

## Adding a strategy

Create `src/strategies/{name}.py`. The name is an identifier and does not start with `_`. Define one `Strategy` subclass in that module.

```python
from src.strategies import Strategy


class Example(Strategy):
    feature_set = "momentum_volatility_volume"
    feature_parameters = {"ma_window": 20, "vol_window": 20, "volume_window": 20}

    def __init__(self, parameters):
        super().__init__(parameters)
        self.threshold = parameters.get("threshold", 0)

    def decide(self, row, position):
        if position == "out" and row["dist_ma"] > self.threshold:
            return "BUY"
        if position == "in" and row["dist_ma"] < self.threshold:
            return "SELL"
        return "HOLD"
```

- Set `feature_set` and `feature_parameters` on the class when the strategy needs a set other than `void`, or parameters other than none. The base reads those two keys from the parameters dict and keeps the class values when a key is absent.
- Read every other key by name in `__init__`, after `super().__init__(parameters)`, and store it on `self`. A key the strategy must have fails there, as `llm` does without a `prompt`. Building happens before a run folder is made.
- `decide(row, position)` gets every column of the feature set as finite numbers, by name, and `in` or `out`. It returns `BUY`, `SELL`, or `HOLD`. The run owns the position. A strategy may keep earlier rows on `self`.
- A strategy does not get the feature frame, the portfolio, or a later bar. It does not download, resample, call vectorbt, or assign `execute_on`.
- `detail()` is optional. Return a dict and it is merged into that bar's `decisions.jsonl` line.

Pass parameters as one JSON object with `--parameters`, or override keys with repeated `--param key=value`. See Parameters and prompts for the file format.

```powershell
.\.venv\Scripts\python.exe -m src.cli backtest --symbol SPY --start 2024-01-01 --end 2024-12-31 --strategy example --param threshold=0.01
```

Add a test in `tests/test_strategies.py`: build it with `load_strategy` and assert one `decide` result. The test does not call the model and does not write under `runs/`.
