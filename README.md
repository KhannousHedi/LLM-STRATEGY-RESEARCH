# LLM strategy backtest

This runs one local model, or an ordinary rule, on one symbol's daily bars. The position is long or flat. A finished run writes its own folder. Objective 1 is one study that uses this engine. Its write-up is published separately and is not in this repository.

Copy `params/default.json` to start a new experiment. Commands below are `python -m src.cli`. On Windows, after the virtualenv step, that interpreter is `.\.venv\Scripts\python.exe`. Later examples use that path. The same commands work as `python -m ...` once the virtualenv is active.

## Scope

- One symbol per run. The bars are daily, or those daily bars resampled to a week or a month. The symbol is whatever Yahoo serves as a daily series, a stock or a fund.
- The book is long or flat and fully invested. The order fills at the next session's open. Slippage defaults to 0.05% of price. The transaction cost defaults to 0.10% of order value. Cash starts at 100,000.
- `decide` receives that day's feature row and whether the position is `in` or `out`. The symbol, the date, and the raw price stay out of the prompt. Showing the model the ticker means changing the run loop and this contract.
- Shorts, position sizing, and a series that is not in the bar file sit outside this runner. The `llm` strategy sends one filled prompt per day. A strategy that needs its own recent rows can store them on itself. News needs a new input to the runner.
- The model call is Ollama on this machine. Temperature is 0 and thinking is off. The model name is a field in the parameters file. Another host means a change to `src/llm.py`.

## Setup

Tested with Python 3.12.

Install [Ollama](https://ollama.com), then pull the example model:

```text
ollama pull qwen3:1.7b
```

Ollama has to be running before an `llm` backtest. If it is down, the run exits. A rule strategy does not call it.

```text
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.cli download SPY --start 2024-01-01 --end 2024-12-31
.\.venv\Scripts\python.exe -m src.cli backtest --symbol SPY --start 2024-01-01 --end 2024-12-31 --parameters params/default.json
```

macOS and Linux:

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m src.cli download SPY --start 2024-01-01 --end 2024-12-31
.venv/bin/python -m src.cli backtest --symbol SPY --start 2024-01-01 --end 2024-12-31 --parameters params/default.json
```

The backtest prints a folder under `runs/`. Open `summary.txt` and `equity.png` there. Bars land in `data/bars/` and are not committed. The run folder is not committed either.

## Studies

Objective 1 asked `qwen3:1.7b` to trade from the `classic_indicators` set. Those prompts and the rule parameters are in `params/objective1/`. `scripts/objective1.py` repeats that study: one parameter file, on the symbols and periods named in the script.

A new experiment copies `params/default.json`, which uses `momentum_volatility_volume`, and runs `python -m src.cli backtest`.

`comparisons/`, `data/`, and `docs/` are gitignored. The comparison tables, the bar files, and the research notes stay on the machine that ran them.

The sections below are the contract for a run and for adding a feature set or a strategy.

## Invariants

- One rule lives in one file. The next file trusts that result.
- Bars are downloaded once into `data/bars/`. A finished backtest writes only under its own folder in `runs/`.
- The row at time t uses only bars timestamped at or before t. `place` names the next session, and `simulate` fills that session's open.
- The symbol and the calendar date are not inputs, so they cannot reach the model.
- A strategy returns `BUY`, `SELL`, or `HOLD`. It does not download, resample, call vectorbt, or assign `execute_on`. The run loop owns the position.
- A new feature set or strategy is a new module under the existing base class. A change to what `FeatureSet` or `Strategy` guarantees changes that base class and this file.

## Data store

Daily bars are stored as one UTC year per file at `data/bars/{SYMBOL}/{YYYY}.csv` (local only, not tracked). One row is one session.

```python
from src.download import download_bars
from src.data import load_prices

download_bars(["SPY"], "2024-01-01", "2024-12-31")
prices = load_prices(["SPY"], "2024-01-01", "2024-12-31", "1D")
```

`download_bars` asks Yahoo for `1d` bars, unadjusted, regular session. A missing year is always fetched from January 1 through December 31, even when the requested range is shorter. A finished year that already has a file is left as it is. The current UTC year is replaced through the last finished session. Today and later dates are not requested. `load_prices` reads the store and does not call Yahoo. It warns once per symbol about missing weekdays, with a truncated list of the dates, and returns the rows inside the requested range. `1D` is those daily rows. Weekly and monthly resample them: open is the first, high is the max, low is the min, close is the last, and volume is the sum.

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

`classic_indicators` returns `r1`, `r5`, `r20`, `ma50_gap`, `ma200_gap`, `high52_gap`, `rsi14`, `vol20`, `volume_ratio`, `open_gap`, `day_range`, and `close_location`. Its windows are keyword parameters with textbook defaults: `r1_window` 1, `r5_window` 5, `r20_window` 20, `ma50_window` 50, `ma200_window` 200, `high52_window` 252, `rsi14_window` 14, `vol20_window` 20, and `volume_window` 20. `open_gap`, `day_range`, and `close_location` have no window. The printed definition of each windowed column names that parameter. Objective 1's parameter files leave these windows at the defaults. Any other run may pass different windows.

`void` returns the bar timestamps and no columns. A strategy that does not name a set gets it. `momentum_volatility_volume` is the set in `params/default.json`. Objective 1 uses `classic_indicators`. The row at time t uses only bars timestamped at or before t.

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
3. Write a description, then the slot it describes: `one-bar return {r1}`. A slot is `{column}` (six decimal places) or `{column:spec}` with any Python format spec. The model never sees the column name.
4. Use only the columns you want. A prompt does not have to use every column, and it does not have to use `{position}`.
5. `{position}` is filled with `in` or `out`. It takes no format spec.

A spec on a fraction column needs `%`, because the stored number is a fraction (`0.021` with `{r1:+.2%}` is `+2.10%`). Nothing checks the scale.

| Columns | Slot | Rendered |
|---|---|---|
| `r1`, `r5`, `r20`, `ma50_gap`, `ma200_gap`, `high52_gap`, `open_gap` | `{r1:+.2%}` | `-1.23%`, `+2.10%`, zero is `+0.00%` |
| `vol20`, `day_range` | `{vol20:.2%}` | `18.20%`, `1.10%` |
| `rsi14` | `{rsi14:.1f}` | `71.4` |
| `close_location` | `{close_location:.2f}` | `0.82` |
| `volume_ratio` | `{volume_ratio:.2f}x` | `1.45x` |

The `x` after `{volume_ratio:.2f}` is a literal character. Building the strategy checks each `{key}` against the set's columns and dry-runs each spec. An unknown key or a bad spec fails before a run folder is made, and the unknown-key error lists what the set returns. Nothing checks the description in front of a slot, so keep it next to the slot it describes. If you change `feature_parameters`, check the descriptions still match: `ma_window` changes what `dist_ma` measures. `classic_indicators` descriptions name their parameters, so changing a window does not make those lines stale.

The symbol and the calendar date are not inputs, so they cannot reach the model.

## Strategies

A strategy is a `Strategy` subclass in `src/strategies/{name}.py`, built from one parameters dict. The base reads `feature_set` (default `void`) and `feature_parameters` from that dict, and a subclass reads its own keys. `decide(row, position)` gets every column of that feature set, in the set's order, as finite numbers, plus `in` or `out`, and returns `BUY`, `SELL`, or `HOLD`. The run does not know which strategy it is running.

- `llm` reads `prompt` (required — a run without it raises `ValueError`) and `model` (default `qwen3:1.7b`) from the same parameters dict as `feature_set` and `feature_parameters`. Its class defaults are the `momentum_volatility_volume` set with windows of 20. It fills each `{key}` slot with that column's value and `{position}` with the position. Each decision line keeps the filled `text`.
- `dist_ma` buys while out when `dist_ma` is above `buy_above` (default 0), and sells while in when it is below. Its windows default to 20.
- `buy_and_hold` uses the `void` feature set, which returns no feature columns, and no parameter keys. The first decision is `BUY`. Every later decision is `HOLD`.
- `trend_ma200` reads `ma200_gap` from the `classic_indicators` feature set. The feature set computes that column from the feature parameters, including `ma200_window`. The strategy adds no parameter keys. While out, it buys when `ma200_gap > 0`. While in, it sells when `ma200_gap < 0`. A gap of 0 holds. The zero level is a constant.
- `rsi_reversion` reads `rsi14` from the `classic_indicators` feature set. The feature set computes that column from the feature parameters, including `rsi14_window`. The strategy adds no parameter keys. While out, it buys when `rsi14 < 30`. While in, it sells when `rsi14 > 70`. A reading of 30 or 70 holds. The 30 and 70 levels are constants.
- `random` uses the `void` feature set, which returns no feature columns. While out, it buys with probability `p_buy`. While in, it sells with probability `p_sell`. The same `seed` and the same position sequence repeat the same actions. It reads `p_buy`, `p_sell`, and `seed`.

## Model

`decide` sends that text to Ollama: model `qwen3:1.7b` at `http://127.0.0.1:11434`, temperature 0, thinking off. A later pull of that model name can change the weights, so a rerun can differ. The run does not record the model digest. The reply has to be a JSON object with a string `analysis` and a `decision` of `BUY`, `SELL`, or `HOLD`. Ollama writes the keys in alphabetical order, whatever order the schema or the prompt gives, so the model writes its `analysis` before it picks the `decision`. A renamed key must keep the reasoning key sorting first. A prompt's reply line lists them in that order. Anything else is sent once more, then becomes `HOLD`. If Ollama is down, the call exits.

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

`simulate` passes those booleans to vectorbt unchanged. The fill price is that session's open. Cash is 100,000. Slippage defaults to 0.0005. The transaction cost defaults to 0.001, a fraction of order value, and is passed to vectorbt as fees.

```python
from src.simulate import simulate

pf = simulate(close, opens, entries, exits, "1D")
```

## Runs

A finished backtest writes only under `runs/YYYYMMDD-HHMMSS-` plus six hex characters. The folder holds `config.json`, `decisions.jsonl`, `orders.csv`, `equity.csv`, `summary.txt`, `summary.json`, and `equity.png`. A backtest loads bars from `--start` minus the warmup offset for features only. Decisions and simulation begin at `--start`. `config.json` records the strategy name, the parameters dict passed to it, `warmup`, and `transaction_cost` immediately after `slippage`. Non-finite features in the window raise one warning. A `decisions.jsonl` line has `as_of`, `execute_on`, `action`, `features`, and whatever the strategy's `detail()` adds. `summary.txt` is the vectorbt stats table, including the Sharpe ratio, win rate, and max drawdown, plus `time_in_market` and `avg_holding_period`. `summary.json` is the same stats plus those two fields. `Total Trades` remains the trade count. `avg_holding_period` is null when there are no trades. A run that raises or exits is removed. Tests write outside `runs/`.

## CLI

```powershell
.\.venv\Scripts\python.exe -m src.cli download SPY --start 2024-01-01 --end 2024-12-31
.\.venv\Scripts\python.exe -m src.cli features momentum_volatility_volume
.\.venv\Scripts\python.exe -m src.cli backtest --symbol SPY --start 2024-01-01 --end 2024-12-31 --parameters params/default.json
.\.venv\Scripts\python.exe -m src.cli backtest --symbol SPY --start 2024-01-01 --end 2024-12-31 --parameters params/default.json --param model=qwen3:4b-instruct
.\.venv\Scripts\python.exe -m src.cli backtest --symbol SPY --start 2024-01-01 --end 2024-12-31 --strategy dist_ma --param buy_above=0
```

`download` takes one or more symbols. `features` takes one feature set name and prints its columns with their definitions. `backtest` takes one `--symbol`, `--start`, `--end`, `--freq` (default `1D`), `--strategy` (default `llm`), `--warmup` (a pandas offset such as `400D` or `13MS`, default `400D`), `--slippage` (default `0.0005`), and `--transaction-cost` (a fraction of order value, default `0.001`). Because the default `llm` strategy requires a `prompt`, a bare `llm` run without parameters fails; pass `--parameters params/default.json` (or `--param prompt=...`). The parameters dict starts from `--parameters PATH`, a JSON object, or `{}`. Each repeated `--param key=value` replaces that key: a value that parses as JSON is that value, so `20` is an integer and an object stays an object, and anything else is a string. It prints the run folder when the backtest finishes. It does not download bars.

## Objective 1 runner

This section repeats the Objective 1 study. A new experiment uses the setup above and `params/default.json`.

`scripts/objective1.py` runs one parameter file on SPY, QQQ, TLT, and GLD for one named period and writes `comparisons/objective1/{period}.csv` and `.md`. It is not a `src.cli` subcommand. The periods are fixed: `validation` is 2023-01-01 to 2024-12-31, `test` is 2025-01-01 to 2026-09-30, and `stress` is 2019-01-01 to 2022-12-31. Download the bars first. The `llm` strategy calls Ollama, so that process has to be up. The other strategies do not.

```powershell
.\.venv\Scripts\python.exe -m src.cli download SPY QQQ TLT GLD --start 2018-01-01 --end 2026-09-30
.\.venv\Scripts\python.exe -m scripts.objective1 --period validation --strategy buy_and_hold --parameters params/objective1/buy_and_hold.json
.\.venv\Scripts\python.exe -m scripts.objective1 --period validation --strategy trend_ma200 --parameters params/objective1/trend_ma200.json
.\.venv\Scripts\python.exe -m scripts.objective1 --period validation --strategy rsi_reversion --parameters params/objective1/rsi_reversion.json
.\.venv\Scripts\python.exe -m scripts.objective1 --period validation --strategy llm --parameters params/objective1/llm_v0.json
.\.venv\Scripts\python.exe -m scripts.objective1 --period validation --diagnostics
```

Finish one command before starting the next. They share the comparison file. A later run with the same strategy name replaces that strategy's rows. `llm_v0.json`, `llm_v1.json`, `llm_v2.json`, and `llm_v3.json` are all strategy `llm`, so a later prompt file replaces the `llm` rows. `--diagnostics` reads the table and the `llm` run folders already written. It does not call the model. It adds the matched random seeds, agreement with the two rules, decision counts, and a depth-2 tree. `params/objective1/random.json` is not a run. `comparisons/` and `runs/` stay on this machine. Research notes under `docs/` are local too.

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
scripts/objective1.py       one period, four symbols, one comparison table
scripts/test_objective1.py  tests for that script
params/         strategy parameter files (tracked)
data/bars/      daily year files (local only)
runs/{id}/      one folder per finished backtest (local only)
comparisons/objective1/     Objective 1 comparison tables (local only)
docs/           research notes (local only)
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
- A set may take window keywords with defaults, as `classic_indicators` does.
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
