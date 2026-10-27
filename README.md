# LLM strategy backtest

One local LLM decides `BUY`, `SELL`, or `HOLD` on a single stock, and vectorbt simulates the result. The data store, feature sets, prompts, the model call, the simulation, and the CLI are implemented. A real backtest waits for its own request.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Data store

5-minute bars are stored as one UTC day per file at `data/bars/{SYMBOL}/{YYYY-MM-DD}.csv` (local only, not tracked).

```python
from src.download import download_bars
from src.data import load_prices

download_bars(["SPY"], "2026-09-01", "2026-10-23")
prices = load_prices(["SPY"], "2026-09-01", "2026-10-23", "1D")
```

`download_bars` only asks Yahoo for weekdays that have no file yet, in chunks of at most 60 days. Yahoo serves 5-minute bars for roughly the last 60 days, so older days stay missing. `load_prices` reads the store, warns about missing weekdays, and resamples to the requested frequency.

## Features

`build_features` loops over symbols and stacks one dataframe. The index levels are `symbol` and `timestamp`, and `timestamp` is datetimes. A prompt names a feature set and passes `parameters` into this call. Several prompts can name the same set and pass different parameters. The set does not list those prompts.

```python
from src.features import build_features

features = build_features(
    prices, "default", ma_window=20, vol_window=20, volume_window=20
)
```

`default` returns `r1`, `r5`, `r20`, `dist_ma`, `vol`, and `vol_ratio`, in that order. The row at time t uses only bars timestamped at or before t. A new set is a `FeatureSet` subclass. It implements `compute` and may add its own arguments. It can inherit `columns` or `compute` from another set.

## Prompts

A prompt is a JSON file in `prompts/` with `feature_set`, `parameters`, and `wording`. `default` names the `default` feature set and passes 20 for `ma_window`, `vol_window`, and `volume_window`. The wording leaves `{features#1}` through `{features#6}` and `{position}`.

```python
from src.prompt import load_template, render

template = load_template("default")
features = build_features(prices, template.feature_set, **template.parameters)
text = render(template, row, "out")
```

`render` fills a row only when its keys are that set's columns and every value is a finite number. Position is `in` or `out`. Each number is written with six decimal places. The symbol and the calendar date are not inputs.

## Model

`decide` sends that text to Ollama: model `qwen3:4b-instruct` at `http://127.0.0.1:11434`, temperature 0, thinking off. The reply has to be a JSON object with `action` of `BUY`, `SELL`, or `HOLD` and a string `reason`. Anything else is sent once more, then becomes `HOLD`. If Ollama is down, the call exits.

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

A finished backtest writes only under `runs/YYYYMMDD-HHMMSS-` plus six hex characters. The folder holds `config.json`, `prompt.txt`, `decisions.jsonl`, `orders.csv`, `equity.csv`, `summary.txt`, and `equity.png`. `config.json` records the prompt name. `prompt.txt` is a copy of that prompt file. Filled session text goes to `decisions.jsonl`. `summary.txt` is the vectorbt stats for the portfolio, including the Sharpe ratio, win rate, and max drawdown. A run that raises or exits is removed. Tests write outside `runs/`.

## CLI

```powershell
.\.venv\Scripts\python.exe -m src.cli download SPY --start 2026-09-01 --end 2026-10-23
.\.venv\Scripts\python.exe -m src.cli backtest --symbol SPY --start 2026-09-01 --end 2026-10-23
```

`download` takes one or more symbols. `backtest` takes one `--symbol`, `--start`, `--end`, `--model` (default `qwen3:4b-instruct`), `--freq` (default `1D`), and `--prompt` (default `default`). It prints the run folder when the backtest finishes. It does not download bars.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Layout

```text
src/            source
src/features/   one feature computation per set; prompts name a set
tests/          tests
prompts/        prompt templates (tracked)
data/bars/      5-minute day files (local only)
runs/{id}/      one folder per finished backtest (local only)
```
