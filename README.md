# LLM strategy backtest

One local LLM decides `BUY`, `SELL`, or `HOLD` on a single stock, and vectorbt simulates the result. Work in progress: the data store and feature sets are implemented; prompts, the model call, simulation, and the CLI are not yet.

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

`build_features` loops over symbols and stacks one dataframe. The index levels are `symbol` and `timestamp`, and `timestamp` is datetimes. A phase 4 prompt is JSON with `feature_set` and `parameters`. Several prompts can name the same set and pass different parameters. The set does not list those prompts.

```python
from src.features import build_features

features = build_features(
    prices, "default", ma_window=20, vol_window=20, volume_window=20
)
```

`default` returns `r1`, `r5`, `r20`, `dist_ma`, `vol`, and `vol_ratio`, in that order. The row at time t uses only bars timestamped at or before t. A new set is a `FeatureSet` subclass. It implements `compute` and may add its own arguments. It can inherit `columns` or `compute` from another set.

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
