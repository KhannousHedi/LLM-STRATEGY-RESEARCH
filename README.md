# LLM strategy backtest

One local LLM decides `BUY`, `SELL`, or `HOLD` on a single stock, and vectorbt simulates the result. Work in progress: the data store is implemented; features, prompts, the model call, simulation, and the CLI are not yet.

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

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Layout

```text
src/            source
tests/          tests
prompts/        prompt templates (tracked)
data/bars/      5-minute day files (local only)
runs/{id}/      one folder per finished backtest (local only)
```
