"""Create one run folder and keep it only when the backtest finishes.

``record`` makes an id, ``YYYYMMDD-HHMMSS-`` plus six hex characters, and
writes ``config.json`` under that folder. The config holds the strategy
name and the parameters dict passed to it, unchanged. ``record`` does not
load the strategy, check coverage, or check timestamps.

``backtest`` loads the named strategy, calls ``record``, then calls the
loader, the strategy's feature set, ``decide``, ``place``, and
``simulate``. It writes the remaining files in the same folder. A row
that is not all finite is not sent. The last bar is not sent. If
``backtest`` raises or exits, the folder is removed, partial files
included.

``root`` defaults to ``runs/``. A test passes another directory.
"""

import json
import math
import secrets
import shutil
from datetime import datetime
from pathlib import Path

from src.data import load_prices
from src.decisions import place, signals
from src.features import build_features
from src.simulate import INIT_CASH, SLIPPAGE, simulate
from src.strategies import load_strategy

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def _run_id():
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3)


def _write_text(path, text):
    path.write_text(text, encoding="utf-8", newline="\n")


def record(symbol, window, freq, strategy, parameters, slippage, cash, root=RUNS_DIR):
    folder = Path(root) / _run_id()
    folder.mkdir(parents=True)
    config = {
        "symbol": symbol,
        "window": window,
        "freq": freq,
        "strategy": strategy,
        "parameters": parameters,
        "slippage": slippage,
        "cash": cash,
    }
    try:
        _write_text(folder / "config.json", json.dumps(config, indent=2) + "\n")
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    return folder


def _finite_values(row, columns):
    values = {}
    for name in columns:
        value = row[name]
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        values[name] = number
    return values


def _write_jsonl(path, rows):
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    _write_text(path, "".join(line + "\n" for line in lines))


def _write_equity_png(portfolio, path):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    try:
        portfolio.value().plot(ax=axis)
        figure.autofmt_xdate()
        figure.tight_layout()
        figure.savefig(path)
    finally:
        plt.close(figure)


def _execute(folder, symbol, start, end, freq, strategy):
    prices = load_prices(symbol, start, end, freq)
    bars = prices[symbol]
    features = build_features(prices, strategy.feature_set, **strategy.feature_parameters)
    rows = features.xs(symbol, level="symbol")
    sessions = bars.index

    placed = []
    logged = []
    held = False
    # The last bar has no next session. Warmup rows are not finite, so they are not sent.
    for as_of in sessions[:-1]:
        values = _finite_values(rows.loc[as_of], rows.columns)
        if values is None:
            continue
        action = strategy.decide(values, "in" if held else "out")
        decision = place(action, as_of, sessions)
        placed.append(decision)
        logged.append(
            {
                "as_of": decision["as_of"].isoformat(),
                "execute_on": decision["execute_on"].isoformat(),
                "action": decision["action"],
                "features": values,
                **strategy.detail(),
            }
        )
        if action == "BUY":
            held = True
        elif action == "SELL":
            held = False

    _write_jsonl(folder / "decisions.jsonl", logged)
    close = bars["close"]
    opens = bars["open"]
    entries, exits = signals(placed, sessions)
    portfolio = simulate(close, opens, entries, exits, freq)
    portfolio.orders.records_readable.to_csv(folder / "orders.csv", index=False)
    equity = portfolio.value()
    equity.name = "equity"
    equity.to_csv(folder / "equity.csv")
    stats = portfolio.stats()
    _write_text(folder / "summary.txt", stats.to_string() + "\n")
    _write_text(folder / "summary.json", stats.to_json(date_format="iso", indent=2) + "\n")
    _write_equity_png(portfolio, folder / "equity.png")


def backtest(symbol, start, end, freq, strategy, parameters, root=RUNS_DIR):
    loaded = load_strategy(strategy, parameters)
    folder = record(
        symbol=symbol,
        window={"start": start, "end": end},
        freq=freq,
        strategy=strategy,
        parameters=parameters,
        slippage=SLIPPAGE,
        cash=INIT_CASH,
        root=root,
    )
    try:
        _execute(folder, symbol, start, end, freq, loaded)
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    return folder
