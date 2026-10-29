"""Create one run folder and keep it only when the backtest finishes.

``record`` makes an id, ``YYYYMMDD-HHMMSS-`` plus six hex characters, and
writes ``config.json`` under that folder. The config holds the strategy
name, the parameters dict passed to it, unchanged, the warmup offset,
the slippage, and the transaction cost. ``warmup`` defaults to
``WARMUP``. ``record`` does not load the strategy, check coverage, or
check timestamps.

``backtest`` resolves ``warmup`` to ``WARMUP`` when it is omitted,
``slippage`` to ``SLIPPAGE`` when it is omitted, and ``transaction_cost``
to ``TRANSACTION_COST`` when it is omitted, loads the named strategy, and
calls ``record``. ``_execute`` subtracts that
offset from ``start`` and loads bars from the earlier date through
``end``. A string that does not parse, or an offset that moves the start
forward, raises ``ValueError`` and the folder is removed. Features are
computed on the loaded bars. Sessions before ``start`` are dropped.
``decide``, ``place``, and ``simulate`` see only the requested window,
so cash starts at ``start``. A window session whose features are not
all finite is not sent. One warning lists those dates and the columns.
The last bar is not sent. If ``backtest`` raises or exits, the folder
is removed, partial files included.

``root`` defaults to ``runs/``. A test passes another directory.
"""

import json
import math
import secrets
import shutil
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
from pandas.tseries.frequencies import to_offset

from src.data import as_date, load_prices
from src.decisions import place, signals
from src.features import build_features
from src.simulate import INIT_CASH, SLIPPAGE, TRANSACTION_COST, simulate
from src.strategies import load_strategy

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
WARMUP = "400D"
_MAX_LISTED_SKIPPED = 5


def _run_id():
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3)


def _write_text(path, text):
    path.write_text(text, encoding="utf-8", newline="\n")


def record(symbol, window, freq, strategy, parameters, slippage, transaction_cost, cash, root=RUNS_DIR, warmup=WARMUP):
    folder = Path(root) / _run_id()
    folder.mkdir(parents=True)
    config = {
        "symbol": symbol,
        "window": window,
        "warmup": warmup,
        "freq": freq,
        "strategy": strategy,
        "parameters": parameters,
        "slippage": slippage,
        "transaction_cost": transaction_cost,
        "cash": cash,
    }
    try:
        _write_text(folder / "config.json", json.dumps(config, indent=2) + "\n")
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    return folder


def _warm_start(start, warmup):
    try:
        offset = to_offset(warmup)
    except TypeError as error:
        raise ValueError(f"warmup must be a pandas offset, got {warmup!r}") from error
    if offset is None:
        raise ValueError(f"warmup must be a pandas offset, got {warmup!r}")
    warm_start = as_date(pd.Timestamp(start) - offset)
    if warm_start > as_date(start):
        raise ValueError(f"warmup must not move the start forward, got {warmup!r}")
    return warm_start


def _finite_values(row, columns):
    values = {}
    bad_columns = []
    for name in columns:
        value = row[name]
        kind = getattr(getattr(value, "dtype", None), "kind", None)
        if isinstance(value, bool) or kind == "b":
            bad_columns.append(name)
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            bad_columns.append(name)
            continue
        if not math.isfinite(number):
            bad_columns.append(name)
            continue
        values[name] = number
    if bad_columns:
        return None, bad_columns
    return values, bad_columns


def _feature_warning(symbol, skipped, total, columns, warmup):
    count = len(skipped)
    noun = "session" if total == 1 else "sessions"
    shown = skipped[:_MAX_LISTED_SKIPPED]
    listed = ", ".join(as_date(day).isoformat() for day in shown)
    if count > _MAX_LISTED_SKIPPED:
        listed = f"{listed}, and {count - _MAX_LISTED_SKIPPED} more"
    names = ", ".join(columns)
    return (
        f"{symbol}: {count} of {total} {noun} skipped, features not finite on {listed} ({names}). "
        f"Increase warmup (now {warmup})."
    )


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


def _execute(folder, symbol, start, end, freq, strategy, warmup, slippage, transaction_cost):
    warm_start = _warm_start(start, warmup)
    prices = load_prices(symbol, warm_start, end, freq)
    features = build_features(prices, strategy.feature_set, **strategy.feature_parameters)
    bars = prices[symbol]
    keep = bars.index.normalize() >= pd.Timestamp(as_date(start), tz="UTC")
    bars = bars[keep]
    rows = features.xs(symbol, level="symbol").loc[bars.index]
    sessions = bars.index

    placed = []
    logged = []
    held = False
    skipped = []
    bad_columns = set()
    # The last bar has no next session.
    for as_of in sessions[:-1]:
        values, missing = _finite_values(rows.loc[as_of], rows.columns)
        if values is None:
            skipped.append(as_of)
            bad_columns.update(missing)
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

    if skipped:
        ordered = [name for name in rows.columns if name in bad_columns]
        warnings.warn(
            _feature_warning(symbol, skipped, len(sessions), ordered, warmup),
            stacklevel=3,
        )

    _write_jsonl(folder / "decisions.jsonl", logged)
    close = bars["close"]
    opens = bars["open"]
    entries, exits = signals(placed, sessions)
    portfolio = simulate(close, opens, entries, exits, freq, slippage, transaction_cost)
    portfolio.orders.records_readable.to_csv(folder / "orders.csv", index=False)
    equity = portfolio.value()
    equity.name = "equity"
    equity.to_csv(folder / "equity.csv")
    stats = portfolio.stats()
    _write_text(folder / "summary.txt", stats.to_string() + "\n")
    _write_text(folder / "summary.json", stats.to_json(date_format="iso", indent=2) + "\n")
    _write_equity_png(portfolio, folder / "equity.png")


def backtest(symbol, start, end, freq, strategy, parameters, root=RUNS_DIR, warmup=None, slippage=None, transaction_cost=None):
    if warmup is None:
        warmup = WARMUP
    if slippage is None:
        slippage = SLIPPAGE
    if transaction_cost is None:
        transaction_cost = TRANSACTION_COST
    loaded = load_strategy(strategy, parameters)
    folder = record(
        symbol=symbol,
        window={"start": start, "end": end},
        freq=freq,
        strategy=strategy,
        parameters=parameters,
        slippage=slippage,
        transaction_cost=transaction_cost,
        cash=INIT_CASH,
        root=root,
        warmup=warmup,
    )
    try:
        _execute(folder, symbol, start, end, freq, loaded, warmup, slippage, transaction_cost)
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    return folder
