"""Backtest one strategy on the Objective 1 symbols and write a comparison table.

``run_period`` accepts ``validation``, ``test``, and ``stress``. The dates,
the symbols, and the frequency are fixed. Each symbol is one ``backtest``
call, so each call loads a fresh strategy. Warmup, slippage, and the
transaction cost stay the backtest defaults.

The table is ``comparisons/objective1/{period}.csv`` and ``.md``. Each
metric is copied from that run's ``summary.json``. A row belongs to a
strategy by the ``strategy`` cell. A strategy already in the file is
replaced where that block started. A new strategy is appended.
``llm_v0`` and ``llm_v1`` are both strategy ``llm``, so a later prompt
version replaces the ``llm`` rows. ``comparisons`` defaults to that
directory. A test passes another one. ``runs`` defaults to the backtest's
``runs/``.
"""

import argparse
import csv
import json
import math
from pathlib import Path

from src.run import backtest

PERIODS = {
    "validation": ("2023-01-01", "2024-12-31"),
    "test": ("2025-01-01", "2026-09-30"),
    "stress": ("2019-01-01", "2022-12-31"),
}
SYMBOLS = ("SPY", "QQQ", "TLT", "GLD")
FREQ = "1D"
COLUMNS = (
    "strategy",
    "symbol",
    "total_return",
    "sharpe_ratio",
    "max_drawdown",
    "time_in_market",
    "n_trades",
    "avg_holding_period",
    "run_id",
)
SUMMARY_KEYS = {
    "total_return": "Total Return [%]",
    "sharpe_ratio": "Sharpe Ratio",
    "max_drawdown": "Max Drawdown [%]",
    "time_in_market": "time_in_market",
    "n_trades": "Total Trades",
    "avg_holding_period": "avg_holding_period",
}
COMPARISONS_DIR = Path(__file__).resolve().parent.parent / "comparisons" / "objective1"


def _write_text(path, text):
    path.write_text(text, encoding="utf-8", newline="\n")


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"cannot write {value!r}")
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return format(value, ".16g")
    return str(value)


def _summary(folder):
    data = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    missing = [key for key in SUMMARY_KEYS.values() if key not in data]
    if missing:
        raise KeyError(f"{folder.name} summary.json is missing {', '.join(missing)}")
    return data


def _row(strategy, symbol, folder):
    summary = _summary(folder)
    row = {column: _cell(summary[source]) for column, source in SUMMARY_KEYS.items()}
    row["strategy"] = strategy
    row["symbol"] = symbol
    row["run_id"] = folder.name
    return row


def _read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return []
        missing = [column for column in COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path.name} is missing {', '.join(missing)}")
        return [{column: row[column] for column in COLUMNS} for row in reader]


def _merge(existing, new_rows):
    strategy = new_rows[0]["strategy"]
    if not any(row["strategy"] == strategy for row in existing):
        return [*existing, *new_rows]
    start = next(index for index, row in enumerate(existing) if row["strategy"] == strategy)
    before = existing[:start]
    after = [row for row in existing[start:] if row["strategy"] != strategy]
    return [*before, *new_rows, *after]


def _table_text(rows):
    header = "| " + " | ".join(COLUMNS) + " |"
    rule = "|" + "|".join("---" for _ in COLUMNS) + "|"
    lines = [header, rule]
    for row in rows:
        lines.append("| " + " | ".join(row[column] for column in COLUMNS) + " |")
    return "\n".join(lines) + "\n"


def _csv_text(rows):
    lines = [",".join(COLUMNS)]
    for row in rows:
        lines.append(",".join(row[column] for column in COLUMNS))
    return "\n".join(lines) + "\n"


def _write_tables(directory, period, new_rows):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / f"{period}.csv"
    existing = _read_rows(csv_path) if csv_path.exists() else []
    rows = _merge(existing, new_rows)
    _write_text(csv_path, _csv_text(rows))
    _write_text(directory / f"{period}.md", _table_text(rows))
    return directory / f"{period}.md"


def run_period(period, strategy, parameters, comparisons=COMPARISONS_DIR, runs=None):
    if period not in PERIODS:
        raise ValueError(f"period must be validation, test, or stress, got {period!r}")
    start, end = PERIODS[period]
    rows = []
    for symbol in SYMBOLS:
        kwargs = {}
        if runs is not None:
            kwargs["root"] = runs
        folder = backtest(symbol, start, end, FREQ, strategy, parameters, **kwargs)
        rows.append(_row(strategy, symbol, folder))
    return _write_tables(comparisons, period, rows)


def _parameters(parser, path):
    try:
        parameters = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        parser.error(f"--parameters: {error}")
    if not isinstance(parameters, dict):
        parser.error("--parameters must contain a JSON object")
    return parameters


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m scripts.objective1")
    parser.add_argument("--period", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--parameters", required=True, metavar="PATH")
    args = parser.parse_args(argv)
    if args.period not in PERIODS:
        parser.error("period must be validation, test, or stress")
    print(run_period(args.period, args.strategy, _parameters(parser, args.parameters)))


if __name__ == "__main__":
    main()
