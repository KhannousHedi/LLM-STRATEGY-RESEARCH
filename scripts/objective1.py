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

``run_diagnostics`` reads that table. It does not backtest. For each ``llm``
row it reads ``decisions.jsonl``, sets ``p_buy`` and ``p_sell`` from that
log, and simulates seeds 0 through 199 in memory when both probabilities
are inside 0 to 1. The markdown table stays the nine comparison columns.
A ``## Diagnostics`` section follows it. Replacing a baseline keeps that
section. Replacing the ``llm`` rows drops it and clears the scalar columns.
"""

import argparse
import csv
import json
import math
import random
import warnings
from pathlib import Path

import pandas as pd

from src.data import load_prices
from src.decisions import place, signals
from src.features.classic_indicators import ClassicIndicators
from src.run import RUNS_DIR, backtest
from src.simulate import simulate
from src.strategies import load_strategy

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
DIAGNOSTIC_COLUMNS = (
    "n_entries",
    "d_in",
    "d_out",
    "p_buy",
    "p_sell",
    "chance_percentile",
    "agree_trend_ma200",
    "agree_rsi_reversion",
    "n_buy",
    "n_sell",
    "n_hold",
    "n_pointless",
    "n_parse_failures",
    "tree_accuracy",
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
FEATURE_COLUMNS = ClassicIndicators.columns
RULE_STRATEGIES = ("trend_ma200", "rsi_reversion")
_DECISION_KEYS = ("action", "as_of", "parsed", "features", "analysis", "text")
_DIAGNOSTICS_MARKER = "## Diagnostics\n"


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


def _finite_number(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


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
        fields = list(COLUMNS)
        if any(column in reader.fieldnames for column in DIAGNOSTIC_COLUMNS):
            missing = [column for column in DIAGNOSTIC_COLUMNS if column not in reader.fieldnames]
            if missing:
                raise ValueError(f"{path.name} is missing {', '.join(missing)}")
            fields = [*COLUMNS, *DIAGNOSTIC_COLUMNS]
        return [{column: row[column] for column in fields} for row in reader]


def _has_diagnostics(rows):
    return bool(rows) and all(column in rows[0] for column in DIAGNOSTIC_COLUMNS)


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


def _csv_text(rows, columns=COLUMNS):
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(row.get(column, "") for column in columns))
    return "\n".join(lines) + "\n"


def _diagnostics_section(text):
    index = text.find(_DIAGNOSTICS_MARKER)
    if index == -1:
        return ""
    return text[index:]


def _write_tables(directory, period, new_rows):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / f"{period}.csv"
    md_path = directory / f"{period}.md"
    existing = _read_rows(csv_path) if csv_path.exists() else []
    section = _diagnostics_section(md_path.read_text(encoding="utf-8")) if md_path.exists() else ""
    rows = _merge(existing, new_rows)
    if new_rows[0]["strategy"] == "llm":
        columns = COLUMNS
        section = ""
    elif _has_diagnostics(existing):
        columns = (*COLUMNS, *DIAGNOSTIC_COLUMNS)
    else:
        columns = COLUMNS
    _write_text(csv_path, _csv_text(rows, columns))
    text = _table_text(rows)
    if section:
        text = text + "\n" + section
    _write_text(md_path, text)
    return md_path


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


def match_probabilities(n, d_in, d_out):
    p_buy = None if d_out == 0 else n / d_out
    p_sell = None if d_in == 0 else n / d_in
    return p_buy, p_sell


def matched_runs(p_buy, p_sell):
    buy = _finite_number(p_buy)
    sell = _finite_number(p_sell)
    if buy is None or sell is None:
        return False
    return 0 <= buy <= 1 and 0 <= sell <= 1


def chance_percentile(llm_sharpe, sharpes):
    score = _finite_number(llm_sharpe)
    if score is None:
        return None
    finite = []
    for value in sharpes:
        number = _finite_number(value)
        if number is not None:
            finite.append(number)
    if not finite:
        return None
    below = sum(value < score for value in finite)
    ties = sum(value == score for value in finite)
    return 100.0 * (below + 0.5 * ties) / len(finite)


def _apply(position, action):
    if action == "BUY":
        return "in"
    if action == "SELL":
        return "out"
    if action == "HOLD":
        return position
    raise ValueError(f"action must be BUY, SELL, or HOLD, got {action!r}")


def _walk(lines):
    position = "out"
    walked = []
    for line in lines:
        before = position
        after = _apply(before, line["action"])
        walked.append((line, before, after))
        position = after
    return walked


def decision_counts(lines):
    n_entries = d_in = d_out = 0
    n_buy = n_sell = n_hold = n_pointless = n_parse_failures = 0
    for index, (line, before, _after) in enumerate(_walk(lines)):
        if "parsed" not in line:
            raise KeyError(f"decision line {index} is missing parsed")
        if not isinstance(line["parsed"], bool):
            raise ValueError(f"decision line {index} parsed must be true or false")
        if line["parsed"] is False:
            n_parse_failures += 1
        action = line["action"]
        if before == "in":
            d_in += 1
        else:
            d_out += 1
        if action == "BUY":
            n_buy += 1
            if before == "out":
                n_entries += 1
            else:
                n_pointless += 1
        elif action == "SELL":
            n_sell += 1
            if before == "out":
                n_pointless += 1
        else:
            n_hold += 1
    return {
        "n_entries": n_entries,
        "d_in": d_in,
        "d_out": d_out,
        "n_buy": n_buy,
        "n_sell": n_sell,
        "n_hold": n_hold,
        "n_pointless": n_pointless,
        "n_parse_failures": n_parse_failures,
    }


def agreement(llm_lines, rule_lines):
    if not rule_lines:
        return None
    rule_at = {line["as_of"]: after for line, _before, after in _walk(rule_lines)}
    shared = 0
    matches = 0
    for line, _before, after in _walk(llm_lines):
        if line["as_of"] not in rule_at:
            continue
        shared += 1
        if after == rule_at[line["as_of"]]:
            matches += 1
    if shared == 0:
        return None
    return matches / shared


def fit_tree(lines):
    walked = _walk(lines)
    if not walked:
        return None, None
    from sklearn.tree import DecisionTreeClassifier, export_text

    rows = []
    labels = []
    for line, _before, after in walked:
        features = line["features"]
        rows.append([features[name] for name in FEATURE_COLUMNS])
        labels.append(after)
    frame = pd.DataFrame(rows, columns=list(FEATURE_COLUMNS))
    clf = DecisionTreeClassifier(max_depth=2, random_state=0)
    clf.fit(frame, labels)
    accuracy = float(clf.score(frame, labels))
    text = export_text(clf, feature_names=list(FEATURE_COLUMNS))
    return text, accuracy


def sample_analyses(symbol, lines):
    if len(lines) <= 20:
        chosen = list(lines)
    else:
        chosen = random.Random(symbol).sample(list(lines), 20)
    chosen.sort(key=lambda line: line["as_of"])
    return [
        {
            "symbol": symbol,
            "session": line["as_of"],
            "action": line["action"],
            "analysis": line["analysis"],
            "text": line["text"],
        }
        for line in chosen
    ]


def _read_jsonl(path):
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            lines.append(json.loads(line))
    return lines


def _require_decision(line, index, run_id):
    missing = [key for key in _DECISION_KEYS if key not in line]
    if missing:
        raise KeyError(f"{run_id} decisions.jsonl line {index} is missing {', '.join(missing)}")
    if not isinstance(line["parsed"], bool):
        raise ValueError(f"{run_id} decisions.jsonl line {index} parsed must be true or false")
    features = line["features"]
    if not isinstance(features, dict):
        raise KeyError(f"{run_id} decisions.jsonl line {index} features must be an object")
    missing_features = [name for name in FEATURE_COLUMNS if name not in features]
    if missing_features:
        raise KeyError(
            f"{run_id} decisions.jsonl line {index} features missing {', '.join(missing_features)}"
        )


def _equity_sessions(folder):
    frame = pd.read_csv(folder / "equity.csv", index_col=0)
    sessions = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True))
    if len(sessions) == 0:
        raise ValueError(f"{folder.name} equity.csv has no sessions")
    return sessions


def _as_of_label(value, sessions):
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    loc = int(sessions.get_indexer([stamp])[0])
    if loc < 0:
        raise ValueError(f"as_of {value} is not an equity session")
    return sessions[loc]


def _aligned_prices(config, sessions):
    symbol = config["symbol"]
    window = config["window"]
    loaded = load_prices(symbol, window["start"], window["end"], config["freq"])
    frame = loaded[symbol].copy()
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True))
    aligned = frame.reindex(sessions)
    if aligned["open"].isna().any() or aligned["close"].isna().any():
        raise ValueError(f"{symbol} prices do not cover the equity sessions")
    return aligned["close"], aligned["open"]


def _matched_sharpes(folder, lines, p_buy, p_sell):
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    sessions = _equity_sessions(folder)
    close, opens = _aligned_prices(config, sessions)
    stamps = [_as_of_label(line["as_of"], sessions) for line in lines]
    freq = config["freq"]
    slippage = config["slippage"]
    transaction_cost = config["transaction_cost"]
    sharpes = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=pd.errors.Pandas4Warning)
        for seed in range(200):
            strategy = load_strategy("random", {"p_buy": p_buy, "p_sell": p_sell, "seed": seed})
            position = "out"
            placed = []
            for stamp in stamps:
                action = strategy.decide({}, position)
                placed.append(place(action, stamp, sessions))
                position = _apply(position, action)
            entries, exits = signals(placed, sessions)
            portfolio = simulate(close, opens, entries, exits, freq, slippage, transaction_cost)
            sharpes.append(_finite_number(portfolio.stats()["Sharpe Ratio"]))
    return sharpes


def _rule_lines(runs, rules, strategy, symbol):
    run_id = rules.get((strategy, symbol))
    if run_id is None:
        return None
    path = Path(runs) / run_id / "decisions.jsonl"
    if not path.is_file():
        return None
    return _read_jsonl(path)


def _symbol_markdown(symbol, cells, avg_holding_period, tree_text, sharpes, sample):
    lines = [f"### {symbol}", ""]
    for column in DIAGNOSTIC_COLUMNS:
        lines.append(f"- {column}: {cells[column]}")
    lines.append(f"- avg_holding_period: {avg_holding_period}")
    lines.append("")
    lines.append("#### Tree")
    lines.append("")
    if tree_text:
        lines.append(tree_text.rstrip("\n"))
        lines.append("")
    lines.append("#### Sharpe distribution")
    lines.append("")
    if sharpes:
        lines.append(f"{len(sharpes)} seeds")
        lines.append("")
        for seed, value in enumerate(sharpes):
            lines.append(f"seed {seed}: {_cell(value)}")
        lines.append("")
    lines.append("#### Sample")
    lines.append("")
    for item in sample:
        lines.append(json.dumps(item, ensure_ascii=False))
    lines.append("")
    return "\n".join(lines)


def _diagnostics_markdown(blocks):
    text = "\n".join(["## Diagnostics", "", *blocks])
    if not text.endswith("\n"):
        text += "\n"
    return text


def _diagnose(row, runs, rules):
    folder = Path(runs) / row["run_id"]
    if not folder.is_dir():
        raise FileNotFoundError(f"llm run folder {row['run_id']} is missing")
    path = folder / "decisions.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{row['run_id']} is missing decisions.jsonl")
    lines = _read_jsonl(path)
    for index, line in enumerate(lines):
        _require_decision(line, index, row["run_id"])
    counts = decision_counts(lines)
    p_buy, p_sell = match_probabilities(counts["n_entries"], counts["d_in"], counts["d_out"])
    if matched_runs(p_buy, p_sell):
        sharpes = _matched_sharpes(folder, lines, p_buy, p_sell)
    else:
        sharpes = []
    percentile = chance_percentile(row["sharpe_ratio"], sharpes)
    agreements = {
        strategy: agreement(lines, _rule_lines(runs, rules, strategy, row["symbol"]))
        for strategy in RULE_STRATEGIES
    }
    tree_text, accuracy = fit_tree(lines)
    sample = sample_analyses(row["symbol"], lines)
    raw = {
        "n_entries": counts["n_entries"],
        "d_in": counts["d_in"],
        "d_out": counts["d_out"],
        "p_buy": p_buy,
        "p_sell": p_sell,
        "chance_percentile": percentile,
        "agree_trend_ma200": agreements["trend_ma200"],
        "agree_rsi_reversion": agreements["rsi_reversion"],
        "n_buy": counts["n_buy"],
        "n_sell": counts["n_sell"],
        "n_hold": counts["n_hold"],
        "n_pointless": counts["n_pointless"],
        "n_parse_failures": counts["n_parse_failures"],
        "tree_accuracy": accuracy,
    }
    cells = {key: _cell(value) for key, value in raw.items()}
    block = _symbol_markdown(row["symbol"], cells, row["avg_holding_period"], tree_text, sharpes, sample)
    return {"cells": cells, "block": block}


def _write_diagnostic_report(directory, period, rows, section):
    directory = Path(directory)
    csv_path = directory / f"{period}.csv"
    md_path = directory / f"{period}.md"
    _write_text(csv_path, _csv_text(rows, (*COLUMNS, *DIAGNOSTIC_COLUMNS)))
    _write_text(md_path, _table_text(rows) + "\n" + section)
    return md_path


def run_diagnostics(period, comparisons=None, runs=None):
    if period not in PERIODS:
        raise ValueError(f"period must be validation, test, or stress, got {period!r}")
    if comparisons is None:
        comparisons = COMPARISONS_DIR
    if runs is None:
        runs = RUNS_DIR
    comparisons = Path(comparisons)
    runs = Path(runs)
    csv_path = comparisons / f"{period}.csv"
    if not csv_path.is_file():
        raise ValueError(f"{period}.csv has no llm row")
    rows = _read_rows(csv_path)
    llm_indexes = [index for index, row in enumerate(rows) if row["strategy"] == "llm"]
    if not llm_indexes:
        raise ValueError(f"{period}.csv has no llm row")
    rules = {
        (row["strategy"], row["symbol"]): row["run_id"]
        for row in rows
        if row["strategy"] in RULE_STRATEGIES
    }
    prepared = [_diagnose(rows[index], runs, rules) for index in llm_indexes]
    diagnosed = dict(zip(llm_indexes, prepared))
    for index, row in enumerate(rows):
        result = diagnosed.get(index)
        if result is None:
            for column in DIAGNOSTIC_COLUMNS:
                row[column] = ""
        else:
            row.update(result["cells"])
    section = _diagnostics_markdown([diagnosed[index]["block"] for index in llm_indexes])
    return _write_diagnostic_report(comparisons, period, rows, section)


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
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--strategy")
    parser.add_argument("--parameters", metavar="PATH")
    args = parser.parse_args(argv)
    if args.period not in PERIODS:
        parser.error("period must be validation, test, or stress")
    if args.diagnostics:
        if args.strategy is not None or args.parameters is not None:
            parser.error("--diagnostics does not take --strategy or --parameters")
        print(run_diagnostics(args.period))
        return
    if args.strategy is None or args.parameters is None:
        parser.error("--strategy and --parameters are required")
    print(run_period(args.period, args.strategy, _parameters(parser, args.parameters)))


if __name__ == "__main__":
    main()
