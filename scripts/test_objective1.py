import csv
import json
import warnings
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

from scripts.objective1 import COLUMNS, main, run_period
from src.data import as_date

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs"
COMPARISONS = REPO / "comparisons"
BUY = json.loads((REPO / "params" / "objective1" / "buy_and_hold.json").read_text(encoding="utf-8"))
RANDOM = {"feature_set": "void", "feature_parameters": {}, "p_buy": 0, "p_sell": 0, "seed": 1}
SYMBOLS = ["SPY", "QQQ", "TLT", "GLD"]


def _names(path):
    return sorted(item.name for item in path.iterdir())


def _ignore_vectorbt_rename_warning():
    # records_readable calls DataFrame.rename(inplace=True), deprecated on pandas 3.1.
    warnings.filterwarnings("ignore", category=pd.errors.Pandas4Warning)


def _rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(COLUMNS)
        return list(reader)


def _markdown_rows(path):
    text = path.read_text(encoding="utf-8")
    assert "\r" not in text
    lines = text.splitlines()
    assert lines[0] == "| " + " | ".join(COLUMNS) + " |"
    assert lines[1] == "|" + "|".join("---" for _ in COLUMNS) + "|"
    parsed = []
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        parsed.append(dict(zip(COLUMNS, cells)))
    assert len(lines) == len(parsed) + 2
    return parsed


def _metric(cell, value):
    if value is None:
        assert cell == ""
    else:
        assert float(cell) == pytest.approx(value)


def _matches_run(row, runs, parameters):
    folder = runs / row["run_id"]
    assert folder.is_dir()
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    _metric(row["total_return"], summary["Total Return [%]"])
    _metric(row["sharpe_ratio"], summary["Sharpe Ratio"])
    _metric(row["max_drawdown"], summary["Max Drawdown [%]"])
    _metric(row["time_in_market"], summary["time_in_market"])
    _metric(row["n_trades"], summary["Total Trades"])
    _metric(row["avg_holding_period"], summary["avg_holding_period"])
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    assert config["window"] == {"start": "2023-01-01", "end": "2024-12-31"}
    assert config["freq"] == "1D"
    assert config["strategy"] == row["strategy"]
    assert config["symbol"] == row["symbol"]
    assert config["parameters"] == parameters
    assert config["warmup"] == "400D"
    assert config["slippage"] == 0.0005
    assert config["transaction_cost"] == 0.001


def test_unknown_period_exits_before_a_run(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("scripts.objective1.backtest", lambda *args, **kwargs: calls.append(args))
    runs_before = _names(RUNS)
    comparisons_before = _names(COMPARISONS) if COMPARISONS.exists() else None

    with pytest.raises(SystemExit):
        main(
            [
                "--period",
                "train",
                "--strategy",
                "buy_and_hold",
                "--parameters",
                "params/objective1/buy_and_hold.json",
            ]
        )
    with pytest.raises(ValueError):
        run_period(
            "train",
            "buy_and_hold",
            BUY,
            comparisons=tmp_path / "comparisons",
            runs=tmp_path / "runs",
        )

    assert calls == []
    assert not (tmp_path / "comparisons").exists()
    assert not (tmp_path / "runs").exists()
    assert _names(RUNS) == runs_before
    if comparisons_before is None:
        assert not COMPARISONS.exists()
    else:
        assert _names(COMPARISONS) == comparisons_before


def _load_prices(symbol, start, end, freq):
    start_day, end_day = as_date(start), as_date(end)
    days = []
    day = start_day
    while day <= end_day:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    index = pd.DatetimeIndex([pd.Timestamp(day, tz="UTC") for day in days], name="timestamp")
    close = pd.Series([100.0 + offset for offset in range(len(days))], index=index, dtype="float64")
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1000.0,
        },
        index=index,
    )
    return {symbol: frame}


def test_table_appends_and_replaces_under_temp_directories(tmp_path, monkeypatch):
    monkeypatch.setattr("src.run.load_prices", _load_prices)
    runs_before = _names(RUNS)
    comparisons_before = _names(COMPARISONS) if COMPARISONS.exists() else None
    comparisons = tmp_path / "comparisons"
    runs = tmp_path / "runs"

    def run(strategy, parameters):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _ignore_vectorbt_rename_warning()
            return run_period("validation", strategy, parameters, comparisons=comparisons, runs=runs)

    markdown = run("buy_and_hold", BUY)
    assert markdown == comparisons / "validation.md"
    held = _rows(comparisons / "validation.csv")
    assert held == _markdown_rows(markdown)
    assert [row["symbol"] for row in held] == SYMBOLS
    assert [row["strategy"] for row in held] == ["buy_and_hold"] * 4
    for row in held:
        _matches_run(row, runs, BUY)
    buy_ids = [row["run_id"] for row in held]
    assert len(set(buy_ids)) == 4

    run("random", RANDOM)
    appended = _rows(comparisons / "validation.csv")
    assert appended[:4] == held
    assert [row["strategy"] for row in appended[4:]] == ["random"] * 4
    assert [row["symbol"] for row in appended[4:]] == SYMBOLS
    for row in appended[4:]:
        _matches_run(row, runs, RANDOM)
    random_rows = appended[4:]
    assert _markdown_rows(markdown) == appended

    run("buy_and_hold", BUY)
    replaced = _rows(comparisons / "validation.csv")
    assert len(replaced) == 8
    assert [row["strategy"] for row in replaced[:4]] == ["buy_and_hold"] * 4
    assert [row["symbol"] for row in replaced[:4]] == SYMBOLS
    assert [row["run_id"] for row in replaced[:4]] != buy_ids
    assert replaced[4:] == random_rows
    for row in replaced[:4]:
        _matches_run(row, runs, BUY)
    for run_id in buy_ids:
        assert (runs / run_id).is_dir()
    assert _markdown_rows(markdown) == replaced
    assert _names(RUNS) == runs_before
    if comparisons_before is None:
        assert not COMPARISONS.exists()
    else:
        assert _names(COMPARISONS) == comparisons_before
