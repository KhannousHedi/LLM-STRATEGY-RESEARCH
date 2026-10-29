import json
import re
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.tseries.frequencies import to_offset

from src.data import as_date
from src.run import WARMUP, _finite_values, backtest, record

RUNS = Path(__file__).resolve().parent.parent / "runs"
STRATEGY = "dist_ma"
PARAMETERS = {"buy_above": 0, "feature_parameters": {"ma_window": 10, "vol_window": 20, "volume_window": 20}}
SYMBOL = "SPY"
START = "2024-06-03"
END = "2024-07-12"
FREQ = "1D"


def test_run_records_parameters(tmp_path):
    before = sorted(path.name for path in RUNS.iterdir())
    folder = record(
        symbol="SPY",
        window={"start": "2024-03-04", "end": "2024-03-05"},
        freq="1D",
        strategy=STRATEGY,
        parameters=PARAMETERS,
        slippage=0.0005,
        transaction_cost=0.001,
        cash=100_000,
        root=tmp_path,
    )
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    assert config["strategy"] == STRATEGY
    assert config["parameters"] == PARAMETERS
    assert config["warmup"] == WARMUP
    assert config["slippage"] == 0.0005
    assert config["transaction_cost"] == 0.001
    keys = list(config)
    assert keys.index("transaction_cost") == keys.index("slippage") + 1
    assert config["window"] == {"start": "2024-03-04", "end": "2024-03-05"}
    assert folder.parent == tmp_path
    assert re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{6}", folder.name)
    assert sorted(path.name for path in RUNS.iterdir()) == before


def test_finite_values_names_bool_and_non_numeric_cells():
    row = {"flag": True, "gap": float("nan"), "text": "x", "ok": 1.5}
    values, bad_columns = _finite_values(row, ["flag", "gap", "text", "ok"])
    assert values is None
    assert bad_columns == ["flag", "gap", "text"]

    values, bad_columns = _finite_values({"flag": np.True_}, ["flag"])
    assert values is None
    assert bad_columns == ["flag"]

    values, bad_columns = _finite_values({"ok": 1.5}, ["ok"])
    assert bad_columns == []
    assert values == {"ok": 1.5}


def _weekdays(start, end):
    day, end = as_date(start), as_date(end)
    days = []
    while day <= end:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def _install_prices(monkeypatch):
    calls = []

    def load_prices(symbol, start, end, freq):
        start_day, end_day = as_date(start), as_date(end)
        calls.append((symbol, start_day, end_day, freq))
        days = _weekdays(start_day, end_day)
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

    monkeypatch.setattr("src.run.load_prices", load_prices)
    return calls


def _run(tmp_path, warmup=None, slippage=None, transaction_cost=None):
    kwargs = {}
    if warmup is not None:
        kwargs["warmup"] = warmup
    if slippage is not None:
        kwargs["slippage"] = slippage
    if transaction_cost is not None:
        kwargs["transaction_cost"] = transaction_cost
    return backtest(SYMBOL, START, END, FREQ, STRATEGY, PARAMETERS, root=tmp_path, **kwargs)


def _decisions(folder):
    text = (folder / "decisions.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def _equity(folder):
    frame = pd.read_csv(folder / "equity.csv", index_col=0)
    column = frame.columns[0]
    return [(as_date(label), float(value)) for label, value in frame[column].items()]


def _warm_day(warmup):
    return as_date(pd.Timestamp(START) - to_offset(warmup))


def _ignore_vectorbt_rename_warning():
    # records_readable calls DataFrame.rename(inplace=True), deprecated on pandas 3.1.
    warnings.filterwarnings("ignore", category=pd.errors.Pandas4Warning)


def test_backtest_loads_default_warmup_and_decides_from_start(tmp_path, monkeypatch):
    calls = _install_prices(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _ignore_vectorbt_rename_warning()
        folder = _run(tmp_path)

    assert calls == [(SYMBOL, _warm_day(WARMUP), as_date(END), FREQ)]
    decisions = _decisions(folder)
    assert decisions
    assert as_date(decisions[0]["as_of"]) == as_date(START)
    assert all(as_date(row["as_of"]) >= as_date(START) for row in decisions)
    assert all(as_date(row["execute_on"]) >= as_date(START) for row in decisions)
    equity = _equity(folder)
    assert equity[0][0] == as_date(START)
    assert equity[0][1] == pytest.approx(100_000)
    assert all(day >= as_date(START) for day, _value in equity)
    orders = pd.read_csv(folder / "orders.csv")
    assert all(as_date(stamp) >= as_date(START) for stamp in orders["Timestamp"])
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    assert config["window"] == {"start": START, "end": END}
    assert config["warmup"] == WARMUP
    assert config["slippage"] == 0.0005
    assert config["transaction_cost"] == 0.001


def test_backtest_records_a_custom_slippage(tmp_path, monkeypatch):
    _install_prices(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _ignore_vectorbt_rename_warning()
        folder = _run(tmp_path, slippage=0.001)

    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    assert config["slippage"] == 0.001


def test_backtest_records_a_custom_transaction_cost(tmp_path, monkeypatch):
    _install_prices(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _ignore_vectorbt_rename_warning()
        folder = _run(tmp_path, transaction_cost=0.002)

    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    assert config["transaction_cost"] == 0.002


def test_backtest_loads_a_custom_warmup(tmp_path, monkeypatch):
    calls = _install_prices(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _ignore_vectorbt_rename_warning()
        folder = _run(tmp_path, warmup="30D")

    assert calls == [(SYMBOL, _warm_day("30D"), as_date(END), FREQ)]
    decisions = _decisions(folder)
    assert as_date(decisions[0]["as_of"]) == as_date(START)
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    assert config["window"] == {"start": START, "end": END}
    assert config["warmup"] == "30D"


def test_backtest_warns_once_when_warmup_leaves_features_non_finite(tmp_path, monkeypatch):
    _install_prices(monkeypatch)
    with pytest.warns(UserWarning, match="features not finite") as caught:
        _ignore_vectorbt_rename_warning()
        folder = _run(tmp_path, warmup="0D")

    assert len(caught) == 1
    assert str(caught[0].message) == (
        "SPY: 20 of 30 sessions skipped, features not finite on "
        "2024-06-03, 2024-06-04, 2024-06-05, 2024-06-06, 2024-06-07, "
        "and 15 more (r1, r5, r20, dist_ma, vol, vol_ratio). "
        "Increase warmup (now 0D)."
    )
    decisions = _decisions(folder)
    assert as_date(decisions[0]["as_of"]) > as_date(START)
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    assert config["window"] == {"start": START, "end": END}
    assert config["warmup"] == "0D"


def test_backtest_rejects_a_bad_or_negative_warmup(tmp_path, monkeypatch):
    calls = _install_prices(monkeypatch)
    for warmup in ("nope", "-5D"):
        with pytest.raises(ValueError):
            _run(tmp_path, warmup=warmup)
    assert calls == []
    assert list(tmp_path.iterdir()) == []
