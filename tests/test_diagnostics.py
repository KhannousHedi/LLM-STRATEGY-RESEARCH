import csv
import json
import random
from pathlib import Path

import pandas as pd
import pytest

from scripts.objective1 import (
    COLUMNS,
    DIAGNOSTIC_COLUMNS,
    agreement,
    chance_percentile,
    decision_counts,
    fit_tree,
    main,
    match_probabilities,
    matched_runs,
    run_diagnostics,
    run_period,
    sample_analyses,
)
from src.features.classic_indicators import ClassicIndicators

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs"
COMPARISONS = REPO / "comparisons"
SESSIONS = [
    "2024-01-02T00:00:00+00:00",
    "2024-01-03T00:00:00+00:00",
    "2024-01-04T00:00:00+00:00",
    "2024-01-05T00:00:00+00:00",
    "2024-01-08T00:00:00+00:00",
]


def _names(path):
    return sorted(item.name for item in path.iterdir())


def _features(r1):
    values = {name: 0.0 for name in ClassicIndicators.columns}
    values["r1"] = r1
    return values


def _line(as_of, action, parsed=True, r1=0.0, analysis="note", text="prompt"):
    return {
        "as_of": as_of,
        "action": action,
        "parsed": parsed,
        "features": _features(r1),
        "analysis": analysis,
        "text": text,
    }


def test_match_probabilities():
    p_buy, p_sell = match_probabilities(2, 4, 8)
    assert p_buy == pytest.approx(0.25)
    assert p_sell == pytest.approx(0.5)
    assert matched_runs(p_buy, p_sell) is True


def test_undefined_match_skips_matched_runs():
    p_buy, p_sell = match_probabilities(1, 0, 4)
    assert p_sell is None
    assert p_buy == pytest.approx(0.25)
    assert matched_runs(p_buy, p_sell) is False

    p_buy, p_sell = match_probabilities(0, 5, 0)
    assert p_buy is None
    assert p_sell == pytest.approx(0)
    assert matched_runs(p_buy, p_sell) is False

    assert match_probabilities(0, 0, 0) == (None, None)
    assert matched_runs(None, None) is False

    p_buy, p_sell = match_probabilities(3, 2, 3)
    assert p_sell == pytest.approx(1.5)
    assert p_buy == pytest.approx(1)
    assert matched_runs(p_buy, p_sell) is False


def test_chance_percentile():
    sharpes = [0.0] * 50 + [1.0] * 98 + [2.0] * 50 + [None, None]
    assert len(sharpes) == 200
    assert chance_percentile(1.0, sharpes) == pytest.approx(50)
    assert chance_percentile(None, sharpes) is None
    assert chance_percentile(float("nan"), sharpes) is None
    assert chance_percentile(1.0, [None] * 200) is None


def test_agreement_drops_unmatched_sessions():
    llm = [
        {"as_of": "t1", "action": "BUY"},
        {"as_of": "t2", "action": "HOLD"},
        {"as_of": "t3", "action": "SELL"},
        {"as_of": "t4", "action": "HOLD"},
    ]
    rule = [
        {"as_of": "t1", "action": "BUY"},
        {"as_of": "t2", "action": "SELL"},
        {"as_of": "t3", "action": "HOLD"},
    ]
    assert agreement(llm, rule) == pytest.approx(2 / 3)
    assert agreement(llm, []) is None
    assert agreement(llm, [{"as_of": "t9", "action": "HOLD"}]) is None


def test_pointless_actions_and_parse_failures():
    lines = [
        {"action": "SELL", "parsed": True, "analysis": ""},
        {"action": "HOLD", "parsed": False, "analysis": ""},
        {"action": "BUY", "parsed": True, "analysis": "enter"},
        {"action": "BUY", "parsed": True, "analysis": "again"},
        {"action": "HOLD", "parsed": True, "analysis": ""},
        {"action": "SELL", "parsed": True, "analysis": "exit"},
    ]
    assert decision_counts(lines) == {
        "n_entries": 1,
        "d_in": 3,
        "d_out": 3,
        "n_buy": 2,
        "n_sell": 2,
        "n_hold": 2,
        "n_pointless": 2,
        "n_parse_failures": 1,
    }
    with pytest.raises(KeyError, match="parsed"):
        decision_counts([{"action": "HOLD"}])


def test_tree_on_fixture_decisions():
    lines = [
        {"action": "BUY", "features": _features(1.0)},
        {"action": "HOLD", "features": _features(1.0)},
        {"action": "SELL", "features": _features(-1.0)},
        {"action": "HOLD", "features": _features(-1.0)},
    ]
    text, accuracy = fit_tree(lines)
    again, again_accuracy = fit_tree(lines)
    assert text == again
    assert accuracy == pytest.approx(1)
    assert again_accuracy == pytest.approx(1)
    assert "r1" in text
    assert fit_tree([]) == (None, None)


def test_sample_is_stable():
    start = pd.Timestamp("2024-01-01", tz="UTC")
    lines = []
    for number in range(25):
        lines.append(
            {
                "as_of": (start + pd.Timedelta(days=number)).isoformat(),
                "action": "HOLD",
                "analysis": f"note {number}",
                "text": f"prompt {number}",
            }
        )
    state = random.getstate()
    drawn = sample_analyses("SPY", lines)
    assert random.getstate() == state
    assert drawn == sample_analyses("SPY", lines)
    assert len(drawn) == 20
    assert [item["session"] for item in drawn] == sorted(item["session"] for item in drawn)
    assert set(drawn[0]) == {"symbol", "session", "action", "analysis", "text"}
    assert drawn[0]["symbol"] == "SPY"

    short = list(reversed(lines[:3]))
    kept = sample_analyses("QQQ", short)
    assert len(kept) == 3
    assert [item["session"] for item in kept] == sorted(item["as_of"] for item in short)
    assert [item["analysis"] for item in kept] == ["note 0", "note 1", "note 2"]


def _write_jsonl(path, lines):
    text = "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines)
    path.write_text(text, encoding="utf-8", newline="\n")


def _write_equity(folder):
    body = ["timestamp,equity"]
    for stamp in SESSIONS:
        body.append(f"{stamp},100000.0")
    (folder / "equity.csv").write_text("\n".join(body) + "\n", encoding="utf-8", newline="\n")


def _write_config(folder, symbol):
    config = {
        "symbol": symbol,
        "window": {"start": "2024-01-02", "end": "2024-01-08"},
        "freq": "1D",
        "slippage": 0.0005,
        "transaction_cost": 0.001,
    }
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")


def _comparison_row(strategy, symbol, run_id, sharpe="1.5", holding="6.5"):
    return {
        "strategy": strategy,
        "symbol": symbol,
        "total_return": "1.25",
        "sharpe_ratio": sharpe,
        "max_drawdown": "-2",
        "time_in_market": "0.4",
        "n_trades": "1",
        "avg_holding_period": holding,
        "run_id": run_id,
    }


def _write_csv(path, rows, columns):
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(row.get(column, "") for column in columns))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _load_prices(symbol, start, end, freq):
    index = pd.DatetimeIndex([pd.Timestamp(stamp) for stamp in SESSIONS], name="timestamp")
    close = pd.Series([100.0 + offset for offset in range(len(index))], index=index, dtype="float64")
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


def _spy_log():
    return [
        _line(SESSIONS[0], "HOLD", r1=-1.0, analysis="spy-wait", text="prompt wait"),
        _line(SESSIONS[1], "BUY", r1=1.0, analysis="spy-note", text="prompt buy"),
        _line(SESSIONS[2], "HOLD", r1=1.0, analysis="spy-hold", text="prompt hold"),
        _line(SESSIONS[3], "SELL", parsed=False, r1=-1.0, analysis="", text="prompt sell"),
    ]


def _flat_log(analysis):
    return [_line(stamp, "HOLD", r1=0.0, analysis=analysis, text="flat prompt") for stamp in SESSIONS[:4]]


def _rules_for(actions):
    return [{"as_of": SESSIONS[index], "action": action} for index, action in enumerate(actions)]


def _fields(block):
    found = {}
    for line in block.splitlines():
        if line.startswith("#### "):
            break
        if line.startswith("- ") and ": " in line:
            key, value = line[2:].split(": ", 1)
            found[key] = value
    return found


def _blocks(text):
    section = text.split("## Diagnostics", 1)[1]
    parts = section.split("\n### ")
    return {part.splitlines()[0].strip(): part for part in parts[1:]}


def test_diagnostics_report(tmp_path, monkeypatch):
    runs_before = _names(RUNS)
    comparisons_before = _names(COMPARISONS) if COMPARISONS.exists() else None
    comparisons = tmp_path / "comparisons"
    runs = tmp_path / "runs"
    monkeypatch.setattr("scripts.objective1.COMPARISONS_DIR", comparisons)
    monkeypatch.setattr("scripts.objective1.RUNS_DIR", runs)
    monkeypatch.setattr("scripts.objective1.load_prices", _load_prices)

    folders = {
        "llm-spy": _spy_log(),
        "llm-qqq": _flat_log("qqq-flat"),
        "trend-spy": [
            {"as_of": SESSIONS[0], "action": "BUY"},
            {"as_of": SESSIONS[2], "action": "HOLD"},
        ],
        "trend-qqq": _rules_for(["HOLD", "HOLD", "HOLD", "HOLD"]),
        "rsi-spy": _rules_for(["HOLD", "BUY", "HOLD", "SELL"]),
    }
    for name, lines in folders.items():
        folder = runs / name
        folder.mkdir(parents=True)
        _write_jsonl(folder / "decisions.jsonl", lines)
    _write_config(runs / "llm-spy", "SPY")
    _write_equity(runs / "llm-spy")
    _write_config(runs / "llm-qqq", "QQQ")

    rows = [
        _comparison_row("buy_and_hold", "SPY", "bh-spy"),
        _comparison_row("trend_ma200", "SPY", "trend-spy"),
        _comparison_row("trend_ma200", "QQQ", "trend-qqq"),
        _comparison_row("rsi_reversion", "SPY", "rsi-spy"),
        _comparison_row("llm", "SPY", "llm-spy", holding="6.5"),
        _comparison_row("llm", "QQQ", "llm-qqq", sharpe="0.2", holding=""),
    ]
    _write_csv(comparisons / "validation.csv", rows, COLUMNS)
    before = _names(runs)

    main(["--period", "validation", "--diagnostics"])

    assert _names(runs) == before
    text = (comparisons / "validation.md").read_text(encoding="utf-8")
    assert "\r" not in text
    assert text.startswith("| " + " | ".join(COLUMNS) + " |")
    assert "Sharpe distribution" in text
    assert "200 seeds" in text
    blocks = _blocks(text)
    spy = _fields(blocks["SPY"])
    qqq = _fields(blocks["QQQ"])
    assert spy["p_buy"] == "0.5"
    assert spy["p_sell"] == "0.5"
    assert spy["n_entries"] == "1"
    assert spy["n_parse_failures"] == "1"
    assert spy["agree_trend_ma200"] == "0.5"
    assert spy["agree_rsi_reversion"] == "1"
    assert spy["avg_holding_period"] == "6.5"
    assert "r1" in blocks["SPY"]
    assert "spy-note" in blocks["SPY"]
    spy_seeds = [line for line in blocks["SPY"].splitlines() if line.startswith("seed ")]
    assert len(spy_seeds) == 200
    assert spy_seeds[0].startswith("seed 0:")
    assert spy_seeds[-1].startswith("seed 199:")
    assert qqq["p_buy"] == "0"
    assert qqq["p_sell"] == ""
    assert qqq["chance_percentile"] == ""
    assert "seed 0:" not in blocks["QQQ"]
    assert "qqq-flat" in blocks["QQQ"]

    with (comparisons / "validation.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [*COLUMNS, *DIAGNOSTIC_COLUMNS]
        written = list(reader)
    by_key = {(row["strategy"], row["symbol"]): row for row in written}
    assert by_key[("llm", "SPY")]["p_sell"] == "0.5"
    assert by_key[("llm", "QQQ")]["p_sell"] == ""
    assert by_key[("llm", "QQQ")]["chance_percentile"] == ""
    assert by_key[("buy_and_hold", "SPY")]["n_entries"] == ""
    assert by_key[("trend_ma200", "SPY")]["tree_accuracy"] == ""
    assert _names(RUNS) == runs_before
    if comparisons_before is None:
        assert not COMPARISONS.exists()
    else:
        assert _names(COMPARISONS) == comparisons_before


def test_missing_inputs_leave_the_table(tmp_path):
    comparisons = tmp_path / "comparisons"
    runs = tmp_path / "runs"
    rows = [_comparison_row("buy_and_hold", "SPY", "bh-spy")]
    csv_path = comparisons / "validation.csv"
    _write_csv(csv_path, rows, COLUMNS)
    before = csv_path.read_bytes()
    with pytest.raises(ValueError, match="llm"):
        run_diagnostics("validation", comparisons=comparisons, runs=runs)
    assert csv_path.read_bytes() == before
    assert not (comparisons / "validation.md").exists()

    folder = runs / "llm-spy"
    folder.mkdir(parents=True)
    _write_jsonl(folder / "decisions.jsonl", [{"action": "HOLD", "as_of": SESSIONS[0]}])
    rows = [_comparison_row("llm", "SPY", "llm-spy")]
    _write_csv(csv_path, rows, COLUMNS)
    md_path = comparisons / "validation.md"
    md_path.write_text("keep\n", encoding="utf-8", newline="\n")
    csv_before = csv_path.read_bytes()
    md_before = md_path.read_bytes()
    with pytest.raises(KeyError, match="parsed"):
        run_diagnostics("validation", comparisons=comparisons, runs=runs)
    assert csv_path.read_bytes() == csv_before
    assert md_path.read_bytes() == md_before

    rows = [_comparison_row("llm", "SPY", "missing-folder")]
    _write_csv(csv_path, rows, COLUMNS)
    csv_before = csv_path.read_bytes()
    with pytest.raises(FileNotFoundError, match="missing-folder"):
        run_diagnostics("validation", comparisons=comparisons, runs=runs)
    assert csv_path.read_bytes() == csv_before
    assert md_path.read_bytes() == md_before


def test_diagnostics_flag_rejects_strategy(tmp_path, monkeypatch):
    runs_before = _names(RUNS)
    comparisons_before = _names(COMPARISONS) if COMPARISONS.exists() else None
    monkeypatch.setattr("scripts.objective1.COMPARISONS_DIR", tmp_path / "comparisons")
    monkeypatch.setattr(
        "scripts.objective1.backtest",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("backtest")),
    )
    with pytest.raises(SystemExit):
        main(
            [
                "--period",
                "validation",
                "--diagnostics",
                "--strategy",
                "buy_and_hold",
                "--parameters",
                "params/objective1/buy_and_hold.json",
            ]
        )
    assert not (tmp_path / "comparisons").exists()
    assert _names(RUNS) == runs_before
    if comparisons_before is None:
        assert not COMPARISONS.exists()
    else:
        assert _names(COMPARISONS) == comparisons_before


def _fake_backtest(symbol, start, end, freq, strategy, parameters, root=None, **kwargs):
    folder = Path(root) / f"{strategy}-{symbol}-{len(list(Path(root).iterdir()))}"
    folder.mkdir(parents=True)
    summary = {
        "Total Return [%]": 1.25,
        "Sharpe Ratio": 0.5,
        "Max Drawdown [%]": -2.0,
        "time_in_market": 0.25,
        "Total Trades": 1,
        "avg_holding_period": 4.0,
    }
    (folder / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return folder


def test_comparison_keeps_diagnostics_until_llm_is_replaced(tmp_path, monkeypatch):
    runs_before = _names(RUNS)
    comparisons_before = _names(COMPARISONS) if COMPARISONS.exists() else None
    comparisons = tmp_path / "comparisons"
    runs = tmp_path / "runs"
    runs.mkdir()
    columns = [*COLUMNS, *DIAGNOSTIC_COLUMNS]
    rows = [
        _comparison_row("buy_and_hold", "SPY", "bh1"),
        _comparison_row("trend_ma200", "SPY", "tr1"),
        _comparison_row("llm", "SPY", "llm1"),
    ]
    for row in rows:
        for column in DIAGNOSTIC_COLUMNS:
            row[column] = ""
    rows[2]["p_buy"] = "0.25"
    _write_csv(comparisons / "validation.csv", rows, columns)
    section = "## Diagnostics\n\n### SPY\n\nseed 199: 1.25\n"
    (comparisons / "validation.md").write_text("old table\n\n" + section, encoding="utf-8", newline="\n")
    monkeypatch.setattr("scripts.objective1.backtest", _fake_backtest)

    run_period("validation", "trend_ma200", {}, comparisons=comparisons, runs=runs)

    text = (comparisons / "validation.md").read_text(encoding="utf-8")
    assert "## Diagnostics" in text
    assert "seed 199: 1.25" in text
    assert text.startswith("| " + " | ".join(COLUMNS) + " |")
    with (comparisons / "validation.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == columns
        kept = list(reader)
    assert kept[0]["run_id"] == "bh1"
    assert kept[-1]["strategy"] == "llm"
    assert kept[-1]["run_id"] == "llm1"
    assert kept[-1]["p_buy"] == "0.25"
    assert all(row["p_buy"] == "" for row in kept if row["strategy"] == "trend_ma200")

    run_period("validation", "llm", {}, comparisons=comparisons, runs=runs)

    text = (comparisons / "validation.md").read_text(encoding="utf-8")
    assert "## Diagnostics" not in text
    assert "seed 199: 1.25" not in text
    with (comparisons / "validation.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(COLUMNS)
        replaced = list(reader)
    assert replaced[0]["run_id"] == "bh1"
    assert "llm1" not in {row["run_id"] for row in replaced}
    assert [row["strategy"] for row in replaced if row["strategy"] == "llm"] == ["llm"] * 4
    assert _names(RUNS) == runs_before
    if comparisons_before is None:
        assert not COMPARISONS.exists()
    else:
        assert _names(COMPARISONS) == comparisons_before
