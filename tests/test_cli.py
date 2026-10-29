import src.cli


def _capture_backtest(monkeypatch):
    calls = []

    def backtest(*args, **kwargs):
        calls.append((args, kwargs))
        return "runs/example"

    monkeypatch.setattr("src.run.backtest", backtest)
    return calls


def test_cli_passes_warmup_to_backtest(monkeypatch, capsys):
    calls = _capture_backtest(monkeypatch)
    src.cli.main(
        [
            "backtest",
            "--symbol",
            "SPY",
            "--start",
            "2024-01-01",
            "--end",
            "2024-12-31",
            "--warmup",
            "30D",
        ]
    )
    assert calls == [
        (
            ("SPY", "2024-01-01", "2024-12-31", "1D", "llm", {}),
            {"warmup": "30D", "slippage": None, "transaction_cost": None},
        )
    ]
    assert capsys.readouterr().out == "runs/example\n"


def test_cli_omits_warmup(monkeypatch, capsys):
    calls = _capture_backtest(monkeypatch)
    src.cli.main(
        [
            "backtest",
            "--symbol",
            "SPY",
            "--start",
            "2024-01-01",
            "--end",
            "2024-12-31",
        ]
    )
    assert calls == [
        (
            ("SPY", "2024-01-01", "2024-12-31", "1D", "llm", {}),
            {"warmup": None, "slippage": None, "transaction_cost": None},
        )
    ]
    assert capsys.readouterr().out == "runs/example\n"


def test_cli_passes_transaction_cost_to_backtest(monkeypatch, capsys):
    calls = _capture_backtest(monkeypatch)
    src.cli.main(
        [
            "backtest",
            "--symbol",
            "SPY",
            "--start",
            "2024-01-01",
            "--end",
            "2024-12-31",
            "--transaction-cost",
            "0.002",
        ]
    )
    assert calls == [
        (
            ("SPY", "2024-01-01", "2024-12-31", "1D", "llm", {}),
            {"warmup": None, "slippage": None, "transaction_cost": 0.002},
        )
    ]
    assert capsys.readouterr().out == "runs/example\n"


def test_cli_passes_slippage_to_backtest(monkeypatch, capsys):
    calls = _capture_backtest(monkeypatch)
    src.cli.main(
        [
            "backtest",
            "--symbol",
            "SPY",
            "--start",
            "2024-01-01",
            "--end",
            "2024-12-31",
            "--slippage",
            "0.001",
        ]
    )
    assert calls == [
        (
            ("SPY", "2024-01-01", "2024-12-31", "1D", "llm", {}),
            {"warmup": None, "slippage": 0.001, "transaction_cost": None},
        )
    ]
    assert capsys.readouterr().out == "runs/example\n"
