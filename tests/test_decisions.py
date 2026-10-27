import pandas as pd

from src.decisions import place, signals
from src.simulate import benchmark, simulate

SLIPPAGE = 0.0005


def test_buy_fills_next_open():
    sessions = pd.date_range("2024-03-04", periods=2, freq="1D", tz="UTC")
    opens = pd.Series([100.0, 110.0], index=sessions)
    close = pd.Series([101.0, 111.0], index=sessions)

    record = place("BUY", sessions[0], sessions)
    assert record["as_of"] == sessions[0]
    assert record["execute_on"] == sessions[1]
    assert record["entry"] is True
    assert record["exit"] is False

    entries, exits = signals([record], sessions)
    assert entries.tolist() == [False, True]
    assert exits.tolist() == [False, False]

    orders = simulate(close, opens, entries, exits, "1D").orders.records_readable
    assert len(orders) == 1
    filled = orders.iloc[0]
    assert filled["Timestamp"] == sessions[1]
    assert filled["Side"] == "Buy"
    assert filled["Price"] == opens.iloc[1] * (1 + SLIPPAGE)

    bench_orders = benchmark(close, opens, record["execute_on"], "1D").orders.records_readable
    assert len(bench_orders) == 1
    assert bench_orders.iloc[0]["Timestamp"] == sessions[1]
    assert bench_orders.iloc[0]["Side"] == "Buy"
