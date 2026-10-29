import pytest

import src.strategies.llm as llm_module
from src.strategies import load_strategy


def test_dist_ma_buys_when_flat():
    strategy = load_strategy("dist_ma", {"buy_above": 0})
    assert strategy.decide({"dist_ma": 0.01}, "out") == "BUY"


def test_llm_requires_a_prompt():
    with pytest.raises(ValueError, match="prompt"):
        load_strategy("llm", {})


def test_llm_rejects_a_key_the_feature_set_does_not_return():
    with pytest.raises(ValueError, match="momentum_volatility_volume"):
        load_strategy("llm", {"prompt": "average {sma} Position: {position}"})


def test_llm_fills_named_slots(monkeypatch):
    seen = {}

    def fake_ask(text, model=None):
        seen["text"] = text
        seen["model"] = model
        return "BUY", "uptrend", True

    monkeypatch.setattr(llm_module, "ask", fake_ask)
    strategy = load_strategy(
        "llm",
        {"prompt": "A {r1} B {r5} Position: {position}"},
    )
    action = strategy.decide({"r1": 0.0125, "r5": -0.034}, "out")
    assert action == "BUY"
    assert seen["text"] == "A 0.012500 B -0.034000 Position: out"
    assert strategy.detail() == {
        "text": "A 0.012500 B -0.034000 Position: out",
        "analysis": "uptrend",
        "parsed": True,
    }


def test_llm_may_use_only_some_features(monkeypatch):
    monkeypatch.setattr(llm_module, "ask", lambda text, model=None: ("HOLD", "", False))
    strategy = load_strategy("llm", {"prompt": "only {dist_ma}"})
    assert strategy.decide({"dist_ma": 0.01}, "in") == "HOLD"
    assert strategy.detail() == {"text": "only 0.010000", "analysis": "", "parsed": False}


def test_llm_key_missing_from_the_row_raises():
    strategy = load_strategy("llm", {"prompt": "X {r20}"})
    with pytest.raises(ValueError, match="not a feature"):
        strategy.decide({"r1": 0.0125, "r5": -0.034}, "out")


def test_buy_and_hold_buys_once_then_holds():
    strategy = load_strategy("buy_and_hold", {})
    assert strategy.decide({}, "out") == "BUY"
    assert strategy.decide({}, "in") == "HOLD"
    assert strategy.decide({}, "out") == "HOLD"


def test_trend_ma200_buys_sells_and_holds_a_zero_gap():
    strategy = load_strategy("trend_ma200", {})
    assert strategy.decide({"ma200_gap": 0.01}, "out") == "BUY"
    assert strategy.decide({"ma200_gap": -0.01}, "in") == "SELL"
    assert strategy.decide({"ma200_gap": 0}, "out") == "HOLD"
    assert strategy.decide({"ma200_gap": 0}, "in") == "HOLD"


def test_rsi_reversion_buys_below_30_sells_above_70_and_holds_the_ties():
    strategy = load_strategy("rsi_reversion", {})
    assert strategy.decide({"rsi14": 29}, "out") == "BUY"
    assert strategy.decide({"rsi14": 71}, "in") == "SELL"
    assert strategy.decide({"rsi14": 30}, "out") == "HOLD"
    assert strategy.decide({"rsi14": 30}, "in") == "HOLD"
    assert strategy.decide({"rsi14": 70}, "out") == "HOLD"
    assert strategy.decide({"rsi14": 70}, "in") == "HOLD"


def test_random_same_seed_same_position_sequence_same_actions():
    parameters = {"p_buy": 0.3, "p_sell": 0.7, "seed": 11}
    positions = ["out", "in", "out", "out", "in", "in", "out", "in"]
    first = load_strategy("random", parameters)
    second = load_strategy("random", dict(parameters))
    actions = [first.decide({}, position) for position in positions]
    assert actions == [second.decide({}, position) for position in positions]
    for position, action in zip(positions, actions):
        if position == "out":
            assert action in ("BUY", "HOLD")
        else:
            assert action in ("SELL", "HOLD")


def test_random_probabilities_apply_to_the_open_side():
    buy_only = load_strategy("random", {"p_buy": 1, "p_sell": 0, "seed": 1})
    assert buy_only.decide({}, "out") == "BUY"
    assert buy_only.decide({}, "in") == "HOLD"
    sell_only = load_strategy("random", {"p_buy": 0, "p_sell": 1, "seed": 1})
    assert sell_only.decide({}, "out") == "HOLD"
    assert sell_only.decide({}, "in") == "SELL"


@pytest.mark.parametrize("missing", ["p_buy", "p_sell", "seed"])
def test_random_missing_key_raises(missing):
    parameters = {"p_buy": 0.5, "p_sell": 0.5, "seed": 0}
    del parameters[missing]
    with pytest.raises(ValueError, match=missing):
        load_strategy("random", parameters)
