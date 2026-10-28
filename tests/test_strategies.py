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
        return "BUY"

    monkeypatch.setattr(llm_module, "ask", fake_ask)
    strategy = load_strategy(
        "llm",
        {"prompt": "A {r1} B {r5} Position: {position}"},
    )
    action = strategy.decide({"r1": 0.0125, "r5": -0.034}, "out")
    assert action == "BUY"
    assert seen["text"] == "A 0.012500 B -0.034000 Position: out"
    assert strategy.detail() == {"text": "A 0.012500 B -0.034000 Position: out"}


def test_llm_may_use_only_some_features(monkeypatch):
    monkeypatch.setattr(llm_module, "ask", lambda text, model=None: "HOLD")
    strategy = load_strategy("llm", {"prompt": "only {dist_ma}"})
    assert strategy.decide({"dist_ma": 0.01}, "in") == "HOLD"
    assert strategy.detail() == {"text": "only 0.010000"}


def test_llm_key_missing_from_the_row_raises():
    strategy = load_strategy("llm", {"prompt": "X {r20}"})
    with pytest.raises(ValueError, match="not a feature"):
        strategy.decide({"r1": 0.0125, "r5": -0.034}, "out")
