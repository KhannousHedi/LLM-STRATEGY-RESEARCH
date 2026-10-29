import pandas as pd
import pytest

import src.strategies.llm as llm_module
from src.features import build_features
from src.strategies import load_strategy


def _capture(monkeypatch):
    seen = {}

    def fake_ask(text, model=None):
        seen["text"] = text
        return "HOLD"

    monkeypatch.setattr(llm_module, "ask", fake_ask)
    return seen


def test_slots_render_with_their_format_specs(monkeypatch):
    seen = _capture(monkeypatch)
    prompt = "{r1:+.2%} {open_gap:+.2%} {vol20:.2%} {rsi14:.1f} {close_location:.2f} {volume_ratio:.2f}x"
    strategy = load_strategy(
        "llm",
        {
            "feature_set": "classic_indicators",
            "feature_parameters": {},
            "prompt": prompt,
        },
    )
    strategy.decide(
        {
            "r1": -0.0123,
            "open_gap": 0.0,
            "vol20": 0.182,
            "rsi14": 71.4,
            "close_location": 0.82,
            "volume_ratio": 1.45,
        },
        "out",
    )
    rendered = "-1.23% +0.00% 18.20% 71.4 0.82 1.45x"
    assert seen["text"] == rendered
    assert strategy.detail() == {"text": rendered}


def test_bare_slot_stays_six_decimal_places(monkeypatch):
    seen = _capture(monkeypatch)
    strategy = load_strategy("llm", {"prompt": "A {r1}"})
    strategy.decide({"r1": 0.0125}, "out")
    assert seen["text"] == "A 0.012500"


def test_reply_schema_json_is_left_alone(monkeypatch):
    seen = _capture(monkeypatch)
    schema = '{"analysis": "...", "decision": "BUY"|"SELL"|"HOLD"}'
    strategy = load_strategy("llm", {"prompt": "Read {r1}. Reply with only this JSON object: " + schema})
    strategy.decide({"r1": 0.0125}, "in")
    assert seen["text"] == "Read 0.012500. Reply with only this JSON object: " + schema


def test_bad_spec_and_position_spec_fail_at_load():
    with pytest.raises(ValueError, match=r"\{r1:.2q\}"):
        load_strategy("llm", {"prompt": "return {r1:.2q}"})
    with pytest.raises(ValueError, match=r"\{position:x\}"):
        load_strategy("llm", {"prompt": "Position: {position:x}"})


def test_classic_indicators_empty_parameters_build_and_unknown_window_fails_later():
    built = load_strategy(
        "llm",
        {
            "prompt": "{rsi14:.1f}",
            "feature_set": "classic_indicators",
            "feature_parameters": {},
        },
    )
    assert built.feature_set == "classic_indicators"
    assert built.feature_parameters == {}

    bad = load_strategy(
        "llm",
        {
            "prompt": "{rsi14:.1f}",
            "feature_set": "classic_indicators",
            "feature_parameters": {"nope_window": 3},
        },
    )
    index = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    prices = {
        "SPY": pd.DataFrame(
            {
                "open": [1.0, 1.0, 1.0],
                "high": [1.0, 1.0, 1.0],
                "low": [1.0, 1.0, 1.0],
                "close": [1.0, 1.0, 1.0],
                "volume": [1.0, 1.0, 1.0],
            },
            index=index,
        )
    }
    with pytest.raises(TypeError):
        build_features(prices, bad.feature_set, **bad.feature_parameters)
