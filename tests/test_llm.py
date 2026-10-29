from src.llm import SCHEMA, decide, parse_reply


def test_unparseable_reply_becomes_hold():
    reply = "not a decision"
    assert decide("unused", complete=lambda text: reply) == "HOLD"


def test_reasoning_key_sorts_before_decision_key():
    # Ollama writes keys alphabetically, so this order is the generation order.
    assert sorted(SCHEMA["properties"]) == ["analysis", "decision"]


def test_reply_returns_decision():
    assert parse_reply('{"analysis": "trend is up", "decision": "BUY"}') == "BUY"
    assert parse_reply('{"action": "BUY", "reason": "old keys"}') is None
