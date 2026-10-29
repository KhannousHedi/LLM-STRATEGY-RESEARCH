from src.llm import SCHEMA, decide, parse_reply


def test_unparseable_reply_becomes_hold():
    reply = "not a decision"
    assert decide("unused", complete=lambda text: reply) == ("HOLD", "", False)


def test_parsed_reply_keeps_its_analysis():
    reply = '{"analysis": "trend is up", "decision": "BUY"}'
    assert decide("unused", complete=lambda text: reply) == ("BUY", "trend is up", True)


def test_second_reply_is_parsed_when_the_first_fails():
    replies = ["not a decision", '{"analysis": "second", "decision": "SELL"}']

    def complete(text):
        return replies.pop(0)

    assert decide("unused", complete=complete) == ("SELL", "second", True)
    assert replies == []


def test_reasoning_key_sorts_before_decision_key():
    # Ollama writes keys alphabetically, so this order is the generation order.
    assert sorted(SCHEMA["properties"]) == ["analysis", "decision"]


def test_reply_returns_decision():
    assert parse_reply('{"analysis": "trend is up", "decision": "BUY"}') == ("BUY", "trend is up")
    assert parse_reply('{"analysis": "", "decision": "HOLD"}') == ("HOLD", "")
    assert parse_reply('{"action": "BUY", "reason": "old keys"}') is None
