from src.llm import decide


def test_unparseable_reply_becomes_hold():
    reply = "not a decision"
    assert decide("unused", complete=lambda text: reply) == "HOLD"
