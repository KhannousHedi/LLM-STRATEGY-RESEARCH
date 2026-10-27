"""Ask Ollama for one action.

``decide(text)`` posts the rendered user text to ``/api/chat``. The model
is ``qwen3:4b-instruct`` at ``http://127.0.0.1:11434``. Temperature is 0
and thinking is off. The reply schema is an object with ``action`` of
``BUY``, ``SELL``, or ``HOLD`` and a string ``reason``. There is no symbol
field.

A reply that does not match that schema is sent once more. A second
failure returns ``HOLD``. If the connection fails, the process exits
before another request is sent. ``complete``, when passed, returns the
reply text instead of posting. The unparseable-reply test uses it.
"""

import json
import re
import sys
import urllib.error
import urllib.request

MODEL = "qwen3:4b-instruct"
BASE_URL = "http://127.0.0.1:11434"
TIMEOUT = 120
ACTIONS = ("BUY", "SELL", "HOLD")

SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": list(ACTIONS)},
        "reason": {"type": "string"},
    },
    "required": ["action", "reason"],
}

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)
# This call is local. An empty proxy map keeps HTTP_PROXY from capturing it.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def parse_reply(reply):
    """Return ``BUY``, ``SELL``, or ``HOLD``, or ``None`` when the reply is not the schema."""
    if not isinstance(reply, str):
        return None
    text = reply.strip().lstrip("\ufeff")
    fenced = _FENCE.fullmatch(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or set(data) != {"action", "reason"}:
        return None
    if data["action"] not in ACTIONS or not isinstance(data["reason"], str):
        return None
    return data["action"]


def _post(text, model, base_url, timeout):
    url = base_url.rstrip("/") + "/api/chat"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "stream": False,
        "think": False,
        "format": SCHEMA,
        "options": {"temperature": 0},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        # HTTPError is a URLError. A status code means the app answered.
        raw = error.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        sys.exit(f"Ollama is down at {base_url}: {error}")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    message = payload.get("message") if isinstance(payload, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    return raw


def decide(text, *, model=MODEL, base_url=BASE_URL, timeout=TIMEOUT, complete=None):
    if complete is None:
        def complete(user_text):
            return _post(user_text, model, base_url, timeout)

    action = parse_reply(complete(text))
    if action is None:
        action = parse_reply(complete(text))
    if action is None:
        return "HOLD"
    return action
