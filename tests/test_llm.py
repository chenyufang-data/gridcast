"""The chat layer: provider fallback chain, daily limits, JSON parsing, Gemini turn mapping."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import UTC, date, datetime

import pytest

from frontend import llm, router

TODAY = date(2026, 9, 20)


class FakeProvider:
    name = "fake"
    model = "fake-1"

    def __init__(self, replies: list[str | Exception]) -> None:
        self.replies = list(replies)
        self.calls: list[Sequence[dict[str, str]]] = []

    def complete(self, system: str, messages: Sequence[dict[str, str]], timeout: float) -> str:
        self.calls.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture(autouse=True)
def fresh_limiter() -> Iterator[None]:
    old = llm.limiter
    llm.limiter = llm.ChatLimiter(per_user=2, global_cap=3)
    yield
    llm.limiter = old
    llm.set_provider(None, "no chat model configured")


def test_no_provider_uses_the_keyword_guide() -> None:
    llm.set_provider(None, "no chat model configured")
    r = llm.chat("nyc forecast tomorrow", [], router.ZONES, "u1", TODAY)
    assert r.intent["zone"] == "N.Y.C." and r.intent["target"] == "2026-09-21"
    assert r.used_model is False and r.remaining is None and r.note is None
    r = llm.chat("tell me a joke", [], router.ZONES, "u1", TODAY)
    assert r.intent["action"] == "answer" and "not available" in r.intent["message"]


def test_model_reply_is_sanitized_and_history_is_sent() -> None:
    fake = FakeProvider(
        [
            'Sure: {"action": "show_zone", "zone": "long island", "view": "prices", "start": "2026-09-13", "end": "2026-09-19"}'
        ]
    )
    llm.set_provider(fake)
    history = [("user", "hi"), ("assistant", "hello"), ("user", "west forecast")]
    r = llm.chat("same for long island but prices, last week", history, router.ZONES, "u1", TODAY)
    assert r.intent == {"action": "show_zone", "zone": "LONGIL", "view": "prices", "start": "2026-09-13", "end": "2026-09-19"}  # fmt: skip
    assert r.used_model and r.remaining == 1 and r.note is None
    sent = fake.calls[0]
    assert [m["role"] for m in sent] == ["user", "assistant", "user", "user"]
    assert "Today: 2026-09-20" in sent[-1]["content"] and "LONGIL" in sent[-1]["content"]


def test_model_failure_and_garbage_fall_back_with_a_note() -> None:
    llm.set_provider(FakeProvider([RuntimeError("boom"), "I have no idea"]))
    r = llm.chat("west prices", [], router.ZONES, "u2", TODAY)
    assert r.intent["zone"] == "WEST" and r.intent["view"] == "prices"
    assert r.used_model and r.note and "keyword guide" in r.note
    r = llm.chat("west prices", [], router.ZONES, "u2", TODAY)
    assert r.intent["view"] == "prices" and r.note and r.remaining == 0


def test_per_user_and_global_limits() -> None:
    fake = FakeProvider(['{"action": "list_zones"}'] * 10)
    llm.set_provider(fake)
    for _ in range(2):
        assert llm.chat("all zones", [], router.ZONES, "alice", TODAY).used_model
    r = llm.chat("nyc forecast", [], router.ZONES, "alice", TODAY)
    assert not r.used_model and r.note and "daily limit" in r.note and r.remaining == 0
    assert r.intent["zone"] == "N.Y.C."  # the guide still navigates
    assert llm.chat("all zones", [], router.ZONES, "bob", TODAY).used_model
    r = llm.chat("all zones", [], router.ZONES, "carol", TODAY)  # 4th call: global cap of 3
    assert not r.used_model and r.note and "budget" in r.note
    assert len(fake.calls) == 3


def test_limiter_rolls_over_at_midnight_utc() -> None:
    lim = llm.ChatLimiter(per_user=1, global_cap=5)
    d1 = datetime(2026, 9, 20, 23, 59, tzinfo=UTC)
    lim.record("u", d1)
    assert lim.check("u", d1) == (False, "daily limit of 1 model replies reached for you")
    d2 = datetime(2026, 9, 21, 0, 1, tzinfo=UTC)
    assert lim.check("u", d2) == (True, None) and lim.remaining("u", d2) == 1


def test_parse_intent_and_gemini_contents() -> None:
    assert llm.parse_intent('```json\n{"action": "help"}\n```') == {"action": "help"}
    assert llm.parse_intent("no json here") is None
    assert llm.parse_intent("[1, 2]") is None
    contents = llm.VertexGemini.contents(
        [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": ""},
        ]
    )
    assert contents == [
        {"role": "user", "parts": [{"text": "a"}]},
        {"role": "model", "parts": [{"text": "b"}]},
    ]
    with pytest.raises(ValueError):
        llm.VertexGemini("", "us-central1", "gemini")
