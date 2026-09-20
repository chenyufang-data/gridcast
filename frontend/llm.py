"""The chat model behind the panel: Gemini on Vertex AI, with limits and a keyword fallback.

Tiers, in order:

1. **Gemini via Vertex AI** (``LLM_PROVIDER=vertex``): the deployed choice. On the GCE VM
   the SDK authenticates with the VM's service account (Application Default
   Credentials), so no key file exists anywhere; locally ``gcloud auth
   application-default login`` does the same. Usage is billable, so every call passes
   :class:`ChatLimiter`: ``CHAT_LIMIT_PER_USER_DAY`` per visitor (IP behind the proxy)
   and ``CHAT_LIMIT_GLOBAL_DAY`` for the whole demo.
2. **GitHub Models** (``LLM_PROVIDER=github``): the free tier, same prompt, when a
   ``GITHUB_TOKEN`` with the Models permission is set.
3. **The keyword guide** (``frontend/router.py``): always available; takes over when no
   provider is configured, the visitor is over the limit, or the model fails or returns
   something that is not an intent.

Every tier produces the same intent dict, so the app does not care which one answered;
``note`` tells the visitor when the guide stepped in.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import requests

from frontend import router

log = logging.getLogger(__name__)


def _today_et() -> date:
    """The market's calendar day when the caller gives none."""
    return datetime.now(ZoneInfo("America/New_York")).date()


PROVIDER = (os.environ.get("LLM_PROVIDER") or "auto").lower()
VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT") or ""
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION") or "us-central1"
VERTEX_MODEL = os.environ.get("VERTEX_MODEL") or "gemini-2.5-flash-lite"
GITHUB_TOKEN = (os.environ.get("GITHUB_TOKEN") or "").strip()
GITHUB_MODELS_URL = os.environ.get(
    "GITHUB_MODELS_URL", "https://models.github.ai/inference/chat/completions"
)
GITHUB_MODELS_MODEL = os.environ.get("GITHUB_MODELS_MODEL", "openai/gpt-4o-mini")
PER_USER_PER_DAY = int(os.environ.get("CHAT_LIMIT_PER_USER_DAY") or 20)
GLOBAL_PER_DAY = int(os.environ.get("CHAT_LIMIT_GLOBAL_DAY") or 300)
HISTORY_TURNS = 10
TIMEOUT = 20.0

SYSTEM_PROMPT = """You are the gridcast assistant, built into a NYISO day-ahead load
forecasting web app, and you drive its navigation while answering questions.

Reply with ONE JSON object and nothing else:
{"action": "list_zones" | "show_zone" | "answer" | "help",
 "zone": "<one of the zone names below, or null>",
 "view": "forecast" | "schedule" | "compare" | "prices" | "load",
 "granularity": "slot" | "hour" | "day",
 "start": "YYYY-MM-DD or null", "end": "YYYY-MM-DD or null", "target": "YYYY-MM-DD or null",
 "message": "<for action=answer: your reply, Markdown allowed, concise but substantive>"}

Choosing the action:
- "show_zone": the user wants a zone's forecast (view=forecast; "target" = the day),
  the DAM bid sheet (bid / schedule -> view=schedule), forecast-vs-actual accuracy or
  dollars (accuracy / error / MAPE / imbalance -> view=compare, with start/end),
  day-ahead vs real-time prices (view=prices) or the actual load history (view=load).
  Resolve zone aliases (NYC = N.Y.C., Long Island = LONGIL, statewide = NYCA) and
  references to earlier turns ("same zone, last week").
- "list_zones": the overview of every zone.
- "answer": greetings, explanations, advice. Write the reply in "message".

App facts you may use: NYISO day-ahead market, bids close 05:00 ET the day before (D-1),
one MW bid per hour; deviations settle at the real-time price, so the cost of an error
is (actual - bid) x (RT - DA) per 15-min slot. Two models: a Temporal Fusion Transformer
(served as ONNX, refit monthly) and LightGBM trees (retrained on demand, the fallback);
every forecast row names its model. The band is P10-P90 rescaled by split conformal on
the trailing 30 scored days. The alpha-bid is the newsvendor quantile of the trailing
30-day RT-DA spread. Accuracy is hourly MAPE next to NYISO's own pre-close forecast
(isolf); over the 12-month backtest the TFT matches the ISO (4.80 vs 4.85 pooled).
Data: NYISO public MIS archive, fetched at runtime; weather: Open-Meteo.

Today's date and the zone list are given with each message. Output ONLY the JSON."""


class Provider(Protocol):
    name: str

    def complete(self, system: str, messages: Sequence[dict[str, str]], timeout: float) -> str: ...


# ─── providers ────────────────────────────────────────────────────────────────────────
class VertexGemini:
    """Gemini through the Vertex AI endpoint of the google-genai SDK (ADC, no key)."""

    name = "vertex"

    def __init__(self, project: str, location: str, model: str) -> None:
        if not project:
            raise ValueError("VERTEX_PROJECT (or GOOGLE_CLOUD_PROJECT) is not set")
        self.project, self.location, self.model = project, location, model
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from google import genai  # lazy: the SDK is optional locally

            self._client = genai.Client(vertexai=True, project=self.project, location=self.location)
        return self._client

    @staticmethod
    def contents(messages: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
        """OpenAI-style turns -> Gemini contents (assistant turns become the model role)."""
        return [
            {
                "role": "model" if m["role"] == "assistant" else "user",
                "parts": [{"text": m["content"]}],
            }
            for m in messages
            if m.get("content")
        ]

    def complete(self, system: str, messages: Sequence[dict[str, str]], timeout: float) -> str:
        from google.genai import types

        client = self._get_client()
        response = client.models.generate_content(
            model=self.model,
            contents=self.contents(messages),
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.3,
                max_output_tokens=600,
                response_mime_type="application/json",
                http_options=types.HttpOptions(timeout=int(timeout * 1000)),
            ),
        )
        return str(response.text or "")


class GitHubModels:
    """The free GitHub Models tier through its OpenAI-compatible endpoint."""

    name = "github"

    def __init__(self, token: str, url: str = GITHUB_MODELS_URL, model: str = GITHUB_MODELS_MODEL):
        if not token:
            raise ValueError("GITHUB_TOKEN is not set")
        self.token, self.url, self.model = token, url, model

    def complete(self, system: str, messages: Sequence[dict[str, str]], timeout: float) -> str:
        resp = requests.post(
            self.url,
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": [{"role": "system", "content": system}, *messages],
                "temperature": 0.3,
                "max_tokens": 600,
            },
            timeout=timeout,
        )
        if resp.status_code == 429:
            raise RuntimeError("GitHub Models rate limit (429)")
        resp.raise_for_status()
        return str(resp.json()["choices"][0]["message"]["content"])


_provider: Provider | None = None
_provider_error: str | None = None
_resolved = False


def provider() -> tuple[Provider | None, str | None]:
    """The configured provider (built once) or ``(None, reason)``."""
    global _provider, _provider_error, _resolved
    if _resolved:
        return _provider, _provider_error
    _resolved = True
    try:
        if PROVIDER == "vertex" or (PROVIDER == "auto" and VERTEX_PROJECT):
            _provider = VertexGemini(VERTEX_PROJECT, VERTEX_LOCATION, VERTEX_MODEL)
        elif PROVIDER == "github" or (PROVIDER == "auto" and GITHUB_TOKEN):
            _provider = GitHubModels(GITHUB_TOKEN)
        elif PROVIDER in ("auto", "none"):
            _provider_error = "no chat model configured"
        else:
            _provider_error = f"unknown LLM_PROVIDER {PROVIDER!r}"
    except ValueError as exc:
        _provider_error = str(exc)
    return _provider, _provider_error


def set_provider(p: Provider | None, error: str | None = None) -> None:
    """Tests inject a fake provider (or none)."""
    global _provider, _provider_error, _resolved
    _provider, _provider_error, _resolved = p, error, True


# ─── limits ───────────────────────────────────────────────────────────────────────────
class ChatLimiter:
    """Daily request counts per visitor and for the whole demo (UTC day, in memory)."""

    def __init__(self, per_user: int = PER_USER_PER_DAY, global_cap: int = GLOBAL_PER_DAY):
        self.per_user, self.global_cap = per_user, global_cap
        self._day: date | None = None
        self._users: dict[str, int] = {}
        self._total = 0
        self._lock = threading.Lock()

    def _roll(self, now: datetime) -> None:
        if self._day != now.date():
            self._day, self._users, self._total = now.date(), {}, 0

    def check(self, user_key: str, now: datetime | None = None) -> tuple[bool, str | None]:
        """Whether one more model call is allowed for `user_key` (and why not)."""
        now = now or datetime.now(UTC)
        with self._lock:
            self._roll(now)
            if self._users.get(user_key, 0) >= self.per_user:
                return False, f"daily limit of {self.per_user} model replies reached for you"
            if self._total >= self.global_cap:
                return False, "the demo's daily model budget is used up"
            return True, None

    def record(self, user_key: str, now: datetime | None = None) -> None:
        now = now or datetime.now(UTC)
        with self._lock:
            self._roll(now)
            self._users[user_key] = self._users.get(user_key, 0) + 1
            self._total += 1

    def remaining(self, user_key: str, now: datetime | None = None) -> int:
        now = now or datetime.now(UTC)
        with self._lock:
            self._roll(now)
            return max(0, self.per_user - self._users.get(user_key, 0))


limiter = ChatLimiter()


# ─── the chat entry point ─────────────────────────────────────────────────────────────
@dataclass
class ChatResult:
    intent: dict[str, Any]
    note: str | None  # shown in italics when a fallback stepped in
    used_model: bool
    remaining: int | None  # model replies left today for this visitor (None = no model)


def parse_intent(content: str) -> dict[str, Any] | None:
    """First JSON object in a model reply, or None."""
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _clip_history(
    history: Sequence[tuple[str, str]], limit: int = HISTORY_TURNS
) -> list[dict[str, str]]:
    return [
        {"role": role, "content": str(content)}
        for role, content in list(history)[-limit:]
        if role in ("user", "assistant") and content
    ]


def _fallback(
    text: str, zone_names: Sequence[str], today: date, note: str | None
) -> dict[str, Any]:
    intent = router.keyword_route(text, tuple(zone_names), today)
    if intent is not None:
        return intent
    return {
        "action": "answer",
        "message": (
            ("" if note else "The chat model is not available right now. ")
            + "I can still navigate: name a zone and a view, for example "
            '"NYC forecast" or "West prices last week". Type **help** for the list.'
        ),
    }


def chat(
    text: str,
    history: Sequence[tuple[str, str]],
    zone_names: Sequence[str],
    user_key: str,
    today: date | None = None,
) -> ChatResult:
    """Route one typed message: the model when configured and allowed, else the keyword guide."""
    today = today or _today_et()
    p, why = provider()
    if p is None:
        return ChatResult(_fallback(text, zone_names, today, None), None, False, None)
    allowed, reason = limiter.check(user_key)
    if not allowed:
        note = f"{reason}; the keyword guide is answering"
        return ChatResult(_fallback(text, zone_names, today, note), note, False, 0)
    user_msg = f"Today: {today.isoformat()}\nZones: {', '.join(zone_names)}\n\nUser message: {text}"
    messages = [*_clip_history(history), {"role": "user", "content": user_msg}]
    limiter.record(user_key)
    try:
        raw = p.complete(SYSTEM_PROMPT, messages, TIMEOUT)
        intent = parse_intent(raw)
    except Exception as exc:  # any provider trouble: the guide answers
        log.warning("chat model %s failed: %s", p.name, exc)
        intent = None
    if intent is None:
        note = "the chat model did not answer; the keyword guide is answering"
        return ChatResult(
            _fallback(text, zone_names, today, note), note, True, limiter.remaining(user_key)
        )
    return ChatResult(
        router.sanitize(intent, tuple(zone_names)), None, True, limiter.remaining(user_key)
    )


def status() -> dict[str, Any]:
    p, why = provider()
    return {
        "provider": p.name if p else None,
        "model": getattr(p, "model", None),
        "error": why,
        "per_user_per_day": limiter.per_user,
        "global_per_day": limiter.global_cap,
    }
