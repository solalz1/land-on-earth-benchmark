"""OpenRouter client: one question per request, pinned provider, retries with backoff.

Every request pins its provider: `only` that provider, no fallback to another one, and
`require_parameters` so a provider that would silently ignore logprobs or the reasoning switch
is refused instead of answering differently. The provider that actually served each answer
comes back in the response and is stored with the prediction.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import httpx

from loe.config import OPENROUTER, REPO_URL, Model, Strategy
from loe.prompts import question
from loe.templates import Renderer

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}
FATAL_STATUS = {401: "clé API refusée", 402: "crédits OpenRouter épuisés"}


class Fatal(Exception):
    """Stops the whole run: bad key or no credits left."""


@dataclass
class Reply:
    ok: bool
    body: dict[str, Any] | None
    status: int | None
    error: str | None
    attempts: int
    latency: float


def api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise Fatal(
            "OPENROUTER_API_KEY absente. Ajoute `export OPENROUTER_API_KEY=...` à ~/.zshrc, "
            "ouvre un nouveau terminal, puis relance."
        )
    return key


def request(model: Model, strategy: Strategy, lat: float, lon: float, render: Renderer | None = None):
    """(path, JSON body) of the request for one point."""
    provider: dict[str, Any] = {
        "only": [model.provider],
        "allow_fallbacks": False,
        "require_parameters": True,
    }
    if model.quantizations:
        provider["quantizations"] = list(model.quantizations)
    body: dict[str, Any] = {
        "model": model.id,
        "temperature": 0,
        "max_tokens": strategy.max_tokens,
        "provider": provider,
        "usage": {"include": True},
    }
    if strategy.logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = model.top_logprobs
    if strategy.reasoning:
        body["reasoning"] = dict(strategy.reasoning)
    if strategy.mode == "raw":
        if render is None:
            raise ValueError(f"{model.key} : pas de template pour le prompt brut")
        body["prompt"] = render(lat, lon)
        return "/completions", body
    body["messages"] = [{"role": "user", "content": question(lat, lon)}]
    return "/chat/completions", body


def _error_text(body: Any, fallback: str) -> str:
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        err = body["error"]
        meta = err.get("metadata") or {}
        raw = meta.get("raw") if isinstance(meta, dict) else None
        msg = str(err.get("message") or fallback)
        if raw and str(raw) not in msg:
            msg += f" | {raw}"
        return msg[:500]
    return fallback[:500]


class RateLimiter:
    """Spaces request starts evenly: at most `rpm` per minute, shared by all workers of a model."""

    def __init__(self, rpm: float):
        self.interval = 60.0 / rpm
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.interval
        if start > now:
            await asyncio.sleep(start - now)


class Client:
    def __init__(
        self,
        key: str,
        base_url: str = OPENROUTER,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 90.0,
        max_attempts: int = 8,
        backoff: float = 1.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rate_limits: bool = True,
    ):
        self.http = httpx.AsyncClient(
            base_url=base_url,
            transport=transport,
            timeout=httpx.Timeout(timeout, connect=15.0),
            limits=httpx.Limits(max_connections=512, max_keepalive_connections=128),
            headers={
                "Authorization": f"Bearer {key}",
                "HTTP-Referer": REPO_URL,
                "X-Title": "Land on Earth",
            },
        )
        self.max_attempts = max_attempts
        self.backoff = backoff
        self.sleep = sleep
        self.rate_limits = rate_limits
        self._limiters: dict[str, RateLimiter] = {}

    def limiter(self, model: Model) -> RateLimiter | None:
        """The model's request pacing (configs/models.yaml `rpm`), shared by every caller."""
        if not (self.rate_limits and model.rpm):
            return None
        if model.id not in self._limiters:
            self._limiters[model.id] = RateLimiter(model.rpm)
        return self._limiters[model.id]

    async def close(self) -> None:
        await self.http.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()

    def _delay(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(120.0, float(retry_after))
            except ValueError:
                pass
        return min(60.0, self.backoff * 2**attempt) * (0.5 + random.random())

    async def post(self, path: str, body: dict[str, Any], limiter: RateLimiter | None = None) -> Reply:
        t0 = time.monotonic()
        error, status = "aucune tentative", None
        for attempt in range(1, self.max_attempts + 1):
            retry_after = None
            if limiter is not None:  # every attempt, retries included, respects the model's pace
                await limiter.wait()
            try:
                r = await self.http.post(path, json=body)
                status = r.status_code
                retry_after = r.headers.get("retry-after")
                try:
                    data = r.json()
                except ValueError:
                    data = None
                if status in FATAL_STATUS:
                    raise Fatal(f"{FATAL_STATUS[status]} ({_error_text(data, r.text)})")
                if status == 200 and isinstance(data, dict) and not data.get("error"):
                    if data.get("choices"):
                        return Reply(True, data, status, None, attempt, time.monotonic() - t0)
                    error = "réponse sans choix"
                else:
                    if status == 200 and isinstance(data, dict):  # error reported inside a 200
                        err = data["error"]
                        code = err.get("code") if isinstance(err, dict) else None
                        status = int(code) if isinstance(code, int) or str(code).isdigit() else 502
                    error = f"HTTP {status}: {_error_text(data, r.text)}"
                    if status not in RETRY_STATUS:
                        return Reply(False, data, status, error, attempt, time.monotonic() - t0)
            except Fatal:
                raise
            except httpx.HTTPError as e:
                error, status = f"{type(e).__name__}: {e}"[:500], None
            if attempt < self.max_attempts:
                await self.sleep(self._delay(attempt - 1, retry_after))
        return Reply(False, None, status, error, self.max_attempts, time.monotonic() - t0)

    async def ask(self, model: Model, strategy: Strategy, lat: float, lon: float, render: Renderer | None = None) -> Reply:
        path, body = request(model, strategy, lat, lon, render)
        return await self.post(path, body, self.limiter(model))

    async def get(self, path: str) -> Any:
        r = await self.http.get(path)
        if r.status_code in FATAL_STATUS:
            raise Fatal(FATAL_STATUS[r.status_code])
        r.raise_for_status()
        return r.json()
