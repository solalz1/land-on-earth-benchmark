import json

import httpx
import pytest

from loe import config
from loe.client import Client, Fatal, api_key, request
from loe.templates import generic_renderer, renderer_from

CFG = config.load()


async def nosleep(_):
    return None


def ok_body(text="Land"):
    return {
        "provider": "Parasail",
        "model": "x",
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"cost": 1e-6},
    }


def test_config_has_the_25_models():
    assert len(CFG.models) == 25  # the 20 of the tweet reproduction + Mistral's size ladder
    assert len({m.provider for m in CFG.models}) == 6
    assert sum(not m.logprobs for m in CFG.models) == 6  # the Mistral models
    assert {m.key for m in CFG.models if m.rpm} == {"deepseek-v4-pro", "kimi-k3", "kimi-k2.6", "qwen3.5-397b-a17b"}
    assert {m.key for m in CFG.models if m.top_logprobs == 5} == {"qwen3.5-122b-a10b", "qwen3.5-27b", "qwen3.6-27b"}
    assert CFG.strategies["chat_low"].reasons and CFG.strategies["text_low"].reasons
    assert not any(CFG.strategies[s].reasons for s in ("chat_none", "chat", "raw", "text_none", "text"))


def test_top_logprobs_follow_the_model():
    _, body = request(CFG.model("qwen3.6-27b"), CFG.strategies["chat_none"], 0.0, 0.0)
    assert body["top_logprobs"] == 5


async def test_rate_limiter_spaces_requests():
    import time

    from loe.client import RateLimiter

    lim = RateLimiter(rpm=600)  # one request every 0.1 s
    t0 = time.monotonic()
    for _ in range(4):
        await lim.wait()
    assert 0.29 < time.monotonic() - t0 < 0.6


def test_request_pins_the_provider():
    m = CFG.model("glm-5.3")
    path, body = request(m, CFG.strategies["chat_none"], 41.0, -73.0)
    assert path == "/chat/completions"
    assert body["provider"] == {
        "only": ["parasail"],
        "allow_fallbacks": False,
        "require_parameters": True,
        "quantizations": ["fp8"],
    }
    assert body["logprobs"] is True and body["top_logprobs"] == 20
    assert body["reasoning"] == {"effort": "none"}
    assert body["temperature"] == 0 and body["max_tokens"] == 8
    assert body["messages"][0]["content"].startswith("Coordinates: 41.0, -73.0")


def test_text_strategy_has_no_logprobs_and_no_quantization_filter():
    m = CFG.model("mistral-large-3")
    _, body = request(m, CFG.strategies["text"], 0.0, 0.0)
    assert "logprobs" not in body and "reasoning" not in body
    assert "quantizations" not in body["provider"]


def test_raw_strategy_uses_completions_and_prefill():
    m = CFG.model("gpt-oss-20b")
    path, body = request(m, CFG.strategies["raw"], 1.0, 2.0, generic_renderer(m.raw_prefill))
    assert path == "/completions"
    assert body["prompt"].endswith("<|im_start|>assistant\n<|channel|>final<|message|>")
    assert "Coordinates: 1.0, 2.0" in body["prompt"]
    with pytest.raises(ValueError):
        request(m, CFG.strategies["raw"], 1.0, 2.0, None)


def test_template_rendering_with_thinking_switch():
    spec = {
        "template": "{{ bos_token }}{% for m in messages %}<u>{{ m.content }}</u>{% endfor %}"
        "{% if add_generation_prompt %}<a>{% if not enable_thinking %}<think></think>{% endif %}{% endif %}",
        "bos_token": "<s>",
    }
    out = renderer_from(spec, "")(10.0, 20.0)
    assert out.startswith("<s><u>Coordinates: 10.0, 20.0") and out.endswith("<a><think></think>")


def test_missing_key_is_explained(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(Fatal, match="zshrc"):
        api_key()


async def test_retries_rate_limits_then_succeeds():
    calls = []

    def handler(req):
        calls.append(json.loads(req.content))
        if len(calls) < 3:
            return httpx.Response(429, json={"error": {"code": 429, "message": "slow down"}})
        return httpx.Response(200, json=ok_body())

    async with Client("k", transport=httpx.MockTransport(handler), sleep=nosleep) as c:
        r = await c.ask(CFG.model("qwen3.5-9b"), CFG.strategies["chat"], 0.0, 0.0)
    assert r.ok and r.attempts == 3


async def test_error_inside_a_200_is_retried():
    n = 0

    def handler(req):
        nonlocal n
        n += 1
        if n == 1:
            return httpx.Response(200, json={"error": {"code": 502, "message": "provider hiccup"}})
        return httpx.Response(200, json=ok_body())

    async with Client("k", transport=httpx.MockTransport(handler), sleep=nosleep) as c:
        r = await c.ask(CFG.model("qwen3.5-9b"), CFG.strategies["chat"], 0.0, 0.0)
    assert r.ok and r.attempts == 2


async def test_bad_request_is_not_retried():
    n = 0

    def handler(req):
        nonlocal n
        n += 1
        return httpx.Response(400, json={"error": {"code": 400, "message": "Reasoning is mandatory"}})

    async with Client("k", transport=httpx.MockTransport(handler), sleep=nosleep) as c:
        r = await c.ask(CFG.model("gpt-oss-20b"), CFG.strategies["chat_none"], 0.0, 0.0)
    assert not r.ok and n == 1 and "mandatory" in r.error


async def test_gives_up_after_max_attempts():
    def handler(req):
        raise httpx.ConnectError("boom")

    async with Client("k", transport=httpx.MockTransport(handler), sleep=nosleep, max_attempts=4) as c:
        r = await c.ask(CFG.model("qwen3.5-9b"), CFG.strategies["chat"], 0.0, 0.0)
    assert not r.ok and r.attempts == 4 and "ConnectError" in r.error


@pytest.mark.parametrize("status", [401, 402])
async def test_bad_key_or_no_credit_stops_everything(status):
    def handler(req):
        return httpx.Response(status, json={"error": {"code": status, "message": "nope"}})

    async with Client("k", transport=httpx.MockTransport(handler), sleep=nosleep) as c:
        with pytest.raises(Fatal):
            await c.ask(CFG.model("qwen3.5-9b"), CFG.strategies["chat"], 0.0, 0.0)


async def test_key_is_sent_only_as_bearer_header():
    seen = {}

    def handler(req):
        seen["auth"] = req.headers["authorization"]
        seen["body"] = req.content.decode()
        return httpx.Response(200, json=ok_body())

    async with Client("sk-secret", transport=httpx.MockTransport(handler), sleep=nosleep) as c:
        await c.ask(CFG.model("qwen3.5-9b"), CFG.strategies["chat"], 0.0, 0.0)
    assert seen["auth"] == "Bearer sk-secret"
    assert "sk-secret" not in seen["body"]
