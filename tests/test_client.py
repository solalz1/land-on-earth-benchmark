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


def test_config_has_the_24_models():
    keys = {m.key for m in CFG.models}
    assert len(CFG.models) == 24  # 19 of the starting list + Mistral's size ladder
    assert "mistral-large-3" not in keys  # 0.5 request per second at Mistral: 9 hours on its own
    assert {m.provider for m in CFG.models} == {"parasail", "novita", "alibaba", "dekallm", "mistral", "siliconflow"}
    assert {m.key for m in CFG.models if m.byok} == {
        "deepseek-v4-pro", "kimi-k2.6", "glm-5.2", "glm-5.3",  # your SiliconFlow key
        "kimi-k3", "qwen3.5-397b-a17b", "qwen3.5-122b-a10b",  # your Alibaba key
        "ministral-3-3b", "ministral-3-8b", "ministral-3-14b", "mistral-small-4", "mistral-medium-3.5",  # Mistral
    }
    assert all(m.provider in ("siliconflow", "alibaba", "mistral") for m in CFG.models if m.byok)
    # the 8 models OpenRouter caps at 20 per minute for new accounts all go through your keys
    capped = {"deepseek-v4-pro", "kimi-k3", "kimi-k2.6", "qwen3.5-397b-a17b",
              "glm-5.2", "glm-5.3", "qwen3.5-122b-a10b", "mistral-medium-3.5"}
    assert capped <= {m.key for m in CFG.models if m.byok}
    assert not any(m.rpm and m.rpm < 30 for m in CFG.models)  # no 19-per-minute pacing left
    assert all(m.rpm and m.rpm <= 480 for m in CFG.models if m.provider == "siliconflow")  # 500 per minute
    assert sum(not m.logprobs for m in CFG.models) == 9  # the 5 Mistral + the 4 at SiliconFlow
    for m in CFG.models:  # a provider without logprobs is only asked without them
        if not m.logprobs:
            assert not any(CFG.strategies[s].logprobs for s in m.strategies), m.key
    assert {m.key for m in CFG.models if m.top_logprobs == 5} == {
        "qwen3.5-122b-a10b", "qwen3.5-27b", "qwen3.6-27b", "qwen3.5-397b-a17b", "kimi-k3"
    }
    assert CFG.model("kimi-k3").temperature is None and CFG.model("qwen3.5-9b").temperature == 0
    per_provider: dict[str, int] = {}
    for m in CFG.models:
        per_provider[m.provider] = per_provider.get(m.provider, 0) + m.concurrency
    assert max(per_provider.values()) <= CFG.provider_concurrency  # no model waits for a provider slot
    assert CFG.strategies["chat_low"].max_tokens == CFG.strategies["text_low"].max_tokens == 2048
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
    m = CFG.model("qwen3.8-27b")
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
    m = CFG.model("mistral-small-4")
    _, body = request(m, CFG.strategies["text"], 0.0, 0.0)
    assert "logprobs" not in body and "reasoning" not in body
    assert "quantizations" not in body["provider"]


def test_temperature_left_out_where_the_provider_refuses_it():
    _, body = request(CFG.model("kimi-k3"), CFG.strategies["chat_none"], 0.0, 0.0)
    assert "temperature" not in body and body["top_logprobs"] == 5
    assert body["provider"] == {"only": ["alibaba"], "allow_fallbacks": False, "require_parameters": True}
    _, body = request(CFG.model("glm-5.3"), CFG.strategies["text_low"], 0.0, 0.0)
    assert body["temperature"] == 0 and body["reasoning"] == {"effort": "low"} and "logprobs" not in body
    assert body["provider"]["only"] == ["siliconflow"] and body["provider"]["quantizations"] == ["fp8"]


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


@pytest.mark.parametrize("status", [401, 403])
async def test_a_provider_refusing_your_key_fails_the_request_not_the_run(status):
    n = 0

    def handler(req):
        nonlocal n
        n += 1
        return httpx.Response(status, json={"error": {"code": status, "message": "Provider returned error",
                                                      "metadata": {"provider_name": "Parasail",
                                                                   "raw": "Unauthorized. Invalid token."}}})

    async with Client("k", transport=httpx.MockTransport(handler), sleep=nosleep) as c:
        r = await c.ask(CFG.model("qwen3.5-9b"), CFG.strategies["chat"], 0.0, 0.0)
    assert not r.ok and r.status == status and n == 1 and "Invalid token" in r.error


async def test_in_flight_budget_is_retried():
    n = 0

    def handler(req):
        nonlocal n
        n += 1
        if n < 3:
            return httpx.Response(402, headers={"retry-after": "1"}, json={"error": {
                "code": 402,
                "message": "This request would exceed your available credits given your current in-flight requests.",
                "metadata": {"reason": "in_flight_budget_exhausted", "limit_source": "openrouter_in_flight_budget"},
            }})
        return httpx.Response(200, json=ok_body())

    async with Client("k", transport=httpx.MockTransport(handler), sleep=nosleep) as c:
        r = await c.ask(CFG.model("qwen3.5-9b"), CFG.strategies["chat"], 0.0, 0.0)
    assert r.ok and r.attempts == 3


async def test_empty_credits_still_stop_everything():
    def handler(req):
        return httpx.Response(402, json={"error": {"code": 402, "message": "Insufficient credits",
                                                   "metadata": {"limit_source": "openrouter_credits"}}})

    async with Client("k", transport=httpx.MockTransport(handler), sleep=nosleep) as c:
        with pytest.raises(Fatal, match="crédits"):
            await c.ask(CFG.model("qwen3.5-9b"), CFG.strategies["chat"], 0.0, 0.0)


def test_cost_counts_what_your_provider_key_billed():
    from loe import store

    openrouter = {"cost": 2e-5, "byok": False, "upstream_cost": 1.9e-5}
    mine = {"cost": 0.0, "byok": True, "upstream_cost": 3e-5}
    unflagged = {"cost": 0.0, "byok": None, "upstream_cost": 3e-5}  # OpenRouter left the flag out
    old = {"cost": 2e-5}  # recorded before the run noted who billed it
    assert [store.is_byok(r) for r in (openrouter, mine, unflagged, old)] == [False, True, True, False]
    assert [store.charged(r) for r in (openrouter, mine, unflagged, old)] == [2e-5, 3e-5, 3e-5, 2e-5]


def test_open_files_limit_is_raised():
    import resource

    from loe.cli import raise_open_files

    raise_open_files()
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    assert soft == resource.RLIM_INFINITY or soft >= min(4096, hard)
