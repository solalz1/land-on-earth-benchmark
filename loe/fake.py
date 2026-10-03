"""A fake OpenRouter, to test the whole pipeline without spending a cent.

It answers from the ground truth with model-dependent mistakes (more of them near coasts), and
reproduces the behaviours the pilot must cope with: a provider without logprobs (Mistral),
reasoning that cannot be switched off (gpt-oss, Kimi K3), a model without the reasoning
parameter (Llama), markdown around the answer (Gemma), the legacy logprobs format of raw
completions (Qwen), transient 429/502 errors, and the endpoints/key listings used by `check`.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any

import httpx
import numpy as np

from loe import grid
from loe.config import Config, Model
from loe.prompts import coords


def _h(*parts: Any) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def _err(status: int, message: str, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(status, json={"error": {"code": status, "message": message}}, headers=headers)


class FakeOpenRouter:
    def __init__(self, cfg: Config, seed: int = 0, fail_rate: float = 0.01):
        self.cfg = cfg
        self.by_id = {m.id: m for m in cfg.models}
        df = grid.load()
        self.truth = df["truth"].to_numpy()
        self.coast = grid.coastal(self.truth)
        self.rng = random.Random(seed)
        self.fail_rate = fail_rate
        self.requests: list[dict[str, Any]] = []

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    # ---- behaviour of each fake model -------------------------------------------------------
    @staticmethod
    def flags(m: Model) -> dict[str, bool]:
        return {
            "no_logprobs": m.id.startswith("mistralai/"),
            "reasoning_mandatory": "gpt-oss" in m.id,
            "always_thinks": "kimi-k3" in m.id,
            "no_reasoning_param": "llama-4" in m.id,
            "markdown": "gemma" in m.id,
            "legacy_raw": m.id.startswith("qwen/"),
        }

    def skill(self, m: Model) -> float:
        return 0.25 + 0.7 * (_h(m.id, "skill") % 1000) / 1000

    def p_land(self, m: Model, lat: float, lon: float) -> float:
        row = int(round((89 - lat) / 2))
        col = int(round((lon + 179) / 2))
        i = row * grid.N_COLS + col
        t = self.truth[i]
        s = self.skill(m)
        strength = (0.3 + 2.0 * s) if self.coast[i] else (2.5 + 3.0 * s)
        noise = random.Random(_h(m.id, i)).gauss(0, 1.2)
        logit = (2 * t - 1) * strength + noise
        return 1 / (1 + math.exp(-logit))

    # ---- routing ------------------------------------------------------------------------------
    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/api/v1")
        if request.method == "GET" and path == "/key":
            return httpx.Response(
                200, json={"data": {"label": "fake", "limit": 20.0, "usage": 0.0, "limit_remaining": 20.0}}
            )
        if request.method == "GET" and path.startswith("/models/") and path.endswith("/endpoints"):
            return self.endpoints(path.removeprefix("/models/").removesuffix("/endpoints"))
        if request.method == "POST" and path in ("/chat/completions", "/completions"):
            body = json.loads(request.content)
            self.requests.append(body)
            return self.complete(path, body)
        return _err(404, f"not found: {path}")

    def endpoints(self, model_id: str) -> httpx.Response:
        m = self.by_id.get(model_id)
        if m is None:
            return _err(404, "model not found")
        f = self.flags(m)
        params = ["max_tokens", "temperature", "top_p", "stop", "seed", "response_format"]
        if not f["no_logprobs"]:
            params += ["logprobs", "top_logprobs"]
        if not f["no_reasoning_param"]:
            params += ["reasoning", "include_reasoning"]
        quant = (m.quantizations or ("unknown",))[0]
        ep = {
            "name": f"{m.provider.title()} | {m.id}",
            "provider_name": m.provider.title(),
            "tag": f"{m.provider}/{quant}" if quant != "unknown" else m.provider,
            "quantization": quant,
            "context_length": 131072,
            "pricing": {"prompt": f"{m.price[0] / 1e6:.10f}", "completion": f"{m.price[1] / 1e6:.10f}"},
            "supported_parameters": params,
            "status": 0,
        }
        decoy = dict(ep, provider_name="Cheapo", tag="cheapo/fp4", quantization="fp4")
        return httpx.Response(200, json={"data": {"id": m.id, "name": m.name, "endpoints": [decoy, ep]}})

    def complete(self, path: str, body: dict[str, Any]) -> httpx.Response:
        m = self.by_id.get(body.get("model"))
        if m is None:
            return _err(400, "unknown model")
        prov = body.get("provider") or {}
        if prov.get("only") != [m.provider] or prov.get("allow_fallbacks") is not False:
            return _err(400, "fake: provider must be pinned without fallbacks")
        if self.rng.random() < self.fail_rate:
            if self.rng.random() < 0.6:
                return _err(429, "rate limited", {"retry-after": "0"})
            return _err(502, "upstream error")
        f = self.flags(m)
        if f["no_logprobs"] and body.get("logprobs") and prov.get("require_parameters"):
            return _err(404, "No endpoints found that can handle the requested parameters.")
        if f["no_reasoning_param"] and "reasoning" in body and prov.get("require_parameters"):
            return _err(404, "No endpoints found that can handle the requested parameters.")
        effort = (body.get("reasoning") or {}).get("effort")
        raw = path == "/completions"
        if f["reasoning_mandatory"] and effort == "none":
            return _err(400, "Reasoning is mandatory for this endpoint and cannot be disabled.")
        thinks = not raw and (
            (f["reasoning_mandatory"]) or (f["always_thinks"] and effort in ("none", None, "low"))
        )
        if thinks and body.get("max_tokens", 0) < 64:
            # all the budget goes to reasoning: no answer, as with a real thinking model
            return self._reply(m, body, raw, tokens=[], p=None, reasoning="Let me think about where this is", stop="length")

        text = body.get("prompt") if raw else body["messages"][-1]["content"]
        c = coords(text or "")
        if c is None:
            return _err(400, "fake: no coordinates in prompt")
        p = self.p_land(m, *c)
        answer = "Land" if p > 0.5 else "Water"
        lead = " " if _h(m.id, "space") % 2 else ""
        tokens = ["**", answer, "**"] if f["markdown"] else [lead + answer]
        reasoning = "Short reasoning about the coordinates." if thinks else None
        return self._reply(m, body, raw, tokens=tokens, p=p, reasoning=reasoning, stop="stop", lead=lead)

    def _reply(self, m: Model, body, raw: bool, tokens: list[str], p: float | None, reasoning, stop, lead=""):
        want_lp = bool(body.get("logprobs"))
        logprobs = None
        if want_lp and tokens:
            p = min(max(p, 1e-6), 1 - 1e-6)
            alts = [
                (lead + "Land", math.log(p * 0.97)),
                (lead + "Water", math.log((1 - p) * 0.97)),
                (lead + "The", math.log(0.012)),
                (lead + "It", math.log(0.006)),
                (lead + "Ocean", math.log(0.004)),
            ]
            alts.sort(key=lambda a: -a[1])
            top_n = int(body.get("top_logprobs") or 1)
            positions = []
            for tok in tokens:
                if tok.strip() in ("Land", "Water"):
                    top = alts[:top_n]
                    chosen = next(lp for t, lp in alts if t.strip() == tok.strip())
                else:
                    top, chosen = [(tok, -0.01)], -0.01
                positions.append((tok, chosen, top))
            if raw and self.flags(m)["legacy_raw"]:
                logprobs = {
                    "tokens": [t for t, _, _ in positions],
                    "token_logprobs": [lp for _, lp, _ in positions],
                    "top_logprobs": [{t: lp for t, lp in top} for _, _, top in positions],
                }
            else:
                logprobs = {
                    "content": [
                        {
                            "token": t,
                            "logprob": lp,
                            "top_logprobs": [{"token": a, "logprob": b} for a, b in top],
                        }
                        for t, lp, top in positions
                    ]
                }
        content = "".join(tokens).strip()
        n_reason = 40 if reasoning else 0
        in_tok = 61 if not raw else max(20, len(body.get("prompt", "")) // 4)
        out_tok = len(tokens) + n_reason
        cost = (in_tok * m.price[0] + out_tok * m.price[1]) / 1e6
        choice: dict[str, Any] = {"index": 0, "finish_reason": stop, "logprobs": logprobs}
        if raw:
            choice["text"] = content
        else:
            choice["message"] = {"role": "assistant", "content": content, "reasoning": reasoning}
        return httpx.Response(
            200,
            json={
                "id": f"gen-{_h(body.get('model'), len(self.requests))}",
                "model": m.id,
                "provider": m.provider.title() if m.provider != "dekallm" else "DekaLLM",
                "choices": [choice],
                "usage": {
                    "prompt_tokens": in_tok,
                    "completion_tokens": out_tok,
                    "total_tokens": in_tok + out_tok,
                    "cost": cost,
                    "completion_tokens_details": {"reasoning_tokens": n_reason},
                },
            },
        )


def accuracy_of(fake: FakeOpenRouter, m: Model) -> float:
    """Area-weighted accuracy the fake model should reach, for tests."""
    df = grid.load()
    p = np.array([fake.p_land(m, a, b) for a, b in zip(df["lat"], df["lon"])])
    pred = (p > 0.5).astype(int)
    w = df["weight"].to_numpy()
    return float((w * (pred == df["truth"].to_numpy())).sum() / w.sum())
