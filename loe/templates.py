"""Raw prompts: render each model's official chat template, then prefill the answer.

The template comes from the model's Hugging Face repo (chat_template.jinja or
tokenizer_config.json) and is cached in .cache/templates/. Gated repos (Llama, Gemma) need
HF_TOKEN in the environment; without it the "raw" strategy is simply unavailable for them.
"""

from __future__ import annotations

import json
import os
from datetime import date
from typing import Any, Callable

import httpx
from jinja2 import TemplateError
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

from loe.config import CACHE, Model
from loe.prompts import question

HF = "https://huggingface.co"
Renderer = Callable[[float, float], str]


def _token(value: Any) -> str:
    if isinstance(value, dict):
        return value.get("content") or ""
    return value or ""


def _cache_path(repo: str):
    return CACHE / "templates" / (repo.replace("/", "__") + ".json")


def fetch(repo: str, timeout: float = 20.0) -> dict[str, str]:
    """{'template', 'bos_token', 'eos_token'} for a repo, from cache or Hugging Face."""
    path = _cache_path(repo)
    if path.exists():
        return json.loads(path.read_text())
    headers = {}
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
        r = client.get(f"{HF}/{repo}/resolve/main/tokenizer_config.json")
        if r.status_code in (401, 403):
            raise ValueError(f"{repo} : dépôt protégé, il faut HF_TOKEN")
        cfg = r.json() if r.status_code == 200 else {}
        template = cfg.get("chat_template")
        if isinstance(template, list):  # named templates: keep the default one
            named = {t.get("name"): t.get("template") for t in template}
            template = named.get("default") or next(iter(named.values()), None)
        j = client.get(f"{HF}/{repo}/resolve/main/chat_template.jinja")
        if j.status_code == 200 and j.text.strip():
            template = j.text
    if not template:
        raise ValueError(f"{repo} : pas de chat template")
    out = {
        "template": template,
        "bos_token": _token(cfg.get("bos_token")),
        "eos_token": _token(cfg.get("eos_token")),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False))
    return out


def _env() -> ImmutableSandboxedEnvironment:
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])

    def raise_exception(message: str):
        raise TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: date.today().strftime(fmt)
    return env


def renderer_from(spec: dict[str, str], prefill: str = "") -> Renderer:
    template = _env().from_string(spec["template"])

    def render(lat: float, lon: float) -> str:
        text = template.render(
            messages=[{"role": "user", "content": question(lat, lon)}],
            add_generation_prompt=True,
            enable_thinking=False,
            thinking=False,
            reasoning_effort="low",
            tools=None,
            bos_token=spec.get("bos_token", ""),
            eos_token=spec.get("eos_token", ""),
        )
        return text + prefill

    return render


def renderer(model: Model) -> Renderer | None:
    """Raw-prompt renderer for a model, or None if its template cannot be fetched."""
    if not model.hf_repo:
        return None
    try:
        r = renderer_from(fetch(model.hf_repo), model.raw_prefill)
        r(0.0, 0.0)  # fail now rather than mid-run
        return r
    except (httpx.HTTPError, ValueError, TemplateError, KeyError, TypeError):
        return None


def generic_renderer(prefill: str = "") -> Renderer:
    """ChatML stand-in used by the fake API, which has no Hugging Face access."""
    spec = {
        "template": "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}",
    }
    return renderer_from(spec, prefill)
