"""Paths and configs/models.yaml loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "models.yaml"
GRID = ROOT / "data" / "grid.csv"
RESULTS = ROOT / "results"
CACHE = ROOT / ".cache"

OPENROUTER = "https://openrouter.ai/api/v1"
REPO_URL = "https://github.com/solalz1/land-on-earth-benchmark"
TOP_LOGPROBS = 20
EUR_PER_USD = 1 / 1.1235  # rate used in the spec, 3 October 2026
CREDIT_FEE = 0.055  # OpenRouter fee on credit purchases


@dataclass(frozen=True)
class Strategy:
    name: str
    mode: str  # "chat" or "raw"
    logprobs: bool
    max_tokens: int
    reasoning: dict[str, Any] | None = None

    @property
    def reasons(self) -> bool:
        """True when the strategy lets the model reason (minimal effort), False when it switches it off."""
        return (self.reasoning or {}).get("effort") not in (None, "none")


@dataclass(frozen=True)
class Model:
    key: str
    name: str
    lab: str
    id: str
    provider: str
    price: tuple[float, float]
    strategies: tuple[str, ...]
    quantizations: tuple[str, ...] | None = None
    logprobs: bool = True
    hf_repo: str | None = None
    raw_prefill: str = ""
    concurrency: int = 8
    top_logprobs: int = TOP_LOGPROBS
    rpm: float | None = None  # requests per minute allowed for this model (None = no limit)
    temperature: float | None = 0.0  # None: the provider does not let it be set, the request omits it
    byok: bool = False  # served through your own key at the provider (OpenRouter > Integrations)


@dataclass
class Config:
    models: list[Model]
    strategies: dict[str, Strategy]
    provider_concurrency: int = 128
    raw: dict[str, Any] = field(default_factory=dict)

    def model(self, key: str) -> Model:
        for m in self.models:
            if m.key == key:
                return m
        raise KeyError(f"modèle inconnu : {key}")

    def select(self, keys: str | None) -> list[Model]:
        """Models named in a comma-separated list, or all of them."""
        if not keys:
            return list(self.models)
        return [self.model(k.strip()) for k in keys.split(",") if k.strip()]


def load(path: Path = CONFIG) -> Config:
    raw = yaml.safe_load(path.read_text())
    defaults = raw.get("defaults", {})
    strategies = {
        name: Strategy(
            name=name,
            mode=s["mode"],
            logprobs=bool(s.get("logprobs", True)),
            max_tokens=int(s.get("max_tokens", 8)),
            reasoning=s.get("reasoning"),
        )
        for name, s in raw["strategies"].items()
    }
    models = []
    for m in raw["models"]:
        strats = tuple(m.get("strategies", defaults.get("strategies", ["chat"])))
        unknown = [s for s in strats if s not in strategies]
        if unknown:
            raise ValueError(f"{m['key']}: stratégies inconnues {unknown}")
        quant = m.get("quantizations")
        models.append(
            Model(
                key=m["key"],
                name=m["name"],
                lab=m["lab"],
                id=m["id"],
                provider=m["provider"],
                price=(float(m["price"][0]), float(m["price"][1])),
                strategies=strats,
                quantizations=tuple(quant) if quant else None,
                logprobs=bool(m.get("logprobs", True)),
                hf_repo=m.get("hf_repo"),
                raw_prefill=m.get("raw_prefill", ""),
                concurrency=int(m.get("concurrency", defaults.get("concurrency", 8))),
                top_logprobs=int(m.get("top_logprobs", TOP_LOGPROBS)),
                rpm=float(m["rpm"]) if m.get("rpm") else None,
                temperature=None if m.get("temperature", 0.0) is None else float(m.get("temperature", 0.0)),
                byok=bool(m.get("byok", False)),
            )
        )
    keys = [m.key for m in models]
    if len(set(keys)) != len(keys):
        raise ValueError("clés de modèles en double dans configs/models.yaml")
    return Config(
        models=models,
        strategies=strategies,
        provider_concurrency=int(defaults.get("provider_concurrency", 128)),
        raw=raw,
    )
