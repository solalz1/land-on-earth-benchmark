"""Free checks before spending anything: the key, its cap, and each pinned provider.

For every model, OpenRouter's public endpoint list must contain the pinned provider, at an
accepted precision, with logprobs and top_logprobs among its supported parameters. Live prices
give the cost of the full run. Raw-prompt templates are fetched from Hugging Face too.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from loe.client import Client
from loe.config import CREDIT_FEE, EUR_PER_USD, Config, Model
from loe.runner import PROMPT_TOKENS

FULL = 16_200


def _norm(s: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def match(model: Model, endpoints: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The endpoint OpenRouter will use for this model's pinned provider."""
    want = _norm(model.provider)
    cands = [
        e
        for e in endpoints
        if _norm((e.get("tag") or "").split("/")[0]) == want or _norm(e.get("provider_name")) == want
    ]
    if model.quantizations:
        good = [e for e in cands if (e.get("quantization") or "").lower() in model.quantizations]
        cands = good or cands
    if not cands:
        return None
    return min(cands, key=lambda e: float((e.get("pricing") or {}).get("prompt") or 1))


def verdict(model: Model, ep: dict[str, Any] | None) -> dict[str, Any]:
    row: dict[str, Any] = {"key": model.key, "name": model.name, "provider": model.provider}
    if ep is None:
        row["problems"] = [f"fournisseur « {model.provider} » absent de la fiche OpenRouter"]
        return row
    pricing = ep.get("pricing") or {}
    pin = float(pricing.get("prompt") or 0) * 1e6
    pout = float(pricing.get("completion") or 0) * 1e6
    params = set(ep.get("supported_parameters") or [])
    quant = (ep.get("quantization") or "unknown").lower()
    problems = []
    if model.quantizations and quant not in model.quantizations:
        problems.append(f"précision {quant}, attendu {'/'.join(model.quantizations)}")
    if model.logprobs and not {"logprobs", "top_logprobs"} <= params:
        problems.append("pas de logprobs/top_logprobs")
    if ep.get("status") not in (None, 0):
        problems.append(f"statut {ep.get('status')}")
    row.update(
        endpoint=ep.get("name"),
        tag=ep.get("tag"),
        quantization=quant,
        reasoning_param="reasoning" in params,
        logprobs="logprobs" in params and "top_logprobs" in params,
        price_in=round(pin, 4),
        price_out=round(pout, 4),
        cost_usd=round(FULL * (PROMPT_TOKENS * pin + pout) / 1e6, 4),
        problems=problems,
    )
    return row


async def run_check(cfg: Config, models: list[Model], client: Client, templates: dict[str, bool]) -> dict[str, Any]:
    try:
        key = (await client.get("/key")).get("data") or {}
    except Exception as e:  # the key listing is informative only
        key = {"error": str(e)}

    async def one(m: Model) -> dict[str, Any]:
        try:
            data = (await client.get(f"/models/{m.id}/endpoints")).get("data") or {}
        except Exception as e:
            return {"key": m.key, "name": m.name, "provider": m.provider, "problems": [f"fiche illisible : {e}"]}
        return verdict(m, match(m, data.get("endpoints") or []))

    rows = await asyncio.gather(*(one(m) for m in models))
    for r in rows:
        m = cfg.model(r["key"])
        if "raw" in m.strategies:
            r["raw_template"] = templates.get(m.key, False)
    total = sum(r.get("cost_usd") or 0 for r in rows)
    return {
        "key": {k: key.get(k) for k in ("label", "limit", "usage", "limit_remaining", "error") if k in key},
        "models": list(rows),
        "total_usd": round(total, 2),
        "total_eur_with_fee": round(total * (1 + CREDIT_FEE) * EUR_PER_USD, 2),
        "ok": all(not r.get("problems") for r in rows),
    }


def show(report: dict[str, Any]) -> str:
    key = report["key"]
    lines = []
    if "error" in key:
        lines.append(f"Clé : impossible de lire ses limites ({key['error']})")
    else:
        limit = key.get("limit")
        lines.append(
            f"Clé « {key.get('label')} » : plafond {limit if limit is not None else 'AUCUN'} $, "
            f"déjà utilisé {key.get('usage', 0):.2f} $, reste {key.get('limit_remaining')} $"
        )
        if limit is None:
            lines.append("  ⚠ Clé sans plafond : fixe une limite de 20 $ sur openrouter.ai/settings/keys.")
    lines.append("")
    lines.append(f"{'Modèle':<22}{'Fournisseur':<22}{'Précision':<11}{'Logprobs':<10}{'Coût run':>10}  Problèmes")
    for r in report["models"]:
        lp = "oui" if r.get("logprobs") else "non"
        cost = f"{r['cost_usd']:.2f} $" if "cost_usd" in r else "—"
        tag = r.get("tag") or r["provider"]
        probs = list(r.get("problems") or [])
        if r.get("raw_template") is False:
            probs.append("template brut indisponible (sans gravité)")
        lines.append(f"{r['name']:<22}{tag:<22}{r.get('quantization', '—'):<11}{lp:<10}{cost:>10}  {'; '.join(probs) or 'ok'}")
    lines.append("")
    lines.append(
        f"Coût estimé du run : {report['total_usd']:.2f} $ de crédits "
        f"(≈ {report['total_eur_with_fee']:.2f} € frais d'achat compris)"
    )
    return "\n".join(lines)


def save(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
