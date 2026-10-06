"""Preflight: a dress rehearsal of the run, with a go / no-go verdict before the real thing.

After the pilot, every validated model gets 300 fresh points (never asked before), all models at
once, with exactly the run's settings: strategy, pinned provider, your provider keys, pacing,
requests in flight, top logprobs, token budget. 300 points keep each model busy for half a
minute to a minute, long enough to see a provider's per-minute limit. The rehearsal checks what
the pilot could hide:

- no request refused by the provider (other than rate limits), and no reasoning where none is
  expected;
- no empty answer (a reasoning budget too short cuts the answer off);
- your keys: a model meant for your key at its provider (`byok: true`) must be served by it,
  and a model meant for OpenRouter's credits should not be;
- rate limits: a model that hits OpenRouter's new-account limit is not using your key, or must
  be paced; the provider's own refusals are counted;
- money: the projection under the run budget (with a 10 % margin), and the part billed on
  OpenRouter credits under the credits left on the key;
- time: each model's throughput, hence the run's duration.

It writes results/pilot/preflight.md and preflight.json; `loe run` refuses to start while the
verdict is not green, or when configs/models.yaml changed since the rehearsal.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from loe import grid, store
from loe.client import Client
from loe.config import CONFIG, CREDIT_FEE, EUR_PER_USD, Config, Model
from loe.runner import Job, Runner

FULL = 16_200
N_POINTS = 300
MARGIN = 0.9  # the projection must stay under 90 % of the budget
OPENROUTER = "OpenRouter"
LABELS = {"siliconflow": "SiliconFlow", "alibaba": "Alibaba", "mistral": "Mistral", "novita": "Novita",
          "parasail": "Parasail", "dekallm": "DekaLLM", "deepinfra": "DeepInfra", "gmicloud": "GMICloud"}


def label(provider: str) -> str:
    return LABELS.get(provider, provider.title())


def config_hash(path: Path = CONFIG) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def duration(hours: float | None) -> str:
    """'≈ 35 min' or '≈ 1 h 10' (French)."""
    if not hours:
        return "—"
    minutes = round(hours * 60)
    if minutes < 60:
        return f"≈ {minutes} min"
    return f"≈ {minutes // 60} h {minutes % 60:02d}"


def rehearsal_points(n: int = N_POINTS, seed: int = 1) -> pd.DataFrame:
    """Half land, half water, none of them among the pilot's 200 points."""
    g = grid.load()
    used = set(grid.pilot_points(200)["point_id"])
    fresh = g[~g["point_id"].isin(used)]
    land = fresh[fresh["truth"] == 1].sample(n // 2, random_state=seed)
    water = fresh[fresh["truth"] == 0].sample(n - n // 2, random_state=seed)
    return pd.concat([land, water]).sort_values("point_id").reset_index(drop=True)


def _minutes(records: list[dict[str, Any]]) -> float:
    ts = sorted(datetime.fromisoformat(r["ts"]) for r in records if r.get("ts"))
    if len(ts) < 2:
        return 0.0
    return (ts[-1] - ts[0]).total_seconds() / 60


def model_check(model: Model, strategy, pilot: dict[str, Any], records: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    ok = [r for r in records if r.get("ok")]
    errors = [r for r in records if not r.get("ok")]
    account_429 = [r for r in errors if r.get("status") == 429 and "new-account" in (r.get("error") or "")]
    other_429 = [r for r in errors if r.get("status") == 429 and r not in account_429]
    in_flight = [r for r in errors if r.get("status") == 402]
    refused = [r for r in errors if r.get("status") not in (429, 402)]
    empty = [r for r in ok if r.get("pred") is None]
    retried = [r for r in records if (r.get("attempts") or 1) > 1]
    reasoning = [r for r in ok if (r.get("reason_tok") or 0) > 0 or (r.get("reasoning_chars") or 0) > 0]
    byok = [r for r in ok if store.is_byok(r)]
    # requests per minute, all workers together: over the rehearsal's window when it lasted long enough,
    # else from the answers' latency and the requests in flight
    latencies = [r["latency"] for r in ok if r.get("latency")]
    if elapsed >= 0.25:
        measured = len(records) / elapsed
    elif latencies:
        measured = model.concurrency * 60 / (sum(latencies) / len(latencies))
    else:
        measured = None
    rate = min(x for x in (model.rpm, measured) if x) if (model.rpm or measured) else None
    cost = sum(store.charged(r) for r in records)
    per_answer = cost / max(len(ok), 1)
    projected = max(per_answer * FULL, float(pilot.get("projected_usd") or 0.0))
    paid_by_key = bool(ok) and len(byok) >= 0.5 * len(ok)
    row: dict[str, Any] = {
        "key": model.key,
        "name": model.name,
        "provider": model.provider,
        "strategy": strategy.name,
        "reasoning": strategy.reasons,
        "rpm": model.rpm,
        "concurrency": model.concurrency,
        "byok_expected": model.byok,
        "byok": len(byok),
        "payer": model.provider if paid_by_key else OPENROUTER,
        "requests": len(records),
        "answered": len(ok) - len(empty),
        "empty": len(empty),
        "refused": len(refused),
        "rate_limited_account": len(account_429),
        "rate_limited_upstream": len(other_429),
        "in_flight_budget": len(in_flight),
        "retried": len(retried),
        "rate_per_min": None if rate is None else round(rate, 1),
        "hours": None if not rate else round(FULL / rate / 60, 2),
        "cost_per_request_usd": cost / max(len(records), 1),
        "rehearsal_cost_usd": round(cost, 6),
        "projected_usd": round(projected, 4),
        "pilot_projected_usd": pilot.get("projected_usd"),
    }
    problems, warnings = [], []
    if pilot.get("status") != "ok":
        problems.append(f"pilote non validé ({pilot.get('status') or 'absent'})")
    if refused:
        problems.append(f"{len(refused)} requêtes refusées : {refused[0].get('error')}"[:300])
    if len(empty) > 1:
        finish = empty[0].get("finish")
        problems.append(f"{len(empty)} réponses vides (fin : {finish}) : budget de tokens trop court ?")
    if reasoning and not strategy.reasons:
        problems.append(f"le modèle réfléchit sur {len(reasoning)} réponses alors que la réflexion est coupée")
    if model.byok and ok and len(byok) < len(ok):
        problems.append(
            f"ta clé {label(model.provider)} n'a servi que {len(byok)}/{len(ok)} réponses : dans OpenRouter > "
            f"Integrations, mets-la en Prioritized et ajoute {model.id} à son filtre"
        )
    if not model.byok and byok:
        warnings.append(
            f"{len(byok)} réponses servies par ta clé {label(model.provider)} alors que la config prévoit les crédits "
            "OpenRouter : vérifie le filtre de cette clé"
        )
    if account_429 and not model.rpm:
        problems.append(
            "plafonné à 20 requêtes/min par OpenRouter (compte récent) : ta clé fournisseur ne sert pas ce modèle, "
            "ou ajoute `rpm: 19` dans configs/models.yaml"
        )
    elif account_429:
        problems.append(f"{len(account_429)} refus malgré le rythme de {model.rpm:g}/min : baisser `rpm`")
    if other_429:
        share = len(other_429) / max(len(records), 1)
        hint = " : baisser `rpm` ou `concurrency`" if share > 0.1 else ", réessayés"
        warnings.append(f"{len(other_429)} refus passagers du fournisseur (429){hint}")
    if in_flight:
        warnings.append(f"{len(in_flight)} requêtes retenues par le budget en vol d'OpenRouter (402), réessayées")
    if len(empty) == 1:
        warnings.append("1 réponse vide")
    row["problems"], row["warnings"] = problems, warnings
    row["status"] = "bloquant" if problems else ("attention" if warnings else "ok")
    return row


async def run_preflight(
    cfg: Config,
    models: list[Model],
    client: Client,
    renderers: dict,
    pilot_dir: Path,
    budget: float,
    credits_left: float | None = None,
    n_points: int = N_POINTS,
    progress: bool = True,
) -> dict[str, Any]:
    path = pilot_dir / "strategies.json"
    chosen = json.loads(path.read_text()) if path.exists() else {}
    session = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = pilot_dir / "rehearsal" / session
    n = 1
    while run_dir.exists():  # never mix two rehearsals' answers
        n += 1
        run_dir = pilot_dir / "rehearsal" / f"{session}-{n}"
    session = run_dir.name
    jobs, rows = [], []
    for m in models:
        pilot = chosen.get(m.key) or {}
        name = pilot.get("strategy")
        if not name or name not in cfg.strategies:
            rows.append({"key": m.key, "name": m.name, "status": "bloquant", "warnings": [],
                         "problems": [f"pilote non validé ({pilot.get('status') or 'absent'})"]})
            continue
        jobs.append(Job(m, cfg.strategies[name], renderers.get(m.key), expected_usd=pilot.get("projected_usd")))
    points = rehearsal_points(n_points)
    if progress:
        print(f"Répétition générale : {len(points)} nouveaux points × {len(jobs)} modèles, réglages du run…")
    runner = Runner(client, run_dir, budget=3.0, provider_concurrency=cfg.provider_concurrency, progress=progress)
    if jobs:
        await runner.run(jobs, points)
    for j in jobs:
        recs = store.read(run_dir, j.model.key)
        rows.append(model_check(j.model, j.strategy, chosen.get(j.model.key) or {}, recs, _minutes(recs)))
    order = {m.key: i for i, m in enumerate(cfg.models)}
    rows.sort(key=lambda r: order.get(r["key"], 99))

    projected = sum(r.get("projected_usd") or 0.0 for r in rows)
    payers: dict[str, float] = {}
    for r in rows:
        if r.get("projected_usd"):
            payers[r.get("payer", OPENROUTER)] = payers.get(r.get("payer", OPENROUTER), 0.0) + r["projected_usd"]
    on_credits = payers.get(OPENROUTER, 0.0)
    hours = max((r.get("hours") or 0.0 for r in rows), default=0.0)
    slowest = [r["name"] for r in rows if r.get("hours") and r["hours"] >= hours - 0.1]
    checks = []
    blocking = [r for r in rows if r["status"] == "bloquant"]
    checks.append(("Modèles", not blocking, f"{len(rows) - len(blocking)} sur {len(rows)} prêts"))
    checks.append(
        ("Budget", projected <= MARGIN * budget, f"projection {projected:.2f} $ pour un budget de {budget:.2f} $")
    )
    if credits_left is not None:
        checks.append(
            ("Crédits", on_credits <= float(credits_left),
             f"{on_credits:.2f} $ sur tes crédits OpenRouter, {float(credits_left):.2f} $ restants sur la clé")
        )
    report = {
        "session": session,
        "points": n_points,
        "config": config_hash(),
        "ready": all(ok for _, ok, _ in checks),
        "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in checks],
        "projected_usd": round(projected, 2),
        "projected_eur": round((on_credits * (1 + CREDIT_FEE) + (projected - on_credits)) * EUR_PER_USD, 2),
        "by_payer": {k: round(v, 2) for k, v in sorted(payers.items(), key=lambda kv: -kv[1])},
        "hours": round(hours, 2),
        "slowest": slowest,
        "budget_usd": budget,
        "rehearsal_cost_usd": round(sum(r.get("rehearsal_cost_usd") or 0.0 for r in rows), 4),
        "models": rows,
    }
    pilot_dir.mkdir(parents=True, exist_ok=True)
    (pilot_dir / "preflight.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (pilot_dir / "preflight.md").write_text(markdown(report))
    return report


def _payer(name: str) -> str:
    return "crédits OpenRouter" if name == OPENROUTER else f"ta clé {label(name)}"


def markdown(report: dict[str, Any]) -> str:
    verdict = "FEU VERT : le run peut partir." if report["ready"] else "FEU ROUGE : corriger avant le run."
    split = ", ".join(f"{v:.2f} $ {_payer(k)}" for k, v in report.get("by_payer", {}).items()) or "—"
    lines = [
        "# Répétition générale avant le run",
        "",
        f"**{verdict}**",
        "",
        f"Estimation du run : {report['projected_usd']:.2f} $ (≈ {report['projected_eur']:.2f} €, frais d'achat de "
        f"crédits compris), {duration(report['hours'])}, durée fixée par : {', '.join(report['slowest']) or '—'}.",
        "",
        f"Qui paie : {split}.",
        "",
        f"Répétition : {report.get('points', '—')} nouveaux points par modèle, tous en même temps, aux réglages du run, "
        f"pour {report.get('rehearsal_cost_usd', 0):.2f} $.",
        "",
    ]
    for c in report["checks"]:
        lines.append(f"- {'✅' if c['ok'] else '❌'} {c['name']} : {c['detail']}")
    lines += [
        "",
        "| Modèle | Statut | Réponses | Vides | Refus | 429 compte | 429 fournisseur | Payé par | Simultanées "
        "| Débit (req/min) | Durée | Coût projeté | Problèmes |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report["models"]:
        notes = "; ".join(r.get("problems", []) + r.get("warnings", [])) or "—"
        rate = f"{r['rate_per_min']:.0f}" if r.get("rate_per_min") else "—"
        proj = f"{r['projected_usd']:.2f} $" if r.get("projected_usd") is not None else "—"
        payer = _payer(r["payer"]) if r.get("payer") else "—"
        lines.append(
            f"| {r['name']} | {r['status']} | {r.get('answered', '—')}/{r.get('requests', '—')} | {r.get('empty', '—')} "
            f"| {r.get('refused', '—')} | {r.get('rate_limited_account', '—')} | {r.get('rate_limited_upstream', '—')} "
            f"| {payer} | {r.get('concurrency', '—')} | {rate} | {duration(r.get('hours'))} | {proj} | {notes} |"
        )
    lines.append("")
    return "\n".join(lines)


def gate(pilot_dir: Path) -> str | None:
    """None when the run may start, else the reason it may not (in French)."""
    path = pilot_dir / "preflight.json"
    if not path.exists():
        return "aucune répétition générale : lance make pilot"
    report = json.loads(path.read_text())
    if report.get("config") != config_hash():
        return "configs/models.yaml a changé depuis la répétition générale : relance make pilot"
    if not report.get("ready"):
        return f"la répétition générale est au rouge : voir {pilot_dir / 'preflight.md'}"
    return None
