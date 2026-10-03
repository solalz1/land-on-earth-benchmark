"""Pilot: pick a working strategy per model on 8 points, then ask 200 points to project the run.

A strategy passes when at least 7 of the 8 probe answers come back, are readable, carry Land
and Water among the top logprobs (when the strategy asks for them), show no reasoning (except
chat_low, the last resort), and come from the pinned provider. The pilot writes
results/pilot/strategies.json, which `make run` reads, and results/pilot/report.md.
"""

from __future__ import annotations

import asyncio
import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from loe import grid, store
from loe.client import Client
from loe.config import CREDIT_FEE, EUR_PER_USD, Config, Model
from loe.runner import Job, Runner, estimate, make_record, spent_in, write_meta
from loe.templates import Renderer

PROBE = 8
FULL = 16_200


def _norm(s: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


@dataclass
class Choice:
    key: str
    strategy: str | None = None
    tried: list[tuple[str, str]] = field(default_factory=list)  # (strategy, verdict)


def judge(model: Model, strategy_name: str, logprobs: bool, recs: list[dict[str, Any]]) -> str:
    """'ok' or the reason the strategy fails, in French for the report."""
    n = len(recs)
    ok = [r for r in recs if r.get("ok")]
    if len(ok) < n - 1:
        err = next((r.get("error") for r in recs if not r.get("ok")), "")
        return f"{n - len(ok)}/{n} erreurs — {err}"
    readable = [r for r in ok if r.get("pred") is not None]
    if len(readable) < len(ok) - 1:
        sample = next((r.get("text") for r in ok if r.get("pred") is None), "")
        finish = next((r.get("finish") for r in ok if r.get("pred") is None), "")
        return f"réponse illisible (fin : {finish}) : {sample!r}"
    if strategy_name != "chat_low":
        if any((r.get("reason_tok") or 0) > 0 or (r.get("reasoning_chars") or 0) > 0 for r in ok):
            return "le modèle raisonne malgré la consigne"
    if logprobs:
        with_lp = [r for r in ok if r.get("source") == "logprobs"]
        if len(with_lp) < len(ok) - 1:
            return "pas de logprobs exploitables dans la réponse"
        mass = statistics.median(r["mass"] for r in with_lp)
        if mass < 0.5:
            return f"Land/Water peu présents dans le top-20 (masse médiane {mass:.2f})"
    served = {r.get("provider") for r in ok if r.get("provider")}
    wrong = [s for s in served if _norm(model.provider) not in _norm(s)]
    if wrong:
        return f"servi par {', '.join(wrong)} au lieu de {model.provider}"
    return "ok"


async def choose(
    cfg: Config, model: Model, client: Client, render: Renderer | None, probe: pd.DataFrame, out_dir: Path, sem
) -> Choice:
    choice = Choice(model.key)
    writer = store.Appender(out_dir / "probe", model.key)
    try:
        for name in model.strategies:
            strategy = cfg.strategies[name]
            if strategy.mode == "raw" and render is None:
                choice.tried.append((name, "template indisponible (dépôt Hugging Face introuvable ou protégé)"))
                continue
            job = Job(model, strategy, render)

            async def one(p):
                async with sem(model.provider):
                    reply = await client.ask(model, strategy, float(p.lat), float(p.lon), render)
                rec = make_record(job, p, reply)
                writer.write(rec)
                return rec

            recs = await asyncio.gather(*(one(p) for p in probe.itertuples(index=False)))
            verdict = judge(model, name, strategy.logprobs, list(recs))
            choice.tried.append((name, verdict))
            if verdict == "ok":
                choice.strategy = name
                break
    finally:
        writer.close()
    return choice


def model_report(model: Model, choice: Choice, records: list[dict[str, Any]], n_points: int) -> dict[str, Any]:
    latest = store.latest(records)
    row: dict[str, Any] = {
        "key": model.key,
        "name": model.name,
        "provider": model.provider,
        "strategy": choice.strategy,
        "tried": choice.tried,
        "estimate_usd": round(estimate(model, FULL), 4),
    }
    if latest.empty or choice.strategy is None:
        row["status"] = "échec"
        return row
    ok = latest[latest["ok"].astype(bool)]
    answered = ok[ok["pred"].notna()] if "pred" in ok else ok.iloc[0:0]
    pts = grid.load().set_index("point_id")
    truth = pts.loc[answered["point_id"], "truth"].to_numpy()
    cost = spent_in(records)
    n_req = len(records)
    lp = answered[answered["source"] == "logprobs"] if "source" in answered else answered.iloc[0:0]
    row.update(
        points=n_points,
        answered=int(len(answered)),
        coverage=round(len(answered) / n_points, 4),
        accuracy=round(float((answered["pred"].astype(int).to_numpy() == truth).mean()), 4) if len(answered) else None,
        logprobs_share=round(len(lp) / max(len(answered), 1), 4),
        median_mass=round(float(lp["mass"].median()), 4) if len(lp) else None,
        providers=sorted({str(p) for p in ok.get("provider", pd.Series(dtype=str)).dropna()}),
        errors=int((~latest["ok"].astype(bool)).sum()),
        requests=n_req,
        cost_usd=round(cost, 6),
        cost_per_request_usd=cost / max(n_req, 1),
        projected_usd=round(cost / max(n_req, 1) * FULL, 4),
        reasoning_tokens=int(pd.to_numeric(ok.get("reason_tok", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
        median_latency_s=round(float(ok["latency"].median()), 3) if len(ok) else None,
    )
    problems = []
    if row["coverage"] < 0.98:
        problems.append(f"couverture {row['coverage']:.0%}")
    if row["projected_usd"] > 2.5 * max(row["estimate_usd"], 0.2):
        problems.append(f"coût projeté {row['projected_usd']:.2f} $ contre {row['estimate_usd']:.2f} $ estimé")
    if model.logprobs and row["logprobs_share"] < 0.98:
        problems.append(f"logprobs sur {row['logprobs_share']:.0%} des réponses")
    row["status"] = "ok" if not problems else "à vérifier : " + ", ".join(problems)
    return row


async def run_pilot(
    cfg: Config,
    models: list[Model],
    client: Client,
    renderers: dict[str, Renderer | None],
    out_dir: Path,
    n_points: int = 200,
    budget: float = 1.0,
    credits_left: float | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    points = grid.pilot_points(n_points)
    probe = points.head(PROBE)
    runner = Runner(client, out_dir, budget=budget, provider_concurrency=cfg.provider_concurrency, progress=progress)

    if progress:
        print(f"Choix de la stratégie sur {PROBE} points pour {len(models)} modèles…")
    choices = await asyncio.gather(
        *(choose(cfg, m, client, renderers.get(m.key), probe, out_dir, runner.sem) for m in models)
    )
    jobs = [
        Job(m, cfg.strategies[c.strategy], renderers.get(m.key)) for m, c in zip(models, choices) if c.strategy
    ]
    if progress:
        print(f"Pilote : {n_points} points sur {len(jobs)} modèles…")
    outcomes = await runner.run(jobs, points) if jobs else []
    write_meta(out_dir, jobs, outcomes, {"kind": "pilot", "points": n_points})

    rows = [
        model_report(m, c, [r for r in store.read(out_dir, m.key) if r.get("strategy") == c.strategy], n_points)
        for m, c in zip(models, choices)
    ]
    total_projected = sum(r.get("projected_usd") or 0.0 for r in rows)
    report = {
        "points": n_points,
        "models": rows,
        "valid": sum(r["status"] == "ok" for r in rows),
        "total": len(rows),
        "projected_usd": round(total_projected, 2),
        "projected_eur": round(total_projected * (1 + CREDIT_FEE) * EUR_PER_USD, 2),
        "credits_left_usd": credits_left,
        "pilot_cost_usd": round(sum(r.get("cost_usd") or 0.0 for r in rows), 4),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    strategies = {
        r["key"]: {"strategy": r["strategy"], "status": r["status"], "projected_usd": r.get("projected_usd")}
        for r in rows
    }
    old = out_dir / "strategies.json"
    merged = json.loads(old.read_text()) if old.exists() else {}
    merged.update(strategies)
    old.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    (out_dir / "report.md").write_text(markdown(report))
    return report


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Pilote Land on Earth",
        "",
        f"{report['valid']} modèles valides sur {report['total']}, {report['points']} points chacun. "
        f"Coût du pilote : {report['pilot_cost_usd']:.3f} $. "
        f"Projection pour le run complet (16 200 points) : {report['projected_usd']:.2f} $ de crédits.",
        "",
        "| Modèle | Stratégie | Fournisseur servi | Précision | Couverture | Masse Land+Water | Coût projeté | Statut |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in report["models"]:
        acc = f"{r['accuracy']:.1%}" if r.get("accuracy") is not None else "—"
        cov = f"{r['coverage']:.0%}" if r.get("coverage") is not None else "—"
        mass = f"{r['median_mass']:.2f}" if r.get("median_mass") is not None else "—"
        proj = f"{r['projected_usd']:.2f} $" if r.get("projected_usd") is not None else "—"
        prov = ", ".join(r.get("providers") or []) or "—"
        lines.append(
            f"| {r['name']} | {r.get('strategy') or '—'} | {prov} | {acc} | {cov} | {mass} | {proj} | {r['status']} |"
        )
    lines += ["", "## Stratégies essayées", ""]
    for r in report["models"]:
        tried = "; ".join(f"`{s}` : {v}" for s, v in r["tried"]) or "aucune"
        lines.append(f"- **{r['name']}** — {tried}")
    lines.append("")
    return "\n".join(lines)
