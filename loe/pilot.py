"""Pilot: pick a working strategy per model on 8 points, then ask 200 points to project the run.

A strategy passes when at least 7 of the 8 probe answers come back, are readable, show no
reasoning (unless the strategy allows minimal reasoning, the last resort), and come from the
pinned provider. Logprobs are used when the provider returns them but are not required: the
tweet's map only needs the answer. When they are there, Land and Water must carry most of the
probability. The pilot writes results/pilot/strategies.json, which `make run` reads, and
results/pilot/report.md.

The 200 pilot points are half land, half water, many of them near the poles, so their raw
accuracy is not the tweet's score: the report re-weights them by class frequency and cell area
to estimate the area-weighted accuracy of the full map.
"""

from __future__ import annotations

import asyncio
import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from loe import grid, store
from loe.client import Client
from loe.config import CREDIT_FEE, EUR_PER_USD, Config, Model, Strategy
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


def judge(model: Model, strategy: Strategy, recs: list[dict[str, Any]]) -> str:
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
    if not strategy.reasons:
        if any((r.get("reason_tok") or 0) > 0 or (r.get("reasoning_chars") or 0) > 0 for r in ok):
            return "le modèle raisonne malgré la consigne"
    with_lp = [r for r in ok if r.get("source") == "logprobs"]
    if len(with_lp) >= 3:  # logprobs are optional, but when present they must make sense
        mass = statistics.median(r["mass"] for r in with_lp)
        if mass < 0.5:
            return f"Land/Water peu présents dans le top des logprobs (masse médiane {mass:.2f})"
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
            verdict = judge(model, strategy, list(recs))
            choice.tried.append((name, verdict))
            if verdict == "ok":
                choice.strategy = name
                break
    finally:
        writer.close()
    return choice


def area_estimate(answered: pd.DataFrame, seed: int = 0, n_boot: int = 300) -> tuple[float, float]:
    """Area-weighted accuracy of the full map estimated from the pilot points, and its 90 % half-width.

    Each pilot point stands for the grid points of its class (land or water) and counts in
    proportion to its cell's area, cos(latitude), as in the tweet's score.
    """
    g = grid.load()
    n_grid = g["truth"].value_counts()
    truth = g.set_index("point_id").loc[answered["point_id"], "truth"].to_numpy()
    correct = answered["pred"].astype(int).to_numpy() == truth
    n_pilot = pd.Series(truth).value_counts()
    w = np.cos(np.radians(answered["lat"].to_numpy())) * np.array([n_grid[t] / n_pilot[t] for t in truth])
    acc = float((w * correct).sum() / w.sum())
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(w), len(w))
        boots.append((w[i] * correct[i]).sum() / w[i].sum())
    lo, hi = np.percentile(boots, [5, 95])
    return acc, float(hi - lo) / 2


def model_report(
    model: Model, choice: Choice, strategy: Strategy | None, records: list[dict[str, Any]], n_points: int
) -> dict[str, Any]:
    latest = store.latest(records)
    row: dict[str, Any] = {
        "key": model.key,
        "name": model.name,
        "provider": model.provider,
        "strategy": choice.strategy,
        "reasoning": bool(strategy and strategy.reasons),
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
    area, half = area_estimate(answered) if len(answered) >= 20 else (None, None)
    row.update(
        points=n_points,
        answered=int(len(answered)),
        coverage=round(len(answered) / n_points, 4),
        accuracy_sample=round(float((answered["pred"].astype(int).to_numpy() == truth).mean()), 4)
        if len(answered)
        else None,
        accuracy_area=None if area is None else round(area, 4),
        accuracy_area_halfwidth=None if half is None else round(half, 4),
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
    est = row["estimate_usd"]
    if row["projected_usd"] > max(3 * est, est + 2.5):  # minimal reasoning may cost a few times more, not 20x
        problems.append(f"coût projeté {row['projected_usd']:.2f} $ contre {est:.2f} $ estimé")
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
        model_report(
            m,
            c,
            cfg.strategies.get(c.strategy) if c.strategy else None,
            [r for r in store.read(out_dir, m.key) if r.get("strategy") == c.strategy],
            n_points,
        )
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
        r["key"]: {
            "strategy": r["strategy"],
            "reasoning": r["reasoning"],
            "status": r["status"],
            "projected_usd": r.get("projected_usd"),
        }
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
        "Précision estimée : la précision pondérée par la surface que le modèle aurait sur la carte complète, "
        "estimée à partir des 200 points (± = intervalle à 90 %). Ce n'est pas encore le score final.",
        "",
        "| Modèle | Stratégie | Réflexion | Fournisseur servi | Précision estimée | Couverture | Logprobs "
        "| Coût projeté | Statut |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report["models"]:
        if r.get("accuracy_area") is not None:
            acc = f"{r['accuracy_area']:.1%} ± {100 * r['accuracy_area_halfwidth']:.1f}"
        else:
            acc = "—"
        cov = f"{r['coverage']:.0%}" if r.get("coverage") is not None else "—"
        lp = f"{r['logprobs_share']:.0%}" if r.get("logprobs_share") is not None else "—"
        proj = f"{r['projected_usd']:.2f} $" if r.get("projected_usd") is not None else "—"
        prov = ", ".join(r.get("providers") or []) or "—"
        reasoning = "—" if not r.get("strategy") else ("minimale" if r.get("reasoning") else "aucune")
        lines.append(
            f"| {r['name']} | {r.get('strategy') or '—'} | {reasoning} | {prov} | {acc} | {cov} | {lp} "
            f"| {proj} | {r['status']} |"
        )
    lines += ["", "## Stratégies essayées", ""]
    for r in report["models"]:
        tried = "; ".join(f"`{s}` : {v}" for s, v in r["tried"]) or "aucune"
        lines.append(f"- **{r['name']}** — {tried}")
    lines.append("")
    return "\n".join(lines)
