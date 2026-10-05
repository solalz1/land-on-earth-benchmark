"""Probe: try one or more strategies for a model at providers other than its pinned one.

Used to find a provider that serves gpt-oss without reasoning (a raw prompt that starts in the
answer channel only works if the provider passes it to the model as it is). Each combination
costs 8 requests. Nothing in configs/models.yaml changes: the result says which provider and
strategy to pin, and the answers are kept in results/probe/ for inspection.
"""

from __future__ import annotations

import asyncio
import json
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any

from loe import grid, store
from loe.client import Client
from loe.config import Config, Model
from loe.pilot import PROBE, judge
from loe.runner import Job, make_record
from loe.templates import Renderer


async def run_probe(
    cfg: Config,
    model: Model,
    providers: list[str],
    strategies: list[str],
    client: Client,
    render: Renderer | None,
    out_dir: Path,
    quantizations: list[str] | None = None,
    n_points: int = PROBE,
) -> list[dict[str, Any]]:
    points = grid.pilot_points(200).head(n_points)
    rows: list[dict[str, Any]] = []
    for provider in providers:
        # another provider rarely shares the pinned one's precision: accept what it declares
        m = replace(model, provider=provider, quantizations=tuple(quantizations) if quantizations else None)
        writer = store.Appender(out_dir, f"{model.key}@{provider}")
        try:
            for name in strategies:
                strategy = cfg.strategies[name]
                row: dict[str, Any] = {"model": model.key, "provider": provider, "strategy": name}
                if strategy.mode == "raw" and render is None:
                    rows.append(row | {"verdict": "template indisponible"})
                    continue
                job = Job(m, strategy, render)

                async def one(p):
                    reply = await client.ask(m, strategy, float(p.lat), float(p.lon), render)
                    rec = make_record(job, p, reply)
                    writer.write(rec)
                    return rec

                recs = list(await asyncio.gather(*(one(p) for p in points.itertuples(index=False))))
                ok = [r for r in recs if r.get("ok")]
                row.update(
                    verdict=judge(m, strategy, recs),
                    answered=sum(r.get("pred") is not None for r in ok),
                    requests=len(recs),
                    logprobs=sum(r.get("source") == "logprobs" for r in ok),
                    reasoning_tokens=statistics.mean([r.get("reason_tok") or 0 for r in ok]) if ok else None,
                    served=sorted({str(r.get("provider")) for r in ok if r.get("provider")}),
                    texts=sorted({(r.get("text") or "")[:30] for r in ok})[:4],
                    cost_usd=sum(r.get("cost") or 0.0 for r in recs),
                )
                rows.append(row)
        finally:
            writer.close()
    report = out_dir / "report.json"
    old = json.loads(report.read_text()) if report.exists() else []
    out_dir.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(old + rows, indent=2, ensure_ascii=False))
    return rows


def show(rows: list[dict[str, Any]]) -> str:
    lines = [f"{'Modèle':<16}{'Fournisseur':<14}{'Stratégie':<11}{'Réponses':>9}{'Logprobs':>9}{'Réflexion':>10}  Verdict"]
    for r in rows:
        rt = r.get("reasoning_tokens")
        rt = "—" if rt is None else f"{rt:.0f} tok"
        ans = f"{r.get('answered', 0)}/{r.get('requests', 0)}"
        lp = str(r.get("logprobs", "—"))
        lines.append(f"{r['model']:<16}{r['provider']:<14}{r['strategy']:<11}{ans:>9}{lp:>9}{rt:>10}  {r['verdict']}")
    good = [r for r in rows if r["verdict"] == "ok"]
    lines.append("")
    if good:
        lines.append("Combinaisons qui marchent : " + ", ".join(f"{r['provider']} + {r['strategy']}" for r in good))
    else:
        lines.append("Aucune combinaison ne marche.")
    return "\n".join(lines)
