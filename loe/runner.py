"""Ask every point to every model, concurrently, with resume and cost guards.

- Resume: points already answered (in raw/<model>.jsonl) are skipped; failed ones are asked again.
- Budget: the run stops once what this run directory cost you reaches the budget: OpenRouter
  credits plus what your own provider keys were billed (store.charged).
- Per-model guard: after 100 answers, a model whose projected cost exceeds `cap_factor` times
  its estimate (or 1.5 times the pilot's projection, when known) is stopped: a model that starts
  reasoning would otherwise burn credits.
- A model that fails 25 requests in a row is stopped, except for rate limits (HTTP 429) and
  OpenRouter's in-flight budget (402), which only slow it down; a bad OpenRouter key or empty
  credits stop everything.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from loe import store
from loe.client import Client, Fatal, Reply
from loe.config import Model, Strategy
from loe.parse import choice_text, parse
from loe.templates import Renderer

PROMPT_TOKENS = 61  # chat-formatted prompt, measured on the spec's tokenizers
MAX_CONSECUTIVE_ERRORS = 25
GUARD_AFTER = 100


@dataclass
class Job:
    model: Model
    strategy: Strategy
    render: Renderer | None = None
    expected_usd: float | None = None  # the pilot's projection for the full run, if any


@dataclass
class Outcome:
    key: str
    total: int
    done_before: int = 0
    answered: int = 0
    errors: int = 0
    cost: float = 0.0  # spent in this session
    stopped: str | None = None
    last_error: str | None = None


@dataclass
class Budget:
    limit: float
    spent: float = 0.0
    stopped: str | None = None

    def add(self, cost: float) -> None:
        self.spent += cost
        if self.spent >= self.limit and not self.stopped:
            self.stopped = f"budget atteint : {self.spent:.2f} $ dépensés sur {self.limit:.2f} $"


def estimate(model: Model, n_points: int, prompt_tokens: int = PROMPT_TOKENS, out_tokens: int = 1) -> float:
    """Expected credits (USD) for n_points one-token answers."""
    pin, pout = model.price
    return n_points * (prompt_tokens * pin + out_tokens * pout) / 1e6


def make_record(job: Job, point: Any, reply: Reply) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "point_id": int(point.point_id),
        "lat": float(point.lat),
        "lon": float(point.lon),
        "model": job.model.key,
        "strategy": job.strategy.name,
        "pinned": job.model.provider,
        "ok": reply.ok,
        "status": reply.status,
        "attempts": reply.attempts,
        "latency": round(reply.latency, 3),
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if not reply.ok:
        rec["error"] = reply.error
        return rec
    body = reply.body or {}
    choice = body["choices"][0]
    parsed = parse(choice)
    usage = body.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    msg = choice.get("message") or {}
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    rec.update(
        pred=parsed.pred,
        p_land=None if parsed.p_land is None else round(parsed.p_land, 6),
        mass=None if parsed.mass is None else round(parsed.mass, 6),
        source=parsed.source,
        pos=parsed.pos,
        top=None if parsed.top is None else [[t, round(lp, 4)] for t, lp in parsed.top],
        text=choice_text(choice)[:160],
        reasoning_chars=len(reasoning) if isinstance(reasoning, str) else len(json.dumps(reasoning)),
        finish=choice.get("finish_reason") or choice.get("native_finish_reason"),
        provider=body.get("provider"),
        served=body.get("model"),
        cost=usage.get("cost"),
        byok=usage.get("is_byok"),
        upstream_cost=(usage.get("cost_details") or {}).get("upstream_inference_cost"),
        in_tok=usage.get("prompt_tokens"),
        out_tok=usage.get("completion_tokens"),
        reason_tok=details.get("reasoning_tokens"),
    )
    return rec


def spent_in(records: list[dict[str, Any]]) -> float:
    """What these requests cost you: OpenRouter credits plus your provider keys' bills."""
    return float(sum(store.charged(r) for r in records))


class Runner:
    def __init__(
        self,
        client: Client,
        run_dir: Path,
        budget: float,
        provider_concurrency: int = 32,
        cap_factor: float = 2.5,
        progress: bool = True,
    ):
        self.client = client
        self.run_dir = run_dir
        self.provider_concurrency = provider_concurrency
        self.cap_factor = cap_factor
        self.progress = progress
        self.budget = Budget(budget)
        self._sems: dict[str, asyncio.Semaphore] = {}
        self._fatal: Fatal | None = None
        self._bar = None

    def sem(self, provider: str) -> asyncio.Semaphore:
        if provider not in self._sems:
            self._sems[provider] = asyncio.Semaphore(self.provider_concurrency)
        return self._sems[provider]

    async def _model(self, job: Job, points: pd.DataFrame) -> Outcome:
        m = job.model
        records = store.for_job(store.read(self.run_dir, m.key), job.strategy.name, m.provider)
        done = store.done_ids(records)
        todo = points[~points["point_id"].isin(done)]
        out = Outcome(m.key, total=len(points), done_before=len(points) - len(todo))
        if todo.empty:
            return out
        answered_before = len(done)
        spent_before = spent_in(records)
        est = estimate(m, len(points))
        cap = max(self.cap_factor * est, est + 0.5)
        if job.expected_usd:
            cap = max(cap, 1.5 * job.expected_usd * len(points) / 16_200)
        queue: asyncio.Queue = asyncio.Queue()
        for p in todo.itertuples(index=False):
            queue.put_nowait(p)
        writer = store.Appender(self.run_dir, m.key)
        streak = 0

        async def worker() -> None:
            nonlocal streak
            while not (out.stopped or self.budget.stopped or self._fatal):
                try:
                    p = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    async with self.sem(m.provider):
                        reply = await self.client.ask(m, job.strategy, float(p.lat), float(p.lon), job.render)
                except Fatal as e:
                    self._fatal = self._fatal or e
                    return
                rec = make_record(job, p, reply)
                writer.write(rec)
                cost = store.charged(rec)
                out.cost += cost
                self.budget.add(cost)
                if reply.ok:
                    out.answered += 1
                    streak = 0
                else:
                    out.errors += 1
                    out.last_error = reply.error
                    if reply.status not in (429, 402):  # rate limit or in-flight budget: slower, not failing
                        streak += 1
                    if streak >= MAX_CONSECUTIVE_ERRORS and not out.stopped:
                        out.stopped = f"{streak} erreurs d'affilée — {reply.error}"
                n = answered_before + out.answered
                if out.answered >= GUARD_AFTER and not out.stopped:
                    projected = (spent_before + out.cost) / max(n, 1) * len(points)
                    if projected > cap:
                        out.stopped = f"coût projeté {projected:.2f} $ > plafond {cap:.2f} $ pour ce modèle"
                if self._bar is not None:
                    self._bar.update(1)
                    self._bar.set_postfix_str(f"{self.budget.spent:.2f} $", refresh=False)

        try:
            await asyncio.gather(*(worker() for _ in range(max(1, m.concurrency))))
        finally:
            writer.close()
        return out

    async def run(self, jobs: list[Job], points: pd.DataFrame) -> list[Outcome]:
        prior = sum(spent_in(store.read(self.run_dir, j.model.key)) for j in jobs)
        self.budget.spent = prior
        if prior >= self.budget.limit:
            self.budget.stopped = f"budget déjà atteint : {prior:.2f} $ sur {self.budget.limit:.2f} $"
            return [Outcome(j.model.key, total=len(points), stopped=self.budget.stopped) for j in jobs]
        todo, self._bar = 0, None
        for j in jobs:
            done = store.done_ids(store.for_job(store.read(self.run_dir, j.model.key), j.strategy.name, j.model.provider))
            todo += len(points) - len(done & set(points["point_id"]))
        if self.progress and todo:
            from tqdm import tqdm

            self._bar = tqdm(total=todo, unit="req", smoothing=0.05, dynamic_ncols=True, mininterval=1.0)
        t0 = time.monotonic()
        try:
            outcomes = await asyncio.gather(*(self._model(j, points) for j in jobs))
        finally:
            if self._bar is not None:
                self._bar.close()
        if self._fatal:
            raise self._fatal
        for o in outcomes:
            if self.budget.stopped and not o.stopped and o.done_before + o.answered < o.total:
                o.stopped = self.budget.stopped
        self.elapsed = time.monotonic() - t0
        return list(outcomes)


def write_meta(run_dir: Path, jobs: list[Job], outcomes: list[Outcome], extra: dict[str, Any]) -> None:
    path = run_dir / "run.json"
    meta = json.loads(path.read_text()) if path.exists() else {"sessions": []}
    meta["models"] = {
        j.model.key: {
            "id": j.model.id,
            "provider": j.model.provider,
            "quantizations": list(j.model.quantizations or []),
            "strategy": j.strategy.name,
            "reasoning": j.strategy.reasons,
            "rpm": j.model.rpm,
            "concurrency": j.model.concurrency,
            "byok": j.model.byok,
        }
        for j in jobs
    } | {k: v for k, v in meta.get("models", {}).items() if k not in {j.model.key for j in jobs}}
    meta["sessions"].append(
        {
            "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **extra,
            "outcomes": [o.__dict__ for o in outcomes],
        }
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
