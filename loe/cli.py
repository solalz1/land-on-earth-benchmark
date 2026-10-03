"""Command line: loe check | pilot | run | score | render | pack | status | demo.

Every command that can spend credits reads OPENROUTER_API_KEY from the environment; the key is
never printed or written anywhere. `--fake` swaps OpenRouter for the in-process fake API and
writes under results/demo/, so the whole pipeline can be tried for free.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path

from loe import config, grid, store
from loe.client import Client, Fatal, api_key
from loe.config import CREDIT_FEE, EUR_PER_USD, RESULTS, Config

FULL = 16_200


def _out(args) -> Path:
    return RESULTS / "demo" if args.fake else RESULTS


def _client(cfg: Config, fake: bool):
    if fake:
        from loe.fake import FakeOpenRouter

        api = FakeOpenRouter(cfg)
        return Client("fake-key", transport=api.transport(), backoff=0.0, base_url="https://openrouter.ai/api/v1")
    return Client(api_key())


def _renderers(cfg: Config, models, fake: bool) -> dict:
    from loe import templates

    out = {}
    for m in models:
        if not any(cfg.strategies[s].mode == "raw" for s in m.strategies):
            continue
        out[m.key] = templates.generic_renderer(m.raw_prefill) if fake else templates.renderer(m)
    return out


async def _check(args, cfg: Config) -> int:
    from loe import check

    models = cfg.select(args.models)
    print("Lecture des fiches OpenRouter et des templates (gratuit)…")
    rend = await asyncio.to_thread(_renderers, cfg, models, args.fake)
    async with _client(cfg, args.fake) as client:
        report = await check.run_check(cfg, models, client, {k: v is not None for k, v in rend.items()})
    print(check.show(report))
    check.save(report, _out(args) / "check.json")
    return 0 if report["ok"] else 1


async def _pilot(args, cfg: Config) -> int:
    from loe import pilot

    models = cfg.select(args.models)
    out_dir = _out(args) / "pilot"
    rend = await asyncio.to_thread(_renderers, cfg, models, args.fake)
    async with _client(cfg, args.fake) as client:
        credits = None
        try:
            credits = ((await client.get("/key")).get("data") or {}).get("limit_remaining")
        except Exception:
            pass
        report = await pilot.run_pilot(cfg, models, client, rend, out_dir, args.points, args.budget, credits)
    print()
    print((out_dir / "report.md").read_text())
    proj = report["projected_usd"]
    print(f"Rapport : {out_dir / 'report.md'}")
    if credits is not None and proj > 0.9 * float(credits):
        print(f"⚠ Projection {proj:.2f} $ proche ou au-dessus des crédits restants ({credits} $).")
    return 0 if report["valid"] == report["total"] else 1


def _jobs(cfg: Config, args, out: Path):
    from loe.runner import Job

    path = out / "pilot" / "strategies.json"
    chosen = json.loads(path.read_text()) if path.exists() else {}
    models = cfg.select(args.models)
    rend = _renderers(cfg, models, args.fake)
    jobs, skipped = [], []
    for m in models:
        pilot = chosen.get(m.key) or {}
        name = args.strategy or pilot.get("strategy")
        if not name or (not args.strategy and pilot.get("status") != "ok" and not args.force):
            skipped.append(f"{m.key} ({pilot.get('status') or 'pas de pilote'})")
            continue
        jobs.append(Job(m, cfg.strategies[name], rend.get(m.key)))
    return jobs, skipped


async def _run(args, cfg: Config) -> int:
    from loe.runner import Runner, write_meta

    out = _out(args)
    run_dir = out / args.run
    jobs, skipped = await asyncio.to_thread(_jobs, cfg, args, out)
    if skipped:
        print("⚠ Modèles ignorés, faute de pilote réussi (make pilot, ou --force pour les lancer quand même) :")
        for s in skipped:
            print(f"    {s}")
    if not jobs:
        print("Aucun modèle à lancer.")
        return 1
    points = grid.load()
    if args.limit:
        points = points.sample(args.limit, random_state=1).sort_values("point_id")
    print(f"Run « {args.run} » : {len(jobs)} modèles × {len(points)} points, budget {args.budget:.2f} $ de crédits.")
    async with _client(cfg, args.fake) as client:
        runner = Runner(
            client, run_dir, budget=args.budget, provider_concurrency=cfg.provider_concurrency, cap_factor=args.cap
        )
        if args.concurrency:
            jobs = [replace(j, model=replace(j.model, concurrency=args.concurrency)) for j in jobs]
        outcomes = await runner.run(jobs, points)
    write_meta(run_dir, jobs, outcomes, {"kind": "run", "points": len(points), "budget": args.budget})
    print()
    complete = True
    for o in outcomes:
        n = o.done_before + o.answered
        line = f"  {o.key:<22} {n:>6}/{o.total}  +{o.cost:.3f} $"
        if o.errors:
            line += f"  {o.errors} erreurs"
        if o.stopped:
            line += f"  ARRÊT : {o.stopped}"
        if n < o.total:
            complete = False
        print(line)
    print(f"Dépensé dans ce run : {runner.budget.spent:.2f} $ de crédits.")
    if not complete:
        print("Run incomplet : relance la même commande pour reprendre là où il s'est arrêté.")
        return 1
    return 0


def _score(args, cfg: Config) -> int:
    from loe.render import render_run
    from loe.score import score_run

    run_dir = _out(args) / args.run
    board = score_run(cfg, run_dir)
    if board.empty:
        print(f"Rien à noter dans {run_dir}")
        return 1
    print((run_dir / "leaderboard.md").read_text())
    if not args.no_maps:
        paths = render_run(run_dir, board)
        print(f"Cartes : {run_dir / 'maps'} (figure du tweet : {paths[0]})")
    return 0


def _pack(args, cfg: Config) -> int:
    for sub in ("pilot", "pilot/probe", args.run):
        d = _out(args) / sub
        for p in store.pack(d):
            print(f"compressé : {p.relative_to(config.ROOT)}")
    return 0


def _status(args, cfg: Config) -> int:
    run_dir = _out(args) / args.run
    keys = store.models_in(run_dir)
    if not keys:
        print(f"Aucun résultat dans {run_dir}")
        return 0
    total_cost = 0.0
    print(f"{'Modèle':<22}{'Faits':>14}{'Erreurs':>9}{'Coût':>10}")
    for key in keys:
        recs = store.read(run_dir, key)
        done = len(store.done_ids(store.current(recs)))
        errs = sum(1 for r in recs if not r.get("ok"))
        cost = sum(r.get("cost") or 0.0 for r in recs)
        total_cost += cost
        print(f"{key:<22}{done:>8}/{FULL:<5}{errs:>9}{cost:>9.3f} $")
    print(f"Total : {total_cost:.2f} $ de crédits (≈ {total_cost * (1 + CREDIT_FEE) * EUR_PER_USD:.2f} €)")
    return 0


def _demo(args, cfg: Config) -> int:
    """Whole pipeline against the fake API: check, pilot, run, score, maps."""
    args.fake = True
    steps = [
        ("check", lambda: asyncio.run(_check(args, cfg))),
        ("pilot", lambda: asyncio.run(_pilot(args, cfg))),
        ("run", lambda: asyncio.run(_run(args, cfg))),
        ("score", lambda: _score(args, cfg)),
    ]
    for name, step in steps:
        print(f"\n===== démo : {name} =====")
        code = step()
        if code and name != "check":
            return code
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="loe", description="Land on Earth — Land or Water? sur 20 modèles open-weight")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(s, run=True):
        s.add_argument("--models", help="clés séparées par des virgules (défaut : les 20)")
        s.add_argument("--fake", action="store_true", help="fausse API, résultats dans results/demo/")
        if run:
            s.add_argument("--run", default="tweet", help="nom du dossier de run (défaut : tweet)")
        return s

    common(sub.add_parser("check", help="clé, fournisseurs, précisions, prix : gratuit"), run=False)
    s = common(sub.add_parser("pilot", help="choisit la stratégie de chaque modèle, 200 points"), run=False)
    s.add_argument("--points", type=int, default=200)
    s.add_argument("--budget", type=float, default=1.0, help="crédits max du pilote, en $ (défaut 1)")
    s = common(sub.add_parser("run", help="les 16 200 points sur chaque modèle, reprend après coupure"))
    s.add_argument("--budget", type=float, default=18.0, help="crédits max du run, en $ (défaut 18)")
    s.add_argument("--cap", type=float, default=2.5, help="arrête un modèle dont le coût projeté dépasse cap × estimation")
    s.add_argument("--strategy", help="force une stratégie pour tous les modèles choisis")
    s.add_argument("--force", action="store_true", help="lance aussi les modèles dont le pilote est « à vérifier »")
    s.add_argument("--concurrency", type=int, help="requêtes simultanées par modèle (défaut : config)")
    s.add_argument("--limit", type=int, help="n'interroger qu'un échantillon de N points (tests)")
    s = common(sub.add_parser("score", help="classement et cartes"))
    s.add_argument("--no-maps", action="store_true")
    common(sub.add_parser("pack", help="compresse les réponses brutes pour le repo"))
    common(sub.add_parser("status", help="avancement et coût du run"))
    s = common(sub.add_parser("demo", help="tout le pipeline contre la fausse API"))
    s.add_argument("--points", type=int, default=200)
    s.add_argument("--budget", type=float, default=18.0)
    s.add_argument("--cap", type=float, default=2.5)
    s.add_argument("--strategy", default=None)
    s.add_argument("--force", action="store_true")
    s.add_argument("--concurrency", type=int, default=None)
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--no-maps", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    cfg = config.load()
    try:
        if args.cmd == "check":
            return asyncio.run(_check(args, cfg))
        if args.cmd == "pilot":
            return asyncio.run(_pilot(args, cfg))
        if args.cmd == "run":
            return asyncio.run(_run(args, cfg))
        if args.cmd == "score":
            return _score(args, cfg)
        if args.cmd == "pack":
            return _pack(args, cfg)
        if args.cmd == "status":
            return _status(args, cfg)
        if args.cmd == "demo":
            return _demo(args, cfg)
    except Fatal as e:
        print(f"\nArrêt : {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrompu. Les réponses déjà reçues sont enregistrées : relance la même commande pour reprendre.")
        return 130
    return 1


if __name__ == "__main__":
    sys.exit(main())
