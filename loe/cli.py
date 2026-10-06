"""Command line: loe check | pilot | preflight | probe | run | score | figure | hf | pack | status | demo.

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
        return Client(
            "fake-key",
            transport=api.transport(),
            backoff=0.0,
            base_url="https://openrouter.ai/api/v1",
            rate_limits=False,  # the fake API has no per-minute limit, keep the demo fast
        )
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
        expected = None if args.strategy else pilot.get("projected_usd")
        jobs.append(Job(m, cfg.strategies[name], rend.get(m.key), expected_usd=expected))
    return jobs, skipped


async def _preflight(args, cfg: Config) -> int:
    from loe import preflight

    models = cfg.select(args.models)
    rend = await asyncio.to_thread(_renderers, cfg, models, args.fake)
    async with _client(cfg, args.fake) as client:
        credits = None
        try:
            credits = ((await client.get("/key")).get("data") or {}).get("limit_remaining")
        except Exception:
            pass
        report = await preflight.run_preflight(
            cfg, models, client, rend, _out(args) / "pilot", args.budget, credits, args.points or preflight.N_POINTS
        )
    print()
    print(preflight.markdown(report))
    return 0 if report["ready"] else 1


async def _run(args, cfg: Config) -> int:
    from loe import preflight
    from loe.runner import Runner, write_meta

    out = _out(args)
    run_dir = out / args.run
    if not args.no_preflight:
        reason = preflight.gate(out / "pilot")
        if reason:
            print(f"Run bloqué : {reason}. (--no-preflight pour passer outre)")
            return 1
        pf = json.loads((out / "pilot" / "preflight.json").read_text())
        print(
            f"Répétition générale au vert : environ {pf['projected_usd']:.2f} $ et {preflight.duration(pf['hours'])} "
            f"(durée fixée par {', '.join(pf['slowest'])})."
        )
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
    print(f"Run « {args.run} » : {len(jobs)} modèles × {len(points)} points, budget {args.budget:.2f} $ (crédits et clés).")
    projected = sum(j.expected_usd or 0.0 for j in jobs) * len(points) / FULL
    if projected > 0.9 * args.budget:
        print(
            f"⚠ Le pilote projette {projected:.2f} $ pour ce run, proche du budget de {args.budget:.2f} $ : "
            f"le run s'arrêtera au budget. Pour le relever : uv run python -m loe run --budget {projected * 1.2:.0f}"
        )
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
    print(f"Dépensé dans ce run : {runner.budget.spent:.2f} $ (crédits OpenRouter et clés fournisseurs).")
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


def _figure(args, cfg: Config) -> int:
    from loe import figure

    run_dir = _out(args) / args.run
    try:
        paths = figure.make(run_dir)
    except FileNotFoundError as e:
        print(e)
        return 1
    for p in paths:
        print(f"écrit : {p.relative_to(config.ROOT) if p.is_relative_to(config.ROOT) else p}")
    return 0


def _hf(args, cfg: Config) -> int:
    from loe import hf

    out = config.ROOT / "hf"
    try:
        paths = hf.build(_out(args) / args.run, out)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return 1
    size = sum(p.stat().st_size for p in paths) / 1e6
    print(f"{len(paths)} fichiers dans hf/ ({size:.0f} Mo). Pour les publier sur Hugging Face :\n")
    print(f"  hf upload {config.HF_DATASET} hf --repo-type=dataset\n")
    return 0


async def _probe(args, cfg: Config) -> int:
    from loe import probe, templates

    model = cfg.model(args.model)
    strategies = [s.strip() for s in args.strategy.split(",") if s.strip()]
    unknown = [s for s in strategies if s not in cfg.strategies]
    if unknown:
        print(f"Stratégies inconnues : {', '.join(unknown)} (voir configs/models.yaml)")
        return 1
    providers = [p.strip() for p in args.provider.split(",") if p.strip()]
    quants = [q.strip() for q in args.quantizations.split(",")] if args.quantizations else None
    render = None
    if any(cfg.strategies[s].mode == "raw" for s in strategies):
        render = templates.generic_renderer(model.raw_prefill) if args.fake else templates.renderer(model)
    print(f"{model.name} : {len(providers)} fournisseurs × {len(strategies)} stratégies, 8 points chacun…")
    async with _client(cfg, args.fake) as client:
        rows = await probe.run_probe(cfg, model, providers, strategies, client, render, _out(args) / "probe", quants)
    print(probe.show(rows))
    return 0 if any(r["verdict"] == "ok" for r in rows) else 1


def _pack(args, cfg: Config) -> int:
    base = _out(args)
    for raw in sorted({p.parent for p in base.rglob("raw/*.jsonl")}):
        if not args.fake and "demo" in raw.relative_to(base).parts:
            continue
        for p in store.pack(raw.parent):
            print(f"compressé : {p.relative_to(config.ROOT)}")
    return 0


def _status(args, cfg: Config) -> int:
    run_dir = _out(args) / args.run
    keys = store.models_in(run_dir)
    if not keys:
        print(f"Aucun résultat dans {run_dir}")
        return 0
    total_cost, by_key = 0.0, 0.0
    print(f"{'Modèle':<22}{'Faits':>14}{'Erreurs':>9}{'Coût':>10}  Payé par")
    for key in keys:
        recs = store.read(run_dir, key)
        done = len(store.done_ids(store.current(recs)))
        errs = sum(1 for r in recs if not r.get("ok"))
        cost = sum(store.charged(r) for r in recs)
        mine = sum(store.charged(r) for r in recs if store.is_byok(r))
        total_cost += cost
        by_key += mine
        payer = "ta clé" if mine and mine >= 0.5 * cost else "OpenRouter"
        print(f"{key:<22}{done:>8}/{FULL:<5}{errs:>9}{cost:>9.3f} $  {payer}")
    credits = total_cost - by_key
    print(
        f"Total : {total_cost:.2f} $, dont {credits:.2f} $ de crédits OpenRouter "
        f"(≈ {credits * (1 + CREDIT_FEE) * EUR_PER_USD:.2f} €) et {by_key:.2f} $ facturés par tes clés fournisseurs"
    )
    return 0


def _demo(args, cfg: Config) -> int:
    """Whole pipeline against the fake API: check, pilot, run, score, maps."""
    args.fake = True
    steps = [
        ("check", lambda: asyncio.run(_check(args, cfg))),
        ("pilot", lambda: asyncio.run(_pilot(args, cfg))),
        ("preflight", lambda: asyncio.run(_preflight(argparse.Namespace(**{**vars(args), "points": 40}), cfg))),
        ("run", lambda: asyncio.run(_run(args, cfg))),
        ("score", lambda: _score(args, cfg)),
        ("figure", lambda: _figure(args, cfg)),
    ]
    for name, step in steps:
        print(f"\n===== démo : {name} =====")
        code = step()
        if code and name != "check":
            return code
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="loe", description="Land on Earth — Land or Water? sur des modèles open-weight")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(s, run=True):
        s.add_argument("--models", help="clés séparées par des virgules (défaut : les 24)")
        s.add_argument("--fake", action="store_true", help="fausse API, résultats dans results/demo/")
        if run:
            s.add_argument("--run", default="tweet", help="nom du dossier de run (défaut : tweet)")
        return s

    common(sub.add_parser("check", help="clé, fournisseurs, précisions, prix : gratuit"), run=False)
    s = common(sub.add_parser("pilot", help="choisit la stratégie de chaque modèle, 200 points"), run=False)
    s.add_argument("--points", type=int, default=200)
    s.add_argument("--budget", type=float, default=2.0, help="coût max du pilote, en $ (défaut 2)")
    s = common(sub.add_parser("preflight", help="répétition générale : 300 nouveaux points par modèle aux réglages du run, feu vert ou rouge"), run=False)
    s.add_argument("--budget", type=float, default=22.0, help="budget du run à vérifier, en $ (défaut 22)")
    s.add_argument("--points", type=int, default=None, help="points par modèle (défaut 300)")
    s = sub.add_parser("probe", help="essaie un modèle chez d'autres fournisseurs, 8 points par essai")
    s.add_argument("--model", required=True, help="clé du modèle (configs/models.yaml)")
    s.add_argument("--provider", required=True, help="fournisseurs séparés par des virgules")
    s.add_argument("--strategy", default="raw", help="stratégies séparées par des virgules (défaut : raw)")
    s.add_argument("--quantizations", help="précisions acceptées, séparées par des virgules (défaut : toutes)")
    s.add_argument("--fake", action="store_true", help="fausse API, résultats dans results/demo/")
    s.add_argument("--run", default="tweet", help=argparse.SUPPRESS)
    s = common(sub.add_parser("run", help="les 16 200 points sur chaque modèle, reprend après coupure"))
    s.add_argument("--budget", type=float, default=22.0, help="coût max du run, crédits et clés fournisseurs, en $ (défaut 22)")
    s.add_argument("--cap", type=float, default=2.5, help="arrête un modèle dont le coût projeté dépasse cap × estimation")
    s.add_argument("--strategy", help="force une stratégie pour tous les modèles choisis")
    s.add_argument("--force", action="store_true", help="lance aussi les modèles dont le pilote est « à vérifier »")
    s.add_argument("--no-preflight", action="store_true", help="lance le run sans répétition générale au vert")
    s.add_argument("--concurrency", type=int, help="requêtes simultanées par modèle (défaut : config)")
    s.add_argument("--limit", type=int, help="n'interroger qu'un échantillon de N points (tests)")
    s = common(sub.add_parser("score", help="classement et cartes"))
    s.add_argument("--no-maps", action="store_true")
    common(sub.add_parser("figure", help="figures et chiffres de RESULTS.md, à partir du classement"))
    common(sub.add_parser("hf", help="dossier hf/ du dataset Hugging Face, à partir du classement et des figures"))
    common(sub.add_parser("pack", help="compresse les réponses brutes pour le repo"))
    common(sub.add_parser("status", help="avancement et coût du run"))
    s = common(sub.add_parser("demo", help="tout le pipeline contre la fausse API"))
    s.add_argument("--points", type=int, default=200)
    s.add_argument("--budget", type=float, default=22.0)
    s.add_argument("--cap", type=float, default=2.5)
    s.add_argument("--strategy", default=None)
    s.add_argument("--force", action="store_true")
    s.add_argument("--no-preflight", action="store_true")
    s.add_argument("--concurrency", type=int, default=None)
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--no-maps", action="store_true")
    return p


def raise_open_files(target: int = 4096) -> None:
    """Hundreds of requests run at once, each with its socket: macOS allows only 256 open files by default."""
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        want = target if hard == resource.RLIM_INFINITY else min(target, hard)
        if soft != resource.RLIM_INFINITY and soft < want:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
    except (ImportError, ValueError, OSError):
        pass  # Windows, or a system that refuses: the default limit stays


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    raise_open_files()
    cfg = config.load()
    try:
        if args.cmd == "check":
            return asyncio.run(_check(args, cfg))
        if args.cmd == "pilot":
            return asyncio.run(_pilot(args, cfg))
        if args.cmd == "probe":
            return asyncio.run(_probe(args, cfg))
        if args.cmd == "preflight":
            return asyncio.run(_preflight(args, cfg))
        if args.cmd == "run":
            return asyncio.run(_run(args, cfg))
        if args.cmd == "score":
            return _score(args, cfg)
        if args.cmd == "figure":
            return _figure(args, cfg)
        if args.cmd == "hf":
            return _hf(args, cfg)
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
