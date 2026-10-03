"""End to end against the fake API: check, pilot, run, score, maps — without spending anything."""

import json

from loe import check, config, grid, pilot
from loe.client import Client
from loe.fake import FakeOpenRouter, accuracy_of
from loe.render import render_run
from loe.runner import Job, Runner
from loe.score import score_run
from loe.templates import generic_renderer

CFG = config.load()
KEYS = ["qwen3.5-9b", "gpt-oss-20b", "kimi-k3", "llama-4-maverick", "gemma-4-31b", "mistral-large-3"]


async def nosleep(_):
    return None


def renderers(models):
    return {m.key: generic_renderer(m.raw_prefill) for m in models}


async def test_check_finds_every_pinned_provider():
    fake = FakeOpenRouter(CFG)
    async with Client("k", transport=fake.transport(), sleep=nosleep) as c:
        report = await check.run_check(CFG, CFG.models, c, {})
    assert report["ok"], [r["problems"] for r in report["models"] if r.get("problems")]
    assert report["key"]["limit"] == 20.0
    # the decoy FP4 endpoint of the fake is never picked
    assert all(r["tag"].startswith(r["provider"]) for r in report["models"])
    assert 12 < report["total_usd"] < 13.5  # the spec's ~12.7 $ for the tweet run
    text = check.show(report)
    assert "Kimi K3" in text and "Coût estimé" in text


def test_check_flags_wrong_precision_and_missing_logprobs():
    m = CFG.model("glm-5.3")
    ep = {"tag": "parasail/fp4", "quantization": "fp4", "provider_name": "Parasail",
          "supported_parameters": ["max_tokens"], "pricing": {"prompt": "0.0000014", "completion": "0.0000044"}}
    assert check.match(m, [ep]) is ep
    v = check.verdict(m, ep)
    assert any("précision" in p for p in v["problems"]) and any("logprobs" in p for p in v["problems"])
    assert check.verdict(m, None)["problems"]


async def test_pilot_picks_the_right_strategy_for_each_behaviour(tmp_path):
    models = [CFG.model(k) for k in KEYS]
    fake = FakeOpenRouter(CFG, fail_rate=0.02)
    async with Client("k", transport=fake.transport(), sleep=nosleep) as c:
        report = await pilot.run_pilot(CFG, models, c, renderers(models), tmp_path, n_points=60, progress=False)
    chosen = {r["key"]: r["strategy"] for r in report["models"]}
    assert chosen == {
        "qwen3.5-9b": "chat_none",
        "gpt-oss-20b": "raw",  # reasoning cannot be switched off through the API
        "kimi-k3": "raw",  # reasons despite effort none
        "llama-4-maverick": "chat",  # no reasoning parameter
        "gemma-4-31b": "chat_none",  # markdown around the answer is fine
        "mistral-large-3": "text",  # no logprobs at the provider
    }
    assert report["valid"] == len(KEYS)
    saved = json.loads((tmp_path / "strategies.json").read_text())
    assert saved["gpt-oss-20b"]["strategy"] == "raw"
    md = (tmp_path / "report.md").read_text()
    assert "Reasoning is mandatory" in md and "| Kimi K3 | raw |" in md
    assert (tmp_path / "probe" / "raw" / "kimi-k3.jsonl").exists()


async def test_run_score_and_maps(tmp_path):
    models = [CFG.model(k) for k in ("qwen3.5-9b", "mistral-large-3")]
    jobs = [Job(models[0], CFG.strategies["chat_none"]), Job(models[1], CFG.strategies["text"])]
    fake = FakeOpenRouter(CFG, fail_rate=0.01)
    async with Client("k", transport=fake.transport(), sleep=nosleep) as c:
        out = await Runner(c, tmp_path, budget=5, progress=False).run(jobs, grid.load())
    assert all(o.done_before + o.answered == 16_200 for o in out)
    board = score_run(CFG, tmp_path)
    assert list(board.columns[:3]) == ["rank", "key", "name"]
    for row in board.itertuples():
        expected = accuracy_of(fake, CFG.model(row.key))
        assert abs(row.accuracy - expected) < 1e-9
        assert row.coverage == 1.0
        assert 0.70 < row.always_water < 0.72
    assert board.set_index("key").loc["mistral-large-3", "logprobs_share"] == 0
    assert board.set_index("key").loc["qwen3.5-9b", "logprobs_share"] == 1
    paths = render_run(tmp_path, board)
    assert paths[0].exists() and paths[0].stat().st_size > 10_000
    for sub in ("", "prob/", "errors/"):
        assert (tmp_path / "maps" / f"{sub}qwen3.5-9b.png").exists()
    assert (tmp_path / "leaderboard.md").read_text().count("|") > 20
    pred = (tmp_path / "predictions" / "qwen3.5-9b.csv").read_text().splitlines()
    assert len(pred) == 16_201
