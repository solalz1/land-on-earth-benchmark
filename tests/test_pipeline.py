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
KEYS = [
    "qwen3.5-9b",
    "gpt-oss-20b",
    "kimi-k3",
    "llama-4-maverick",
    "gemma-4-31b",
    "ministral-3-14b",
    "deepseek-v4-pro",
    "qwen3.6-27b",
    "glm-5.3",
    "glm-5.2",
    "minimax-m3",
    "mistral-small-4",
    "kimi-k2.6",
    "qwen3.5-122b-a10b",
]


async def nosleep(_):
    return None


def renderers(models):
    return {m.key: generic_renderer(m.raw_prefill) for m in models}


async def test_check_finds_every_pinned_provider():
    fake = FakeOpenRouter(CFG)
    async with Client("k", transport=fake.transport(), sleep=nosleep, rate_limits=False) as c:
        report = await check.run_check(CFG, CFG.models, c, {})
    assert report["ok"], [r["problems"] for r in report["models"] if r.get("problems")]
    assert report["key"]["limit"] == 20.0
    # the decoy FP4 endpoint of the fake is never picked
    assert all(r["tag"].startswith(r["provider"]) for r in report["models"])
    assert 13 < report["total_usd"] < 15  # ~12.7 $ for the tweet's 20 models + ~1.4 $ of Mistral
    text = check.show(report)
    assert "Kimi K3" in text and "Coût estimé" in text


def test_check_flags_wrong_precision_and_missing_logprobs():
    m = CFG.model("qwen3.8-27b")
    ep = {"tag": "parasail/fp4", "quantization": "fp4", "provider_name": "Parasail",
          "supported_parameters": ["max_tokens"], "pricing": {"prompt": "0.0000014", "completion": "0.0000044"}}
    assert check.match(m, [ep]) is ep
    v = check.verdict(m, ep)
    assert any("précision" in p for p in v["problems"]) and any("logprobs" in p for p in v["problems"])
    assert check.verdict(m, None)["problems"]


async def test_pilot_picks_the_right_strategy_for_each_behaviour(tmp_path):
    models = [CFG.model(k) for k in KEYS]
    fake = FakeOpenRouter(CFG, fail_rate=0.02)
    async with Client("k", transport=fake.transport(), sleep=nosleep, rate_limits=False) as c:
        report = await pilot.run_pilot(CFG, models, c, renderers(models), tmp_path, n_points=60, progress=False)
    chosen = {r["key"]: r["strategy"] for r in report["models"]}
    assert chosen == {
        "qwen3.5-9b": "chat_none",
        "gpt-oss-20b": "chat_low",  # reasoning mandatory, and Novita does not pass raw prompts through
        "kimi-k3": "chat_none",
        "llama-4-maverick": "chat",  # no reasoning parameter
        "gemma-4-31b": "chat_none",  # markdown around the answer is fine
        "ministral-3-14b": "text",  # no logprobs at the provider
        "deepseek-v4-pro": "text_none",  # SiliconFlow: no logprobs
        "qwen3.6-27b": "chat_none",  # with 5 top logprobs instead of 20
        "glm-5.3": "text_low",  # reasoning mandatory, no logprobs at SiliconFlow
        "glm-5.2": "text_none",  # SiliconFlow: no logprobs
        "minimax-m3": "chat_low",  # reasons whatever the switch says
        "mistral-small-4": "text_none",  # hybrid: reasons unless told not to
        "kimi-k2.6": "text_none",  # SiliconFlow: no logprobs
        "qwen3.5-122b-a10b": "chat_none",  # Alibaba: logprobs, 5 at most
    }
    by_key = {r["key"]: r for r in report["models"]}
    assert report["valid"] == len(KEYS), [(k, r["status"]) for k, r in by_key.items() if r["status"] != "ok"]
    assert {k for k, r in by_key.items() if r["reasoning"]} == {"gpt-oss-20b", "glm-5.3", "minimax-m3"}
    assert by_key["deepseek-v4-pro"]["logprobs_share"] == 0
    assert by_key["kimi-k3"]["logprobs_share"] == 1  # Kimi K3 at Alibaba, without the temperature it refuses
    # your keys served exactly the models configured for them
    assert {k for k, r in by_key.items() if r["byok_share"] == 1} == {k for k in KEYS if CFG.model(k).byok}
    assert {k for k, r in by_key.items() if r["byok_share"] == 0} == {k for k in KEYS if not CFG.model(k).byok}
    for r in by_key.values():  # the pilot's area-weighted estimate is near the full map's accuracy
        assert abs(r["accuracy_area"] - accuracy_of(fake, CFG.model(r["key"]))) < 0.08, r["key"]
    saved = json.loads((tmp_path / "strategies.json").read_text())
    assert saved["gpt-oss-20b"] == saved["gpt-oss-20b"] | {"strategy": "chat_low", "reasoning": True}
    md = (tmp_path / "report.md").read_text()
    assert "Reasoning is mandatory" in md
    assert "| Kimi K3 | chat_none | aucune | Alibaba | oui |" in md and "| Qwen3.5-9B | chat_none | aucune | Parasail | non |" in md
    assert "Précision estimée" in md
    assert (tmp_path / "probe" / "raw" / "kimi-k3.jsonl").exists()


async def test_probe_finds_a_provider_that_passes_raw_prompts(tmp_path):
    from loe import probe

    m = CFG.model("gpt-oss-20b")
    fake = FakeOpenRouter(CFG, fail_rate=0)
    async with Client("k", transport=fake.transport(), sleep=nosleep, rate_limits=False) as c:
        rows = await probe.run_probe(
            CFG, m, ["novita", "deepinfra"], ["raw"], c, generic_renderer(m.raw_prefill), tmp_path
        )
    verdicts = {r["provider"]: r["verdict"] for r in rows}
    assert verdicts["deepinfra"] == "ok" and verdicts["novita"] != "ok"
    assert all(b["provider"]["only"] in (["novita"], ["deepinfra"]) for b in fake.requests)
    assert (tmp_path / "raw" / "gpt-oss-20b@deepinfra.jsonl").exists()
    assert "deepinfra + raw" in probe.show(rows)


async def test_run_score_and_maps(tmp_path):
    models = [CFG.model(k) for k in ("qwen3.5-9b", "ministral-3-14b")]
    jobs = [Job(models[0], CFG.strategies["chat_none"]), Job(models[1], CFG.strategies["text"])]
    fake = FakeOpenRouter(CFG, fail_rate=0.01)
    async with Client("k", transport=fake.transport(), sleep=nosleep, rate_limits=False) as c:
        out = await Runner(c, tmp_path, budget=5, progress=False).run(jobs, grid.load())
    assert all(o.done_before + o.answered == 16_200 for o in out)
    board = score_run(CFG, tmp_path)
    assert list(board.columns[:3]) == ["rank", "key", "name"]
    for row in board.itertuples():
        expected = accuracy_of(fake, CFG.model(row.key))
        assert abs(row.accuracy - expected) < 1e-9
        assert row.coverage == 1.0
        assert 0.70 < row.always_water < 0.72
    assert board.set_index("key").loc["ministral-3-14b", "logprobs_share"] == 0
    assert not board["reasoning"].any()
    assert "| Réflexion |" in (tmp_path / "leaderboard.md").read_text()
    assert board.set_index("key").loc["qwen3.5-9b", "logprobs_share"] == 1
    paths = render_run(tmp_path, board)
    assert paths[0].exists() and paths[0].stat().st_size > 10_000
    for sub in ("", "prob/", "errors/"):
        assert (tmp_path / "maps" / f"{sub}qwen3.5-9b.png").exists()
    assert (tmp_path / "leaderboard.md").read_text().count("|") > 20
    pred = (tmp_path / "predictions" / "qwen3.5-9b.csv").read_text().splitlines()
    assert len(pred) == 16_201
