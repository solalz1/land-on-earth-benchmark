"""The dress rehearsal must catch every problem the pilots and the key tests ran into, and only those."""

import json
from dataclasses import replace

import pytest

from loe import config, grid, preflight, store
from loe.client import Client
from loe.fake import FakeOpenRouter
from loe.runner import Job, Runner

CFG = config.load()


async def nosleep(_):
    return None


def write_strategies(pilot_dir, entries):
    pilot_dir.mkdir(parents=True, exist_ok=True)
    (pilot_dir / "strategies.json").write_text(json.dumps(entries))


def ok(strategy, projected=0.1):
    return {"strategy": strategy, "status": "ok", "projected_usd": projected}


async def rehearse(tmp_path, models, entries, fake, budget=22.0, credits=None, rate_limits=False, n_points=40):
    write_strategies(tmp_path, entries)
    async with Client("k", transport=fake.transport(), sleep=nosleep, rate_limits=rate_limits) as c:
        return await preflight.run_preflight(CFG, models, c, {}, tmp_path, budget, credits, n_points, progress=False)


def by_key(report):
    return {r["key"]: r for r in report["models"]}


async def test_green_light_when_everything_works(tmp_path):
    models = [CFG.model(k) for k in ("qwen3.5-9b", "ministral-3-8b", "gpt-oss-20b")]
    entries = {"qwen3.5-9b": ok("chat_none"), "ministral-3-8b": ok("text"), "gpt-oss-20b": ok("chat_low", 0.3)}
    report = await rehearse(tmp_path, models, entries, FakeOpenRouter(CFG, fail_rate=0))
    assert report["ready"], report["checks"]
    rows = by_key(report)
    assert all(r["status"] == "ok" and r["requests"] == 40 and r["answered"] == 40 for r in rows.values())
    assert rows["gpt-oss-20b"]["reasoning"] is True
    assert report["projected_usd"] == pytest.approx(0.5)  # the pilot's projections, higher than the rehearsal's here
    # who pays: Ministral through your Mistral key, the others on OpenRouter's credits
    assert rows["ministral-3-8b"]["payer"] == "mistral" and rows["qwen3.5-9b"]["payer"] == "OpenRouter"
    assert report["by_payer"] == {"OpenRouter": pytest.approx(0.4), "mistral": pytest.approx(0.1)}
    md = (tmp_path / "preflight.md").read_text()
    assert "FEU VERT" in md and "| Qwen3.5-9B | ok | 40/40 |" in md and "ta clé Mistral" in md
    # the rehearsal used fresh points, none of the pilot's
    session = next((tmp_path / "rehearsal").iterdir())
    asked = {r["point_id"] for r in store.read(session, "qwen3.5-9b")}
    assert asked.isdisjoint(set(grid.pilot_points(200)["point_id"]))
    assert preflight.gate(tmp_path) is None


async def test_rehearsal_uses_the_run_settings(tmp_path):
    m = CFG.model("deepseek-v4-pro")  # your SiliconFlow key, 24 in flight, 480 per minute
    fake = FakeOpenRouter(CFG, fail_rate=0)
    report = await rehearse(tmp_path, [m], {m.key: ok("text_none", 0.8)}, fake, n_points=preflight.N_POINTS)
    row = by_key(report)[m.key]
    assert row["requests"] == 300 and row["status"] == "ok" and row["byok"] == 300 and row["concurrency"] == 24
    assert all(b["provider"] == {"only": ["siliconflow"], "allow_fallbacks": False, "require_parameters": True,
                                 "quantizations": ["fp8"]} and "logprobs" not in b for b in fake.requests)
    # the rehearsal measures more than the pilot's 0.80 $: the higher projection is kept, billed by SiliconFlow
    assert report["by_payer"] == {"siliconflow": pytest.approx(16_200 * (40 * 1.50162 + 3.135) / 1e6, abs=0.01)}


async def test_unpaced_model_hitting_the_account_limit_is_blocking(tmp_path):
    # the second pilot's mistake: a capped model, without your key and without pacing
    m = replace(CFG.model("kimi-k2.6"), provider="parasail", byok=False, rpm=None)
    fake = FakeOpenRouter(CFG, fail_rate=0, rate_window=0.5, rate_limit=5)
    report = await rehearse(tmp_path, [m], {m.key: ok("text_none")}, fake)
    row = by_key(report)[m.key]
    assert not report["ready"] and row["status"] == "bloquant"
    assert row["rate_limited_account"] > 0
    assert any("rpm: 19" in p and "ta clé" in p for p in row["problems"])
    assert "FEU ROUGE" in (tmp_path / "preflight.md").read_text()
    assert "rouge" in preflight.gate(tmp_path)


async def test_paced_model_passes_the_same_limit(tmp_path):
    m = replace(CFG.model("kimi-k2.6"), provider="parasail", byok=False, rpm=480)  # 1 every 125 ms: under 5 per 0.5 s
    fake = FakeOpenRouter(CFG, fail_rate=0, rate_window=0.5, rate_limit=5)
    report = await rehearse(tmp_path, [m], {m.key: ok("text_none")}, fake, rate_limits=True)
    row = by_key(report)[m.key]
    assert row["status"] == "ok" and row["rate_limited_account"] == 0 and fake.rate_limited == 0
    assert 400 <= row["rate_per_min"] <= 480


async def test_a_key_that_does_not_serve_its_model_is_blocking(tmp_path):
    m = CFG.model("glm-5.2")  # meant for your SiliconFlow key...
    fake = FakeOpenRouter(CFG, fail_rate=0, byok=set())  # ...which is missing from OpenRouter's integrations
    report = await rehearse(tmp_path, [m], {m.key: ok("text_none")}, fake)
    row = by_key(report)[m.key]
    assert row["status"] == "bloquant" and row["byok"] == 0
    assert any("ta clé SiliconFlow n'a servi que 0/40" in p and "z-ai/glm-5.2" in p for p in row["problems"])


def test_a_few_answers_on_openrouters_capacity_are_only_a_warning():
    # the third pilot: Qwen3.5-397B answered 297 times through your Alibaba key, 3 times through
    # OpenRouter's own capacity at Alibaba while your key was busy
    m = CFG.model("qwen3.5-397b-a17b")

    def records(n_byok, n):
        return [{"point_id": i, "ok": True, "pred": 1, "latency": 1.0, "ts": f"2026-10-06T14:28:{i % 60:02d}+00:00",
                 "byok": i < n_byok, "cost": 0.0 if i < n_byok else 1.8e-5, "upstream_cost": 1.8e-5}
                for i in range(n)]

    row = preflight.model_check(m, CFG.strategies["chat_none"], ok("chat_none"), records(297, 300), 0.8)
    assert row["status"] == "attention" and not row["problems"]
    assert "3 réponses sur 300 servies par la capacité d'OpenRouter chez Alibaba" in row["warnings"][0]
    row = preflight.model_check(m, CFG.strategies["chat_none"], ok("chat_none"), records(200, 300), 0.8)
    assert row["status"] == "bloquant" and "n'a servi que 200/300" in row["problems"][0]


async def test_a_key_serving_an_unplanned_model_is_a_warning(tmp_path):
    m = CFG.model("qwen3.5-27b")  # OpenRouter's credits at Novita, but a key of yours there has no model filter
    fake = FakeOpenRouter(CFG, fail_rate=0, byok={(m.id, "novita")})
    report = await rehearse(tmp_path, [m], {m.key: ok("chat_none")}, fake)
    row = by_key(report)[m.key]
    assert row["status"] == "attention" and report["ready"]
    assert "filtre" in row["warnings"][0]


async def test_provider_refusals_are_blocking(tmp_path):
    m = replace(CFG.model("glm-5.2"), provider="gmicloud", byok=False, rpm=None)  # the second pilot's GLM-5.2
    fake = FakeOpenRouter(CFG, fail_rate=0, provider_errors={"gmicloud": 0.5})
    report = await rehearse(tmp_path, [m], {m.key: ok("text_none")}, fake)
    row = by_key(report)[m.key]
    assert row["status"] == "bloquant" and row["refused"] > 0
    assert "Arrearage" in row["problems"][0]
    # the same model at SiliconFlow, as configured now, is clean
    report = await rehearse(tmp_path, [CFG.model("glm-5.2")], {m.key: ok("text_none")},
                            FakeOpenRouter(CFG, fail_rate=0))
    assert by_key(report)[m.key]["status"] == "ok"


async def test_empty_answers_are_blocking(tmp_path):
    m = CFG.model("minimax-m3")  # reasons whatever the switch: 8 tokens are not enough
    report = await rehearse(tmp_path, [m], {m.key: ok("chat_none")}, FakeOpenRouter(CFG, fail_rate=0))
    row = by_key(report)[m.key]
    assert row["status"] == "bloquant" and row["empty"] == 40


async def test_budget_and_credits_are_checked(tmp_path):
    m = CFG.model("qwen3.5-9b")
    report = await rehearse(tmp_path, [m], {m.key: ok("chat_none", 21.0)}, FakeOpenRouter(CFG, fail_rate=0),
                            budget=22.0, credits=15.0)
    checks = {c["name"]: c["ok"] for c in report["checks"]}
    assert checks == {"Modèles": True, "Budget": False, "Crédits": False} and not report["ready"]


async def test_credits_only_cover_what_openrouter_bills(tmp_path):
    models = [CFG.model("qwen3.5-9b"), CFG.model("ministral-3-8b")]  # the second one through your Mistral key
    entries = {"qwen3.5-9b": ok("chat_none", 1.0), "ministral-3-8b": ok("text", 12.0)}
    report = await rehearse(tmp_path, models, entries, FakeOpenRouter(CFG, fail_rate=0), budget=22.0, credits=5.0)
    checks = {c["name"]: c["ok"] for c in report["checks"]}
    assert checks == {"Modèles": True, "Budget": True, "Crédits": True} and report["ready"]
    assert report["by_payer"]["mistral"] == pytest.approx(12.0)


async def test_model_without_a_validated_pilot_is_blocking(tmp_path):
    m = CFG.model("qwen3.5-9b")
    report = await rehearse(tmp_path, [m], {m.key: {"strategy": None, "status": "échec"}}, FakeOpenRouter(CFG))
    assert not report["ready"] and by_key(report)[m.key]["status"] == "bloquant"


def test_gate_requires_a_fresh_rehearsal(tmp_path, monkeypatch):
    assert "make pilot" in preflight.gate(tmp_path)
    (tmp_path / "preflight.json").write_text(json.dumps({"ready": True, "config": "an old config"}))
    assert "a changé" in preflight.gate(tmp_path)
    (tmp_path / "preflight.json").write_text(json.dumps({"ready": True, "config": preflight.config_hash()}))
    assert preflight.gate(tmp_path) is None


def test_durations_read_in_french():
    assert preflight.duration(0.58) == "≈ 35 min"
    assert preflight.duration(1.17) == "≈ 1 h 10"
    assert preflight.duration(None) == "—"


async def test_a_new_provider_asks_every_point_again(tmp_path):
    m_old = replace(CFG.model("glm-5.2"), provider="gmicloud", byok=False, rpm=None)
    m_new = CFG.model("glm-5.2")  # siliconflow
    pts = preflight.rehearsal_points(20)
    fake = FakeOpenRouter(CFG, fail_rate=0, provider_errors={})
    async with Client("k", transport=fake.transport(), sleep=nosleep, rate_limits=False) as c:
        await Runner(c, tmp_path, budget=5, progress=False).run([Job(m_old, CFG.strategies["text_none"])], pts)
        out = await Runner(c, tmp_path, budget=5, progress=False).run([Job(m_new, CFG.strategies["text_none"])], pts)
    assert out[0].done_before == 0 and out[0].answered == 20  # GMICloud's answers are not reused
    recs = store.read(tmp_path, m_new.key)
    assert {r["pinned"] for r in recs} == {"gmicloud", "siliconflow"}
    assert {r["pinned"] for r in store.current(recs)} == {"siliconflow"}  # the score uses SiliconFlow only
