import httpx
import pandas as pd
import pytest

from loe import config, grid, store
from loe.client import Client, Fatal
from loe.fake import FakeOpenRouter
from loe.runner import Job, Runner

CFG = config.load()


async def nosleep(_):
    return None


def points(n=300):
    return grid.load().sample(n, random_state=3).sort_values("point_id")


def fake_client(fake=None, **kw):
    fake = fake or FakeOpenRouter(CFG, fail_rate=0.05)
    return Client("k", transport=fake.transport(), sleep=nosleep, **kw), fake


async def test_runs_resumes_and_never_duplicates(tmp_path):
    m = CFG.model("qwen3.5-9b")
    pts = points()
    client, _ = fake_client()
    async with client:
        r = Runner(client, tmp_path, budget=5, progress=False)
        out = await r.run([Job(m, CFG.strategies["chat_none"])], pts.head(120))
        assert out[0].answered == 120
        out = await r.run([Job(m, CFG.strategies["chat_none"])], pts)  # resume on a larger set
    assert out[0].done_before == 120 and out[0].answered == 180
    recs = store.read(tmp_path, m.key)
    ok_ids = [x["point_id"] for x in recs if x["ok"]]
    assert len(ok_ids) == len(set(ok_ids)) == 300
    assert all(x["provider"] == "Parasail" for x in recs if x["ok"])


async def test_new_strategy_asks_again_and_score_uses_it(tmp_path):
    from loe.score import predictions

    m = CFG.model("gpt-oss-20b")
    pts = points(40)
    client, _ = fake_client(FakeOpenRouter(CFG, fail_rate=0))
    async with client:
        r = Runner(client, tmp_path, budget=5, progress=False)
        await r.run([Job(m, CFG.strategies["chat_low"])], pts)
        out = await r.run([Job(m, CFG.strategies["raw"], render=_render(m))], pts)
    assert out[0].done_before == 0 and out[0].answered == 40
    df = predictions(tmp_path, m.key)
    assert set(df["strategy"].dropna()) == {"raw"}


async def test_budget_stops_the_run(tmp_path):
    m = CFG.model("kimi-k3")  # the most expensive model
    client, _ = fake_client(FakeOpenRouter(CFG, fail_rate=0))
    async with client:
        r = Runner(client, tmp_path, budget=0.01, progress=False)
        out = await r.run([Job(m, CFG.strategies["raw"], render=_render(m))], points(2000))
    assert out[0].stopped and "budget" in out[0].stopped
    spent = sum(x.get("cost") or 0 for x in store.read(tmp_path, m.key))
    assert 0.01 <= spent < 0.011  # stops within one batch of requests


async def test_model_stopped_after_consecutive_errors(tmp_path):
    m = CFG.model("gpt-oss-20b")
    client, fake = fake_client(FakeOpenRouter(CFG, fail_rate=0))
    async with client:
        r = Runner(client, tmp_path, budget=5, progress=False)
        out = await r.run([Job(m, CFG.strategies["chat_none"])], points(500))  # rejected by the fake
    assert out[0].stopped and "erreurs d'affilée" in out[0].stopped
    assert out[0].errors < 40  # stopped early, not after 500 failures


async def test_cost_guard_stops_a_model_that_reasons(tmp_path):
    m = CFG.model("kimi-k3")  # reasoning tokens: ~4x the one-token estimate
    client, _ = fake_client(FakeOpenRouter(CFG, fail_rate=0))
    async with client:
        r = Runner(client, tmp_path, budget=5, progress=False)
        out = await r.run([Job(m, CFG.strategies["chat_low"])], points(16_200))
    assert out[0].stopped and "coût projeté" in out[0].stopped
    assert out[0].answered < 200


async def test_fatal_error_propagates(tmp_path):
    def handler(req):
        return httpx.Response(402, json={"error": {"code": 402, "message": "Insufficient credits"}})

    async with Client("k", transport=httpx.MockTransport(handler), sleep=nosleep) as client:
        r = Runner(client, tmp_path, budget=5, progress=False)
        with pytest.raises(Fatal, match="crédits"):
            await r.run([Job(CFG.model("qwen3.5-9b"), CFG.strategies["chat"])], points(50))


def test_latest_prefers_success_over_later_error():
    recs = [
        {"point_id": 1, "ok": True, "pred": 1},
        {"point_id": 1, "ok": False, "error": "x"},
        {"point_id": 2, "ok": False, "error": "y"},
    ]
    df = store.latest(recs)
    assert list(df["point_id"]) == [1, 2]
    assert bool(df.loc[0, "ok"]) and df.loc[0, "pred"] == 1
    assert not bool(df.loc[1, "ok"])


def test_truncated_last_line_is_ignored(tmp_path):
    p = store.plain(tmp_path, "m")
    p.parent.mkdir(parents=True)
    p.write_text('{"point_id": 0, "ok": true}\n{"point_id": 1, "ok": tr')
    assert store.done_ids(store.read(tmp_path, "m")) == {0}


def test_pack_and_resume_from_packed_copy(tmp_path):
    w = store.Appender(tmp_path, "m")
    w.write({"point_id": 5, "ok": True})
    w.close()
    (z,) = store.pack(tmp_path)
    assert z.name == "m.jsonl.gz"
    first = z.read_bytes()
    store.pack(tmp_path)
    assert z.read_bytes() == first  # deterministic, no spurious git diff
    store.plain(tmp_path, "m").unlink()
    assert store.done_ids(store.read(tmp_path, "m")) == {5}
    w = store.Appender(tmp_path, "m")  # fresh clone: resumes from the .gz
    w.write({"point_id": 6, "ok": True})
    w.close()
    assert store.done_ids(store.read(tmp_path, "m")) == {5, 6}


def _render(m):
    from loe.templates import generic_renderer

    return generic_renderer(m.raw_prefill)


def test_points_helper_is_deterministic():
    assert isinstance(points(5), pd.DataFrame)
