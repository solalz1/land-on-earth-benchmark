"""Figures and numbers of RESULTS.md, and the leaderboard's note on indirect answers."""

import json

import numpy as np
import pandas as pd

from loe import config, figure, grid, hf
from loe.client import Client
from loe.fake import FakeOpenRouter
from loe.parse import answers_first
from loe.runner import Job, Runner
from loe.score import leaderboard_md, score_run

CFG = config.load()


async def nosleep(_):
    return None


def test_weighted_auc():
    y = np.array([0, 0, 1, 1])
    w = np.ones(4)
    assert figure.weighted_auc(y, np.array([0.1, 0.2, 0.8, 0.9]), w) == 1.0
    assert figure.weighted_auc(y, np.array([0.9, 0.8, 0.2, 0.1]), w) == 0.0
    assert figure.weighted_auc(y, np.full(4, 0.5), w) == 0.5  # ties count half
    # one land point ranked below a water point: only that pair is wrong, in proportion to its area
    s = np.array([0.1, 0.6, 0.5, 0.9])
    assert figure.weighted_auc(y, s, w) == 0.75
    assert figure.weighted_auc(y, s, np.array([1, 3, 1, 1])) == 0.625


def test_answers_first():
    assert answers_first("Land") and answers_first("**Water**") and answers_first(" water.")
    assert answers_first("<think>hmm</think>\nLand")
    assert not answers_first("To determine whether the given")
    assert not answers_first("The answer is Land") and not answers_first("") and not answers_first(None)
    assert not answers_first("Landmass")  # a word that only starts like the answer


def test_leaderboard_notes_indirect_answers():
    row = dict(rank=1, key="a", name="Model A", lab="Lab", reasoning=False, accuracy=0.8, skill=0.3,
               accuracy_lakes=0.8, land_recall=0.6, coverage=1.0, logprobs_share=1.0, strategy="chat",
               provider_served="X", cost_usd=0.2, always_water=0.71)
    board = pd.DataFrame([
        row | {"indirect_share": 0.35, "indirect_median_mass": 0.156, "indirect_example": "To determine…"},
        row | {"rank": 2, "key": "b", "name": "Model B", "indirect_share": 0.0001},
    ])
    md = leaderboard_md(board)
    assert "| Model A ¹ |" in md and "| Model B |" in md
    assert "¹ Model A : 35.0 % des réponses commencent par une phrase" in md and "« To determine… »" in md
    assert "15.6 %" in md and md.count("¹") == 2


async def test_figures_and_dataset_from_a_fake_run(tmp_path):
    run = tmp_path / "tweet"
    models = [CFG.model("qwen3.5-9b"), CFG.model("ministral-3-14b")]
    jobs = [Job(models[0], CFG.strategies["chat_none"]), Job(models[1], CFG.strategies["text"])]
    fake = FakeOpenRouter(CFG, fail_rate=0)
    async with Client("k", transport=fake.transport(), sleep=nosleep, rate_limits=False) as c:
        await Runner(c, run, budget=5, progress=False).run(jobs, grid.load())
    board = score_run(CFG, run)
    paths = figure.make(run)
    names = {p.name for p in paths}
    assert names == {"stats.json", "land_on_earth.png", "ranking.png", "difficulty.png", "error_sources.png",
                     "knowledge_vs_bias.png"}
    assert all(p.stat().st_size > 5_000 for p in paths if p.suffix == ".png")
    stats = json.loads((run / "figures" / "stats.json").read_text())
    assert stats["models"] == 2 and 70.5 < stats["always_water"] < 71.5
    assert abs(stats["always_water"] + stats["always_land"] - 100) < 1e-6
    assert 19 < stats["coast_area"] < 22 and 2 < stats["ice_area"] < 4
    by_key = {m["key"]: m for m in stats["models_detail"]}
    for key, m in by_key.items():
        acc = 100 * float(board.set_index("key").loc[key, "accuracy"])
        assert abs(m["accuracy"] - acc) < 1e-3
        assert len(m["subgrids"]) == 4 and m["subgrid_min"] <= m["accuracy"] <= m["subgrid_max"]
        assert abs(sum(m["lost"].values()) - (100 - m["accuracy"])) < 1e-3  # the three sources add up
        assert m["indirect_share"] == 0
    assert "auc" in by_key["qwen3.5-9b"] and 0.5 < by_key["qwen3.5-9b"]["auc"] <= 1
    assert "auc" not in by_key["ministral-3-14b"]  # text answers only: no P(Land)
    # the fake model knows the map: forcing the true land share cannot move it far
    m = by_key["qwen3.5-9b"]
    assert abs(m["accuracy_at_true_land_share"] - m["accuracy"]) < 15
    assert set(stats["cost_by_payer_usd"]) <= {"openrouter", "mistral", "alibaba", "siliconflow"}

    # the Hugging Face dataset, built from the same run
    files = hf.build(run, tmp_path / "hf", results_md=config.ROOT / "RESULTS.md")
    names = {str(p.relative_to(tmp_path / "hf")) for p in files}
    assert {"README.md", "data/predictions.csv", "data/leaderboard.csv", "data/grid.csv", "figures/stats.json",
            "figures/ranking.png", "raw/qwen3.5-9b.jsonl.gz", "raw/ministral-3-14b.jsonl.gz"} <= names
    pred = pd.read_csv(tmp_path / "hf" / "data" / "predictions.csv")
    assert len(pred) == 2 * 16_200 and list(pred.columns) == hf.PREDICTION_COLUMNS
    assert set(pred["model"]) == {"qwen3.5-9b", "ministral-3-14b"} and pred["pred"].isin([0, 1]).all()
    text = (tmp_path / "hf" / "README.md").read_text()
    assert text.startswith("---\nlicense: cc-by-4.0\n") and "data_files: data/predictions.csv" in text
    assert "](figures/ranking.png)" in text and "](results/" not in text
    assert "32,400 rows" in text
