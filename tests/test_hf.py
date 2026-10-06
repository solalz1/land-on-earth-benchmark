"""The Hugging Face dataset card."""

from loe import hf
from loe.config import HF_DATASET, REPO_URL

RESULTS = """# Land on Earth — results

Intro, see [the config](configs/models.yaml) and [maps](results/tweet/maps/).

![Poster](results/tweet/figures/land_on_earth.png)

## Setup

Text with [an anchor](#leaderboard) and [a site](https://example.com).

## Files

| the repo's own list |
"""


def test_card_fixes_links_and_adds_the_dataset_section():
    card = hf.card(RESULTS, "tweet", 32_400)
    assert card.startswith("---\nlicense: cc-by-4.0\n")
    assert "\n# Land on Earth\n\nIntro" in card and "— results" not in card
    assert "](figures/land_on_earth.png)" in card
    assert f"]({REPO_URL}/blob/main/configs/models.yaml)" in card
    assert f"]({REPO_URL}/tree/main/results/tweet/maps)" in card
    assert "](#leaderboard)" in card and "](https://example.com)" in card
    assert "the repo's own list" not in card  # replaced by the dataset's files
    # the guide to the files comes after the summary, before the setup
    assert card.index("![Poster]") < card.index("## Using this dataset") < card.index("## Setup")
    assert f'load_dataset("{HF_DATASET}", "predictions", split="train")' in card
    assert "32,400 rows" in card
