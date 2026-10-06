"""The Hugging Face dataset: `make hf` builds hf/, which `hf upload` sends as it is.

hf/ holds the dataset card (README.md: metadata, then RESULTS.md with its links fixed for the
Hub), every model's answer at every point in one table, the leaderboard, the grid with its truth,
the figures, and every request and answer as recorded. It is rebuilt from scratch each time from
results/<run>/, after `make score` and `make figure`; it is not committed (see .gitignore).
"""

from __future__ import annotations

import gzip
import json
import re
import shutil
from pathlib import Path

import pandas as pd

from loe import grid, store
from loe.config import GRID, HF_DATASET, REPO_URL, ROOT

LICENSE = "cc-by-4.0"
PREDICTION_COLUMNS = ["model", "point_id", "lat", "lon", "truth", "truth_lakes", "pred", "p_land", "mass", "source",
                      "provider"]

FRONT_MATTER = f"""---
license: {LICENSE}
pretty_name: Land on Earth
language:
- en
tags:
- benchmark
- llm-evaluation
- geography
- world-model
- open-weight
size_categories:
- 100K<n<1M
configs:
- config_name: predictions
  data_files: data/predictions.csv
  default: true
- config_name: leaderboard
  data_files: data/leaderboard.csv
- config_name: grid
  data_files: data/grid.csv
---
"""

DATASET_SECTION = """## Using this dataset

The code, the configuration of every model and the way to run it all again are on GitHub:
[{repo_name}]({repo}).

```python
from datasets import load_dataset

predictions = load_dataset("{dataset}", "predictions", split="train")  # one row per model and point
leaderboard = load_dataset("{dataset}", "leaderboard", split="train")
grid = load_dataset("{dataset}", "grid", split="train")  # the 16,200 points and their truth
```

| Path | Contents |
|---|---|
| `data/predictions.csv` | {rows:,} rows: each model's answer at each point (columns below) |
| `data/leaderboard.csv` | one row per model, every metric of the leaderboard |
| `data/grid.csv` | the 16,200 points: latitude, longitude, area weight, `truth` (1 = land), `truth_lakes` (large lakes as water) |
| `figures/` | the figures on this page, and `stats.json` with every number it quotes |
| `raw/<model>.jsonl.gz` | every request as recorded: answer text, P(Land), provider, tokens, cost, errors and retries |

Columns of `data/predictions.csv`: `model` (key in the leaderboard), `point_id`, `lat`, `lon`, `truth` and
`truth_lakes` (1 = land), `pred` (1 = the model answered Land, 0 = Water), `p_land` (P(Land) from the logprobs,
empty when the answer was read from the text), `mass` (probability of Land + Water at the answer token),
`source` (`logprobs` or `text`) and `provider` (who served the answer).

"""


def _github(path: str) -> str:
    kind = "tree" if path.endswith("/") else "blob"
    return f"{REPO_URL}/{kind}/main/{path.rstrip('/')}"


def card(results_md: str, run: str, rows: int) -> str:
    """The dataset card: metadata, RESULTS.md with links that work on the Hub, and a guide to the files
    after its summary (in place of the repo's own file list)."""
    body = results_md.split("\n## Files", 1)[0].rstrip() + "\n"
    body = re.sub(r"\A# [^\n]*\n+", "# Land on Earth\n\n", body)
    figures = f"results/{run}/figures/"

    def link(m: re.Match) -> str:
        target = m.group(2)
        if re.match(r"[a-z]+://|#|mailto:", target):
            return m.group(0)
        if target.startswith(figures):
            return f"{m.group(1)}(figures/{target[len(figures):]})"
        return f"{m.group(1)}({_github(target)})"

    body = re.sub(r"(\]|!\[[^\]]*\])\(([^)\s]+)\)", link, body)
    section = DATASET_SECTION.format(repo=REPO_URL, repo_name=REPO_URL.removeprefix("https://github.com/"),
                                     dataset=HF_DATASET, rows=rows)
    head, sep, rest = body.partition("\n## Setup")
    body = head.rstrip() + "\n\n" + section + sep.lstrip("\n") + rest if sep else body + "\n" + section
    return FRONT_MATTER + "\n" + body


def predictions_table(run_dir: Path, keys: list[str]) -> pd.DataFrame:
    parts = []
    for key in keys:
        df = pd.read_csv(run_dir / "predictions" / f"{key}.csv")
        df.insert(0, "model", key)
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    for col in ("truth", "truth_lakes", "pred"):
        out[col] = out[col].astype("Int8")
    return out.reindex(columns=PREDICTION_COLUMNS)


def build(run_dir: Path, out: Path, results_md: Path | None = None) -> list[Path]:
    """Writes the whole dataset into `out`, emptied first."""
    board_path = run_dir / "leaderboard.csv"
    figures = run_dir / "figures"
    results_md = results_md or ROOT / "RESULTS.md"
    missing = [p for p in (board_path, figures / "stats.json", results_md) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "manquant : " + ", ".join(str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p) for p in missing)
            + " (lance d'abord `make score figure`)"
        )
    board = pd.read_csv(board_path)
    keys = list(board["key"])
    pred = predictions_table(run_dir, keys)
    if len(pred) != len(keys) * len(grid.load()):
        raise ValueError(f"{len(pred)} prédictions pour {len(keys)} modèles : relance `make score`")
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    pred.to_csv(out / "data" / "predictions.csv", index=False, float_format="%.6g")
    board.to_csv(out / "data" / "leaderboard.csv", index=False, float_format="%.6g")
    shutil.copy(GRID, out / "data" / "grid.csv")
    shutil.copytree(figures, out / "figures")
    (out / "raw").mkdir()
    for key in keys:  # the working .jsonl, when there is one, is the latest version
        plain, packed = store.plain(run_dir, key), store.packed(run_dir, key)
        dst = out / "raw" / packed.name
        if plain.exists():
            with open(plain, "rb") as src, open(dst, "wb") as f, gzip.GzipFile(fileobj=f, mode="wb", mtime=0) as z:
                shutil.copyfileobj(src, z)
        elif packed.exists():
            shutil.copy(packed, dst)
    (out / "README.md").write_text(card(results_md.read_text(), run_dir.name, len(pred)))
    json.loads((out / "figures" / "stats.json").read_text())  # valid, or the copy failed
    return sorted(p for p in out.rglob("*") if p.is_file())
