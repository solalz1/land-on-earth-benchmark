"""One JSON line per answered request, appended as answers arrive, so a run can always resume.

results/<run>/raw/<model>.jsonl is the working file (ignored by git); `pack` writes the
compressed copy raw/<model>.jsonl.gz that goes to the repo. A fresh clone with only the .gz
resumes from it.
"""

from __future__ import annotations

import gzip
import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterator

import pandas as pd


def raw_dir(run_dir: Path) -> Path:
    return run_dir / "raw"


def plain(run_dir: Path, key: str) -> Path:
    return raw_dir(run_dir) / f"{key}.jsonl"


def packed(run_dir: Path, key: str) -> Path:
    return raw_dir(run_dir) / f"{key}.jsonl.gz"


def _lines(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # a line cut by a crash: that point is simply asked again
    except (EOFError, gzip.BadGzipFile):
        return


def read(run_dir: Path, key: str) -> list[dict[str, Any]]:
    p, z = plain(run_dir, key), packed(run_dir, key)
    if p.exists():
        return list(_lines(p))
    if z.exists():
        return list(_lines(z))
    return []


def models_in(run_dir: Path) -> list[str]:
    d = raw_dir(run_dir)
    if not d.exists():
        return []
    keys = {p.name.removesuffix(".gz").removesuffix(".jsonl") for p in d.glob("*.jsonl*")}
    return sorted(keys)


def _norm(s: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def is_byok(r: dict[str, Any]) -> bool:
    """Whether your own key at the provider served this answer (OpenRouter's usage.is_byok).

    When OpenRouter leaves the flag out, the upstream cost tells: with your key, the provider
    bills it and OpenRouter charges (almost) nothing.
    """
    if r.get("byok") is not None:
        return bool(r["byok"])
    upstream = float(r.get("upstream_cost") or 0.0)
    return upstream > 0 and upstream > 2 * float(r.get("cost") or 0.0)


def charged(r: dict[str, Any]) -> float:
    """What this request cost you in USD: OpenRouter's charge, plus the provider's bill when your key served it."""
    cost = float(r.get("cost") or 0.0)
    if is_byok(r):
        cost += float(r.get("upstream_cost") or 0.0)
    return cost


def served_by(r: dict[str, Any], provider: str) -> bool:
    """Whether a record was asked through this pinned provider (older records only name the server)."""
    if r.get("pinned"):
        return r["pinned"] == provider
    return bool(r.get("provider")) and _norm(provider) in _norm(r.get("provider"))


def for_job(records: list[dict[str, Any]], strategy: str, provider: str) -> list[dict[str, Any]]:
    """Records of one strategy at one provider: a change of either asks every point again."""
    return [r for r in records if r.get("strategy") == strategy and (not r.get("ok") or served_by(r, provider))]


def done_ids(records: list[dict[str, Any]], strategy: str | None = None) -> set[int]:
    """Points answered, by this strategy if given (a new strategy asks every point again)."""
    return {int(r["point_id"]) for r in records if r.get("ok") and (strategy is None or r.get("strategy") == strategy)}


def current(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Records of the strategy and provider used last, so that a change never mixes two maps."""
    last = next((r for r in reversed(records) if r.get("ok")), None)
    if last is None:
        return records
    if last.get("pinned"):
        return for_job(records, last["strategy"], last["pinned"])
    return [
        r
        for r in records
        if r.get("strategy") == last.get("strategy") and (not r.get("ok") or r.get("provider") == last.get("provider"))
    ]


def latest(records: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per point: the last successful answer, else the last error."""
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["_ok"] = df["ok"].astype(bool)
    df["_i"] = range(len(df))
    df = df.sort_values(["point_id", "_ok", "_i"]).groupby("point_id").tail(1)
    return df.drop(columns=["_ok", "_i"]).sort_values("point_id").reset_index(drop=True)


class Appender:
    def __init__(self, run_dir: Path, key: str):
        p, z = plain(run_dir, key), packed(run_dir, key)
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists() and z.exists():  # resume from a packed copy (fresh clone)
            with gzip.open(z, "rb") as src, open(p, "wb") as dst:
                shutil.copyfileobj(src, dst)
        self.f = open(p, "a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self.f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.f.flush()

    def close(self) -> None:
        self.f.close()


def pack(run_dir: Path) -> list[Path]:
    out = []
    for p in sorted(raw_dir(run_dir).glob("*.jsonl")):
        z = p.with_suffix(".jsonl.gz")
        # mtime=0: the same content always gives the same bytes, so git sees no false change
        with open(p, "rb") as src, open(z, "wb") as f, gzip.GzipFile(fileobj=f, mode="wb", mtime=0) as dst:
            shutil.copyfileobj(src, dst)
        out.append(z)
    return out
