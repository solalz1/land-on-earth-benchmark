"""Scores of the tweet: area-weighted accuracy against the 1 km land mask, plus context.

- accuracy: share of the globe's area answered correctly; an unanswered point counts as wrong.
- skill: gain over always answering Water (~71 %): (acc - base) / (1 - base). 0 = no better
  than the constant answer, 1 = perfect, negative = worse.
- accuracy_lakes: same, with large lakes and the Caspian Sea counted as water.
- land_recall / land_precision: area-weighted, on the Land answers.
- brier: mean squared error of P(Land), for models that return logprobs.
- indirect_share: answers read from the logprobs of a first token that was not Land or Water
  (the model started a sentence, cut by max_tokens): there the prediction is only its preference
  between the two words at that token, a weaker signal. The leaderboard flags it from 1 %.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from loe import grid, store
from loe.config import Config
from loe.parse import answers_first

INDIRECT_FLAG = 0.01  # share of indirect answers from which the leaderboard adds a note


def _stamp(run_dir: Path, key: str) -> float:
    for p in (store.plain(run_dir, key), store.packed(run_dir, key)):
        if p.exists():
            return p.stat().st_mtime
    return 0.0


def predictions(run_dir: Path, key: str) -> pd.DataFrame:
    """All 16,200 points with this model's latest answer (NaN where none)."""
    return _predictions(str(run_dir), key, _stamp(run_dir, key)).copy()


@lru_cache(maxsize=64)
def _predictions(run_dir: str, key: str, stamp: float) -> pd.DataFrame:
    run_dir = Path(run_dir)
    g = grid.load()
    latest = store.latest(store.current(store.read(run_dir, key)))
    cols = ["point_id", "pred", "p_land", "mass", "source", "provider", "strategy", "text"]
    if latest.empty:
        latest = pd.DataFrame(columns=cols)
    ok = latest[latest["ok"].astype(bool)] if "ok" in latest else latest
    ok = ok.reindex(columns=cols)
    df = g[["point_id", "lat", "lon", "row", "col", "weight", "truth", "truth_lakes"]].merge(
        ok, on="point_id", how="left"
    )
    df["pred"] = pd.to_numeric(df["pred"], errors="coerce")
    df["p_land"] = pd.to_numeric(df["p_land"], errors="coerce")
    df["mass"] = pd.to_numeric(df["mass"], errors="coerce")
    return df


def metrics(df: pd.DataFrame) -> dict[str, Any]:
    w = df["weight"].to_numpy()
    W = w.sum()
    truth = df["truth"].to_numpy()
    lakes = df["truth_lakes"].to_numpy()
    pred = df["pred"].to_numpy()
    answered = ~np.isnan(pred)
    p = np.where(answered, pred, -1)
    base = float((w * (truth == 0)).sum() / W)
    acc = float((w * (p == truth)).sum() / W)
    acc_l = float((w * (p == lakes)).sum() / W)
    base_l = float((w * (lakes == 0)).sum() / W)
    land_pred = p == 1
    tp = float((w * (land_pred & (truth == 1))).sum())
    out: dict[str, Any] = {
        "points": int(len(df)),
        "answered": int(answered.sum()),
        "coverage": float(w[answered].sum() / W),
        "accuracy": acc,
        "always_water": base,
        "skill": (acc - base) / (1 - base),
        "accuracy_lakes": acc_l,
        "skill_lakes": (acc_l - base_l) / (1 - base_l),
        "land_recall": tp / float((w * (truth == 1)).sum()),
        "land_precision": tp / float((w * land_pred).sum()) if land_pred.any() else float("nan"),
        "land_share_predicted": float((w * land_pred).sum() / W),
    }
    prob = df["p_land"].to_numpy()
    has_p = ~np.isnan(prob)
    out["logprobs_share"] = float(has_p.sum() / max(answered.sum(), 1))
    if has_p.any():
        out["brier"] = float((w[has_p] * (prob[has_p] - truth[has_p]) ** 2).sum() / w[has_p].sum())
        out["median_mass"] = float(np.nanmedian(df["mass"].to_numpy()))
    indirect = indirect_answers(df)
    out["indirect_share"] = float(indirect.sum() / max(answered.sum(), 1))
    if indirect.any():
        out["indirect_median_mass"] = float(np.nanmedian(df["mass"].to_numpy()[indirect]))
        starts = pd.Series([" ".join(str(t).split()[:2]) for t in df["text"].to_numpy()[indirect]])
        out["indirect_example"] = f"{starts.value_counts().index[0]}…"
    return out


def indirect_answers(df: pd.DataFrame) -> np.ndarray:
    """Answers read from logprobs whose text does not start with Land or Water."""
    if "text" not in df or "source" not in df:
        return np.zeros(len(df), dtype=bool)
    lp = (df["source"] == "logprobs").to_numpy()
    first = np.array([answers_first(t) if isinstance(t, str) else True for t in df["text"]])
    return lp & ~first


def score_run(cfg: Config, run_dir: Path, keys: list[str] | None = None) -> pd.DataFrame:
    keys = keys or store.models_in(run_dir)
    known = {m.key: m for m in cfg.models}
    rows = []
    pred_dir = run_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for key in keys:
        df = predictions(run_dir, key)
        records = store.read(run_dir, key)
        m = metrics(df)
        served = df["provider"].dropna().astype(str).value_counts()
        strategies = df["strategy"].dropna().astype(str).value_counts()
        model = known.get(key)
        strategy = strategies.index[0] if len(strategies) else None
        rows.append(
            {
                "key": key,
                "name": model.name if model else key,
                "lab": model.lab if model else "",
                **m,
                "strategy": strategy,
                "reasoning": bool(strategy in cfg.strategies and cfg.strategies[strategy].reasons),
                "provider_served": " + ".join(served.index) if len(served) else None,
                "cost_usd": float(sum(store.charged(r) for r in records)),
                "requests": len(records),
            }
        )
        out = df[["point_id", "lat", "lon", "truth", "truth_lakes", "pred", "p_land", "mass", "source", "provider"]]
        out.to_csv(pred_dir / f"{key}.csv", index=False, float_format="%.6g")
    board = pd.DataFrame(rows)
    if board.empty:
        return board
    board = board.sort_values("accuracy", ascending=False).reset_index(drop=True)
    board.insert(0, "rank", range(1, len(board) + 1))
    board.to_csv(run_dir / "leaderboard.csv", index=False, float_format="%.6g")
    (run_dir / "leaderboard.md").write_text(leaderboard_md(board))
    (run_dir / "leaderboard.json").write_text(json.dumps(board.to_dict(orient="records"), indent=2, default=str))
    return board


def _pct(x: Any) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f} %"


def leaderboard_md(board: pd.DataFrame) -> str:
    base = board["always_water"].iloc[0]
    lines = [
        "# Land on Earth — reproduction du tweet",
        "",
        f"Précision pondérée par la surface contre le masque terre à 1 km (GLOBE). "
        f"Répondre toujours « Water » donne {_pct(base)}. Skill = gain sur cette réponse constante. "
        "Réflexion « minimale » : le modèle ne peut pas répondre sans réfléchir, il réfléchit au minimum "
        "avant de répondre ; les autres répondent sans réflexion.",
        "",
        "| # | Modèle | Labo | Réflexion | Précision | Skill | Avec lacs | Rappel terre | Couverture | Logprobs "
        "| Stratégie | Fournisseur | Coût |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    notes = []
    for r in board.itertuples():
        reasoning = "minimale" if getattr(r, "reasoning", False) else "aucune"
        name = r.name
        share = getattr(r, "indirect_share", 0.0)
        if isinstance(share, float) and share >= INDIRECT_FLAG:
            notes.append(r)
            name += " " + "¹²³⁴⁵⁶⁷⁸⁹"[min(len(notes), 9) - 1]
        lines.append(
            f"| {r.rank} | {name} | {r.lab} | {reasoning} | {_pct(r.accuracy)} | {r.skill:.3f} | "
            f"{_pct(r.accuracy_lakes)} | {_pct(r.land_recall)} | {_pct(r.coverage)} | {_pct(r.logprobs_share)} | "
            f"{r.strategy or '—'} | {r.provider_served or '—'} | {r.cost_usd:.2f} $ |"
        )
    lines.append("")
    for i, r in enumerate(notes[:9]):
        mass = getattr(r, "indirect_median_mass", float("nan"))
        lines.append(
            f"{'¹²³⁴⁵⁶⁷⁸⁹'[i]} {r.name} : {_pct(r.indirect_share)} des réponses commencent par une phrase "
            f"au lieu de Land ou Water (« {getattr(r, 'indirect_example', '…')} »), coupée par la limite de "
            "tokens. La prédiction y est lue dans les logprobs du premier "
            f"token, où Land et Water ne pèsent que {_pct(mass)} en médiane : un signal plus faible, "
            "que la colonne Couverture ne montre pas."
        )
        lines.append("")
    return "\n".join(lines)
