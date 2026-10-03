"""The tweet's 2-degree grid (data/grid.csv) and the subsets used by the pilot."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from loe.config import GRID

N_ROWS, N_COLS = 90, 180  # 2-degree cells, north row first, from longitude -180


@lru_cache(maxsize=1)
def load() -> pd.DataFrame:
    df = pd.read_csv(GRID, keep_default_na=False)
    assert len(df) == N_ROWS * N_COLS, f"{GRID}: {len(df)} points au lieu de 16 200"
    return df


def pilot_points(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Half land, half water, drawn once with a fixed seed so every model sees the same points."""
    df = load()
    land = df[df["truth"] == 1].sample(n // 2, random_state=seed)
    water = df[df["truth"] == 0].sample(n - n // 2, random_state=seed)
    # interleave so that the first few points (the strategy probe) mix both answers
    rows = [r for pair in zip(land.itertuples(), water.itertuples()) for r in pair]
    return pd.DataFrame(rows).drop(columns="Index").reset_index(drop=True)


def to_image(values: pd.Series | np.ndarray, point_ids: np.ndarray | None = None) -> np.ndarray:
    """Values indexed by point_id -> 90 x 180 array (row 0 = latitude 89, col 0 = longitude -179)."""
    img = np.full(N_ROWS * N_COLS, np.nan)
    if point_ids is None:
        img[: len(values)] = np.asarray(values, dtype=float)
    else:
        img[np.asarray(point_ids, dtype=int)] = np.asarray(values, dtype=float)
    return img.reshape(N_ROWS, N_COLS)


def coastal(truth: np.ndarray) -> np.ndarray:
    """True where one of the 8 neighbouring cells (longitude wraps) has the other answer."""
    t = truth.reshape(N_ROWS, N_COLS)
    out = np.zeros_like(t, dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == dc == 0:
                continue
            shifted = np.roll(t, dc, axis=1)
            if dr:
                shifted = np.roll(shifted, dr, axis=0)
                edge = 0 if dr == 1 else -1  # no wrap across the poles
                shifted[edge, :] = t[edge, :]
            out |= shifted != t
    return out.ravel()
