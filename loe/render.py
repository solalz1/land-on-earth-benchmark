"""Maps in the tweet's format: white = Land, black = Water, one pixel per 2-degree point."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from loe import grid  # noqa: E402
from loe.score import predictions  # noqa: E402

EXTENT = (-180, 180, -90, 90)
MISSING = 0.5  # grey where a model gave no usable answer
ERR_COLORS = ListedColormap(["#000000", "#ffffff", "#e4572e", "#4ea8de", "#808080"])


def binary_image(df: pd.DataFrame, column: str = "pred") -> np.ndarray:
    v = df[column].to_numpy(dtype=float)
    img = grid.to_image(np.where(np.isnan(v), MISSING, v), df["point_id"].to_numpy())
    return img


def prob_image(df: pd.DataFrame) -> np.ndarray:
    p = df["p_land"].to_numpy(dtype=float)
    p = np.where(np.isnan(p), df["pred"].to_numpy(dtype=float), p)
    return grid.to_image(np.where(np.isnan(p), MISSING, p), df["point_id"].to_numpy())


def error_image(df: pd.DataFrame) -> np.ndarray:
    """0 water ok, 1 land ok, 2 land answered on water, 3 water answered on land, 4 no answer."""
    t = df["truth"].to_numpy()
    p = df["pred"].to_numpy(dtype=float)
    code = np.where(np.isnan(p), 4, np.where(p == t, t, np.where(p == 1, 2, 3)))
    return grid.to_image(code, df["point_id"].to_numpy())


def _save(img: np.ndarray, path: Path, cmap="gray", vmin=0.0, vmax=1.0, scale: int = 8) -> None:
    """One map, no frame: 180 x 90 points upscaled `scale` times with sharp pixels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    big = np.kron(np.nan_to_num(img, nan=MISSING), np.ones((scale, scale)))
    plt.imsave(path, big, cmap=cmap, vmin=vmin, vmax=vmax)


def render_model(run_dir: Path, key: str) -> pd.DataFrame:
    df = predictions(run_dir, key)
    _save(binary_image(df), run_dir / "maps" / f"{key}.png")
    _save(prob_image(df), run_dir / "maps" / "prob" / f"{key}.png")
    _save(error_image(df), run_dir / "maps" / "errors" / f"{key}.png", cmap=ERR_COLORS, vmin=0, vmax=4)
    return df


def render_truth(run_dir: Path) -> None:
    g = grid.load()
    _save(grid.to_image(g["truth"].to_numpy(), g["point_id"].to_numpy()), run_dir / "maps" / "truth.png")
    _save(grid.to_image(g["truth_lakes"].to_numpy(), g["point_id"].to_numpy()), run_dir / "maps" / "truth_lakes.png")


def montage(run_dir: Path, board: pd.DataFrame, path: Path | None = None, cols: int = 4) -> Path:
    """The tweet's figure: every model's map with its accuracy, best first, truth on top."""
    g = grid.load()
    panels = [("Vérité terrain (masque 1 km)", grid.to_image(g["truth"].to_numpy(), g["point_id"].to_numpy()), None)]
    for r in board.itertuples():
        df = predictions(run_dir, r.key)
        panels.append((r.name, binary_image(df), r.accuracy))
    rows = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 2.45), facecolor="white")
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, (name, img, acc) in zip(np.atleast_1d(axes).ravel(), panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, extent=EXTENT, interpolation="nearest")
        title = name if acc is None else f"{name} — {100 * acc:.1f} %"
        ax.set_title(title, fontsize=10, pad=3)
    fig.suptitle("Land or Water? — 16 200 points, un tous les 2°", fontsize=13, y=0.995)
    fig.tight_layout()
    path = path or run_dir / "maps" / "montage.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def render_run(run_dir: Path, board: pd.DataFrame) -> list[Path]:
    render_truth(run_dir)
    for key in board["key"]:
        render_model(run_dir, key)
    return [montage(run_dir, board)]
