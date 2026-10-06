"""Figures and numbers of RESULTS.md: ranking, maps, where the errors are, knowledge against bias.

`make figure` reads the leaderboard written by `make score` and every model's predictions, and
writes results/<run>/figures/: land_on_earth.png (the whole story on one page), ranking.png,
difficulty.png, error_sources.png, knowledge_vs_bias.png, and stats.json with every number that
RESULTS.md quotes.

Diagnostics, all weighted by area like the score:
- sub-grids: the 2-degree grid split into four 4-degree grids (every other row and column). A
  model's spread across them shows how much its accuracy depends on where the grid falls; two
  models whose spreads overlap are tied.
- coasts: points with one of their 8 neighbours on the other side (grid.coastal).
- ice sheets: land south of 60°S (Antarctica) and inside a rough outline of Greenland, which the
  1 km mask counts as land.
- AUC, for models with P(Land) at every point: the chance that a land point gets a higher P(Land)
  than a water point (1 = perfect, 0.5 = chance). It measures what the model knows, apart from
  how readily it says Land.
- accuracy at the true land share: the accuracy if the model answered Land on the 29 % of the area
  where its P(Land) is highest. It uses the true share, so it is a diagnostic, not a fair score.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.path import Path as Outline  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from loe import grid, store  # noqa: E402
from loe.config import REPO_URL  # noqa: E402
from loe.score import INDIRECT_FLAG, indirect_answers, predictions  # noqa: E402

AUTHOR = "Solal Zana"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
RULE = "#e4e3df"
BLUE = "#2a78d6"  # answers without reasoning; on maps, Water answered on land
ORANGE = "#eb6834"  # minimal reasoning; on maps, Land answered on water
SOURCES = {"ice_sheets": ("#6fb7d9", "ice sheets"), "coasts": ("#e0a458", "coasts"), "elsewhere": ("#b9b8b3", "elsewhere")}
EXTENT = (-180, 180, -90, 90)
ERRORS = ListedColormap(["#000000", "#ffffff", ORANGE, BLUE, "#9a9994"])
WRONG = LinearSegmentedColormap.from_list("wrong", ["#fbf8f3", "#f3c08a", "#d9572b", "#7a1b0c"])
LOGPROBS_FULL = 0.99  # P(Land) on at least this share of the answers for the AUC

# Greenland, roughly (longitude, latitude): enough to sort 2-degree points, not a coastline
GREENLAND = [
    (-73.0, 78.2), (-68.0, 80.5), (-62.0, 81.8), (-52.0, 82.4), (-40.0, 83.6), (-30.0, 83.6),
    (-20.0, 82.6), (-11.5, 81.5), (-17.0, 79.5), (-18.5, 76.5), (-19.5, 74.0), (-21.5, 71.0),
    (-24.0, 69.5), (-30.0, 68.2), (-35.0, 66.0), (-39.0, 64.5), (-41.5, 62.0), (-43.5, 59.8),
    (-46.5, 60.5), (-49.5, 62.5), (-51.5, 64.5), (-53.5, 66.5), (-54.0, 69.0), (-55.5, 71.5),
    (-57.0, 74.0), (-60.0, 75.8), (-66.0, 76.2), (-72.0, 77.0),
]


# Regions where the models above the always-Water line go wrong most, for RESULTS.md:
# (south, north, west, east) in degrees, and which points count (1 land, 0 water, None both)
HOTSPOTS = {
    "amazon_land": ((-6, 4, -74, -50), 1),
    "equatorial_africa_land": ((-6, 4, 10, 40), 1),
    "land_within_5_of_equator": ((-5, 5, -180, 180), 1),
    "land_11_to_25_from_equator": (((-25, -11), (11, 25)), 1),
    "west_african_coast_land": ((10, 30, -18, -10), 1),
    "arctic_ocean_north_of_eurasia": ((77, 90, 0, 110), 0),
    "canadian_arctic_archipelago": ((68, 82, -125, -62), None),
    "siberia_land": ((60, 75, 60, 140), 1),
}


# ---- numbers -------------------------------------------------------------------------------------


def load_board(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "leaderboard.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} manquant : lance d'abord `make score`")
    return pd.DataFrame(json.loads(path.read_text()))


def regions(g: pd.DataFrame) -> dict[str, np.ndarray]:
    truth = g["truth"].to_numpy()
    lon, lat = g["lon"].to_numpy(), g["lat"].to_numpy()
    greenland = Outline(GREENLAND).contains_points(np.c_[lon, lat]) & (truth == 1)
    antarctica = (lat < -60) & (truth == 1)
    return {
        "coast": grid.coastal(truth),
        "antarctica": antarctica,
        "greenland": greenland,
        "ice": antarctica | greenland,
        "lat_eq_lon": lat == lon,
        "lat_eq_minus_lon": lat == -lon,
        # the diagonals' neighbours, 2 or 4 degrees away: same regions, without the repeated number
        "near_lat_eq_lon": (np.abs(lat - lon) > 0) & (np.abs(lat - lon) <= 4),
        "near_lat_eq_minus_lon": (np.abs(lat + lon) > 0) & (np.abs(lat + lon) <= 4),
    }


def hotspot(g: pd.DataFrame, box, which) -> np.ndarray:
    lat, lon, truth = g["lat"].to_numpy(), g["lon"].to_numpy(), g["truth"].to_numpy()
    if isinstance(box[0], tuple):  # two latitude bands, all longitudes
        m = np.zeros(len(g), dtype=bool)
        for lo, hi in box:
            m |= (lat >= lo) & (lat <= hi)
    else:
        south, north, west, east = box
        m = (lat >= south) & (lat <= north) & (lon >= west) & (lon <= east)
    return m if which is None else m & (truth == which)


def weighted_auc(y: np.ndarray, s: np.ndarray, w: np.ndarray) -> float:
    """P(score of a land point > score of a water point), each point counting for its area; ties count half."""
    _, inv = np.unique(s, return_inverse=True)
    land = np.bincount(inv, w * (y == 1))
    water = np.bincount(inv, w * (y == 0))
    below = np.cumsum(water) - water  # water weight with a strictly lower score
    return float((land * (below + water / 2)).sum() / (land.sum() * water.sum()))


def _share(w: np.ndarray, hit: np.ndarray, mask: np.ndarray | None = None) -> float:
    m = np.ones(len(w), dtype=bool) if mask is None else mask
    return float(100 * (w[m] * hit[m]).sum() / w[m].sum()) if m.any() and w[m].sum() > 0 else float("nan")


def model_stats(df: pd.DataFrame, reg: dict[str, np.ndarray]) -> dict[str, Any]:
    w = df["weight"].to_numpy()
    truth = df["truth"].to_numpy()
    pred = df["pred"].to_numpy(dtype=float)
    ok = pred == truth  # no answer counts wrong, as in the score
    W = w.sum()
    rows, cols = df["row"].to_numpy(), df["col"].to_numpy()
    sub = [_share(w, ok, (rows % 2 == a) & (cols % 2 == b)) for a in (0, 1) for b in (0, 1)]
    lost = {
        "ice_sheets": float(100 * (w * ~ok * reg["ice"]).sum() / W),
        "coasts": float(100 * (w * ~ok * (reg["coast"] & ~reg["ice"])).sum() / W),
        "elsewhere": float(100 * (w * ~ok * ~(reg["coast"] | reg["ice"])).sum() / W),
    }
    said_land = pred == 1
    out: dict[str, Any] = {
        "accuracy": _share(w, ok),
        "subgrids": sub,
        "subgrid_min": min(sub),
        "subgrid_max": max(sub),
        "accuracy_inland_and_open_sea": _share(w, ok, ~reg["coast"]),
        "accuracy_coasts": _share(w, ok, reg["coast"]),
        "errors_on_coasts": float(100 * (w * ~ok * reg["coast"]).sum() / (w * ~ok).sum()) if (~ok).any() else 0.0,
        "lost": lost,
        "said_land": _share(w, said_land),
        "antarctica_said_land": _share(w, said_land, reg["antarctica"]),
        "greenland_said_land": _share(w, said_land, reg["greenland"]),
        "lat_eq_lon_said_land": _share(w, said_land, reg["lat_eq_lon"]),
        "near_lat_eq_lon_said_land": _share(w, said_land, reg["near_lat_eq_lon"]),
        "lat_eq_minus_lon_said_land": _share(w, said_land, reg["lat_eq_minus_lon"]),
        "near_lat_eq_minus_lon_said_land": _share(w, said_land, reg["near_lat_eq_minus_lon"]),
    }
    indirect = indirect_answers(df)
    out["indirect_share"] = float(100 * indirect.mean())
    if indirect.any():
        out["accuracy_indirect"] = _share(w, ok, indirect)
        out["accuracy_direct"] = _share(w, ok, ~indirect)
        out["indirect_median_mass"] = float(100 * np.nanmedian(df["mass"].to_numpy()[indirect]))
        direct = ~indirect & (df["source"] == "logprobs").to_numpy()
        if direct.any():
            out["direct_median_mass"] = float(100 * np.nanmedian(df["mass"].to_numpy()[direct]))
    p = df["p_land"].to_numpy(dtype=float)
    has = ~np.isnan(p)
    out["logprobs_share"] = float(100 * has.mean())
    if has.mean() >= LOGPROBS_FULL:
        out["auc"] = weighted_auc(truth[has], p[has], w[has])
        land_area = (w * (truth == 1)).sum() / W
        order = np.argsort(-np.where(has, p, -1.0), kind="stable")
        forced = np.zeros(len(df))
        forced[order[np.cumsum(w[order]) / W <= land_area]] = 1
        out["accuracy_at_true_land_share"] = _share(w, forced == truth)
    return out


def pilot_check(run_dir: Path, key: str, df: pd.DataFrame) -> dict[str, Any] | None:
    """The pilot's estimate against the full run scored on the same points with the same estimator."""
    from loe.pilot import area_estimate

    path = run_dir.parent / "pilot" / "report.json"
    if not path.exists():
        return None
    rows = {r["key"]: r for r in json.loads(path.read_text()).get("models", [])}
    r = rows.get(key)
    if not r or r.get("accuracy_area") is None:
        return None
    pts = grid.pilot_points(int(r.get("points") or 200))
    sub = df.set_index("point_id").loc[pts["point_id"]].reset_index()
    sub = sub[sub["pred"].notna()]
    est, _ = area_estimate(sub[["point_id", "lat", "pred"]]) if len(sub) >= 20 else (float("nan"), None)
    return {
        "pilot_estimate": 100 * float(r["accuracy_area"]),
        "pilot_halfwidth": 100 * float(r.get("accuracy_area_halfwidth") or float("nan")),
        "full_run_on_pilot_points": 100 * float(est),
    }


def cost_by_payer(run_dir: Path, keys: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in keys:
        for r in store.read(run_dir, key):
            payer = (r.get("pinned") or "provider") if store.is_byok(r) else "openrouter"
            out[payer] = out.get(payer, 0.0) + store.charged(r)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def finished(run_dir: Path) -> str | None:
    try:
        sessions = json.loads((run_dir / "run.json").read_text())["sessions"]
        return max(s["finished"] for s in sessions if s.get("finished"))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def diagnostics(run_dir: Path, board: pd.DataFrame, preds: dict[str, pd.DataFrame]) -> dict[str, Any]:
    g = grid.load()
    reg = regions(g)
    w = g["weight"].to_numpy()
    W = w.sum()
    base = 100 * float((w * (g["truth"] == 0)).sum() / W)
    models = []
    for r in board.itertuples():
        s = {
            "key": r.key,
            "name": r.name,
            "lab": getattr(r, "lab", ""),
            "reasoning": bool(getattr(r, "reasoning", False)),
            "strategy": getattr(r, "strategy", None),
            "provider": getattr(r, "provider_served", None),
            "cost_usd": float(getattr(r, "cost_usd", 0.0) or 0.0),
            **model_stats(preds[r.key], reg),
        }
        check = pilot_check(run_dir, r.key, preds[r.key])
        if check:
            s["pilot"] = check
        models.append(s)
    models.sort(key=lambda m: -m["accuracy"])
    above = [m for m in models if m["accuracy"] > base]
    wrong = np.mean([(preds[m["key"]]["pred"] != preds[m["key"]]["truth"]).to_numpy() for m in above], axis=0) if above else None
    out: dict[str, Any] = {
        "run": run_dir.name,
        "finished": finished(run_dir),
        "points": int(len(g)),
        "models": len(models),
        "always_water": base,
        "always_land": 100 - base,
        "coast_area": float(100 * w[reg["coast"]].sum() / W),
        "ice_area": float(100 * w[reg["ice"]].sum() / W),
        "antarctica_area": float(100 * w[reg["antarctica"]].sum() / W),
        "greenland_area": float(100 * w[reg["greenland"]].sum() / W),
        "lat_eq_lon_true_land": _share(w, g["truth"].to_numpy() == 1, reg["lat_eq_lon"]),
        "lat_eq_minus_lon_true_land": _share(w, g["truth"].to_numpy() == 1, reg["lat_eq_minus_lon"]),
        "near_lat_eq_lon_true_land": _share(w, g["truth"].to_numpy() == 1, reg["near_lat_eq_lon"]),
        "near_lat_eq_minus_lon_true_land": _share(w, g["truth"].to_numpy() == 1, reg["near_lat_eq_minus_lon"]),
        "diagonal_points": int((reg["lat_eq_lon"] | reg["lat_eq_minus_lon"]).sum()),
        "above_always_water": len(above),
        "total_cost_usd": float(sum(m["cost_usd"] for m in models)),
        "cost_by_payer_usd": cost_by_payer(run_dir, [m["key"] for m in models]),
        "models_detail": models,
    }
    if wrong is not None:
        out["area_all_above_wrong"] = float(100 * (w * (wrong == 1)).sum() / W)
        out["area_most_above_wrong"] = float(100 * (w * (wrong > 0.5)).sum() / W)
        allw = wrong == 1
        out["all_above_wrong_on_ice"] = float(100 * (w * allw * reg["ice"]).sum() / max((w * allw).sum(), 1e-12))
        out["all_above_wrong_on_coasts"] = float(
            100 * (w * (allw & reg["coast"] & ~reg["ice"])).sum() / max((w * allw).sum(), 1e-12)
        )
        most = wrong > 0.5
        out["most_above_wrong_on_ice"] = float(100 * (w * (most & reg["ice"])).sum() / max((w * most).sum(), 1e-12))
        out["most_above_wrong_on_coasts"] = float(
            100 * (w * (most & reg["coast"] & ~reg["ice"])).sum() / max((w * most).sum(), 1e-12)
        )
        said = np.mean([(preds[m["key"]]["pred"] == 1).to_numpy() for m in above], axis=0)
        out["hotspots_above"] = {}
        for name, (box, which) in HOTSPOTS.items():
            m = hotspot(g, box, which)
            if m.any():
                out["hotspots_above"][name] = {
                    "area": float(100 * w[m].sum() / W),
                    "wrong": _share(w, wrong, m),
                    "said_land": _share(w, said, m),
                }
    return out


# ---- drawing ---------------------------------------------------------------------------------------


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.color": INK,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "axes.facecolor": SURFACE,
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
        }
    )


def _bare(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _pct(v: float, _=None) -> str:
    return f"{v:.0f} %"


def _flagged(m: dict[str, Any]) -> bool:
    return m.get("indirect_share", 0.0) >= 100 * INDIRECT_FLAG


def _image(df: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    return grid.to_image(values, df["point_id"].to_numpy())


def error_codes(df: pd.DataFrame) -> np.ndarray:
    """0 water right, 1 land right, 2 Land answered on water, 3 Water answered on land, 4 no answer."""
    t = df["truth"].to_numpy()
    p = df["pred"].to_numpy(dtype=float)
    return np.where(np.isnan(p), 4, np.where(p == t, t, np.where(p == 1, 2, 3)))


def draw_ranking(ax, stats: dict[str, Any], cost: bool = True, size: float = 10) -> None:
    models = sorted(stats["models_detail"], key=lambda m: m["accuracy"])
    y = np.arange(len(models))
    base, land = stats["always_water"], stats["always_land"]
    lo = min(land, min(m["subgrid_min"] for m in models)) - 4
    for i, m in enumerate(models):
        color, marker = (ORANGE, "D") if m["reasoning"] else (BLUE, "o")
        ax.hlines(i, m["subgrid_min"], m["subgrid_max"], color=color, alpha=0.28, lw=7, capstyle="round", zorder=2)
        ax.scatter(m["accuracy"], i, s=46 if marker == "o" else 40, color=color, marker=marker, zorder=3,
                   edgecolors=SURFACE, linewidths=1.2)
        ax.text(m["subgrid_max"] + 0.9, i, f"{m['accuracy']:.1f} %", va="center", fontsize=size - 1, color=INK2)
        if cost:
            ax.text(1.015, i, f"${m['cost_usd']:.2f}", transform=ax.get_yaxis_transform(), va="center",
                    fontsize=8.5, color=MUTED)
    names = [m["name"] + (" ¹" if _flagged(m) else "") for m in models]
    ax.set_yticks(y, names, fontsize=size)
    for x, label in ((base, f"always Water\n{base:.0f} %"), (land, f"always Land\n{land:.0f} %")):
        ax.axvline(x, color=MUTED, lw=1.1, ls=(0, (4, 3)), zorder=1)
        ax.text(x + 0.6, -1.0, label, fontsize=size - 1.5, color=MUTED, va="top", ha="left", linespacing=1.2)
    if cost:
        ax.text(1.015, len(models) - 0.2, "cost", transform=ax.get_yaxis_transform(), fontsize=8.5,
                color=MUTED, va="bottom")
    ax.set_xlim(lo, 96)
    ax.set_ylim(-2.6, len(models) - 0.3)
    ax.xaxis.set_major_formatter(FuncFormatter(_pct))
    ax.tick_params(axis="x", labelsize=size - 1, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=RULE, lw=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(
        handles=[
            Line2D([], [], ls="", marker="o", color=BLUE, ms=7, label="answers directly"),
            Line2D([], [], ls="", marker="D", color=ORANGE, ms=6, label="minimal reasoning"),
            Line2D([], [], color=MUTED, alpha=0.45, lw=7, solid_capstyle="round", label="spread over four 4° sub-grids"),
        ],
        loc="lower left", bbox_to_anchor=(-0.01, 1.0), ncols=3, frameon=False, fontsize=size - 1,
        handlelength=1.8, handletextpad=0.6, columnspacing=1.4, borderaxespad=0.2,
    )


def draw_maps(fig, region, stats: dict[str, Any], preds: dict[str, pd.DataFrame], cols: int = 5) -> None:
    g = grid.load()
    models = stats["models_detail"]
    rows = int(np.ceil((len(models) + 1) / cols))
    sub = region.subgridspec(rows, cols, hspace=0.42, wspace=0.05)
    panels = [("1 km land mask (truth)", "white = land, black = water", grid.to_image(g["truth"].to_numpy(), g["point_id"].to_numpy()), "gray", 1)]
    for i, m in enumerate(models):
        df = preds[m["key"]]
        tag = " · minimal reasoning" if m["reasoning"] else ""
        panels.append((f"{i + 1}. {m['name']}" + (" ¹" if _flagged(m) else ""), f"{m['accuracy']:.1f} %{tag}",
                       _image(df, error_codes(df)), ERRORS, 4))
    for i, (title, line, img, cmap, vmax) in enumerate(panels):
        ax = fig.add_subplot(sub[i // cols, i % cols])
        ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax, extent=EXTENT, interpolation="nearest")
        _bare(ax)
        ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold", pad=13)
        ax.text(0, 1.012, line, transform=ax.transAxes, fontsize=8.5, color=INK2, va="bottom")


def maps_legend(fig, x: float, y: float) -> None:
    items = [("#ffffff", "land, right"), ("#000000", "water, right"), (ORANGE, "Land answered on water"),
             (BLUE, "Water answered on land")]
    fig.legend(handles=[Patch(facecolor=c, edgecolor="#9a9994", lw=0.6, label=t) for c, t in items],
               loc="upper left", bbox_to_anchor=(x, y), ncols=4, frameon=False, fontsize=10, handlelength=1.1,
               columnspacing=1.6)


def draw_consensus(ax, preds: dict[str, pd.DataFrame]) -> None:
    stack = np.stack([_image(df, df["pred"].to_numpy(dtype=float)) for df in preds.values()])
    ax.imshow(np.nanmean(stack, axis=0), cmap="gray", vmin=0, vmax=1, extent=EXTENT, interpolation="nearest")
    _bare(ax)


def draw_difficulty(ax, stats: dict[str, Any], preds: dict[str, pd.DataFrame]):
    g = grid.load()
    above = [m for m in stats["models_detail"] if m["accuracy"] > stats["always_water"]]
    wrong = np.mean([_image(preds[m["key"]], (preds[m["key"]]["pred"] != preds[m["key"]]["truth"]).to_numpy(dtype=float))
                     for m in above], axis=0)
    im = ax.imshow(wrong, cmap=WRONG, vmin=0, vmax=1, extent=EXTENT, interpolation="nearest")
    truth = grid.to_image(g["truth"].to_numpy(dtype=float), g["point_id"].to_numpy())
    ax.contour(truth, levels=[0.5], extent=EXTENT, origin="upper", colors="#3b3a37", linewidths=0.45)
    _bare(ax)
    return im, len(above)


def draw_sources(ax, stats: dict[str, Any]) -> None:
    models = [m for m in stats["models_detail"] if m["accuracy"] > stats["always_water"]][::-1]
    y = np.arange(len(models))
    left = np.zeros(len(models))
    for part, (color, label) in SOURCES.items():
        v = np.array([m["lost"][part] for m in models])
        ax.barh(y, v, left=left, color=color, height=0.68, label=label, edgecolor=SURFACE, lw=0.6)
        left += v
    for i, m in enumerate(models):
        ax.text(left[i] + 0.25, i, f"{100 - m['accuracy']:.1f}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, [m["name"] for m in models], fontsize=9.5)
    ax.set_xlim(0, max(left) * 1.1)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.set_xlabel("points of accuracy lost (100 − accuracy)", fontsize=9.5)
    ax.tick_params(length=0, labelsize=9)
    ax.grid(axis="x", color=RULE, lw=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 1.0), ncols=3, frameon=False, fontsize=9.5, borderaxespad=0.2)


def draw_knowledge(ax, stats: dict[str, Any]) -> None:
    """One row per model with P(Land) everywhere, best AUC on top: accuracy as answered, and if it said
    Land on the true share of the area. A long arrow is a model that knows more than its answers show."""
    models = sorted([m for m in stats["models_detail"] if "auc" in m], key=lambda m: m["auc"])
    base, land = stats["always_water"], stats["always_land"]
    y = np.arange(len(models))
    for i, m in enumerate(models):
        color = ORANGE if m["reasoning"] else BLUE
        a, b = m["accuracy"], m["accuracy_at_true_land_share"]
        if abs(b - a) > 1.2:
            ax.annotate("", xy=(b, i), xytext=(a, i),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.1, shrinkA=6, shrinkB=6, mutation_scale=10))
        ax.scatter(b, i, s=56, facecolors=SURFACE, edgecolors=color, lw=1.4, zorder=3)
        ax.scatter(a, i, s=56, color=color, zorder=4, edgecolors=SURFACE, lw=1.2)
        for x, text in ((1.02, f"{m['auc']:.3f}"), (1.11, f"{m['said_land']:.0f} %")):
            ax.text(x, i, text, transform=ax.get_yaxis_transform(), va="center", fontsize=9.5, color=INK2)
    for x, text in ((1.02, "AUC"), (1.11, "says Land on")):
        ax.text(x, len(models) - 0.3, text, transform=ax.get_yaxis_transform(), va="bottom", fontsize=9, color=MUTED)
    ax.set_yticks(y, [m["name"] for m in models], fontsize=10)
    ax.axvline(base, color=MUTED, lw=1.1, ls=(0, (4, 3)), zorder=0)
    ax.text(base + 0.6, -0.9, f"always Water {base:.0f} %", fontsize=8.5, color=MUTED, va="top")
    ax.set_xlim(min(land, min(m["accuracy"] for m in models)) - 4, 95)
    ax.set_ylim(-1.6, len(models) - 0.4)
    ax.xaxis.set_major_formatter(FuncFormatter(_pct))
    ax.tick_params(length=0, labelsize=9)
    ax.grid(axis="x", color=RULE, lw=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(
        handles=[
            Line2D([], [], ls="", marker="o", color=INK2, ms=7, label="accuracy as answered"),
            Line2D([], [], ls="", marker="o", mfc=SURFACE, mec=INK2, ms=7,
                   label=f"if it said Land only on the {land:.0f} % of the area it is surest of"),
        ],
        loc="lower left", bbox_to_anchor=(-0.01, 1.0), ncols=2, frameon=False, fontsize=9, borderaxespad=0.2,
    )


def _when(stats: dict[str, Any]) -> str:
    try:
        return datetime.fromisoformat(stats["finished"]).strftime("%B %Y")
    except (TypeError, ValueError):
        return ""


def _note(fig, stats: dict[str, Any], x: float, y: float, size: float = 8.5) -> None:
    flagged = [m for m in stats["models_detail"] if _flagged(m)]
    if flagged:
        m = flagged[0]
        fig.text(x, y, f"¹ {m['name']}: {m['indirect_share']:.0f} % of its answers start with a sentence instead of "
                 "Land or Water; there the answer is read from the first token's odds, a weaker signal.",
                 fontsize=size, color=MUTED, va="top")


def poster(path: Path, stats: dict[str, Any], preds: dict[str, pd.DataFrame]) -> Path:
    """The whole story on one page: what the models see, where they fail, the ranking, every map."""
    n = stats["models"]
    rows = int(np.ceil((n + 1) / 5))
    head, top_h, gap, maps_h, foot = 2.5, 9.4, 1.25, 2.3 * rows, 0.6  # inches
    H = head + top_h + gap + maps_h + foot

    def up(inches: float) -> float:
        return inches / H

    fig = plt.figure(figsize=(16, H))
    fig.text(0.045, 1 - up(0.3), "Land on Earth", fontsize=30, fontweight="bold", va="top")
    fig.text(0.045, 1 - up(0.95), f"What {n} open-weight models know about the world map: “Land or Water?” at "
             f"{stats['points']:,} points, one every 2°, with no image and no tools", fontsize=13, color=INK2, va="top")
    fig.text(0.955, 1 - up(0.3), f"{AUTHOR} · {_when(stats)}\n{REPO_URL.removeprefix('https://')}", fontsize=10,
             color=MUTED, va="top", ha="right", linespacing=1.5)
    top = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[1, 1], wspace=0.3, hspace=0.32,
                           left=0.045, right=0.955, top=1 - up(head), bottom=1 - up(head + top_h))
    a = fig.add_subplot(top[0, 0])
    draw_consensus(a, preds)
    d = fig.add_subplot(top[1, 0])
    _, n_above = draw_difficulty(d, stats, preds)
    b = fig.add_subplot(top[:, 1])
    draw_ranking(b, stats, cost=False, size=11.5)
    for ax in (a, d):
        ax.set_anchor("SW")
    fig.canvas.draw()
    for ax, title, line, lift in (
        (a, "The world as the models see it", f"share of the {n} models answering “Land”: white = all, black = none", 0.15),
        (d, "Where they go wrong", f"share of the {n_above} models above the always-Water line that are wrong; "
         "grey line: the true coastline", 0.15),
        (b, "Ranking", "accuracy weighted by area, against a 1 km land mask", 0.5),
    ):
        y0 = ax.get_position().y1 + up(lift)
        fig.text(ax.get_position().x0, y0 + up(0.32), title, fontsize=15, fontweight="bold", va="bottom")
        fig.text(ax.get_position().x0, y0, line, fontsize=10.5, color=INK2, va="bottom")
    maps = fig.add_gridspec(1, 1, left=0.045, right=0.955, top=1 - up(head + top_h + gap), bottom=up(foot))
    draw_maps(fig, maps[0], stats, preds)
    y_maps = 1 - up(head + top_h + gap) + up(0.55)
    fig.text(0.045, y_maps, "Each model's map", fontsize=15, fontweight="bold", va="bottom")
    maps_legend(fig, 0.2, y_maps + up(0.3))
    _note(fig, stats, 0.045, up(0.4), size=9)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def ranking(path: Path, stats: dict[str, Any]) -> Path:
    n = stats["models"]
    fig, ax = plt.subplots(figsize=(10, 2.2 + 0.36 * n))
    fig.subplots_adjust(left=0.2, right=0.9, top=1 - 1.25 / (2.2 + 0.36 * n), bottom=0.9 / (2.2 + 0.36 * n))
    draw_ranking(ax, stats)
    fig.text(0.02, 0.985, "Land or Water? — accuracy weighted by area", fontsize=14, fontweight="bold", va="top")
    fig.text(0.02, 0.985 - 0.32 / (2.2 + 0.36 * n), f"{stats['points']:,} points on a 2° grid, against a 1 km land "
             "mask. Spreads that overlap are ties.", fontsize=10, color=INK2, va="top")
    _note(fig, stats, 0.02, 0.35 / (2.2 + 0.36 * n))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def difficulty(path: Path, stats: dict[str, Any], preds: dict[str, pd.DataFrame]) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.9))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.12)
    im, n = draw_difficulty(ax, stats, preds)
    fig.text(0.02, 0.975, "Where the models go wrong", fontsize=14, fontweight="bold", va="top")
    fig.text(0.02, 0.925, f"Share of the {n} models above the always-Water line that answer wrong at each point; "
             "grey line: the true coastline", fontsize=10, color=INK2, va="top")
    cax = fig.add_axes([0.035, 0.07, 0.3, 0.025])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 0.5, 1], labels=["none", "half", f"all {n}"])
    cb.outline.set_visible(False)
    cax.tick_params(labelsize=9, length=0)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def error_sources(path: Path, stats: dict[str, Any]) -> Path:
    n = stats["above_always_water"]
    h = 2.0 + 0.36 * n
    fig, ax = plt.subplots(figsize=(10, h))
    fig.subplots_adjust(left=0.2, right=0.97, top=1 - 1.25 / h, bottom=0.8 / h)
    draw_sources(ax, stats)
    fig.text(0.02, 1 - 0.15 / h, "Where each model loses its points", fontsize=14, fontweight="bold", va="top")
    fig.text(0.02, 1 - 0.47 / h, f"Ice sheets: Antarctica and Greenland, land in the mask ({stats['ice_area']:.1f} % of the "
             f"area). Coasts: a neighbour 2° away is on the other side ({stats['coast_area']:.1f} %).",
             fontsize=10, color=INK2, va="top")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def knowledge_vs_bias(path: Path, stats: dict[str, Any]) -> Path | None:
    n = sum("auc" in m for m in stats["models_detail"])
    if not n:
        return None
    h = 2.3 + 0.42 * n
    fig, ax = plt.subplots(figsize=(10, h))
    fig.subplots_adjust(left=0.2, right=0.8, top=1 - 1.35 / h, bottom=0.55 / h)
    draw_knowledge(ax, stats)
    fig.text(0.02, 1 - 0.15 / h, "Knowing the map vs. saying “Land” too often", fontsize=14, fontweight="bold", va="top")
    fig.text(0.02, 1 - 0.47 / h, "Models that return P(Land) at every point. AUC: the chance that a land point gets a higher "
             "P(Land) than a water point,\nwhatever the threshold (1 = perfect, 0.5 = chance). Similar AUC, very different "
             "accuracy: the low scorers say Land too readily.", fontsize=10, color=INK2, va="top", linespacing=1.4)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _round(x: Any) -> Any:
    if isinstance(x, float):
        return None if np.isnan(x) else round(x, 4)
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v) for v in x]
    return x


def make(run_dir: Path) -> list[Path]:
    """Every figure of RESULTS.md and stats.json, from the scored run."""
    _style()
    board = load_board(run_dir)
    preds = {k: predictions(run_dir, k) for k in board["key"]}
    stats = diagnostics(run_dir, board, preds)
    out = run_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    (out / "stats.json").write_text(json.dumps(_round(stats), indent=2, ensure_ascii=False) + "\n")
    paths = [
        poster(out / "land_on_earth.png", stats, preds),
        ranking(out / "ranking.png", stats),
        difficulty(out / "difficulty.png", stats, preds),
        error_sources(out / "error_sources.png", stats),
        knowledge_vs_bias(out / "knowledge_vs_bias.png", stats),
    ]
    return [out / "stats.json"] + [p for p in paths if p is not None]
