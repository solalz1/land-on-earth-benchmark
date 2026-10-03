"""Build data/grid.csv: the 16,200-point grid of the tweet with its ground truth.

Grid: one point every 2 degrees, at cell centres (latitudes -89..89, longitudes -179..179).
Truth: `truth` comes from the 1 km GLOBE land mask (package global-land-mask), our best guess
for the "1-km land mask" quoted in the tweet. That mask counts inland water such as the Caspian
Sea or Lake Victoria as land, so `truth_lakes` also marks as water every point inside a Natural
Earth 1:10m lake, plus the Caspian Sea (which Natural Earth files as a sea, not a lake).
Weight: cell area, proportional to cos(latitude) for equal-angle cells.

Run once with: uv run --extra truth python scripts/build_truth.py
The resulting CSV is committed, so nobody else needs to run this.
"""

from __future__ import annotations

import json
import math
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "grid.csv"
NE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
LAKES = "ne_10m_lakes.geojson"
MARINE = "ne_10m_geography_marine_polys.geojson"
INLAND_SEAS = {"Caspian Sea"}  # filed as "sea" by Natural Earth but marked land by GLOBE
STEP = 2.0


def grid() -> pd.DataFrame:
    lats = np.arange(90 - STEP / 2, -90, -STEP)  # 89 .. -89, north first
    lons = np.arange(-180 + STEP / 2, 180, STEP)  # -179 .. 179
    lat, lon = np.meshgrid(lats, lons, indexing="ij")
    df = pd.DataFrame({"lat": lat.ravel(), "lon": lon.ravel()})
    df.insert(0, "point_id", np.arange(len(df)))
    df["row"] = np.repeat(np.arange(len(lats)), len(lons))
    df["col"] = np.tile(np.arange(len(lons)), len(lats))
    df["weight"] = np.cos(np.radians(df["lat"]))
    return df


def features(name: str) -> list[dict]:
    cache = ROOT / ".cache" / name
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {NE + name}")
        urllib.request.urlretrieve(NE + name, cache)
    return json.loads(cache.read_text())["features"]


def load_water():
    from shapely.geometry import shape
    from shapely.strtree import STRtree

    feats = features(LAKES)
    feats += [f for f in features(MARINE) if f["properties"].get("name") in INLAND_SEAS]
    geoms = [shape(f["geometry"]) for f in feats]
    names = [f["properties"].get("name") or "" for f in feats]
    return STRtree(geoms), geoms, names


def main() -> int:
    from global_land_mask import globe
    from shapely.geometry import Point

    df = grid()
    df["truth"] = globe.is_land(df["lat"].to_numpy(), df["lon"].to_numpy()).astype(int)

    tree, geoms, names = load_water()
    in_lake = np.zeros(len(df), dtype=bool)
    lake_name = [""] * len(df)
    for i, (la, lo) in enumerate(zip(df["lat"], df["lon"])):
        p = Point(lo, la)
        for j in tree.query(p):
            if geoms[j].contains(p):
                in_lake[i] = True
                lake_name[i] = names[j]
                break
    df["truth_lakes"] = np.where(in_lake, 0, df["truth"]).astype(int)
    df["lake"] = lake_name

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df["weight"] = df["weight"].round(8)
    df.to_csv(OUT, index=False)
    w = df["weight"]
    land = (w * df["truth"]).sum() / w.sum()
    land_l = (w * df["truth_lakes"]).sum() / w.sum()
    changed = df[df["truth"] != df["truth_lakes"]]
    print(f"wrote {OUT} ({len(df)} points)")
    print(f"land share, area-weighted: {land:.4f} (lakes as water: {land_l:.4f})")
    print(f"points switched to water by lakes: {len(changed)}")
    print(changed.groupby("lake").size().sort_values(ascending=False).to_string())
    assert len(df) == 16_200 and math.isclose(df["lat"].max(), 89.0)
    assert (df["weight"] > 0).all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
