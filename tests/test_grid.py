import numpy as np

from loe import grid
from loe.prompts import coords, question


def test_grid_matches_the_tweet():
    df = grid.load()
    assert len(df) == 16_200
    assert df["lat"].max() == 89 and df["lat"].min() == -89
    assert df["lon"].max() == 179 and df["lon"].min() == -179
    assert set(np.unique(np.diff(np.sort(df["lat"].unique())))) == {2.0}
    assert (df["weight"] > 0).all()
    assert (df["point_id"] == np.arange(16_200)).all()


def test_land_share_and_always_water_baseline():
    df = grid.load()
    w = df["weight"]
    land = (w * df["truth"]).sum() / w.sum()
    assert 0.28 < land < 0.30  # ~29 % of the globe is land
    assert ((df["truth_lakes"] <= df["truth"])).all()  # lakes only turn land into water


def test_known_points():
    df = grid.load().set_index(["lat", "lon"])
    assert df.loc[(47.0, 3.0), "truth"] == 1  # central France
    assert df.loc[(-1.0, -151.0), "truth"] == 0  # central Pacific
    assert df.loc[(-81.0, 11.0), "truth"] == 1  # Antarctica
    assert df.loc[(41.0, 51.0), "truth_lakes"] == 0  # Caspian Sea


def test_image_layout():
    df = grid.load()
    img = grid.to_image(df["lat"].to_numpy())
    assert img.shape == (90, 180)
    assert img[0, 0] == 89 and img[-1, 0] == -89
    lon = grid.to_image(df["lon"].to_numpy())
    assert lon[0, 0] == -179 and lon[0, -1] == 179


def test_pilot_points_are_stable_and_balanced():
    a, b = grid.pilot_points(200), grid.pilot_points(200)
    assert (a["point_id"] == b["point_id"]).all()
    assert a["truth"].sum() == 100
    assert list(a["truth"].head(4)) == [1, 0, 1, 0]


def test_prompt_round_trip():
    q = question(-33.0, 151.0)
    assert q.startswith("Coordinates: -33.0, 151.0\nLand or Water?")
    assert coords(q) == (-33.0, 151.0)
