"""The question asked at every point — identical for the 20 models."""

from __future__ import annotations

import re

# The tweet's figure only shows "Land or Water?" and coordinates as text; this is the closest
# wording, with the answer format spelled out so that the first token is the answer.
TEMPLATE = "Coordinates: {lat:.1f}, {lon:.1f}\nLand or Water? Answer with one word: Land or Water."

_COORDS = re.compile(r"Coordinates: (-?\d+(?:\.\d+)?), (-?\d+(?:\.\d+)?)")


def question(lat: float, lon: float) -> str:
    return TEMPLATE.format(lat=lat, lon=lon)


def coords(text: str) -> tuple[float, float] | None:
    """Inverse of question(), used by the fake API."""
    m = _COORDS.search(text)
    return (float(m.group(1)), float(m.group(2))) if m else None
