"""Turn a completion into a prediction: P(Land) from the top logprobs, else the text answer.

The answer position is the first generated token that is not whitespace or markup ("**",
quotes...). There, P(Land) = mass(Land) / (mass(Land) + mass(Water)), summing every variant
of each word among the top-20 alternatives (" Land", "land", "LAND", "Lan"...). The prediction
is the more probable of the two, which is the greedy answer whenever the model answers directly.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

_STRIP = " \t\r\n*\"'`.:,;!?_#-()[]{}<>“”‘’«»"
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_WORD = re.compile(r"\b(land|water)\b", re.I)
MAX_SCAN = 8  # tokens scanned when the model starts with a few words ("The answer is ...")

Position = tuple[str, list[tuple[str, float]]]  # (chosen token, [(alternative, logprob), ...])


@dataclass
class Parsed:
    pred: int | None  # 1 = Land, 0 = Water, None = no usable answer
    p_land: float | None  # from logprobs, None without them
    mass: float | None  # probability mass on Land + Water at the answer position
    source: str | None  # "logprobs" or "text"
    pos: int | None  # index of the answer position among the content tokens
    top: list[tuple[str, float]] | None  # alternatives at the answer position


def norm(token: str) -> str:
    t = token.replace("Ġ", " ").replace("▁", " ").replace("Ċ", "\n")
    return t.strip(_STRIP).lower()


def word_of(token: str) -> str | None:
    """'land' or 'water' if the token spells the start of that word (3+ letters), else None."""
    t = norm(token)
    if len(t) >= 3:
        if "land".startswith(t):
            return "land"
        if "water".startswith(t):
            return "water"
    return None


def positions(choice: dict[str, Any]) -> list[Position]:
    """Normalise the logprobs of one choice: chat format, or legacy completions format."""
    lp = choice.get("logprobs")
    if not lp:
        return []
    out: list[Position] = []
    if isinstance(lp, dict) and lp.get("content"):
        for item in lp["content"]:
            alts = [(a.get("token", ""), float(a.get("logprob", -math.inf))) for a in item.get("top_logprobs") or []]
            chosen = item.get("token", "")
            if not alts and "logprob" in item:
                alts = [(chosen, float(item["logprob"]))]
            out.append((chosen, alts))
    elif isinstance(lp, dict) and lp.get("tokens"):
        tops = lp.get("top_logprobs") or []
        chosen_lps = lp.get("token_logprobs") or []
        for i, tok in enumerate(lp["tokens"]):
            top = tops[i] if i < len(tops) and tops[i] else {}
            alts = [(k, float(v)) for k, v in top.items()]
            if not alts and i < len(chosen_lps) and chosen_lps[i] is not None:
                alts = [(tok, float(chosen_lps[i]))]
            out.append((tok, alts))
    return out


def _masses(alts: list[tuple[str, float]]) -> tuple[float, float]:
    land = water = 0.0
    for tok, lp in alts:
        w = word_of(tok)
        if w == "land":
            land += math.exp(lp)
        elif w == "water":
            water += math.exp(lp)
    return land, water


def from_logprobs(pos: list[Position]) -> tuple[float, float, int, list[tuple[str, float]]] | None:
    """(p_land, mass, index, alternatives) at the answer position, or None."""
    content = [i for i, (tok, _) in enumerate(pos[:MAX_SCAN]) if norm(tok)]
    if not content:
        return None
    # the first token that is Land or Water itself, else the first content token
    pick = next((i for i in content if word_of(pos[i][0])), content[0])
    land, water = _masses(pos[pick][1])
    if land + water <= 0:
        return None
    return land / (land + water), land + water, pick, pos[pick][1]


def from_text(text: str | None) -> int | None:
    if not text:
        return None
    text = _THINK.sub(" ", text)
    if "<think>" in text.lower():  # unterminated reasoning: no answer yet
        return None
    m = _WORD.search(text)
    return None if m is None else int(m.group(1).lower() == "land")


def parse(choice: dict[str, Any]) -> Parsed:
    text = choice_text(choice)
    lp = from_logprobs(positions(choice))
    if lp is not None:
        p, mass, i, alts = lp
        return Parsed(int(p > 0.5), p, mass, "logprobs", i, alts)
    pred = from_text(text)
    return Parsed(pred, None, None, "text" if pred is not None else None, None, None)


def choice_text(choice: dict[str, Any]) -> str:
    msg = choice.get("message")
    if isinstance(msg, dict):
        return msg.get("content") or ""
    return choice.get("text") or ""
