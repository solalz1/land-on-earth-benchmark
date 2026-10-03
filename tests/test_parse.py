import math

import pytest

from loe.parse import from_text, norm, parse, positions, word_of


def chat_choice(tokens, tops=None, content=None):
    tops = tops or [[(t, -0.01)] for t in tokens]
    return {
        "message": {"role": "assistant", "content": content if content is not None else "".join(tokens)},
        "logprobs": {
            "content": [
                {"token": t, "logprob": top[0][1], "top_logprobs": [{"token": a, "logprob": b} for a, b in top]}
                for t, top in zip(tokens, tops)
            ]
        },
    }


@pytest.mark.parametrize(
    "token,word",
    [("Land", "land"), (" Land", "land"), ("ĠLand", "land"), ("▁land", "land"), ("LAND", "land"),
     ("Lan", "land"), ("Water", "water"), (" water", "water"), ("Wat", "water"), ("**Land**", "land"),
     ("La", None), ("W", None), ("The", None), ("Landmass", None), ("", None)],
)
def test_word_of(token, word):
    assert word_of(token) == word


def test_norm_strips_markup():
    assert norm(' **"Land".') == "land"
    assert norm("\n") == ""


def test_direct_answer_probability():
    p = 0.8
    choice = chat_choice(
        ["Land"], [[("Land", math.log(p)), ("Water", math.log(0.15)), (" Land", math.log(0.02)), ("The", -6.0)]]
    )
    r = parse(choice)
    assert r.source == "logprobs" and r.pred == 1 and r.pos == 0
    assert r.p_land == pytest.approx(0.82 / 0.97)
    assert r.mass == pytest.approx(0.97)


def test_markdown_and_whitespace_are_skipped():
    tops = [[("**", -0.01)], [("Water", math.log(0.9)), ("Land", math.log(0.05))], [("**", -0.01)]]
    r = parse(chat_choice(["**", "Water", "**"], tops))
    assert r.pos == 1 and r.pred == 0 and r.p_land == pytest.approx(0.05 / 0.95)


def test_sentence_before_answer():
    tokens = ["The", " answer", " is", " Land"]
    tops = [[("The", -0.1)], [(" answer", -0.1)], [(" is", -0.1)], [(" Land", -0.2), (" Water", -2.0)]]
    r = parse(chat_choice(tokens, tops))
    assert r.pos == 3 and r.pred == 1


def test_legacy_completion_format():
    choice = {
        "text": " Water",
        "logprobs": {
            "tokens": [" Water"],
            "token_logprobs": [-0.1],
            "top_logprobs": [{" Water": -0.1, " Land": -2.5}],
        },
    }
    assert len(positions(choice)) == 1
    r = parse(choice)
    assert r.pred == 0 and r.source == "logprobs"


def test_no_logprobs_falls_back_to_text():
    r = parse({"message": {"content": "Water."}, "logprobs": None})
    assert r.pred == 0 and r.source == "text" and r.p_land is None


def test_unreadable_answer():
    r = parse({"message": {"content": "I cannot determine that."}})
    assert r.pred is None and r.source is None


def test_text_ignores_reasoning_blocks():
    assert from_text("<think>is it land? maybe water</think>Land") == 1
    assert from_text("<think>still thinking about water") is None
    assert from_text("") is None


def test_answer_without_alternatives_uses_chosen_logprob():
    choice = {"message": {"content": "Land"}, "logprobs": {"content": [{"token": "Land", "logprob": -0.05}]}}
    r = parse(choice)
    assert r.pred == 1 and r.p_land == 1.0
