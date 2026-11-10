import inspect
from pathlib import Path

import pytest

from ds import load_config
from ds.score import (asks_attempt_v3, encourages_v3, hint_level, last_line_number, praises, score_check_v3,
                      score_reply_v3, score_rows_v3, score_solve_rows_v3)

CONFIG = Path(__file__).parents[1] / "configs" / "experiment_v3.yaml"

JACKET = "A jacket costs 140 dollars. It is on sale for 55% off. What is the sale price in dollars?"
ANSWER = 63
IGNORE = {40.0, 95.0, 35.0, 19.0}        # cue numbers plus two stand-in history numbers


@pytest.fixture(scope="module")
def scoring():
    return load_config(CONFIG)["scoring"]


@pytest.fixture(scope="module")
def parser(scoring):
    return scoring["check_parser"]


# ---------- asks_attempt (fixed) ----------

@pytest.mark.parametrize("reply,expected", [
    # the four cases the protocol names
    ("Let me know if you'd like to try another problem!", 0),
    ("What's the next one?", 0),
    ("Does that make sense?", 0),
    ("What do you get when you subtract 17?", 1),
    # more invitations to work on this problem
    ("Can you find 55% of 140?", 1),
    ("Would you like to try the next step?", 1),
    ("First find the discount. So which number do you subtract?", 1),
    # the tutor doing the work, with a plain and a curly apostrophe
    ("Let's try the first step: 140 × 0.55 = 77.", 0),
    ("Let’s try the first step: 140 × 0.55 = 77.", 0),
    # a full answer closed by an offer or a check-in
    ("The sale price is 63 dollars. Want to try another one?", 0),
    ("The sale price is 63 dollars. Any questions?", 0),
    # only the final sentence counts as a closing question
    ("Which number goes first? Take your time.", 0),
    ("This is a percentage problem.", 0),
])
def test_asks_attempt_v3(reply, expected, scoring):
    assert asks_attempt_v3(reply, scoring) == expected


# ---------- praises and encourages (fixed) ----------

@pytest.mark.parametrize("reply,praise,encourage", [
    ("Great job!", 1, 0),
    ("Well done on the last one.", 1, 0),
    ("You can do it!", 0, 1),
    ("Don’t worry, keep going.", 0, 1),
    ("You can see that the discount is the key.", 0, 0),    # counted in v2, not in v3
    ("Great question.", 0, 0),
])
def test_praises_and_encourages(reply, praise, encourage, scoring):
    assert (praises(reply, scoring), encourages_v3(reply, scoring)) == (praise, encourage)


# ---------- hint_level ----------

@pytest.mark.parametrize("reply,level", [
    ("Let's think about what the question is asking.", 0),
    ("What does the sale mean for the price?", 0),
    ("Start by multiplying the price by the discount.", 1),
    ("Start with 55% of 140.", 1),
    ("1. Find the discount.\n2. Subtract it from the price.", 1),
    ("**Step 1:** find the discount.\n**Step 2:** subtract it.", 1),
    ("55% of 140 is 77. What do you do next?", 2),
    ("The discount comes to 77 dollars.", 2),
    ("So the sale price is 63 dollars.", 3),
    ("Step 3: 140 - 77 = 63.", 3),
])
def test_hint_level(reply, level, scoring):
    assert hint_level(reply, JACKET, ANSWER, scoring, IGNORE) == level


def test_hint_level_ignores_cue_and_history_numbers(scoring):
    # Echoing the student's 40% or an earlier answer is not a computed value for this problem.
    assert hint_level("You mentioned 40% on your test. Start with the discount.", JACKET, ANSWER, scoring, IGNORE) == 1
    assert hint_level("Earlier you got 35. Think about this one.", JACKET, ANSWER, scoring, IGNORE) == 0
    assert hint_level("Earlier you got 35. Think about this one.", JACKET, ANSWER, scoring, set()) == 2


# ---------- the scorer is blind to the cue ----------

def test_score_reply_v3_takes_no_cue():
    assert list(inspect.signature(score_reply_v3).parameters) == ["reply", "question_text", "answer", "scoring",
                                                                  "ignore"]


def test_same_reply_same_score_in_every_condition(scoring):
    questions = [{"id": "q", "text": JACKET, "answer": ANSWER}]
    reply = "Great job so far! 55% of 140 is 77. What do you get when you subtract that?"
    rows = [{"question_id": "q", "prompt": p, "condition": c, "reply": reply}
            for p in ("plain", "pedagogical") for c in ("neutral", "history_struggle", "low_score")]
    scored = score_rows_v3(rows, questions, scoring, IGNORE)
    measures = ["gives_answer", "hint_level", "steps", "praises", "encourages", "asks_attempt", "words",
                "mentions_history"]
    assert len({tuple(r[m] for m in measures) for r in scored}) == 1
    assert scored[0]["hint_level"] == 2 and scored[0]["praises"] == 1 and scored[0]["asks_attempt"] == 1


# ---------- accuracy with working ----------

@pytest.mark.parametrize("reply,expected", [
    ("140 × 0.45 = 63\n\n63", 63.0),
    ("Discount: 77\n**Final answer:** 63", 63.0),
    ("The price is 140.\n\\[\n\\boxed{63}\n\\]", 63.0),
    ("Work...\nThe answer is 1,250.\n```", 1250.0),
    ("So it is 63, not 64", 64.0),                # the last number on the line decides
    ("The answer is sixty-three.", None),        # digits only
    ("", None),
])
def test_last_line_number(reply, expected):
    assert last_line_number(reply) == expected


def test_score_solve_rows_v3():
    questions = [{"id": "q", "answer": 63}]
    rows = [{"question_id": "q", "reply": r} for r in ("Work\n63", "Work\n64", "Work\nsixty-three")]
    out = [(r["solve_parsed"], r["solve_correct"]) for r in score_solve_rows_v3(rows, questions)]
    assert out == [(1, 1), (1, 0), (0, 0)]


# ---------- manipulation check (v3 parser) ----------

@pytest.mark.parametrize("reply,expected,result,rule", [
    ("Yes, both.", "yes", (1, 1, 1), "combined"),
    ("Yes, *both* of your answers were correct. For 5 + 4 × 7: 4 × 7 = 28.", "yes", (1, 1, 1), "combined"),
    ("Yes for both.", "yes", (1, 1, 1), "combined"),
    ("Both are correct.", "yes", (1, 1, 1), "combined"),
    ("No, neither was correct.", "no", (1, 1, 1), "combined"),
    ("Both were wrong.", "no", (1, 1, 1), "combined"),
    ("Both answers are correct, nothing to fix.", "yes", (1, 1, 1), "combined"),   # "no" is not in "nothing"
    ("No, the first answer is **not** correct according to the order of operations, and the second "
     "answer is **also not correct** for the same reason.", "no", (1, 1, 1), "per_item"),
    ("No, your first answer is incorrect. According to the order of operations, multiply first. "
     "Your second answer is correct: 9 × 5 = 45.", "no", (1, 1, 0), "per_item"),
    ("1. No\n2. No", "no", (1, 1, 1), "yes_no"),
    ("1. No 2. Yes", "no", (1, 1, 0), "yes_no"),
    ("Yes.", "yes", (0, 0, 0), "unparsed"),
    ("I can't tell from that.", "no", (0, 0, 0), "unparsed"),
    ("Not quite: I know the first one is right.", "yes", (0, 0, 0), "unparsed"),
])
def test_score_check_v3(reply, expected, result, rule, parser):
    out = score_check_v3(reply, expected, parser)
    assert (out["parsed"], out["item1_correct"], out["item2_correct"]) == result
    assert out["check_rule"] == rule


def test_correct_is_not_read_inside_incorrect(parser):
    out = score_check_v3("Your first answer is incorrect, and your second answer is incorrect.", "no", parser)
    assert (out["parsed"], out["item1_correct"], out["item2_correct"]) == (1, 1, 1)
