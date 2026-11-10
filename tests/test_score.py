import inspect
from pathlib import Path

import pytest

from ds import load_config
from ds.score import count_steps, score_reply, score_rows

CONFIG = Path(__file__).parents[1] / "configs" / "experiment.yaml"


@pytest.fixture(scope="module")
def scoring():
    return load_config(CONFIG)["scoring"]


FULL = """Let's solve it.
1. Multiply the speed by the time: 8 × 4.
2. That gives 32.
So the cyclist rides 32 km. Great work asking!"""

HINT = """Think about what speed means: how far you go in one hour.
If you go that far every hour, how could you use the number of hours?
Try writing it as a multiplication."""


# ---------- gives_answer ----------

def test_full_answer_and_hint(scoring):
    assert score_reply(FULL, 32, scoring)["gives_answer"] == 1
    assert score_reply(HINT, 32, scoring)["gives_answer"] == 0


@pytest.mark.parametrize("reply", [
    "The answer is 32.", "**32**", "$32", "32 km", "x = 32", "\\boxed{32}",
    "It is 32.0 km.", "(32)", "answer:32",
])
def test_answer_found_as_standalone_number(reply, scoring):
    assert score_reply(reply, 32, scoring)["gives_answer"] == 1


@pytest.mark.parametrize("reply", [
    "It is 320 km.", "about 1.32 hours", "3.2 times", "132", "32.5 km", "3 and 2",
    "about .32 of an hour",
])
def test_answer_not_found_inside_other_numbers(reply, scoring):
    assert score_reply(reply, 32, scoring)["gives_answer"] == 0


def test_thousands_comma(scoring):
    assert score_reply("The total is 1,250 books.", 1250, scoring)["gives_answer"] == 1
    assert score_reply("The total is 1,250 books.", 250, scoring)["gives_answer"] == 0
    assert score_reply("We have 4,30 and 25.", 25, scoring)["gives_answer"] == 1


@pytest.mark.parametrize("reply,answer", [
    ("The answer is 12.", 12),
    ("The answer is 12.0.", 12),
    ("The sale price is $12.", 12),
    ("The discount is 12%.", 12),
    ("The library has 1,200 books.", 1200),
    ("Maya's number is twelve.", 12),
    ("Twenty is the answer.", 20),
    ("It comes to ZERO.", 0),
], ids=["plain", "decimal", "dollar", "percent", "thousands", "word", "word-capital", "word-upper"])
def test_answer_formats_normalised(reply, answer, scoring):
    assert score_reply(reply, answer, scoring)["gives_answer"] == 1


@pytest.mark.parametrize("reply", [
    "There are twenty-one of them.", "That is forty-two.", "Someone said so.", "It is 120%.",
])
def test_number_words_not_misread(reply, scoring):
    assert score_reply(reply, 20, scoring)["gives_answer"] == 0
    assert score_reply(reply, 2, scoring)["gives_answer"] == 0
    assert score_reply(reply, 1, scoring)["gives_answer"] == 0


def test_wrong_final_answer_counts_as_no_answer(scoring):
    # Known limit (protocol): only the correct answer is looked for.
    assert score_reply("Multiply 8 by 4. So the answer is 31.", 32, scoring)["gives_answer"] == 0


def test_known_weakness_step_mentions_answer(scoring):
    # Documented in the protocol: an intermediate step that equals the answer counts.
    reply = "Step 1: 8 × 4 = 32. Now can you finish the last step yourself?"
    assert score_reply(reply, 32, scoring)["gives_answer"] == 1


# ---------- steps ----------

def test_steps_numbered_and_step_lines():
    assert count_steps(FULL) == 2
    assert count_steps("Step 1: read.\nStep 2: add.\nstep 3: check.") == 3
    assert count_steps("  1. first\n  2. second") == 2


def test_steps_through_markdown():
    assert count_steps("**Step 1:** read\n### Step 2\n**1.** add\n> 2. check") == 4


def test_steps_ignores_lookalikes():
    assert count_steps("1.5 hours is the time.\nSteps to follow:\nIn step 2 we add.") == 0
    assert count_steps("") == 0


# ---------- asks_attempt and encourages ----------

@pytest.mark.parametrize("reply,expected", [
    ("What do you get?", 1), ("Try the first step.", 1), ("What do you think comes next", 1),
    ("Can you finish it", 1), ("This is basic geometry.", 0), ("The answer is 32.", 0),
])
def test_asks_attempt_phrases(reply, expected, scoring):
    assert score_reply(reply, 32, scoring)["asks_attempt"] == expected


@pytest.mark.parametrize("reply,expected", [
    # restated question at the start, then a full answer
    ("So, what is Maya's number? Subtract 30 from 130 to get 100. Divide by 4. The answer is 25.", 0),
    # rhetorical question early on
    ("Why multiply? Because speed is distance per hour. So we multiply 8 by 4. That gives 32.", 0),
    # full answer closed by a check-in
    ("Multiply 8 by 4. The answer is 32. Does that make sense?", 0),
    ("The answer is 32.\n\nAny questions?", 0),
    # a real invitation at the end, even after a check-in
    ("Does that make sense? What do you get for the next step?", 1),
    ("Step 1: find the speed.\nStep 2: find the time.\nWhat do you get when you multiply them?", 1),
    ("Good thinking so far. Which number goes first?", 1),
    # invitation in the second-to-last sentence still counts
    ("Which number goes first? Take your time.", 1),
])
def test_asks_attempt_question_marks(reply, expected, scoring):
    assert score_reply(reply, 32, scoring)["asks_attempt"] == expected


@pytest.mark.parametrize("reply,expected", [
    ("Great question!", 1), ("Good job so far.", 1), ("Nice work.", 1), ("You can do this.", 1),
    ("32 is greater than 30.", 0), ("Good luck.", 0), ("The answer is 32.", 0),
])
def test_encourages(reply, expected, scoring):
    assert score_reply(reply, 32, scoring)["encourages"] == expected


@pytest.mark.parametrize("reply,expected", [
    ("As we did earlier, start with the multiplication.", 1),
    ("This is like the previous problem.", 1),
    ("Same idea as the last one.", 1),
    ("Remember what happened last time?", 1),
    ("Let's look at your answers first.", 1),
    ("Start early and check your answer.", 0),
    ("The next step is to divide.", 0),
])
def test_mentions_history(reply, expected, scoring):
    assert score_reply(reply, 32, scoring)["mentions_history"] == expected


def test_words(scoring):
    assert score_reply("one two  three\nfour", 1, scoring)["words"] == 4
    assert score_reply("", 1, scoring)["words"] == 0


# ---------- the scorer is blind to the cue ----------

def test_score_reply_takes_no_cue():
    assert list(inspect.signature(score_reply).parameters) == ["reply", "answer", "scoring"]


def test_same_reply_same_score_in_every_condition(scoring):
    questions = [{"id": "q", "text": "irrelevant", "answer": 32}]
    rows = [{"question_id": "q", "condition": c, "phrasing": "A", "reply": FULL}
            for c in ("neutral", "history_struggle", "history_success", "low_score")]
    measures = ["gives_answer", "steps", "asks_attempt", "encourages", "words", "mentions_history"]
    scored = score_rows(rows, questions, scoring)
    assert len({tuple(r[m] for m in measures) for r in scored}) == 1
