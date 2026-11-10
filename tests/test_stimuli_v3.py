import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest

from ds import load_config
from ds.stimuli import history_numbers, history_setup, make_history_pool, make_questions, numbers_in
from ds.stimuli_v3 import (build_cells_v3, build_check_cells_v3, build_solve_cells_v3, hint_ignore_numbers,
                           make_questions_v3, pilot_questions)

CONFIGS = Path(__file__).parents[1] / "configs"


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIGS / "experiment_v3.yaml")


@pytest.fixture(scope="module")
def v2cfg():
    return load_config(CONFIGS / "experiment.yaml")


@pytest.fixture(scope="module")
def questions(cfg):
    return make_questions_v3(cfg)


@pytest.fixture(scope="module")
def cells(cfg):
    return build_cells_v3(cfg)


def solve(question):
    """Re-derive the answer from the text alone, independently of the generator."""
    n = [float(x) for x in re.findall(r"\d+", question["text"])]
    family = question["family"]
    if family == "two_step":
        return n[0] - n[1] + n[2] * n[3]
    if family == "percentage":
        return n[0] * (1 - n[1] / 100)
    if family == "linear_equation":
        return (n[2] - n[1]) / n[0]
    return n[0] * n[1]


# ---------- questions ----------

def test_200_questions_50_per_family(questions):
    assert len(questions) == 200
    assert Counter(q["family"] for q in questions) == {f: 50 for f in
                                                       ["two_step", "percentage", "linear_equation", "rate"]}
    assert len({q["id"] for q in questions}) == 200 and all(q["id"].startswith("v3_") for q in questions)


def test_answers_correct(questions):
    for q in questions:
        assert solve(q) == pytest.approx(q["answer"]), q["id"]


def test_minimum_answer(questions, cfg):
    assert min(q["answer"] for q in questions) >= cfg["questions"]["min_answer"] == 21
    low, high = cfg["questions"]["linear_answer_range"]
    linear = [q["answer"] for q in questions if q["family"] == "linear_equation"]
    assert all(low <= a <= high for a in linear)
    # The v2 range (3 to 25) would also pass the minimum, but only with the 5 answers 21 to 25.
    assert len(set(linear)) > 10


def test_answer_not_in_question_or_cues(questions):
    for q in questions:
        assert float(q["answer"]) not in numbers_in(q["text"]), q["id"]
        assert q["answer"] not in (40, 95), q["id"]


def test_no_v2_question_repeated(questions, v2cfg):
    v2 = {q["text"] for q in make_questions(v2cfg)}
    assert not v2 & {q["text"] for q in questions}


def test_same_wording_as_v2(questions, v2cfg):
    # Only the numbers differ from v2: replace every number and compare the wordings per family.
    shape = lambda q: (q["family"], re.sub(r"\d+", "N", q["text"]))
    assert {shape(q) for q in questions} == {shape(q) for q in make_questions(v2cfg)}


def test_questions_reproducible(cfg, questions):
    assert make_questions_v3(cfg) == questions


def test_pilot_questions(cfg, questions):
    pilot = pilot_questions(cfg, questions)
    assert len(pilot) == 100 and set(Counter(q["family"] for q in pilot).values()) == {25}


# ---------- history ----------

def test_history_pool_is_v2_pool(cfg, v2cfg):
    pool = lambda c: make_history_pool(c, np.random.default_rng(c["seeds"]["history"]))
    assert pool(cfg) == pool(v2cfg)


def test_history_avoids_answer(cfg, questions):
    pool, assignment = history_setup(cfg, questions)
    for q in questions:
        for picks in assignment[q["id"]].values():
            assert all(float(q["answer"]) not in history_numbers(pool[i]) for i in picks), q["id"]


def test_hint_ignore_numbers(cfg):
    ignore = hint_ignore_numbers(cfg)
    pool, _ = history_setup(cfg, [])
    assert {40.0, 95.0} <= ignore
    assert all(history_numbers(item) <= ignore for item in pool)


# ---------- cells ----------

def by_question(cells):
    out = defaultdict(dict)
    for row in cells:
        out[row["question_id"]][(row["prompt"], row["condition"], row["phrasing"])] = row["messages"]
    return out


def test_cells_per_question(cells, questions):
    assert len(cells) == 200 * 20
    counts = Counter((r["question_id"], r["prompt"]) for r in cells)
    assert set(counts[(q["id"], "pedagogical")] for q in questions) == {15}
    assert set(counts[(q["id"], "plain")] for q in questions) == {5}
    plain = {r["condition"] for r in cells if r["prompt"] == "plain"}
    assert plain == {"neutral", "history_success", "history_struggle"}


def test_system_prompt_per_cell(cells, cfg):
    for r in cells:
        assert r["messages"][0] == {"role": "system", "content": cfg["prompts"]["system"][r["prompt"]]}


def test_question_text_identical_in_every_cell(cells, questions):
    text = {q["id"]: q["text"] for q in questions}
    for r in cells:
        assert r["messages"][-1]["role"] == "user"
        assert r["messages"][-1]["content"].endswith(text[r["question_id"]])


def test_history_conditions_differ_only_in_correctness(cells):
    for cells_of_q in by_question(cells).values():
        for prompt in ("pedagogical", "plain"):
            for ph in ("A", "B"):
                ok, bad = cells_of_q[(prompt, "history_success", ph)], cells_of_q[(prompt, "history_struggle", ph)]
                assert len(ok) == len(bad) == 6
                assert [m["role"] for m in ok] == [m["role"] for m in bad]
                same = [a == b for a, b in zip(ok, bad)]
                assert same == [True, False, True, False, True, True]   # system, work, ack, work, ack, question


def test_cues_and_messages_match_v2_shape(cells, cfg):
    first = next(r for r in cells if (r["prompt"], r["condition"], r["phrasing"]) == ("pedagogical", "low_score", "A"))
    assert first["messages"][-1]["content"].startswith(cfg["cues"]["low_score"]["A"] + "\n\n")
    neutral = next(r for r in cells if r["condition"] == "neutral")
    assert len(neutral["messages"]) == 2


def test_subset_keeps_full_history_assignment(cfg, questions, cells):
    # The pilot is built from a subset; its history turns must match the full design.
    subset = build_cells_v3(cfg, pilot_questions(cfg, questions))
    full = {(r["question_id"], r["prompt"], r["condition"], r["phrasing"]): r["messages"] for r in cells}
    assert all(r["messages"] == full[(r["question_id"], r["prompt"], r["condition"], r["phrasing"])]
               for r in subset)


# ---------- solve and check ----------

def test_solve_cells(cfg, questions):
    rows = build_solve_cells_v3(cfg, questions)
    assert len(rows) == 200
    assert all(r["messages"][0]["content"] == "You are a patient math tutor." for r in rows)
    assert all(r["messages"][1]["content"].endswith(cfg["prompts"]["solve_instruction"]) for r in rows)


def test_check_cells(cfg, questions):
    rows = build_check_cells_v3(cfg, questions)
    assert len(rows) == 200 * 2 * 2
    assert all(r["messages"][0]["content"] == cfg["prompts"]["system"]["pedagogical"] for r in rows)
    assert all(r["messages"][-1]["content"] == cfg["prompts"]["check_question"] for r in rows)
    assert all(text not in r["messages"][-2]["content"] for r in rows for text in ("Maya", "cyclist"))
