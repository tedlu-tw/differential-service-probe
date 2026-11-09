import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from ds import load_config
from ds.stimuli import build_cells, make_history_pool, make_questions, numbers_in

CONFIG = Path(__file__).parents[1] / "configs" / "experiment.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG)


@pytest.fixture(scope="module")
def questions(cfg):
    return make_questions(cfg)


@pytest.fixture(scope="module")
def pool(cfg):
    return make_history_pool(cfg, np.random.default_rng(cfg["seeds"]["history"]))


@pytest.fixture(scope="module")
def cells(cfg):
    return build_cells(cfg)


def by_question(cells):
    out = defaultdict(dict)
    for row in cells:
        out[row["question_id"]][(row["condition"], row["phrasing"])] = row["messages"]
    return out


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


def test_100_questions_25_per_family(questions):
    assert len(questions) == 100
    counts = defaultdict(int)
    for q in questions:
        counts[q["family"]] += 1
    assert set(counts.values()) == {25} and len(counts) == 4
    assert len({q["id"] for q in questions}) == 100
    assert len({q["text"] for q in questions}) == 100


def test_answers_correct(questions):
    for q in questions:
        assert solve(q) == pytest.approx(q["answer"]), q["id"]


def test_answer_not_in_question_text(questions):
    for q in questions:
        assert float(q["answer"]) not in numbers_in(q["text"]), q["id"]


def test_questions_deterministic(cfg, questions):
    assert make_questions(cfg) == questions


def test_15_cells_per_question(cells):
    grouped = by_question(cells)
    assert len(grouped) == 100
    assert all(len(c) == 15 for c in grouped.values())
    assert len(cells) == 1500


def test_question_identical_across_conditions(cfg, questions, cells):
    grouped = by_question(cells)
    for q in questions:
        for (condition, phrasing), messages in grouped[q["id"]].items():
            last = messages[-1]["content"]
            prefix = cfg["cues"][condition][phrasing]
            assert last.endswith(q["text"])
            assert last[: len(last) - len(q["text"])].strip() == prefix


def test_system_prompt_first(cfg, cells):
    for row in cells:
        assert row["messages"][0] == {"role": "system", "content": cfg["prompts"]["system"]}


def test_history_pool_disjoint_from_questions(questions, pool):
    pool_texts = {p["text"] for p in pool}
    assert not pool_texts & {q["text"] for q in questions}
    assert len(pool_texts) == len(pool)


def test_history_only_correctness_differs(cfg, cells):
    for cell in by_question(cells).values():
        for phrasing in ("A", "B"):
            ok, bad = cell[("history_success", phrasing)], cell[("history_struggle", phrasing)]
            assert len(ok) == len(bad) == 2 + 2 * cfg["history"]["turns"]
            assert ok[0] == bad[0] and ok[-1] == bad[-1]
            for m_ok, m_bad in zip(ok[1:-1], bad[1:-1]):
                assert m_ok["role"] == m_bad["role"]
                if m_ok["role"] == "assistant":
                    assert m_ok["content"] == m_bad["content"] == cfg["prompts"]["tutor_ack"]
                else:
                    assert m_ok["content"].split("\n")[0] == m_bad["content"].split("\n")[0]
                    assert m_ok["content"] != m_bad["content"]


def solve_history(item):
    n = [int(x) for x in re.findall(r"\d+", item["text"])]
    if "rectangle" in item["text"]:
        return 2 * (n[0] + n[1])
    return n[0] + n[1] * n[2]


def test_history_work_is_right_and_wrong(pool):
    for item in pool:
        truth = solve_history(item)
        assert f"answer is {truth}" in item["right"]
        assert f"answer is {truth}" not in item["wrong"]
        assert f"answer is {item['wrong_value']}" in item["wrong"]


def test_answer_not_in_history(questions, cells):
    answers = {q["id"]: float(q["answer"]) for q in questions}
    for row in cells:
        if row["condition"].startswith("history_"):
            earlier = " ".join(m["content"] for m in row["messages"][1:-1])
            assert answers[row["question_id"]] not in numbers_in(earlier), row["question_id"]


STYLE_FORBIDDEN = ["stuck", "confused", "not sure how to start"]


def style_phrasings(cfg):
    return [text for cond, phrasings in cfg["cues"].items() if cond.startswith("style_")
            for text in phrasings.values()]


def test_style_phrasings_keep_the_request(cfg):
    phrasings = style_phrasings(cfg)
    assert len(phrasings) == 6
    for text in phrasings:
        # "solv" so that "solving" in the formal B phrasing counts as the same request.
        assert "solv" in text.lower(), text
        assert not any(word in text.lower() for word in STYLE_FORBIDDEN), text


def test_phrasings_use_different_history_problems(cells):
    for cell in by_question(cells).values():
        problems_a = {m["content"].split("\n")[0] for m in cell[("history_success", "A")][1:-1:2]}
        problems_b = {m["content"].split("\n")[0] for m in cell[("history_success", "B")][1:-1:2]}
        assert len(problems_a) == len(problems_b) == 2
        assert not problems_a & problems_b
