"""src/ds/items_v4.py: pre-freeze step a, the item bank tests (docs/protocol_v4.md: Items)."""
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from ds import load_config
from ds.items_v4 import (attempt_text, draw_params, earlier_texts, fixed_numbers, is_valid_current, make_bank,
                         question_text, worked_values)
from ds.stimuli import FAMILIES, history_numbers, numbers_in
from ds.stimuli_v3 import templates

ROOT = Path(__file__).parents[1]
CFG = load_config(ROOT / "configs" / "experiment_v4.yaml")
V3 = load_config(ROOT / "configs" / "experiment_v3.yaml")


@pytest.fixture(scope="module")
def bank():
    return make_bank(CFG)


def pool_items(bank):
    return {h["id"]: h for items in bank["pool"].values() for h in items}


def test_wording_and_draws_are_v3s():
    """Same seed, same text and answer as v2's and v3's own template functions."""
    v3_templates = templates(V3)
    for family in CFG["items"]["families"]:
        assert family in FAMILIES
        for seed in range(300):
            p = draw_params(family, np.random.default_rng(seed), CFG["items"]["linear_answer_range"])
            text, answer = v3_templates[family](np.random.default_rng(seed))
            assert question_text(family, p) == text and worked_values(family, p)["ans"] == answer


def test_sizes_and_ids(bank):
    it = CFG["items"]
    assert Counter(q["family"] for q in bank["current"]) == {f: it["current_per_family"] for f in it["families"]}
    assert {f: len(items) for f, items in bank["pool"].items()} == {f: it["pool_per_family"] for f in it["families"]}
    ids = [q["id"] for q in bank["current"]] + list(pool_items(bank))
    assert len(ids) == len(set(ids)) == 100 + 200


def correct_and_wrong(family, p):
    """The protocol's table, computed independently of items_v4."""
    if family == "two_step":
        return p["a"] - p["b"] + p["c"] * p["d"], (p["a"] - p["b"] + p["c"]) * p["d"]
    if family == "percentage":
        return p["p"] * (100 - p["o"]) / 100, p["p"] * p["o"] / 100
    if family == "linear_equation":
        return (p["r"] - p["b"]) / p["a"], p["r"] / p["a"] - p["b"]
    return p["s"] * p["h"], p["s"] + p["h"]


def test_answers_and_misconception_answers(bank):
    for item in bank["current"] + list(pool_items(bank).values()):
        right, wrong = correct_and_wrong(item["family"], item["params"])
        assert item["answer"] == right and item["wrong_answer"] == wrong, item["id"]
        assert isinstance(item["wrong_answer"], int) and item["wrong_answer"] > 0 and item["wrong_answer"] != item["answer"]
        assert re.fullmatch(r".* So the answer is (\d+)( dollars| km)?\.", item["right"]).group(1) == str(item["answer"])
        assert re.fullmatch(r".* So the answer is (\d+)( dollars| km)?\.", item["wrong"]).group(1) == str(item["wrong_answer"])
        if item["family"] == "linear_equation":
            assert item["params"]["r"] % item["params"]["a"] == 0       # dividing first gives a whole number
    assert min(q["answer"] for q in bank["current"]) >= CFG["items"]["min_answer"]


def test_worked_answers_do_the_stated_arithmetic(bank):
    """Each 'x op y = z' step in a worked answer is true."""
    ops = {"×": lambda x, y: x * y, "−": lambda x, y: x - y, "+": lambda x, y: x + y, "÷": lambda x, y: x / y}
    for item in bank["current"] + list(pool_items(bank).values()):
        for work in (item["right"], item["wrong"]):
            words = work.replace(".", " ").split()
            for i, w in enumerate(words):
                if w in ops and i + 3 < len(words) and words[i + 2] == "=":
                    assert ops[w](float(words[i - 1]), float(words[i + 1])) == float(words[i + 3]), (item["id"], work)


def test_disjoint_from_pool_and_earlier_studies(bank):
    """Protocol: current questions differ from v2 and v3 and from the pool. The pool may repeat a
    v2 or v3 text (the rate template has too few new texts for 50 pool items)."""
    current = {q["text"] for q in bank["current"]}
    pool = {h["text"] for h in pool_items(bank).values()}
    assert len(current) == 100 and len(pool) == 200                        # no repeats within either
    assert not current & pool and not current & earlier_texts(CFG)


def test_number_rules(bank):
    blocked, items = fixed_numbers(CFG), pool_items(bank)
    for q in bank["current"]:
        ans = float(q["answer"])
        assert ans not in numbers_in(q["text"]) and ans not in numbers_in(attempt_text(q, CFG)) and ans not in blocked
        for phrasing in bank["assignment"][q["id"]].values():
            for h in phrasing["matched"] + phrasing["unmatched"]:
                assert ans not in history_numbers(items[h]), (q["id"], h)


def test_attempt_and_fixed_message_rules_enforced(bank):
    """With these templates the two rules never bind (every attempt number that could equal the
    answer is already in the question, and the fixed messages hold no numbers), so they are
    checked here on messages that do contain the answer."""
    q, ans = bank["current"][0], str(bank["current"][0]["answer"])
    assert is_valid_current(q, CFG, fixed_numbers(CFG))
    with_attempt = {**CFG, "prompts": {**CFG["prompts"], "attempt": "I tried: {work} Maybe " + ans + "?"}}
    assert not is_valid_current(q, with_attempt, fixed_numbers(with_attempt))
    with_ack = {**CFG, "prompts": {**CFG["prompts"], "acks": {"A": f"Okay, {ans}.", "B": "All right."}}}
    assert not is_valid_current(q, with_ack, fixed_numbers(with_ack))
    assert not fixed_numbers(CFG)                                            # the fixed messages hold no numbers


def test_history_assignment(bank):
    items = pool_items(bank)
    for q in bank["current"]:
        phrasings = bank["assignment"][q["id"]]
        assert list(phrasings) == CFG["arms"]["arm1"]["phrasings"]
        for a in phrasings.values():
            assert len(a["matched"]) == len(a["unmatched"]) == CFG["arms"]["arm1"]["history_turns"]
            assert {items[h]["family"] for h in a["matched"]} == {q["family"]}
            assert {items[h]["family"] for h in a["unmatched"]} == {a["unmatched_family"]} != {q["family"]}
        used = [h for a in phrasings.values() for h in a["matched"] + a["unmatched"]]
        assert len(used) == len(set(used))                                   # phrasings A and B never share an item
        assert phrasings["A"]["unmatched_family"] != phrasings["B"]["unmatched_family"]


def test_unmatched_families_balanced(bank):
    for family in CFG["items"]["families"]:
        counts = Counter(a["unmatched_family"] for q in bank["current"] if q["family"] == family
                         for a in bank["assignment"][q["id"]].values())
        assert sorted(counts.values()) == [16, 17, 17] and family not in counts


def test_bank_is_reproducible(bank):
    assert make_bank(CFG) == bank


def test_adding_questions_only_appends(bank):
    """Raising current_per_family (step f) appends questions; the earlier questions, the whole pool
    and every earlier history assignment stay exactly as they were."""
    raised = make_bank({**CFG, "items": {**CFG["items"], "current_per_family": 26}})
    for family in CFG["items"]["families"]:
        before = [q for q in bank["current"] if q["family"] == family]
        after = [q for q in raised["current"] if q["family"] == family]
        assert after[:25] == before and len(after) == 26
    assert raised["pool"] == bank["pool"]
    assert all(raised["assignment"][q["id"]] == bank["assignment"][q["id"]] for q in bank["current"])


def test_current_questions_capped_at_38_per_family():
    """Step f may add current questions, up to 38 per family: the rate template allows no more."""
    raised = {**CFG, "items": {**CFG["items"], "current_per_family": 38}}
    assert len(make_bank(raised)["current"]) == 152
    over = {**CFG, "items": {**CFG["items"], "current_per_family": 39}}
    with pytest.raises(ValueError, match="cap"):
        make_bank(over)
