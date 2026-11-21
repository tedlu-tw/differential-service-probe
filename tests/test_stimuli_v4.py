"""src/ds/stimuli_v4.py: Arm 1 and Arm 2 cells, their student messages, and the step e checks."""
from collections import Counter
from pathlib import Path

import pytest

from ds import load_config
from ds.items_v4 import attempt_text, make_bank
from ds.stimuli import numbers_in
from ds.stimuli_v4 import KEY, build_arm1_cells, build_arm2_cells, build_cells_v4, build_check_cells_v4, build_solve_cells_v4

ROOT = Path(__file__).parents[1]
CFG = load_config(ROOT / "configs" / "experiment_v4.yaml")
PR = CFG["prompts"]


@pytest.fixture(scope="module")
def bank():
    return make_bank(CFG)


@pytest.fixture(scope="module")
def cells(bank):
    return {"arm1": build_arm1_cells(CFG, bank), "arm2": build_arm2_cells(CFG, bank)}


def by_question(cells, arm, question_id):
    return {(c["condition"], c["phrasing"], c["prompt"]): c for c in cells[arm] if c["question_id"] == question_id}


def test_counts_and_unique_keys(cells, bank):
    assert len(cells["arm1"]) == 100 * 10 and len(cells["arm2"]) == 100 * 4
    every = cells["arm1"] + cells["arm2"]
    assert len({tuple(c[k] for k in KEY) for c in every}) == len(every)
    assert Counter(c["condition"] for c in cells["arm1"]) == {c: 200 for c in CFG["arms"]["arm1"]["conditions"]}
    assert build_cells_v4(CFG, bank) == every


def test_turn_one_ends_with_the_question_alone(cells, bank):
    text = {q["id"]: q["text"] for q in bank["current"]}
    for c in cells["arm1"] + cells["arm2"]:
        assert c["messages"][0] == {"role": "system", "content": PR["system"][c["prompt"]]}
        assert c["messages"][-1] == {"role": "user", "content": text[c["question_id"]]}
    assert {c["prompt"] for c in cells["arm1"]} == {"A"}


def test_history_turns(cells, bank):
    pool = {h["id"]: h for items in bank["pool"].values() for h in items}
    for c in cells["arm1"]:
        history = c["messages"][1:-1]
        assert [m["role"] for m in history] == ["user", "assistant"] * 2
        assert all(m["content"] == PR["acks"][c["phrasing"]] for m in history[1::2])
        assigned = bank["assignment"][c["question_id"]][c["phrasing"]]
        kind = "unmatched" if "unmatched" in c["condition"] else "matched"
        items = [pool[h] for h in assigned[kind]]
        if c["condition"] == "history_neutral":
            expected = [f"{i['text']}\n{PR['neutral_history_line']}" for i in items]
        else:
            work = "right" if c["condition"].endswith("_success") else "wrong"
            expected = [f"{i['text']}\nMy answer: {i[work]}" for i in items]
        assert [m["content"] for m in history[0::2]] == expected, (c["question_id"], c["condition"])


def test_success_and_struggle_differ_only_in_the_worked_answers(cells, bank):
    for q in bank["current"]:
        cs = by_question(cells, "arm1", q["id"])
        for kind in ("matched", "unmatched"):
            for ph in CFG["arms"]["arm1"]["phrasings"]:
                ok, bad = cs[(f"history_{kind}_success", ph, "A")], cs[(f"history_{kind}_struggle", ph, "A")]
                strip = lambda c: [m["content"].split("\nMy answer:")[0] for m in c["messages"]]
                assert strip(ok) == strip(bad) and ok["messages"] != bad["messages"]
        neutral = [m["content"] for m in cs[("history_neutral", "A", "A")]["messages"]]
        assert not any("My answer:" in t for t in neutral)                       # no correctness information


def test_followups(cells, bank):
    for c in cells["arm1"]:
        assert c["followups"] == PR["pressure"]
    wrong = {q["id"]: q for q in bank["current"]}
    for c in cells["arm2"]:
        assert len(c["followups"]) == CFG["arms"]["arm2"]["max_turns"] - 1
        if c["condition"] == "demand":
            assert c["followups"] == PR["pressure"]
        else:
            q = wrong[c["question_id"]]
            assert c["followups"][0] == attempt_text(q, CFG) and q["wrong"] in c["followups"][0]
            assert c["followups"][1:] == PR["pressure"][1:]


def test_arm2_cells_of_one_question_differ_only_where_intended(cells, bank):
    q = bank["current"][7]
    cs = by_question(cells, "arm2", q["id"])
    assert len(cs) == 4
    assert len({c["messages"][-1]["content"] for c in cs.values()}) == 1           # same turn 1
    for prompt in ("A", "B"):
        assert cs[("demand", "A", prompt)]["messages"] == cs[("after_attempt", "A", prompt)]["messages"]


def test_no_message_contains_the_answer(cells, bank):
    """The number rules end to end: no turn-1 message or scripted student message holds the current answer."""
    answer = {q["id"]: float(q["answer"]) for q in bank["current"]}
    for c in cells["arm1"] + cells["arm2"]:
        texts = [m["content"] for m in c["messages"]] + c["followups"]
        assert answer[c["question_id"]] not in numbers_in(" ".join(texts)), (c["question_id"], c["condition"])


def test_solve_cells_are_v3s_on_the_v4_questions(bank):
    solve = build_solve_cells_v4(CFG, bank["current"])
    assert len(solve) == 100 and {c["arm"] for c in solve} == {"solve"}
    c, q = solve[0], bank["current"][0]
    assert c["messages"] == [{"role": "system", "content": PR["system"]["plain"]},
                             {"role": "user", "content": f"{q['text']}\n\n{PR['solve_instruction']}"}]


def test_check_cells(bank, cells):
    checks = build_check_cells_v4(CFG, bank)
    kinds = Counter((c["kind"], c["condition"]) for c in checks)
    assert all(kinds[("check_correctness", cond)] == 200 for cond in CFG["arms"]["arm1"]["conditions"] if cond != "history_neutral")
    assert {cond for kind, cond in kinds if kind == "check_relevance"} == set(CFG["check"]["relevance_expected"])
    arm1 = {(c["question_id"], c["condition"], c["phrasing"]): c for c in cells["arm1"]}
    for c in checks:
        source = arm1[(c["question_id"], c["condition"], c["phrasing"])]
        assert c["messages"][:-1] == source["messages"][:-1] and c["followups"] == []
        last = c["messages"][-1]["content"]
        if c["kind"] == "check_correctness":
            assert last == PR["check_question"]
        else:
            assert last == f"{source['messages'][-1]['content']}\n\n{PR['relevance_question']}"


def test_subset_for_dry_run_and_pilot(bank):
    picked = bank["current"][:5]
    cells = build_cells_v4(CFG, bank, picked)
    assert len(cells) == 5 * 14 and {c["question_id"] for c in cells} == {q["id"] for q in picked}
