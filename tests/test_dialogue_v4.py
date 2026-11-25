"""src/ds/dialogue_v4.py: the Arm 1 and Arm 2 loop, and Arm 3 through v3's run_dialogues."""
import json
import os
import random
from pathlib import Path

import pytest

from ds import load_config
from ds.dialogue import run_dialogues
from ds.dialogue_v4 import KEY, dialogue_key, run_arm3, run_dialogues_v4, with_raw
from ds.items_v4 import attempt_text, make_bank
from ds.mock import mock_reply_fn, v3_setting
from ds.models_v4 import GenerationFailure
from ds.generate_v3 import single_turn
from ds.stimuli_v3 import build_cells_v3, build_solve_cells_v3, make_questions_v3
from ds.stimuli_v4 import build_cells_v4

ROOT = Path(__file__).parents[1]
CFG = load_config(ROOT / "configs" / "experiment_v4.yaml")
V3 = load_config(ROOT / "configs" / "experiment_v3.yaml")
NOLOG = (os.devnull, {"model": "fake", "run": "arm1"})        # failures are tested separately
ARM3LOG = (os.devnull, {"model": "fake", "run": "arm3"})
MAX_TURNS, BATCH = CFG["arms"]["arm1"]["max_turns"], CFG["generation"]["batch_size"]


@pytest.fixture(scope="module")
def bank():
    return make_bank(CFG)


@pytest.fixture(scope="module")
def setup(bank):
    questions = bank["current"][::25]                          # one question per family
    return build_cells_v4(CFG, bank, questions), {q["id"]: q["answer"] for q in bank["current"]}, \
        {q["id"]: q for q in bank["current"]}


class Planned:
    """A fake model: each dialogue gives the answer at its planned turn (1 to 5, or never).
    Records every state it was asked about. raw_reply names the dialogue and turn it belongs to."""
    def __init__(self, answers, plan):
        self.answers, self.plan, self.seen = answers, plan, []

    def __call__(self, batch):
        self.seen += batch
        out = []
        for s in batch:
            key = dialogue_key(s)
            gives = self.plan(key) == s["turn"]
            reply = f"The answer is {self.answers[s['question_id']]}." if gives else f"Try again ({s['turn']})."
            out.append((reply, False, f"RAW|{'|'.join(key)}|{s['turn']}"))
        return out


def plan_by_hash(key):
    return [1, 2, 3, 4, 5, None][sum(map(ord, "".join(key))) % 6]


def test_followups_come_only_from_the_cell(setup, bank):
    cells, answers, questions = setup
    marked = [{**c, "followups": [f"{c['condition']}-{t}" for t in range(2, 6)]} if c["condition"] != "after_attempt"
              else c for c in cells]
    fake = Planned(answers, lambda key: None)                   # never gives in: every turn is asked
    run_dialogues_v4(marked, 0, fake, answers, MAX_TURNS, BATCH, failure_log=NOLOG)
    for s in fake.seen:
        later_users = [m["content"] for m in s["messages"] if m["role"] == "user"][-(s["turn"] - 1):] if s["turn"] > 1 else []
        cell = next(c for c in marked if all(c[k] == s[k] for k in KEY if k != "seed"))
        assert later_users == cell["followups"][:s["turn"] - 1]
        assert not set(later_users) & set(CFG["prompts"]["pressure"]) or s["condition"] == "after_attempt"
        if s["turn"] >= 2:
            turn2 = later_users[0]
            if s["condition"] == "after_attempt":
                assert turn2 == attempt_text(questions[s["question_id"]], CFG)
            else:
                assert "I tried it myself" not in turn2


def test_stop_rule_and_five_turn_limit(setup):
    cells, answers, _ = setup
    rows = run_dialogues_v4(cells, 0, Planned(answers, plan_by_hash), answers, MAX_TURNS, BATCH, failure_log=NOLOG)
    by_dialogue = {}
    for r in rows:
        by_dialogue.setdefault(dialogue_key(r), []).append(r)
    assert len(by_dialogue) == len(cells)
    for key, rs in by_dialogue.items():
        turns, planned = [r["turn"] for r in rs], plan_by_hash(key)
        assert turns == list(range(1, len(rs) + 1)) and len(rs) <= MAX_TURNS
        assert len(rs) == (planned or MAX_TURNS)                 # stops at the first answer; never gives in -> 5
        assert [r["gives_answer"] for r in rs] == [False] * (len(rs) - 1) + [planned is not None]


def test_resume_is_per_arm_and_reproduces_a_full_run(setup):
    cells, answers, _ = setup
    full = run_dialogues_v4(cells, 0, Planned(answers, plan_by_hash), answers, MAX_TURNS, BATCH, failure_log=NOLOG)
    part = [r for r in full if r["turn"] <= 2]
    resumed = part + run_dialogues_v4(cells, 0, Planned(answers, plan_by_hash), answers, MAX_TURNS, BATCH, done_rows=part, failure_log=NOLOG)
    order = lambda rs: sorted(rs, key=lambda r: (dialogue_key(r), r["turn"]))
    assert order(resumed) == order(full)
    twin = {**cells[0], "arm": "arm2" if cells[0]["arm"] == "arm1" else "arm1"}   # same fields, other arm
    done_other_arm = [r for r in full if dialogue_key(r)[1:] == dialogue_key({**cells[0], "seed": 0})[1:]
                      and r["arm"] == cells[0]["arm"]]
    new = run_dialogues_v4([twin], 0, Planned(answers, plan_by_hash), answers, MAX_TURNS, BATCH, done_rows=done_other_arm, failure_log=NOLOG)
    assert new and new[0]["turn"] == 1 and new[0]["arm"] == twin["arm"]    # the other arm's rows are not its own


def test_raw_replies_aligned_through_batching_by_length(setup):
    cells, answers, _ = setup
    shuffled = cells[:]
    random.Random(3).shuffle(shuffled)
    rows = run_dialogues_v4(shuffled, 1, Planned(answers, plan_by_hash), answers, MAX_TURNS, 3, failure_log=NOLOG)
    for r in rows:
        assert r["raw_reply"] == f"RAW|{'|'.join(dialogue_key(r))}|{r['turn']}"


def test_failed_batch_logged_before_the_run_stops(setup, tmp_path):
    cells, answers, _ = setup
    path = tmp_path / "failures.jsonl"

    def failing(batch):
        if batch[0]["turn"] == 2:
            raise GenerationFailure("non-empty thinking block in a reply", [f"raw {i}" for i in range(len(batch))])
        return Planned(answers, lambda key: None)(batch)
    with pytest.raises(GenerationFailure):
        run_dialogues_v4(cells, 0, failing, answers, MAX_TURNS, BATCH, failure_log=(path, {"model": "m", "run": "dry_run"}))
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines and {r["turn"] for r in lines} == {2} and all(r["model"] == "m" and r["run"] == "dry_run" for r in lines)
    assert [r["raw_reply"] for r in lines] == [f"raw {i}" for i in range(len(lines))]
    assert all({"arm", "question_id", "prompt", "condition", "phrasing", "seed", "error"} <= set(r) for r in lines)


def test_failure_inside_arm3_is_logged_with_the_arm(tmp_path):
    arm3 = CFG["arm3"]
    questions = make_questions_v3(arm3)
    cells = build_cells_v3(arm3, questions[::50])
    answers = {q["id"]: q["answer"] for q in questions}
    path = tmp_path / "failures.jsonl"

    def failing(batch):
        raise GenerationFailure("a thinking block is opened and never closed", [f"raw {i}" for i in range(len(batch))])
    with pytest.raises(GenerationFailure):
        run_arm3(cells, 2, failing, answers, arm3, {}, failure_log=(path, {"model": "Qwen/Qwen3-8B", "run": "arm3"}))
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines and all(r["arm"] == "arm3" and r["model"] == "Qwen/Qwen3-8B" and r["run"] == "arm3" and r["seed"] == 2
                         for r in lines)
    assert all({"question_id", "prompt", "condition", "phrasing", "turn", "raw_reply", "error"} <= set(r) for r in lines)


def test_rows_labelled_by_run(setup):
    """Every row names its run; v3_solve rows (through run_arm3) are not labelled Arm 3."""
    cells, answers, _ = setup
    rows = run_dialogues_v4(cells, 0, Planned(answers, plan_by_hash), answers, MAX_TURNS, BATCH,
                            failure_log=(os.devnull, {"model": "fake", "run": "pilot"}))
    assert {r["run"] for r in rows} == {"pilot"} and {r["arm"] for r in rows} == {"arm1", "arm2"}
    arm3 = CFG["arm3"]
    questions = make_questions_v3(arm3)
    solve = build_solve_cells_v3(single_turn(arm3), questions[:6])
    stash = {}

    def solver(batch):
        stash["raw"] = ["Working. 42" for _ in batch]
        return [("Working. 42", False) for _ in batch]
    rows = run_arm3(solve, 0, solver, {q["id"]: q["answer"] for q in questions}, single_turn(arm3), stash,
                    failure_log=(os.devnull, {"model": "fake", "run": "v3_solve"}))
    assert len(rows) == 6 and all(r["run"] == "v3_solve" and r["arm"] != "arm3" for r in rows)
    for bad in ({"model": "fake"}, {"model": "fake", "run": "Arm3"}, {"model": "fake", "run": "full"}):
        with pytest.raises(ValueError, match="run"):
            run_dialogues_v4(cells, 0, Planned(answers, plan_by_hash), answers, MAX_TURNS, BATCH, failure_log=(os.devnull, bad))
        with pytest.raises(ValueError, match="run"):
            run_arm3(solve, 0, solver, {}, single_turn(arm3), stash, failure_log=(os.devnull, bad))


def test_arm3_label_refuses_solve_cells():
    arm3 = CFG["arm3"]
    questions = make_questions_v3(arm3)
    solve = build_solve_cells_v3(single_turn(arm3), questions[:2])
    for run in ("arm3", "dry_run"):
        with pytest.raises(ValueError, match="solve cells"):
            run_arm3(solve, 0, lambda batch: [], {}, single_turn(arm3), {}, failure_log=(os.devnull, {"model": "m", "run": run}))


def test_with_raw_takes_the_stash_of_the_same_batch():
    stash = {}

    def v3_style(batch):
        stash["raw"] = [f"raw {s['n']}" for s in batch]
        return [(f"reply {s['n']}", False) for s in batch]
    fn = with_raw(v3_style, stash)
    assert fn([{"n": 1}, {"n": 2}]) == [("reply 1", False, "raw 1"), ("reply 2", False, "raw 2")]
    assert "raw" not in stash                                     # used once
    with pytest.raises(RuntimeError, match="raw replies"):
        with_raw(lambda batch: [("r", False)], stash)([{"n": 3}])    # nothing generated into the stash


def test_arm3_rows_equal_v3_code_path_on_the_mock():
    arm3 = CFG["arm3"]
    questions = make_questions_v3(arm3)
    picked = questions[::40]
    cells = build_cells_v3(arm3, picked)
    answers = {q["id"]: q["answer"] for q in questions}
    params, extra = v3_setting(V3, "planted"), {"model": "mock-planted", "system_mode": "mock"}
    v3_rows = run_dialogues(cells, 0, mock_reply_fn(questions, params, arm3), answers, arm3, extra=extra)
    stash, mock = {}, mock_reply_fn(questions, params, arm3)

    def stashing(batch):
        out = mock(batch)
        stash["raw"] = [reply for reply, _ in out]
        return out
    v4_rows = run_arm3(cells, 0, stashing, answers, arm3, stash, extra=extra, failure_log=ARM3LOG)
    assert [{k: v for k, v in r.items() if k not in ("raw_reply", "arm", "run")} for r in v4_rows] == v3_rows
    assert all(r["arm"] == "arm3" and r["run"] == "arm3" and r["raw_reply"] == r["reply"] for r in v4_rows)
    assert len(v3_rows) > len(cells)                              # some dialogues went past turn 1
