from collections import Counter
from pathlib import Path

import pytest

from ds import load_config
from ds.dialogue import group_replies, run_dialogues
from ds.stimuli_v3 import build_cells_v3, make_questions_v3

CONFIG = Path(__file__).parents[1] / "configs" / "experiment_v3.yaml"

# Turn at which the fake tutor gives the answer, per condition; None = never.
GIVE_IN = {"history_struggle": 1, "low_score": 2, "high_score": 3}


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG)


@pytest.fixture(scope="module")
def questions(cfg):
    return make_questions_v3(cfg)[:3]


@pytest.fixture(scope="module")
def cells(cfg, questions):
    return build_cells_v3(cfg, questions)


@pytest.fixture(scope="module")
def answers(questions):
    return {q["id"]: q["answer"] for q in questions}


def fake_reply_fn(answers, calls=None):
    """Deterministic tutor: the reply depends only on the dialogue and the turn."""
    def reply_fn(batch):
        if calls is not None:
            calls.append(batch)
        out = []
        for s in batch:
            if GIVE_IN.get(s["condition"]) == s["turn"]:
                out.append((f"So the answer is {answers[s['question_id']]}.", False))
            else:
                out.append((f"Hint {s['turn']} for {s['question_id']}: what do you get?", s["turn"] == 3))
        return out
    return reply_fn


def run(cells, answers, cfg, seed=0, **kw):
    return run_dialogues(cells, seed, fake_reply_fn(answers), answers, cfg, **kw)


def turns_per_dialogue(rows):
    return Counter((r["question_id"], r["prompt"], r["condition"], r["phrasing"]) for r in rows)


# ---------- stopping ----------

def test_stops_at_first_answer(cells, answers, cfg):
    counts = turns_per_dialogue(run(cells, answers, cfg))
    for (_, _, condition, _), n in counts.items():
        assert n == GIVE_IN.get(condition, 3), condition
    assert len(counts) == len(cells)


def test_gives_answer_logged_on_last_turn_only(cells, answers, cfg):
    rows = run(cells, answers, cfg)
    for r in rows:
        assert r["gives_answer"] == int(GIVE_IN.get(r["condition"]) == r["turn"])


def test_answer_inside_other_number_does_not_stop(cells, answers, cfg):
    # The v2 rule: 1270 is not 127. The dialogue must go on to turn 3.
    reply_fn = lambda batch: [(f"Try {answers[s['question_id']]}0 first.", False) for s in batch]
    rows = run_dialogues(cells[:1], 0, reply_fn, answers, cfg)
    assert [r["turn"] for r in rows] == [1, 2, 3] and not any(r["gives_answer"] for r in rows)


# ---------- messages ----------

def test_turn_messages(cells, answers, cfg):
    calls = []
    run_dialogues(cells, 0, fake_reply_fn(answers, calls), answers, cfg)
    cell = next(c for c in cells if c["condition"] == "neutral")
    seen = {s["turn"]: s["messages"] for batch in calls for s in batch
            if (s["question_id"], s["prompt"], s["condition"]) == (cell["question_id"], cell["prompt"], "neutral")}
    first, second = cfg["prompts"]["followups"]
    assert seen[1] == cell["messages"]
    assert seen[2][:-2] == cell["messages"] and seen[2][-1] == {"role": "user", "content": first}
    assert seen[2][-2]["role"] == "assistant" and seen[2][-2]["content"].startswith("Hint 1")
    assert seen[3][:-2] == seen[2] and seen[3][-1] == {"role": "user", "content": second}


def test_followups_same_in_every_condition(cells, answers, cfg):
    calls = []
    run_dialogues(cells, 0, fake_reply_fn(answers, calls), answers, cfg)
    late = [s["messages"][-1] for batch in calls for s in batch if s["turn"] == 3]
    assert {m["content"] for m in late} == {cfg["prompts"]["followups"][1]}


# ---------- batching ----------

def test_batches_one_turn_at_a_time(cells, answers, cfg):
    calls = []
    run_dialogues(cells, 0, fake_reply_fn(answers, calls), answers, cfg)
    assert all(len(b) <= cfg["generation"]["batch_size"] for b in calls)
    turns = [b[0]["turn"] for b in calls]
    assert all(len({s["turn"] for s in b}) == 1 for b in calls)
    assert turns == sorted(turns)                  # every turn-1 reply before any turn-2 reply


def test_reply_count_must_match(cells, answers, cfg):
    with pytest.raises(ValueError, match="returned"):
        run_dialogues(cells, 0, lambda batch: [("x", False)], answers, cfg)


# ---------- logging and resume ----------

def test_row_fields(cells, answers, cfg):
    rows = run(cells, answers, cfg, seed=1, extra={"model": "m", "system_mode": "system"})
    assert set(rows[0]) == {"model", "system_mode", "question_id", "prompt", "condition", "phrasing", "seed",
                            "turn", "reply", "truncated", "gives_answer"}
    assert {r["seed"] for r in rows} == {1} and {r["model"] for r in rows} == {"m"}


def test_rows_reconstruct_every_dialogue(cells, answers, cfg):
    rows = run(cells, answers, cfg)
    replies = group_replies(rows)
    assert all(len(v) == turns_per_dialogue(rows)[k[:4]] for k, v in replies.items())


def test_resume_after_interrupt(cells, answers, cfg):
    full = run(cells, answers, cfg)
    saved, calls = [], []

    def dies_after_three(batch):
        calls.append(batch)
        if len(calls) > 3:
            raise RuntimeError("pod stopped")
        return fake_reply_fn(answers)(batch)

    with pytest.raises(RuntimeError):
        run_dialogues(cells, 0, dies_after_three, answers, cfg, on_rows=saved.extend)
    assert 0 < len(saved) < len(full)
    resumed = []
    rest = run_dialogues(cells, 0, fake_reply_fn(answers, resumed), answers, cfg, done_rows=saved)
    # After the interrupt, some dialogues wait at turn 1 and others at turn 2: still one turn per batch, lowest first.
    assert {s["turn"] for b in resumed for s in b} >= {1, 2}
    assert all(len({s["turn"] for s in b}) == 1 for b in resumed)
    assert [b[0]["turn"] for b in resumed] == sorted(b[0]["turn"] for b in resumed)
    key = lambda r: (r["question_id"], r["prompt"], r["condition"], r["phrasing"], r["turn"])
    assert sorted(map(key, saved + rest)) == sorted(map(key, full))
    assert run(cells, answers, cfg, done_rows=full) == []      # nothing left to do


def test_other_seed_not_treated_as_done(cells, answers, cfg):
    seed0 = run(cells, answers, cfg, seed=0)
    assert len(run(cells, answers, cfg, seed=1, done_rows=seed0)) == len(seed0)


def test_damaged_log_refused(cells, answers, cfg):
    rows = [r for r in run(cells, answers, cfg) if r["turn"] != 1]
    with pytest.raises(ValueError, match="turn"):
        run(cells, answers, cfg, done_rows=rows)


def test_followups_must_match_turns(cells, answers, cfg):
    bad = {**cfg, "design": {**cfg["design"], "max_turns": 4}}
    with pytest.raises(ValueError, match="followups"):
        run(cells, answers, bad)
