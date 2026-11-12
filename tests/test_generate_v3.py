"""generate_v3 without a GPU: the reply_fn wiring, one-turn prompts and resuming from the CSV.

The GPU parts (gpu_parts, run_model_v3, generate_on_gpu, main) reuse v2's generate.py and are
checked by the dry run on the pod, not here.
"""
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from ds import load_config
from ds.dialogue import run_dialogues
from ds.generate_v3 import (append_rows, dry_run_questions, load_done_rows, make_reply_fn, single_turn,
                            timing_rows)
from ds.stimuli_v3 import build_cells_v3, build_check_cells_v3, make_questions_v3

CONFIG = Path(__file__).parents[1] / "configs" / "experiment_v3.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG)


@pytest.fixture(scope="module")
def questions(cfg):
    return make_questions_v3(cfg)


@pytest.fixture(scope="module")
def answers(questions):
    return {q["id"]: q["answer"] for q in questions}


@pytest.fixture(scope="module")
def picked(questions):
    # One two-step and one linear question: the fake model answers only the linear one, at turn 2.
    return [questions[0], questions[100]]


@pytest.fixture(scope="module")
def cells(cfg, picked):
    return build_cells_v3(cfg, picked)


def fake_parts(questions, cfg, seen=None, timings=None):
    """Template text is every message joined. The 'model' answers Maya questions once asked again (turn 2)."""
    def render(messages):
        if seen is not None:
            seen.append(messages)
        return "\n".join(m["content"] for m in messages)

    def generate(prompts):
        replies = []
        for p in prompts:
            q = next(q for q in questions if q["text"] in p)
            asked_again = cfg["prompts"]["followups"][0] in p
            give = asked_again and q["family"] == "linear_equation"
            replies.append(f"Maya's number is {q['answer']}." if give else "Hint: what do you get?")
        return replies, [False] * len(prompts)
    log = (lambda batch, s: timings.append((len(batch), batch[0]["turn"]))) if timings is not None else (lambda b, s: None)
    return make_reply_fn(lambda m: m, render, generate, log)


def test_module_imports_without_torch():
    assert "torch" not in sys.modules and "transformers" not in sys.modules


def test_reply_fn_wiring(cfg, cells, answers, picked):
    seen, timings = [], []
    rows = run_dialogues(cells, 0, fake_parts(picked, cfg, seen, timings), answers, cfg)
    assert Counter(r["turn"] for r in rows)[1] == len(cells)
    turns = pd.DataFrame(rows).groupby("question_id").turn.max().to_dict()
    assert turns == {picked[0]["id"]: 3, picked[1]["id"]: 2}               # stops when the answer is given
    assert sum(n for n, _ in timings) == len(rows)                     # every reply is in a timed batch
    assert all(n <= cfg["generation"]["batch_size"] for n, _ in timings)
    assert any(m[-1]["content"] == cfg["prompts"]["followups"][0] for m in seen)


def test_prepare_applied_before_render(cfg, cells, answers):
    seen = []
    fold = lambda messages: [{"role": "user", "content": "folded"}] + messages[1:]
    fn = make_reply_fn(fold, lambda m: seen.append(m) or "x", lambda p: (["r"] * len(p), [True] * len(p)),
                       lambda b, s: None)
    out = fn([{"messages": cells[0]["messages"], "seed": 0, "turn": 1}])
    assert out == [("r", True)] and seen[0][0]["content"] == "folded"


def test_single_turn_prompts(cfg, questions, answers):
    check = build_check_cells_v3(cfg, questions[:2])
    fn = make_reply_fn(lambda m: m, lambda m: "x", lambda p: (["Yes, both."] * len(p), [False] * len(p)),
                       lambda b, s: None)
    rows = run_dialogues(check, 0, fn, answers, single_turn(cfg))
    assert len(rows) == len(check) and {r["turn"] for r in rows} == {1}
    assert cfg["design"]["max_turns"] == 3                             # the original config is not changed


def test_dry_run_questions_cover_families(cfg, questions):
    picked = dry_run_questions(cfg, questions)
    assert len(picked) == 5 and {q["family"] for q in picked} == {"two_step", "percentage", "linear_equation", "rate"}


def test_progress_bar_counts_dialogues():
    from ds.generate_v3 import progress_bar
    bar = progress_bar("org/Model-8B", "pilot", 0)(2, 10)
    bar.update(8)
    assert bar.n == 8 and bar.total == 10 and bar.desc.startswith("Model-8B pilot seed=0 turn=2")
    bar.close()


def test_timing_rows():
    batch = [{"seed": 1, "turn": 2}] * 3
    assert timing_rows("m", "pilot", batch, 1.23456) == [{"run": "pilot", "model": "m", "seed": 1, "turn": 2,
                                                           "batch_size": 3, "seconds": 1.235}]


def test_resume_from_csv(cfg, cells, answers, picked, tmp_path):
    out = tmp_path / "replies.csv"
    fn = fake_parts(picked, cfg)
    full = run_dialogues(cells, 0, fn, answers, cfg)

    batches = []
    def dies_after_four(batch):
        batches.append(batch)
        if len(batches) > 4:
            raise RuntimeError("pod stopped")
        return fn(batch)

    extra = {"model": "m", "system_mode": "system_role"}
    with pytest.raises(RuntimeError):
        run_dialogues(cells, 0, dies_after_four, answers, cfg, on_rows=lambda r: append_rows(out, r), extra=extra)
    append_rows(out, [{**extra, "question_id": "x", "prompt": "plain", "condition": "neutral", "phrasing": "A",
                       "seed": 0, "turn": 1, "reply": "None", "truncated": False, "gives_answer": 0,
                       "model": "other"}])
    done = load_done_rows(out, "m")
    assert 0 < len(done) < len(full) and {r["model"] for r in done} == {"m"}
    run_dialogues(cells, 0, fn, answers, cfg, done_rows=done, on_rows=lambda r: append_rows(out, r), extra=extra)

    logged = pd.read_csv(out, keep_default_na=False)
    mine = logged[logged.model == "m"]
    key = ["question_id", "prompt", "condition", "phrasing", "turn", "reply"]
    assert sorted(map(tuple, mine[key].values)) == sorted(tuple(r[k] for k in key) for r in full)
    assert run_dialogues(cells, 0, fn, answers, cfg, done_rows=load_done_rows(out, "m")) == []
    assert [r["reply"] for r in load_done_rows(out, "other")] == ["None"]  # read back as text, not missing


def test_other_seed_resumes_separately(cfg, cells, answers, picked, tmp_path):
    out, fn = tmp_path / "replies.csv", fake_parts(picked, cfg)
    append_rows(out, run_dialogues(cells, 0, fn, answers, cfg, extra={"model": "m"}))
    assert len(run_dialogues(cells, 1, fn, answers, cfg, done_rows=load_done_rows(out, "m"))) > 0
