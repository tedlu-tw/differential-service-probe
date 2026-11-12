from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ds import load_config
from ds.dialogue import run_dialogues
from ds.mock import (dialogue_plan, mock_reply_fn, run_mock_check_v3, run_mock_dialogues, run_mock_solve_v3,
                     sensitivity_params, sensitivity_rows_v3, shift_up, v3_setting)
from ds.score import score_check_rows_v3, score_rows_v3, score_solve_rows_v3
from ds.stimuli_v3 import build_cells_v3, build_check_cells_v3, build_solve_cells_v3, hint_ignore_numbers, \
    make_questions_v3

CONFIG = Path(__file__).parents[1] / "configs" / "experiment_v3.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG)


@pytest.fixture(scope="module")
def questions(cfg):
    return make_questions_v3(cfg)


@pytest.fixture(scope="module")
def cells(cfg, questions):
    return build_cells_v3(cfg, questions)


def scored(rows, questions, cfg):
    return pd.DataFrame(score_rows_v3(rows, questions, cfg["scoring"], hint_ignore_numbers(cfg)))


@pytest.fixture(scope="module")
def planted(cfg, questions, cells):
    rows = run_mock_dialogues(cells, questions, cfg, v3_setting(cfg, "planted"), "mock-planted",
                              cfg["seeds"]["generation"])
    return scored(rows, questions, cfg)


def by_dialogue(df):
    keys = ["question_id", "prompt", "condition", "phrasing", "seed"]
    first = df[df.turn == 1].set_index(keys)
    last = df.groupby(keys).agg(answer_turn=("turn", "max"), gave=("gives_answer", "max"))
    out = first.join(last)
    out["gave_by_turn2"] = ((out.gave == 1) & (out.answer_turn <= 2)).astype(int)
    return out.reset_index()


def expected_rate(p):
    return p["turn1_levels"][3] + (1 - p["turn1_levels"][3]) * p["later_give_in"][0]


def mean_of(probs):
    return float(np.dot(range(len(probs)), probs))


# ---------- the scorer reads what the mock planned ----------

def test_scorer_reads_planned_level_and_steps(planted, cfg):
    params = v3_setting(cfg, "planted")
    for row in planted[planted.turn == 1].itertuples():
        plan = dialogue_plan(row._asdict(), params, cfg, run=0)
        assert (row.hint_level, row.steps) == (plan["level"], plan["steps"]), row.question_id


def test_answer_turn_matches_plan(planted, cfg):
    params = v3_setting(cfg, "planted")
    for row in by_dialogue(planted).itertuples():
        plan = dialogue_plan(row._asdict(), params, cfg, run=0)
        assert (row.answer_turn if row.gave else None) == plan["answer_turn"]


def test_mock_replies_show_no_other_measure(planted):
    # Planted effects must come only from the planned level, steps and give-in turn.
    assert planted.praises.sum() == 0 and planted.encourages.sum() == 0 and planted.mentions_history.sum() == 0


# ---------- planted and no-effect rates ----------

def test_planted_rates(planted, cfg):
    params, d = v3_setting(cfg, "planted"), by_dialogue(planted)
    ped = d[d.prompt == "pedagogical"]
    for condition in ("history_struggle", "history_success", "low_score", "high_score", "neutral"):
        p = params.get(condition, params["default"])
        sub = ped[ped.condition == condition]
        assert sub.gave_by_turn2.mean() == pytest.approx(expected_rate(p), abs=0.07), condition
        assert sub.hint_level.mean() == pytest.approx(mean_of(p["turn1_levels"]), abs=0.15), condition
        assert sub.steps.mean() == pytest.approx(mean_of(p["steps"]), abs=0.15), condition


def test_no_effect_rates_equal(cfg, questions, cells):
    rows = run_mock_dialogues(cells, questions, cfg, v3_setting(cfg, "no_effect"), "m", cfg["seeds"]["generation"])
    d = by_dialogue(scored(rows, questions, cfg))
    rates = d[d.prompt == "pedagogical"].groupby("condition").gave_by_turn2.mean()
    assert rates.max() - rates.min() < 0.12
    assert rates.mean() == pytest.approx(expected_rate(v3_setting(cfg, "no_effect")["default"]), abs=0.05)


# ---------- determinism and resume ----------

def test_batch_size_does_not_change_replies(cfg, questions, cells):
    fn, answers = mock_reply_fn(questions, v3_setting(cfg, "planted"), cfg), {q["id"]: q["answer"] for q in questions}
    small = {**cfg, "generation": {**cfg["generation"], "batch_size": 1}}
    key = lambda r: (r["question_id"], r["prompt"], r["condition"], r["phrasing"], r["turn"], r["reply"])
    a = run_dialogues(cells[:60], 0, fn, answers, cfg)
    b = run_dialogues(cells[:60], 0, fn, answers, small)
    assert sorted(map(key, a)) == sorted(map(key, b))


def test_resumed_mock_run_identical(cfg, questions, cells):
    fn, answers = mock_reply_fn(questions, v3_setting(cfg, "planted"), cfg), {q["id"]: q["answer"] for q in questions}
    full = run_dialogues(cells[:60], 0, fn, answers, cfg)
    part = full[: len(full) // 3]
    rest = run_dialogues(cells[:60], 0, fn, answers, cfg, done_rows=part)
    key = lambda r: (r["question_id"], r["prompt"], r["condition"], r["phrasing"], r["turn"], r["reply"])
    assert sorted(map(key, part + rest)) == sorted(map(key, full))


@pytest.mark.parametrize("other", [{"seed": 1, "run": 0}, {"seed": 0, "run": 1}])
def test_seeds_and_runs_differ(cfg, cells, other):
    # The same cells under another seed (or another simulated run) must be fresh draws.
    params = v3_setting(cfg, "no_effect")
    plan = lambda c, seed, run: tuple(dialogue_plan({**c, "seed": seed}, params, cfg, run).values())
    same = np.mean([plan(c, 0, 0) == plan(c, other["seed"], other["run"]) for c in cells[:400]])
    assert same < 0.3                      # chance agreement of two independent plans is about 0.1
    assert plan(cells[0], 0, 0) == plan(cells[0], 0, 0)


# ---------- sensitivity parameters ----------

@pytest.mark.parametrize("size", [0.05, 0.10, 0.20])
def test_gave_by_turn2_planted_exactly(cfg, size):
    params = sensitivity_params(cfg, "gave_by_turn2", size)
    base = cfg["mock"]["sensitivity"]["gave_by_turn2"]["base"]
    assert expected_rate(params["default"]) == pytest.approx(base)
    assert expected_rate(params["history_struggle"]) == pytest.approx(base + size)


@pytest.mark.parametrize("measure,key", [("hint_level", "turn1_levels"), ("steps", "steps")])
@pytest.mark.parametrize("size", [0.1, 0.2, 0.4])
def test_mean_shift_planted_exactly(cfg, measure, key, size):
    params = sensitivity_params(cfg, measure, size)
    assert sum(params["history_struggle"][key]) == pytest.approx(1)
    assert mean_of(params["history_struggle"][key]) - mean_of(params["default"][key]) == pytest.approx(size)


def test_steps_beyond_three(cfg, questions, cells):
    """Real tutors number up to ten steps (pilot): the mock plans and the scorer reads any count."""
    long_steps = [0.42, 0.0, 0.09, 0.20, 0.17, 0.04, 0.06, 0.0, 0.01, 0.0, 0.01]
    params = {"default": {**v3_setting(cfg, "no_effect")["default"], "steps": long_steps}}
    rows = run_mock_dialogues(cells[:400], questions, cfg, params, "mock-long", [0])
    first = scored(rows, questions, cfg).query("turn == 1")
    assert first.steps.max() > 3
    for row in first.itertuples():
        assert row.steps == dialogue_plan(row._asdict(), params, cfg, run=0)["steps"]
    assert mean_of(shift_up(long_steps, 0.4)) - mean_of(long_steps) == pytest.approx(0.4)


def test_shift_too_large_refused():
    with pytest.raises(ValueError):
        shift_up([0.0, 0.0, 0.1, 0.9], 0.4)


def test_sensitivity_rows_primary_only(cfg, questions, cells):
    rows = pd.DataFrame(sensitivity_rows_v3(cells, questions, cfg, "steps", 0.4, run=0))
    assert set(rows.condition) == {"history_struggle", "history_success"} and set(rows.prompt) == {"pedagogical"}
    first = rows[rows.turn == 1].groupby("condition").steps.mean()
    assert first["history_struggle"] - first["history_success"] == pytest.approx(0.4, abs=0.15)


# ---------- solve and check ----------

def test_mock_solve_parses(cfg, questions):
    rows = run_mock_solve_v3(build_solve_cells_v3(cfg, questions), questions, cfg, "m", [0])
    out = pd.DataFrame(score_solve_rows_v3(rows, questions))
    assert out.solve_parsed.all() and out.solve_correct.mean() == pytest.approx(0.9, abs=0.06)


def test_mock_check_parses_both_formats(cfg, questions):
    rows = run_mock_check_v3(build_check_cells_v3(cfg, questions), cfg, "m", [0])
    out = pd.DataFrame(score_check_rows_v3(rows, cfg["check"]["expected"], cfg["scoring"]["check_parser"]))
    assert out.parsed.all() and set(out.check_rule) == {"combined", "yes_no"}
    accuracy = pd.concat([out.item1_correct, out.item2_correct]).mean()
    assert accuracy == pytest.approx(cfg["mock"]["check_accuracy"], abs=0.05)
