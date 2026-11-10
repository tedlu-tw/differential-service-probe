from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ds import load_config
from ds.analyze import (bootstrap_ci, check_report, diagnostics, excludes_zero, key_measures,
                        paired_diffs, pilot_report, question_means, sensitivity, smallest_detectable,
                        summarize)
from ds.mock import run_mock, sensitivity_run, setting_probs
from ds.score import score_rows
from ds.stimuli import build_cells, make_questions

CONFIG = Path(__file__).parents[1] / "configs" / "experiment.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG)


@pytest.fixture(scope="module")
def questions(cfg):
    return make_questions(cfg)


@pytest.fixture(scope="module")
def cells(cfg):
    return build_cells(cfg)


@pytest.fixture(scope="module")
def scored(cfg, questions, cells):
    """Both mock settings in one frame, as two 'models'."""
    rows = []
    for setting in ("planted", "no_effect"):
        rows += score_rows(run_mock(cells, questions, cfg, setting_probs(cfg, setting), f"mock-{setting}"),
                           questions, cfg["scoring"])
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def summary(scored, cfg):
    return summarize(scored, cfg)


@pytest.fixture(scope="module")
def sens_table(cfg, questions, cells):
    return sensitivity(lambda diff, run: sensitivity_run(cells, questions, cfg, diff, run), cfg)


def pick(summary, model, role):
    rows = summary[(summary.model == model) & (summary.role == role)]
    assert len(rows) == 1
    return rows.iloc[0]


# ---------- the mock checks ----------

@pytest.mark.parametrize("role", ["primary", "control"])
def test_planted_effect_recovered(summary, role):
    row = pick(summary, "mock-planted", role)
    assert row.ci_low > 0                      # detected
    assert row.ci_low < 0.30 < row.ci_high     # the planted 0.60 - 0.30 is inside the interval


@pytest.mark.parametrize("role", ["primary", "control"])
def test_no_effect_mean_near_zero(summary, role):
    assert abs(pick(summary, "mock-no_effect", role).mean_diff) < 0.1


@pytest.mark.parametrize("role", ["primary", "control"])
def test_no_effect_false_positive_rate(cfg, questions, cells, role):
    # One draw excludes zero about 1 time in 20 by chance, so the no-effect check is
    # a rate over many runs, not a single interval.
    c = cfg["analysis"][role]
    keep = [x for x in cells if x["condition"] in (c["a"], c["b"])]
    probs = setting_probs(cfg, "no_effect")
    hits = [excludes_zero(pd.DataFrame(score_rows(run_mock(keep, questions, cfg, probs, "m", run),
                                                  questions, cfg["scoring"])),
                          c["a"], c["b"], c["measure"], cfg)
            for run in range(100)]
    assert np.mean(hits) <= 0.10               # about 5% expected for a 95% interval


def test_resamples_questions_not_replies(summary, scored, cfg):
    row = pick(summary, "mock-planted", "primary")
    assert row.n_questions == 100
    means = question_means(scored, "gives_answer").loc["mock-planted"]
    diffs = means["history_struggle"] - means["history_success"]
    expected = bootstrap_ci(diffs, cfg["analysis"]["n_boot"], cfg["analysis"]["ci"], cfg["seeds"]["bootstrap"])
    assert (row.mean_diff, row.ci_low, row.ci_high) == pytest.approx(expected)


def test_roles_and_shape(summary, cfg):
    n_contrasts = 2 + (len(cfg["cues"]) - 1)
    contrast_sets = []
    for model, rows in summary.groupby("model"):
        assert len(rows) == n_contrasts * len(cfg["analysis"]["measures"])
        assert rows.role.value_counts().to_dict() == {"exploratory": len(rows) - 2, "primary": 1, "control": 1}
        assert set(rows[rows.role != "exploratory"].measure) == {"gives_answer"}
        contrast_sets.append(set(zip(rows.contrast, rows.measure)))
    # Every model gets every contrast: none is left out or singled out.
    assert all(s == contrast_sets[0] for s in contrast_sets)


# ---------- sensitivity ----------

def test_sensitivity_table(sens_table, cfg):
    sens = cfg["mock"]["sensitivity"]
    assert list(sens_table["diff"]) == sens["diffs"]
    assert (sens_table["runs"] == sens["runs"]).all()
    shares = list(sens_table["share_excluding_zero"])
    assert shares == sorted(shares)            # larger planted differences are found more often
    assert shares[-1] >= sens["power"]         # 0.20 is detectable


def test_smallest_detectable():
    table = pd.DataFrame({"diff": [0.05, 0.10, 0.20], "share_excluding_zero": [0.3, 0.85, 1.0]})
    assert smallest_detectable(table, 0.80) == 0.10
    assert np.isnan(smallest_detectable(table.assign(share_excluding_zero=[0.1, 0.2, 0.5]), 0.80))


def test_min_detectable_written_to_primary_row(scored, cfg):
    out = summarize(scored, cfg, min_detectable=0.10)
    marked = out[out.min_detectable_diff.notna()]
    assert set(marked.role) == {"primary"} and set(marked.measure) == {"gives_answer"}
    assert len(marked) == out.model.nunique()


# ---------- ceiling or floor rule ----------

def neutral_rows(model, rate, n=100):
    hits = int(round(rate * n))
    return [{"model": model, "question_id": f"q{i}", "condition": "neutral", "phrasing": "A",
             "seed": 0, "gives_answer": int(i < hits)} for i in range(n)]


def test_key_measures_ceiling_and_floor(cfg):
    df = pd.DataFrame(neutral_rows("high", 0.95) + neutral_rows("low", 0.05)
                      + neutral_rows("mid", 0.50) + neutral_rows("edge", 0.90))
    assert key_measures(df, cfg) == {"high": "steps", "low": "steps", "mid": "gives_answer",
                                     "edge": "gives_answer"}


def test_key_measures_refuses_other_conditions(cfg, scored):
    with pytest.raises(ValueError):
        key_measures(scored, cfg)


def test_switched_model_uses_steps_for_primary_and_control(scored, cfg):
    out = summarize(scored, cfg, keys={"mock-planted": "steps"})
    switched = out[(out.model == "mock-planted") & (out.role != "exploratory")]
    assert set(switched.measure) == {"steps"} and set(switched.role) == {"primary", "control"}
    kept = out[(out.model == "mock-no_effect") & (out.role != "exploratory")]
    assert set(kept.measure) == {"gives_answer"}


# ---------- pilot, manipulation check, diagnostics ----------

def test_pilot_report(cfg):
    neutral = pd.DataFrame(neutral_rows("a", 0.95) + neutral_rows("b", 0.40))
    neutral["steps"] = np.where(neutral["model"] == "a", 3, 1)
    neutral["truncated"] = [i < 2 for i in range(100)] + [False] * 100
    solve = pd.DataFrame({"model": ["a"] * 4 + ["b"] * 4, "solve_correct": [1, 1, 1, 0, 1, 0, 0, 0]})
    out = pilot_report(neutral, solve, cfg).set_index("model")
    assert out.loc["a", "gives_answer_rate"] == 0.95 and out.loc["b", "gives_answer_rate"] == 0.40
    assert out.loc["a", "mean_steps"] == 3 and out.loc["a", "truncation_rate"] == 0.02
    assert out.loc["a", "accuracy"] == 0.75 and out.loc["b", "accuracy"] == 0.25
    assert out.loc["a", "key_measure"] == "steps" and out.loc["b", "key_measure"] == "gives_answer"
    assert not out.loc["a", "truncation_ok"] and out.loc["b", "truncation_ok"]


def test_pilot_refuses_contrasts(cfg, scored):
    solve = pd.DataFrame({"model": ["mock-planted"], "solve_correct": [1]})
    with pytest.raises(ValueError, match="no contrasts"):
        pilot_report(scored, solve, cfg)


def test_check_report():
    df = pd.DataFrame({"model": ["m"] * 4, "condition": ["history_success"] * 2 + ["history_struggle"] * 2,
                       "parsed": [1, 1, 1, 0], "item1_correct": [1, 1, 1, 0], "item2_correct": [1, 0, 1, 0]})
    out = check_report(df).set_index("condition")
    assert out.loc["history_success", "accuracy"] == 0.75
    assert out.loc["history_struggle", "accuracy"] == 1.0         # the unreadable reply is not counted as wrong
    assert out.loc["history_struggle", "n_unparsed"] == 1 and out.loc["history_success", "n_unparsed"] == 0


def test_diagnostics():
    df = pd.DataFrame({"model": "m", "condition": "neutral", "phrasing": "A",
                       "question_id": ["q1", "q1", "q2", "q2"], "seed": [0, 1, 0, 1],
                       "reply": ["same", "same", "one", "two"], "truncated": [True, False, False, False]})
    row = diagnostics(df).iloc[0]
    assert row.truncation_rate == 0.25 and row.identical_across_seeds == 0.5


# ---------- small hand-built cases ----------

def reply_row(model, qid, condition, value, seed):
    return {"model": model, "question_id": qid, "condition": condition,
            "phrasing": "A", "seed": seed, "gives_answer": value}


def tiny_df(model="m", q1_effect=1):
    rows = []
    for seed in (0, 1):
        rows += [reply_row(model, "q1", "history_struggle", q1_effect, seed),
                 reply_row(model, "q1", "history_success", 0, seed),
                 reply_row(model, "q2", "history_struggle", seed, seed),
                 reply_row(model, "q2", "history_success", seed, seed)]
    return pd.DataFrame(rows)


def test_paired_diff_per_question():
    means = question_means(tiny_df(), "gives_answer").loc["m"]
    diffs = paired_diffs(means, "history_struggle", "history_success")
    assert diffs.to_dict() == {"q1": 1.0, "q2": 0.0}


def test_missing_cell_dropped():
    df = tiny_df()
    df = df[~((df.question_id == "q2") & (df.condition == "history_success"))]
    means = question_means(df, "gives_answer").loc["m"]
    assert list(paired_diffs(means, "history_struggle", "history_success").index) == ["q1"]
    assert paired_diffs(means, "history_struggle", "low_score").empty


def test_bootstrap_basic():
    assert bootstrap_ci([0.2] * 10, 500, 0.95, 1) == pytest.approx((0.2, 0.2, 0.2))
    mean, lo, hi = bootstrap_ci([0, 1] * 20, 2000, 0.95, 1)
    assert mean == 0.5 and 0.3 < lo < 0.5 < hi < 0.7
    assert bootstrap_ci([0, 1] * 20, 2000, 0.95, 1) == bootstrap_ci([0, 1] * 20, 2000, 0.95, 1)
    assert all(np.isnan(bootstrap_ci([], 100, 0.95, 1)))


def test_models_not_averaged(cfg):
    df = pd.concat([tiny_df("a", q1_effect=1), tiny_df("b", q1_effect=0)])
    only_answer = {**cfg, "analysis": {**cfg["analysis"], "measures": ["gives_answer"]}}
    out = summarize(df, only_answer)
    primary = out[out.role == "primary"].set_index("model").mean_diff
    assert primary.to_dict() == {"a": 0.5, "b": 0.0}
