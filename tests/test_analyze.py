from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ds import load_config
from ds.analyze import bootstrap_ci, paired_diffs, question_means, summarize
from ds.mock import run_mock
from ds.score import score_rows
from ds.stimuli import build_cells, make_questions

CONFIG = Path(__file__).parents[1] / "configs" / "experiment.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG)


@pytest.fixture(scope="module")
def scored(cfg):
    """Both mock settings in one frame, as two 'models'."""
    questions, cells = make_questions(cfg), build_cells(cfg)
    rows = []
    for setting in ("planted", "no_effect"):
        rows += score_rows(run_mock(cells, questions, cfg, setting), questions, cfg["scoring"])
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def summary(scored, cfg):
    return summarize(scored, cfg)


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
def test_no_effect_interval_covers_zero(summary, role):
    row = pick(summary, "mock-no_effect", role)
    assert row.ci_low < 0 < row.ci_high
    assert abs(row.mean_diff) < 0.1


def test_resamples_questions_not_replies(summary, scored, cfg):
    row = pick(summary, "mock-planted", "primary")
    assert row.n_questions == 100
    means = question_means(scored, "gives_answer").loc["mock-planted"]
    diffs = means["history_struggle"] - means["history_success"]
    expected = bootstrap_ci(diffs, cfg["analysis"]["n_boot"], cfg["analysis"]["ci"], cfg["seeds"]["bootstrap"])
    assert (row.mean_diff, row.ci_low, row.ci_high) == pytest.approx(expected)


def test_roles_and_shape(summary, cfg):
    n_contrasts = 2 + (len(cfg["cues"]) - 1)
    for model, rows in summary.groupby("model"):
        assert len(rows) == n_contrasts * len(cfg["analysis"]["measures"])
        assert rows.role.value_counts().to_dict() == {"exploratory": len(rows) - 2, "primary": 1, "control": 1}
        assert set(rows[rows.role != "exploratory"].measure) == {"gives_answer"}


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
