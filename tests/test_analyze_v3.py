import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ds import load_config
from ds.analyze import (answer_turn_shares, check_report_v3, cohen_kappa, contrasts_v3, dialogues_v3, excludes_zero,
                        hint_level_shares, key_measures_v3, min_detectable_by_model, pilot_report_v3, replication_v3,
                        sensitivity_v3, smallest_detectable_v3, summarize_v3)
from ds.mock import is_plantable, run_mock_dialogues, sensitivity_rows_v3, v3_setting
from ds.score import score_rows_v3
from ds.stimuli_v3 import build_cells_v3, hint_ignore_numbers, make_questions_v3

CONFIG = Path(__file__).parents[1] / "configs" / "experiment_v3.yaml"
PLANTED = {"gave_by_turn2": 0.24, "hint_level": 0.45, "steps": 0.80}     # from the config's planted setting


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG)


@pytest.fixture(scope="module")
def questions(cfg):
    return make_questions_v3(cfg)


@pytest.fixture(scope="module")
def cells(cfg, questions):
    return build_cells_v3(cfg, questions)


def mock_dialogues(cells, questions, cfg, setting, name, run=0, cache=None):
    rows = run_mock_dialogues(cells, questions, cfg, v3_setting(cfg, setting), name, cfg["seeds"]["generation"], run)
    return dialogues_v3(score_rows_v3(rows, questions, cfg["scoring"], hint_ignore_numbers(cfg), cache))


@pytest.fixture(scope="module")
def d(cells, questions, cfg):
    return pd.concat([mock_dialogues(cells, questions, cfg, s, f"mock-{s}") for s in ("planted", "no_effect")])


@pytest.fixture(scope="module")
def summary(d, cfg):
    return summarize_v3(d, cfg)


def pick(summary, model, contrast, measure):
    rows = summary[(summary.model == model) & (summary.contrast == contrast) & (summary.measure == measure)]
    assert len(rows) == 1
    return rows.iloc[0]


# ---------- dialogue table ----------

def reply(turn, gives, **kw):
    base = {"model": "m", "question_id": "q1", "prompt": "pedagogical", "condition": "neutral", "phrasing": "A",
            "seed": 0, "turn": turn, "reply": f"r{turn}", "truncated": False, "gives_answer": gives, "hint_level": 1,
            "steps": 2, "praises": 0, "encourages": 0, "asks_attempt": 1, "words": 10, "mentions_history": 0}
    return {**base, **kw}


def test_dialogue_table():
    rows = ([reply(1, 0), reply(2, 1)]
            + [reply(t, 0, phrasing="B") for t in (1, 2, 3)]
            + [reply(1, 1, condition="low_score", hint_level=3)]
            + [reply(1, 0, seed=1), reply(2, 0, seed=1), reply(3, 1, seed=1, truncated=True)])
    out = dialogues_v3(rows).set_index(["condition", "phrasing", "seed"])
    assert out.loc[("neutral", "A", 0), ["answer_turn", "gave_by_turn2", "n_turns"]].tolist() == [2, 1, 2]
    assert np.isnan(out.loc[("neutral", "B", 0), "answer_turn"]) and out.loc[("neutral", "B", 0), "gave_by_turn2"] == 0
    assert out.loc[("low_score", "A", 0), ["answer_turn", "gave_by_turn2", "gives_answer_turn1"]].tolist() == [1, 1, 1]
    late = out.loc[("neutral", "A", 1)]
    assert late.answer_turn == 3 and late.gave_by_turn2 == 0 and late.any_truncated
    assert out.loc[("neutral", "A", 0), "hint_level"] == 1          # turn-1 measures come from turn 1


def test_dialogue_without_turn1_refused():
    with pytest.raises(ValueError, match="turn-1"):
        dialogues_v3([reply(2, 1)] + [reply(1, 0, phrasing="B")])


# ---------- the mock checks (definition of done) ----------

@pytest.mark.parametrize("measure", ["gave_by_turn2", "hint_level", "steps"])
@pytest.mark.parametrize("contrast", ["history", "explicit"])
def test_planted_effect_recovered(summary, contrast, measure):
    row = pick(summary, "mock-planted", contrast, measure)
    assert row.ci_low > 0 and row.ci_low < PLANTED[measure] < row.ci_high


def test_no_effect_mean_near_zero(summary):
    for contrast in ("history", "explicit"):
        for measure in PLANTED:
            assert abs(pick(summary, "mock-no_effect", contrast, measure).mean_diff) < 0.1 * (1 + (measure != "gave_by_turn2"))


def test_no_effect_false_positive_rate(cfg, questions, cells):
    # One interval excludes zero about 1 time in 20 by chance, so the check is a rate over 100 runs.
    keep = [c for c in cells if c["prompt"] == "pedagogical"
            and c["condition"] in ("history_struggle", "history_success", "low_score", "high_score")]
    hits, cache = {(c, m): [] for c in ("primary", "control") for m in PLANTED}, {}
    for run in range(100):
        one = mock_dialogues(keep, questions, cfg, "no_effect", "m", run, cache)
        for c in ("primary", "control"):
            a, b = cfg["analysis"][c]["a"], cfg["analysis"][c]["b"]
            for m in PLANTED:
                hits[(c, m)].append(excludes_zero(one, a, b, m, cfg))
    rates = {k: np.mean(v) for k, v in hits.items()}
    assert all(r <= 0.10 for r in rates.values()), rates


# ---------- summary shape and roles ----------

def test_roles_and_contrasts(summary, cfg):
    measures = cfg["analysis"]["measures"]
    for model, rows in summary.groupby("model"):
        assert len(rows) == len(contrasts_v3(cfg)) * len(measures)
        assert rows.role.value_counts().to_dict() == {"exploratory": len(rows) - 2, "primary": 1, "control": 1}
        key_rows = rows[rows.role != "exploratory"]
        assert set(key_rows.measure) == {"gave_by_turn2"} and set(key_rows.prompt) == {"pedagogical"}
    pairs = set(zip(summary.prompt, summary.a, summary.b))
    assert pairs == {("pedagogical", "history_struggle", "history_success"), ("pedagogical", "low_score", "high_score"),
                     ("pedagogical", "style_casual", "style_formal"), ("pedagogical", "style_hedged", "style_formal"),
                     ("plain", "history_struggle", "history_success")}
    assert "neutral" not in set(summary.b)                       # no "vs neutral" rows in v3


def test_contrasts_never_mix_prompts(d, cfg):
    # The plain history contrast must use plain dialogues only.
    row = pick(summarize_v3(d, cfg), "mock-planted", "history_plain", "gave_by_turn2")
    plain = d[(d.model == "mock-planted") & (d.prompt == "plain")]
    means = plain.groupby(["question_id", "condition"]).gave_by_turn2.mean().unstack()
    assert row.mean_diff == pytest.approx((means["history_struggle"] - means["history_success"]).mean())


def test_precision_only_for_count_and_ordinal(summary, cfg):
    wide = summary[summary.measure.isin(cfg["analysis"]["precision_measures"])]
    assert np.allclose(wide.precision, (wide.ci_high - wide.ci_low) / 2)
    assert summary[~summary.measure.isin(cfg["analysis"]["precision_measures"])].precision.isna().all()


def test_switched_model_uses_fallback_for_primary_and_control(d, cfg):
    out = summarize_v3(d, cfg, keys={"mock-planted": "hint_level"})
    switched = out[(out.model == "mock-planted") & (out.role != "exploratory")]
    assert set(switched.measure) == {"hint_level"} and set(switched.role) == {"primary", "control"}
    kept = out[(out.model == "mock-no_effect") & (out.role != "exploratory")]
    assert set(kept.measure) == {"gave_by_turn2"}


def test_min_detectable_per_model_and_measure(d, cfg):
    sizes = {"mock-planted": {"gave_by_turn2": 0.10, "hint_level": 0.2, "steps": 0.4},
             "mock-no_effect": {"gave_by_turn2": 0.05, "hint_level": 0.1, "steps": 0.1}}
    out = summarize_v3(d, cfg, keys={"mock-planted": "steps"}, min_detectable=sizes)
    marked = out[out.min_detectable_diff.notna()]
    assert set(marked.role) == {"primary"}
    assert dict(zip(marked.model, marked.min_detectable_diff)) == {"mock-planted": 0.4, "mock-no_effect": 0.05}


def test_min_detectable_by_model():
    found = {("default", "gave_by_turn2"): 0.10, ("default", "hint_level"): 0.2, ("default", "steps"): 0.2,
             ("m", "hint_level"): np.nan, ("m", "steps"): 0.4, ("m", "gave_by_turn2"): 0.99}
    out = min_detectable_by_model(found, {"m": "m", "mock": "default"})
    assert out["m"]["gave_by_turn2"] == 0.10                          # the shared sweep, not the model's own row
    assert np.isnan(out["m"]["hint_level"]) and out["m"]["steps"] == 0.4
    assert out["mock"] == {"gave_by_turn2": 0.10, "hint_level": 0.2, "steps": 0.2}


def test_models_not_averaged(d, cfg):
    out = summarize_v3(d, cfg)
    primary = out[out.role == "primary"].set_index("model").mean_diff
    assert primary["mock-planted"] > 0.15 and abs(primary["mock-no_effect"]) < 0.1


# ---------- replication and distributions ----------

def test_replication(d, cfg):
    out = replication_v3(d, cfg).set_index("model")
    plain = d[(d.prompt == "plain") & (d.condition == "neutral")]
    expected = plain.groupby(["model", "question_id"]).gives_answer_turn1.mean().groupby("model").mean()
    assert np.allclose(out.rate, expected[out.index]) and (out.n_questions == 200).all()


def test_shares_sum_to_one(d):
    for table, prefix in ((answer_turn_shares(d), "answer_turn_"), (hint_level_shares(d), "hint_level_")):
        cols = [c for c in table if c.startswith(prefix)]
        assert np.allclose(table[cols].sum(axis=1), 1)
    assert "answer_turn_none" in answer_turn_shares(d)


# ---------- ceiling or floor rule ----------

def neutral_dialogues(model, rate, n=100):
    hits = int(round(rate * n))
    return [{"model": model, "question_id": f"q{i}", "prompt": "pedagogical", "condition": "neutral",
             "gave_by_turn2": int(i < hits)} for i in range(n)]


def test_key_measures_v3(cfg):
    df = pd.DataFrame(neutral_dialogues("high", 0.95) + neutral_dialogues("low", 0.05)
                      + neutral_dialogues("mid", 0.5) + neutral_dialogues("edge", 0.90))
    assert key_measures_v3(df, cfg) == {"high": "hint_level", "low": "hint_level", "mid": "gave_by_turn2",
                                        "edge": "gave_by_turn2"}
    assert key_measures_v3(df, cfg, hint_valid=False)["high"] == "steps"


@pytest.mark.parametrize("change", [{"condition": "low_score"}, {"prompt": "plain"}])
def test_key_measures_v3_refuses_other_cells(cfg, change):
    df = pd.DataFrame(neutral_dialogues("m", 0.5) + [{**neutral_dialogues("m", 0.5)[0], **change}])
    with pytest.raises(ValueError):
        key_measures_v3(df, cfg)


# ---------- pre-freeze reports ----------

def test_pilot_report_v3(cfg):
    rows = ([reply(1, 1, question_id=f"q{i}") for i in range(3)]
            + [reply(t, int(t == 3), question_id="q3", truncated=(t == 3)) for t in (1, 2, 3)])
    solve = pd.DataFrame({"model": ["m"] * 4, "solve_correct": [1, 1, 0, 0], "solve_parsed": [1, 1, 1, 0]})
    out = pilot_report_v3(rows, solve, cfg).iloc[0]
    assert out.n_dialogues == 4 and out.gave_by_turn2 == 0.75
    assert (out.answer_turn_1, out.answer_turn_3, out.answer_turn_none) == (0.75, 0.25, 0)
    assert out.truncation_rate == pytest.approx(1 / 6)
    assert out.accuracy_with_working == 0.5 and out.solve_unparsed_rate == 0.25 and not out.ceiling_rule_applies


def test_pilot_report_v3_refuses_contrasts(cfg):
    # A valid solve table, so only the guard can stop the report.
    solve = pd.DataFrame({"model": ["m"], "solve_correct": [1], "solve_parsed": [1]})
    rows = [reply(1, 1), reply(1, 1, condition="low_score")]
    with pytest.raises(ValueError, match="no contrasts"):
        pilot_report_v3(rows, solve, cfg)
    with pytest.raises(ValueError, match="no contrasts"):
        pilot_report_v3([reply(1, 1), reply(1, 1, prompt="plain", question_id="q2")], solve, cfg)


def test_check_report_v3():
    df = pd.DataFrame({"model": "m", "condition": ["history_success"] * 3 + ["history_struggle"],
                       "parsed": [1, 1, 0, 1], "item1_correct": [1, 1, 0, 1], "item2_correct": [1, 0, 0, 1],
                       "check_rule": ["combined", "per_item", "unparsed", "yes_no"]})
    out = check_report_v3(df).set_index("condition")
    assert out.loc["history_success", "accuracy"] == 0.75 and out.loc["history_success", "unparsed_rate"] == 1 / 3
    assert out.loc["history_success", ["rule_combined", "rule_per_item", "rule_unparsed"]].tolist() == [1, 1, 1]
    assert out.loc["history_struggle", "rule_yes_no"] == 1


def test_cohen_kappa():
    # 50 items: both yes 20, a yes b no 5, a no b yes 10, both no 15 -> kappa 0.4
    a = [1] * 25 + [0] * 25
    b = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)
    assert cohen_kappa([0, 1, 2, 3], [0, 1, 2, 3]) == 1
    assert cohen_kappa([0, 1, 2, 3], [1, 2, 3, 0]) < 0          # ordinal labels, unweighted
    assert np.isnan(cohen_kappa([1, 1], [1, 1]))


# ---------- sensitivity ----------

OLMO = "allenai/Olmo-3-7B-Instruct"


def test_sensitivity_v3(cfg, questions, cells):
    small = copy.deepcopy(cfg)
    small["mock"]["sensitivity"]["runs"] = 20
    cache = {}
    make = lambda base, m, s, run: sensitivity_rows_v3(cells, questions, small, m, s, run, cache, base)
    table = sensitivity_v3(make, small, ["default", OLMO], lambda base, m, s: is_plantable(small, m, s, base))
    plan = list(zip(table.base, table.measure, table["size"]))
    assert plan[:3] == [("default", "gave_by_turn2", s) for s in (0.05, 0.10, 0.20)]   # one shared sweep
    assert len(plan) == 3 + 2 * 6 and (OLMO, "gave_by_turn2", 0.05) not in plan
    # 19% of Olmo's turn-1 replies are below level 3, so its hint_level can rise by 0.1 but not 0.2 or 0.4.
    hint = table[(table.base == OLMO) & (table.measure == "hint_level")]
    assert list(hint.plantable) == [True, False, False] and list(hint.runs) == [20, 0, 0]
    assert hint.share_excluding_zero.isna().tolist() == [False, True, True]
    for _, t in table[table.plantable].groupby(["base", "measure"]):
        shares = list(t.share_excluding_zero)
        assert shares[0] <= shares[-1]
    assert (table[table.base == "default"].groupby("measure").share_excluding_zero.max() >= 0.8).all()
    assert set(smallest_detectable_v3(table, 0.8)) == {("default", "gave_by_turn2"), ("default", "hint_level"),
                                                       ("default", "steps"), (OLMO, "hint_level"), (OLMO, "steps")}


def test_smallest_detectable_v3():
    table = pd.DataFrame({"base": ["x"] * 4 + ["y"] * 2, "measure": ["a", "a", "b", "b", "a", "a"],
                          "size": [0.1, 0.2, 0.1, 0.2, 0.1, 0.2],
                          "share_excluding_zero": [0.5, 0.9, 0.1, 0.3, 0.95, np.nan]})
    out = smallest_detectable_v3(table, 0.8)
    assert out[("x", "a")] == 0.2 and np.isnan(out[("x", "b")]) and out[("y", "a")] == 0.1
