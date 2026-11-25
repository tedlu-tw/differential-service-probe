"""configs/experiment_v4.yaml (docs/protocol_v4.md): Arm 3's parity with v3, and the shape of the rest."""
import hashlib
from pathlib import Path

import pytest

from ds import load_config
from ds.stimuli_v3 import build_cells_v3, make_questions_v3

ROOT = Path(__file__).parents[1]
V3 = load_config(ROOT / "configs" / "experiment_v3.yaml")
V4 = load_config(ROOT / "configs" / "experiment_v4.yaml")
ARM3 = V4["arm3"]
COPIED = ["questions", "history", "prompts", "cues", "generation", "scoring", "pilot"]
V3_SUMMARY_SHA256 = "fd6e03ad9daa8ccc26f1bcd537bf20282eb920a0d7c8217d30829234e3beb6d9"   # protocol: Why v4


@pytest.mark.parametrize("section", COPIED)
def test_arm3_sections_equal_v3(section):
    assert ARM3[section] == V3[section]


def test_arm3_differs_only_where_intended():
    drop = lambda d, keys: {k: v for k, v in d.items() if k not in keys}
    assert drop(ARM3["seeds"], {"generation"}) == drop(V3["seeds"], {"generation"})
    assert ARM3["seeds"]["generation"] == {"v3": [2, 3], "new": [0, 1]}
    assert drop(ARM3["analysis"], {"exploratory", "key_measures"}) == drop(V3["analysis"], {"exploratory"})
    assert ARM3["design"]["max_turns"] == V3["design"]["max_turns"]
    for prompt, conditions in ARM3["design"]["conditions"].items():
        assert set(conditions) <= set(V3["design"]["conditions"][prompt])


def test_v3_code_gives_v3_questions_and_cells():
    assert make_questions_v3(ARM3) == make_questions_v3(V3)
    wanted = {(p, c) for p, cs in ARM3["design"]["conditions"].items() for c in cs}
    v3_cells = [c for c in build_cells_v3(V3) if (c["prompt"], c["condition"]) in wanted]
    arm3_cells = build_cells_v3(ARM3)
    assert len(arm3_cells) == 200 * 10 and arm3_cells == v3_cells


def test_arm3_key_measures_for_v3_models():
    v3_models = [s["id"] for s in V4["models"] if s["group"] == "v3"]
    assert ARM3["analysis"]["key_measures"] == {"meta-llama/Llama-3.1-8B-Instruct": "gave_by_turn2",
                                                "Qwen/Qwen3-8B": "gave_by_turn2",
                                                "allenai/Olmo-3-7B-Instruct": "hint_level"}
    assert sorted(ARM3["analysis"]["key_measures"]) == sorted(v3_models)


def test_replication_uses_the_fixed_v3_summary():
    rep = V4["analysis"]["arm3_replication"]
    assert rep["v3_summary_sha256"] == V3_SUMMARY_SHA256
    path = ROOT / rep["v3_summary_csv"]
    if path.exists():                                   # private; present only on the author's machine
        assert hashlib.sha256(path.read_bytes()).hexdigest() == V3_SUMMARY_SHA256


def test_arms_and_messages():
    arms, prompts = V4["arms"], V4["prompts"]
    for arm in ("arm1", "arm2"):
        assert len(prompts["pressure"]) == arms[arm]["max_turns"] - 1
    assert len(arms["arm1"]["conditions"]) * len(arms["arm1"]["phrasings"]) == 10
    assert len(arms["arm2"]["prompts"]) * len(arms["arm2"]["pressures"]) == 4
    assert prompts["system"]["A"] == V3["prompts"]["system"]["pedagogical"]
    assert prompts["pressure"][:2] == V3["prompts"]["followups"]
    assert "{work}" in prompts["attempt"] and set(prompts["acks"]) == set(arms["arm1"]["phrasings"])


def test_outputs_private_and_validation_never_sees_hidden_cells():
    assert all(str(p).startswith("results_v4") for p in V4["paths"].values())
    sources = V4["validate"]["sources"]
    assert sources["pilot"] == {"prompt": "A", "pressure": "demand"}
    assert sources["dry_run"]["exclude"] == {"prompt": "B", "pressure": "after_attempt"}


def test_step_d_sample_and_family_pair_analyses():
    v = V4["validate"]
    assert v["n"] == 150 and v["min_per_model"] == 20
    assert v["arm1_turn1"] == {"min": 50, "min_per_family": 10, "min_per_model": 6, "max_per_question": 5}
    a1 = v["arm1_turn1"]
    n_q = len(V4["dry_run"]["arm1_positions"]) * len(V4["items"]["families"])
    assert n_q * a1["max_per_question"] >= a1["min"]                             # the limits can all be met
    assert len(V4["dry_run"]["arm1_positions"]) * a1["max_per_question"] >= a1["min_per_family"]
    assert len(V4["models"]) * a1["min_per_model"] <= a1["min"]
    assert v["later_turns"] == {"turns": [2, 3, 4, 5], "min_per_value": 10}
    ex = V4["analysis"]["exploratory"]
    assert ex["u_by_family_pair"]["by"] == ex["relevance_by_family_pair"]["by"] == ["current_family", "unmatched_family"]
    assert (ex["u_by_family_pair"]["a"], ex["u_by_family_pair"]["b"]) == ("history_unmatched_struggle",
                                                                          "history_unmatched_success")
