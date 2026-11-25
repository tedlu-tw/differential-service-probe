"""src/ds/generate_v4.py: the dry run (step b), on mock parts, so the whole orchestration runs without a GPU."""
import copy
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

import ds.generate_v4 as gv
from ds import load_config
from ds.generate_v3 import dry_run_questions
from ds.items_v4 import make_bank
from ds.models_v4 import resolved_generation
from ds.stimuli_v3 import make_questions_v3

ROOT = Path(__file__).parents[1]
CFG = load_config(ROOT / "configs" / "experiment_v4.yaml")
SPECS = {s["id"]: s for s in CFG["models"]}


@pytest.fixture(scope="module")
def bank():
    return make_bank(CFG)


@pytest.fixture(scope="module")
def dry(bank, tmp_path_factory):
    """One mock dry run of every model, into a temporary folder."""
    cfg = copy.deepcopy(CFG)
    cfg["paths"]["mock_dir"] = str(tmp_path_factory.mktemp("mock"))
    report = gv.run_dry_run(cfg, bank, gv.mock_parts, "mock")
    read = lambda key: pd.read_csv(gv.out_path(cfg, key, "mock"), keep_default_na=False)
    info = [json.loads(line) for line in gv.out_path(cfg, "run_info_jsonl", "mock").read_text().splitlines()]
    return {"cfg": cfg, "report": report, "info": info, "replies": read("dry_run_csv"),
            "arm3": read("dry_run_arm3_csv"), "contexts": read("contexts_csv"), "diag": read("dry_run_diagnostics_csv")}


def test_dry_run_questions(bank):
    v3_questions = make_questions_v3(CFG["arm3"])
    picked = gv.dry_run_questions_v4(CFG, bank, v3_questions)
    assert Counter(q["family"] for q in picked["arm1"]) == {f: 3 for f in CFG["items"]["families"]}
    for f in CFG["items"]["families"]:
        family = [q for q in bank["current"] if q["family"] == f]
        assert [q for q in picked["arm1"] if q["family"] == f] == [family[0], family[8], family[16]]
    assert len(picked["arm2"]) == 5 and picked["arm2"] == dry_run_questions(CFG, bank["current"])
    assert picked["arm3"] == dry_run_questions(CFG, v3_questions)                 # v3's own dry-run questions


def test_arm3_seeds_by_model_group():
    """Full runs: Arm 3 seeds 2 and 3 for the v3 models (test-retest), 0 and 1 for the new models."""
    for spec in CFG["models"]:
        expected = [2, 3] if spec["group"] == "v3" else [0, 1]
        assert gv.seeds_for(CFG, spec, "arm3") == expected, spec["id"]
        assert gv.seeds_for(CFG, spec, "arm1") == gv.seeds_for(CFG, spec, "arm2") == [0, 1]
        assert gv.seeds_for(CFG, spec, "dry_run") == [0]
        assert all(gv.seeds_for(CFG, spec, run) == [0] for run in ("pilot", "solve", "check", "v3_pilot", "v3_solve"))
    with pytest.raises(ValueError, match="unknown run"):
        gv.seeds_for(CFG, CFG["models"][0], "full")


def test_run_info_records_the_resolved_settings_for_every_run(dry):
    info = dry["info"]
    assert len(info) == len(CFG["models"]) * 3                                   # every model, every arm
    for row in info:
        assert row["run"] == "dry_run" and row["settings"] == resolved_generation(dry["cfg"], row["model"], row["arm"])
        assert row["settings"]["max_new_tokens"] == (512 if row["arm"] == "arm3" else 768)
        assert row["effective"]["max_new_tokens"] == row["settings"]["max_new_tokens"]
        assert {"revision", "precision", "packages", "lock_checked", "generation_config_diff", "gpu",
                "system_mode", "forced_text"} <= set(row)


def test_replies_logged_by_arm_and_run(dry, bank):
    replies, arm3 = dry["replies"], dry["arm3"]
    assert set(replies["run"]) == {"dry_run"} and set(replies["arm"]) == {"arm1", "arm2"}
    assert set(arm3["run"]) == {"dry_run"} and set(arm3["arm"]) == {"arm3"}
    per_model = replies.groupby(["model", "arm"])["question_id"].nunique().unstack()
    assert (per_model["arm1"] == 12).all() and (per_model["arm2"] == 5).all()
    assert (arm3.groupby("model")["question_id"].nunique() == 5).all()
    assert replies["turn"].max() == 5 and arm3["turn"].max() == 3                  # the mock never gives in
    assert (replies["raw_reply"] == replies["reply"]).all()


def test_contexts_saved_for_every_turn_one_cell(dry):
    contexts = dry["contexts"]
    per_model = contexts.groupby("model").size()
    assert (per_model == 12 * 10 + 5 * 4 + 5 * 10).all() and len(per_model) == len(CFG["models"])
    assert all(isinstance(json.loads(ids), list) for ids in contexts["token_ids"])
    assert {"model", "revision", "run", "arm", "question_id", "prompt", "condition", "phrasing", "text"} <= set(contexts)


def test_report_is_per_model_only(dry):
    report, diag = dry["report"], dry["diag"]
    assert list(report["model"]) == [s["id"] for s in CFG["models"]]
    hidden = {"condition", "arm", "prompt", "phrasing", "question_id", "gives_answer", "answer_turn", "reply"}
    assert not hidden & set(report.columns) and not hidden & set(diag.columns)
    assert {"replies", "share_ended_on_stop", "share_truncated", "empty_replies", "share_roundtrip_differs_own",
            "seconds_per_reply"} <= set(report.columns)
    assert (report["share_ended_on_stop"] == 1.0).all() and (report["empty_replies"] == 0).all()


def test_resume_does_not_repeat_a_finished_dry_run(dry, bank):
    before = len(dry["replies"])
    gv.run_dry_run(dry["cfg"], bank, gv.mock_parts, "mock", model_id="microsoft/phi-4")
    after = pd.read_csv(gv.out_path(dry["cfg"], "dry_run_csv", "mock"), keep_default_na=False)
    contexts = pd.read_csv(gv.out_path(dry["cfg"], "contexts_csv", "mock"), keep_default_na=False)
    assert len(after) == before and len(contexts) == len(dry["contexts"])        # nothing written twice


def test_ministral_also_round_trips_without_the_fix(monkeypatch):
    loaded = []
    monkeypatch.setattr(gv, "load_tokenizer_v4", lambda spec: loaded.append(spec.get("tokenizer_kwargs")) or "tok")
    ministral = SPECS["mistralai/Ministral-3-8B-Instruct-2512-BF16"]
    assert gv.roundtrip_tokenizers(ministral, CFG, "own") == {"own": "own", "fix_mistral_regex_false": "tok"}
    assert loaded == [{"fix_mistral_regex": False}]
    assert gv.roundtrip_tokenizers(SPECS["microsoft/phi-4"], CFG, "own") == {"own": "own"}
