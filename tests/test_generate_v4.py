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


@pytest.fixture
def fresh_cfg(tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg["paths"]["mock_dir"] = str(tmp_path / "mock")
    return cfg


def test_report_built_from_files_across_separate_runs(fresh_cfg, bank):
    """One model per run: the report covers every model on disk, and a resume with nothing left keeps it."""
    llama, qwen = "meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen3-8B"
    gv.run_dry_run(fresh_cfg, bank, gv.mock_parts, "mock", model_id=llama)
    report = gv.run_dry_run(fresh_cfg, bank, gv.mock_parts, "mock", model_id=qwen)
    assert list(report["model"]) == [llama, qwen] and list(report["replies"]) == [850, 850]
    again = gv.run_dry_run(fresh_cfg, bank, gv.mock_parts, "mock", model_id=qwen)       # nothing left to generate
    assert again.equals(report)
    on_disk = pd.read_csv(gv.out_path(fresh_cfg, "dry_run_report_csv", "mock"))
    assert list(on_disk["model"]) == [llama, qwen]


def test_missing_diagnostics_row_stops_the_report(fresh_cfg, bank):
    gv.run_dry_run(fresh_cfg, bank, gv.mock_parts, "mock", model_id="microsoft/phi-4")
    path = gv.out_path(fresh_cfg, "dry_run_diagnostics_csv", "mock")
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n")                                   # one diagnostics row lost
    with pytest.raises(SystemExit, match="microsoft/phi-4"):
        gv.report_from_files(fresh_cfg, "mock")


def test_diagnostics_written_with_each_batch(fresh_cfg, bank):
    """A crash part-way keeps the diagnostics of every batch whose replies were written."""
    calls = {"n": 0}

    def crashing_parts(spec, cfg):
        parts = gv.mock_parts(spec, cfg)
        make = parts["make_generate"]

        def make_generate(gen, stash):
            generate = make(gen, stash)

            def maybe_crash(prompts):
                calls["n"] += 1
                if calls["n"] == 5:
                    raise RuntimeError("pod stopped")
                return generate(prompts)
            return maybe_crash
        return {**parts, "make_generate": make_generate}
    with pytest.raises(RuntimeError, match="pod stopped"):
        gv.run_dry_run(fresh_cfg, bank, crashing_parts, "mock", model_id="microsoft/phi-4")
    replies = pd.read_csv(gv.out_path(fresh_cfg, "dry_run_csv", "mock"))
    diagnostics = pd.read_csv(gv.out_path(fresh_cfg, "dry_run_diagnostics_csv", "mock"))
    assert len(replies) == len(diagnostics) > 0
    report = gv.run_dry_run(fresh_cfg, bank, gv.mock_parts, "mock", model_id="microsoft/phi-4")   # resume
    assert list(report["replies"]) == [850]


MINISTRAL = "mistralai/Ministral-3-8B-Instruct-2512-BF16"
EXTRA = "roundtrip_differs_fix_mistral_regex_false"


def test_diagnostics_have_fixed_columns_whoever_writes_first(dry):
    """The pod failure: Llama wrote the header, Ministral's rows had a 9th field. Now every row has every column."""
    assert list(dry["diag"].columns) == gv.diagnostics_columns(CFG) and EXTRA in gv.diagnostics_columns(CFG)
    diag = gv.read_diagnostics(gv.out_path(dry["cfg"], "dry_run_diagnostics_csv", "mock"), CFG)
    assert diag.loc[diag["model"] == MINISTRAL, EXTRA].notna().all()
    assert diag.loc[diag["model"] != MINISTRAL, EXTRA].isna().all()
    assert dry["report"].set_index("model")["share_" + EXTRA].notna().sum() == 1


def legacy_file(path):
    """The file as the pod wrote it: the header from Llama's rows, then Ministral's rows with one more field."""
    from ds.generate_v3 import append_rows
    base = {"run": "dry_run", "turn": 1, "n_tokens": 7, "ended_on_stop": True, "truncated": False, "empty": False}
    append_rows(path, [{"model": "meta-llama/Llama-3.1-8B-Instruct", **base, "roundtrip_differs_own": False}] * 2)
    append_rows(path, [{"model": MINISTRAL, **base, "roundtrip_differs_own": False, EXTRA: True}] * 3)
    append_rows(path, [{"model": "microsoft/phi-4", **base, "roundtrip_differs_own": True}])


def test_reads_the_file_the_pod_wrote(tmp_path):
    path = tmp_path / "diag.csv"
    legacy_file(path)
    lines = path.read_text().splitlines()
    assert lines[0].count(",") == 7 and lines[3].count(",") == 8                  # the mismatch pandas refused
    diag = gv.read_diagnostics(path, CFG)
    assert list(diag.columns) == gv.diagnostics_columns(CFG) and len(diag) == 6
    assert list(diag[EXTRA].fillna(-1)) == [-1, -1, 1.0, 1.0, 1.0, -1]
    assert list(diag["roundtrip_differs_own"]) == [0.0] * 5 + [1.0] and (diag["turn"] == 1).all()
    report = gv.dry_run_report(diag, pd.DataFrame(columns=["model", "run", "seconds"]))
    assert list(report["replies"]) == [2, 3, 1] and (report["share_ended_on_stop"] == 1.0).all()


def test_a_row_with_the_wrong_field_count_stops_the_read(tmp_path):
    path = tmp_path / "diag.csv"
    legacy_file(path)
    path.write_text(path.read_text() + "microsoft/phi-4,dry_run,1,7,True,False,False,False,True\n")
    with pytest.raises(SystemExit, match="microsoft/phi-4: 9 fields, expected 8 or 8"):
        gv.read_diagnostics(path, CFG)


def test_report_only_mode_loads_no_model(fresh_cfg, bank, monkeypatch):
    gv.run_dry_run(fresh_cfg, bank, gv.mock_parts, "mock", model_id=MINISTRAL)
    report_path = gv.out_path(fresh_cfg, "dry_run_report_csv", "mock")
    report_path.unlink()
    monkeypatch.setattr(gv, "mock_parts", lambda *a: pytest.fail("a model was loaded"))
    report = gv.write_report(fresh_cfg, "mock")
    assert list(report["model"]) == [MINISTRAL] and report_path.exists()
