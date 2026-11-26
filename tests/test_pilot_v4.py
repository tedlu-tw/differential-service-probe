"""src/ds/pilot_v4.py: the neutral pilot (step c), on mock parts, so it runs without a GPU."""
import copy
import json

import pandas as pd
import pytest

import ds.generate_v4 as gv
import ds.pilot_v4 as pv
from ds import load_config
from ds.items_v4 import make_bank

from pathlib import Path

ROOT = Path(__file__).parents[1]
CFG = load_config(ROOT / "configs" / "experiment_v4.yaml")
PHI = "microsoft/phi-4"
QWEN = "Qwen/Qwen3-8B"


@pytest.fixture(scope="module")
def bank():
    return make_bank(CFG)


@pytest.fixture
def cfg(tmp_path):
    out = copy.deepcopy(CFG)
    out["paths"]["mock_dir"] = str(tmp_path / "mock")
    return out


def read(cfg, key):
    return pd.read_csv(gv.out_path(cfg, key, "mock"), keep_default_na=False)


def faked(change):
    """mock_parts whose generate is passed through change(gen, prompts, replies, truncated)."""
    def make_parts(spec, cfg):
        parts = gv.mock_parts(spec, cfg)
        make = parts["make_generate"]

        def make_generate(gen, stash):
            generate = make(gen, stash)

            def wrapped(prompts):
                replies, truncated = generate(prompts)
                replies, truncated = change(gen, prompts, list(replies), list(truncated))
                stash["raw"] = list(replies)
                return replies, truncated
            return wrapped
        return {**parts, "make_generate": make_generate}
    return make_parts


def test_pilot_cells(bank):
    cells, answers = pv.pilot_cells(CFG, bank)
    questions = bank["current"][::2]
    assert len(questions) == 50 and len(cells) == 50 * 4 and len(answers) == 100
    assert {c["question_id"] for c in cells} == {q["id"] for q in questions}
    assert {(c["arm"], c["prompt"], c["condition"]) for c in cells} == {
        ("arm2", p, c) for p in ("A", "B") for c in ("demand", "after_attempt")}


def test_mock_pilot_every_model(cfg, bank):
    report = pv.run_pilot(cfg, bank, gv.mock_parts, "mock")
    rows = read(cfg, "pilot_csv")
    assert list(report["model"]) == [s["id"] for s in CFG["models"]]
    assert (rows.groupby("model").size() == 200 * 5).all()                       # the mock never gives in
    assert set(rows["run"]) == {"pilot"} and set(rows["arm"]) == {"arm2"} and set(rows["max_new_tokens"]) == {768}
    assert (report["max_new_tokens"] == 768).all() and report["finished"].all() and report["chosen"].all()
    assert (report["demand_a_dialogues"] == 50).all() and (report["gave_by_turn2"] == 0).all()
    assert (report["share_never"] == 1).all() and (report["mean_give_in_turn"] == 6).all()
    assert report["rq2_floor_or_ceiling"].all()                                  # 0 is below the floor
    info = [json.loads(x) for x in gv.out_path(cfg, "run_info_jsonl", "mock").read_text().splitlines()]
    assert [(r["run"], r["arm"], r["settings"]["max_new_tokens"]) for r in info] == [("pilot", "arm2", 768)] * 7
    timing = read(cfg, "timing_csv")
    assert set(timing["run"]) == {"pilot"} and timing["batch_size"].sum() == 7 * 1000
    assert (read(cfg, "contexts_csv").groupby("model").size() == 200).all()


def test_report_shows_nothing_of_after_attempt_or_prompt_b(cfg, bank, tmp_path):
    """A model that gives the answer in every after_attempt (from turn 2) and prompt B dialogue leaves the
    report exactly as the mock's, which never gives it."""
    texts = {q["text"]: q["answer"] for q in bank["current"]}
    prompt_b = CFG["prompts"]["system"]["B"]

    def answer_hidden_cells(gen, prompts, replies, truncated):
        for i, p in enumerate(prompts):
            if prompt_b in p or "I tried it myself" in p:
                answer = texts[max((t for t in texts if t in p), key=len)]
                replies[i] = f"The answer is {answer:g}."
        return replies, truncated
    plain = pv.run_pilot(cfg, bank, gv.mock_parts, "mock", model_id=PHI)
    other = copy.deepcopy(cfg)
    other["paths"]["mock_dir"] = str(tmp_path / "other")
    hidden = pv.run_pilot(other, bank, faked(answer_hidden_cells), "mock", model_id=PHI)
    assert (read(other, "pilot_csv")["gives_answer"] == 1).sum() > 0               # the fake did give answers
    pd.testing.assert_frame_equal(plain, hidden)
    assert not {"n_replies", "after_attempt", "prompt", "condition"} & set(plain.columns)


def truncate_first_batch_at(n_bad):
    seen = {"n": 0}

    def change(gen, prompts, replies, truncated):
        seen["n"] += 1
        if gen["max_new_tokens"] == n_bad and seen["n"] == 1:
            truncated = [True] * len(prompts)                    # 16 of 1000 replies: 1.6%
        return replies, truncated
    return change


def test_rerun_at_the_next_candidate(cfg, bank):
    first = pv.run_pilot(cfg, bank, faked(truncate_first_batch_at(768)), "mock", model_id=PHI)
    assert first["truncation_rate"].iloc[0] == pytest.approx(0.016) and not first["chosen"].iloc[0]
    report = pv.run_pilot(cfg, bank, gv.mock_parts, "mock", model_id=PHI, n_tokens=1024)
    assert list(report["max_new_tokens"]) == [768, 1024] and list(report["chosen"]) == [False, True]
    rows = read(cfg, "pilot_csv")
    assert (rows.groupby("max_new_tokens").size() == 1000).all()                 # 768's rows not taken as done
    assert (read(cfg, "contexts_csv").groupby("model").size() == 200).all()      # contexts written once
    info = [json.loads(x) for x in gv.out_path(cfg, "run_info_jsonl", "mock").read_text().splitlines()]
    assert [r["settings"]["max_new_tokens"] for r in info] == [768, 1024]


def test_rerun_refused_when_not_needed_or_out_of_order(cfg, bank):
    pv.run_pilot(cfg, bank, gv.mock_parts, "mock", model_id=PHI)
    with pytest.raises(SystemExit, match="below 0.01; no rerun"):
        pv.run_pilot(cfg, bank, gv.mock_parts, "mock", model_id=PHI, n_tokens=1024)
    with pytest.raises(SystemExit, match="pilot at 1024 has not finished"):
        pv.run_pilot(cfg, bank, gv.mock_parts, "mock", model_id=PHI, n_tokens=1536)
    with pytest.raises(SystemExit, match="not one of the candidates"):
        pv.run_pilot(cfg, bank, gv.mock_parts, "mock", model_id=PHI, n_tokens=900)
    with pytest.raises(SystemExit, match="pilot at 768 has not finished"):          # no model loaded first
        pv.run_pilot(cfg, bank, lambda *a: pytest.fail("a model was loaded"), "mock", model_id=QWEN, n_tokens=1024)


def test_resume_after_a_crash(cfg, bank):
    calls = {"n": 0}

    def crash(gen, prompts, replies, truncated):
        calls["n"] += 1
        if calls["n"] == 20:
            raise RuntimeError("pod stopped")
        return replies, truncated
    with pytest.raises(RuntimeError, match="pod stopped"):
        pv.run_pilot(cfg, bank, faked(crash), "mock", model_id=PHI)
    partial = read(cfg, "pilot_csv")
    assert 0 < len(partial) < 1000 and read(cfg, "timing_csv")["batch_size"].sum() == len(partial)
    with pytest.raises(SystemExit, match="pilot at 768 has not finished"):
        pv.run_pilot(cfg, bank, gv.mock_parts, "mock", model_id=PHI, n_tokens=1024)
    report = pv.run_pilot(cfg, bank, gv.mock_parts, "mock", model_id=PHI)
    rows = read(cfg, "pilot_csv")
    assert len(rows) == 1000 and not rows.duplicated(["question_id", "prompt", "condition", "turn"]).any()
    assert report["finished"].all()


def test_report_only_mode_loads_no_model(cfg, bank):
    pv.run_pilot(cfg, bank, gv.mock_parts, "mock", model_id=PHI)
    gv.out_path(cfg, "pilot_summary_csv", "mock").unlink()
    report = pv.write_pilot_report(cfg, bank, "mock")
    assert list(report["model"]) == [PHI] and gv.out_path(cfg, "pilot_summary_csv", "mock").exists()


def test_give_in_turns_and_the_floor_or_ceiling_rule():
    """Hand-built demand A dialogues: answered at turns 1, 2, 4 and never."""
    rows = []
    for q, turn in (("q1", 1), ("q2", 2), ("q3", 4), ("q4", None)):
        for t in range(1, (turn or 5) + 1):
            rows.append({"model": "m", "max_new_tokens": 768, "arm": "arm2", "question_id": q, "prompt": "A",
                         "condition": "demand", "phrasing": "A", "seed": 0, "turn": t, "truncated": False,
                         "gives_answer": int(t == turn)})
    report = pv.pilot_report(rows, CFG, {("m", 768): True})
    r = report.iloc[0]
    assert r["gave_by_turn2"] == 0.5 and r["mean_give_in_turn"] == (1 + 2 + 4 + 6) / 4
    assert [r[f"share_turn_{t}"] for t in range(1, 6)] + [r["share_never"]] == [0.25, 0.25, 0, 0.25, 0, 0.25]
    assert not r["rq2_floor_or_ceiling"] and r["chosen"]


def test_chosen_tokens_needs_every_earlier_candidate():
    def report(rows):
        return pd.DataFrame([{"model": "m", "max_new_tokens": n, "finished": f, "truncation_ok": ok}
                             for n, f, ok in rows])
    assert pv.chosen_tokens(report([(768, True, True)]), CFG) == {"m": 768}
    assert pv.chosen_tokens(report([(768, False, True)]), CFG) == {"m": None}        # not finished
    assert pv.chosen_tokens(report([(768, True, False)]), CFG) == {"m": None}        # 1024 still to run
    assert pv.chosen_tokens(report([(768, True, False), (1024, True, True)]), CFG) == {"m": 1024}
    assert pv.chosen_tokens(report([(1024, True, True)]), CFG) == {"m": None}        # 768 never ran
    assert pv.chosen_tokens(report([(n, True, False) for n in (768, 1024, 1536)]), CFG) == {"m": None}
