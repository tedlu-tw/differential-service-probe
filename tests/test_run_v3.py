"""scripts/run_v3.py on the mock and on CSVs, with a small config: 40 questions, 2 sensitivity runs.

The GPU path is checked for its wiring only (generate_on_gpu replaced by a recorder).
"""
import copy
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ds import load_config

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import run_v3  # noqa: E402

CONFIG = Path(__file__).parents[1] / "configs" / "experiment_v3.yaml"


def small_cfg(root):
    cfg = copy.deepcopy(load_config(CONFIG))
    cfg["paths"] = {k: str(root / "real" / Path(v).name) for k, v in cfg["paths"].items()}
    cfg["paths"]["mock_dir"] = str(root / "mock")
    cfg["questions"]["per_family"] = 10
    cfg["mock"]["sensitivity"]["runs"] = 2
    cfg["analysis"]["n_boot"] = 200
    return cfg


@pytest.fixture(scope="module")
def mock_run(tmp_path_factory):
    """Every mode once on the mock, into one folder."""
    cfg = small_cfg(tmp_path_factory.mktemp("run"))
    for mode in (run_v3.run_pilot, run_v3.run_check, run_v3.run_full):
        mode(cfg, "mock")
    return cfg


def mock_file(cfg, key):
    return pd.read_csv(run_v3.out_path(cfg, key, "mock"), keep_default_na=False)


def copy_mock_to_real(cfg, keys):
    for key in keys:
        Path(cfg["paths"][key]).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(run_v3.out_path(cfg, key, "mock"), cfg["paths"][key])


def write_agreement(cfg, kappas):
    path = Path(cfg["paths"]["validate_agreement_csv"])
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"measure": list(kappas), "kappa": list(kappas.values())}).to_csv(path, index=False)


# ---------- mock modes ----------

def test_mock_never_writes_real_paths(mock_run):
    assert not Path(mock_run["paths"]["replies_csv"]).parent.exists()


def test_pilot_mock(mock_run):
    report = mock_file(mock_run, "pilot_summary_csv")
    assert set(report.model) == {"mock-planted", "mock-no_effect"}
    assert (report.n_dialogues == 20).all()                        # every second of 40 questions, 1 seed
    assert report.accuracy_with_working.between(0, 1).all()
    shares = [c for c in report.columns if c.startswith("hint_level_")]
    assert shares == [f"hint_level_{k}" for k in range(4)] and np.allclose(report[shares].sum(axis=1), 1)
    steps = [c for c in report.columns if c.startswith("steps_")]
    assert np.allclose(report[steps].sum(axis=1), 1)
    pilot = mock_file(mock_run, "pilot_csv")
    assert set(pilot.condition) == {"neutral", "solve"} and set(pilot[pilot.condition == "neutral"].prompt) == {"pedagogical"}
    assert not run_v3.out_path(mock_run, "compute_estimate_csv", "mock").exists()   # no timing on the mock


def test_check_mock(mock_run):
    report = mock_file(mock_run, "check_summary_csv")
    assert len(report) == 4 and (report.n_prompts == 40 * 2).all()     # 40 questions x 2 phrasings
    assert set(report.condition) == set(mock_run["check"]["expected"])


def test_full_mock(mock_run):
    d = mock_file(mock_run, "dialogues_csv")
    assert len(d) == 2 * 40 * len(run_v3.list_cells_v3(mock_run)) * 2
    summary = mock_file(mock_run, "summary_csv")
    primary = summary[summary.role == "primary"].set_index("model")
    assert primary.loc["mock-planted", "ci_low"] > 0                  # the planted effect is found
    assert set(primary.min_detectable_diff.astype(str)) <= {"0.05", "0.1", "0.2", ""}
    sensitivity = mock_file(mock_run, "sensitivity_csv")
    assert len(sensitivity) == 9 and set(sensitivity.base) == {"default"}      # mock models use the default base
    replication = mock_file(mock_run, "replication_csv")
    assert set(replication.prompt) == {"plain"} and set(replication.condition) == {"neutral"}
    for key in ("diagnostics_csv", "answer_turns_csv", "hint_levels_csv"):
        assert len(mock_file(mock_run, key)) == 2 * len({(p, c) for p, c, _ in run_v3.list_cells_v3(mock_run)})


# ---------- step c ----------

def test_compute_estimate():
    timing = pd.DataFrame([
        {"run": "pilot", "model": "m", "turn": 1, "batch_size": 8, "seconds": 10.0},
        {"run": "pilot", "model": "m", "turn": 1, "batch_size": 2, "seconds": 5.0},     # 1.5 s per reply
        {"run": "pilot", "model": "m", "turn": 2, "batch_size": 4, "seconds": 12.0},    # 3 s per reply
        {"run": "pilot_solve", "model": "m", "turn": 1, "batch_size": 8, "seconds": 999.0},
        {"run": "dry_run", "model": "m", "turn": 3, "batch_size": 8, "seconds": 999.0},
    ])
    d = pd.DataFrame({"model": ["m"] * 4 + ["untimed"] * 2, "n_turns": [1, 1, 2, 2, 1, 1]})
    table = run_v3.compute_estimate(timing, d, 3600, 3).set_index("model")
    m = table.loc["m"]
    assert m.replies_turn1 == 3600 and m.replies_turn2 == 1800 and m.replies_turn3 == 0
    assert np.isnan(m.seconds_per_reply_turn3)                         # no turn 3 in the pilot, none needed
    assert m.gpu_hours == pytest.approx((3600 * 1.5 + 1800 * 3) / 3600)
    assert np.isnan(table.loc["untimed", "gpu_hours"])                 # missing timing is never zero


def test_pilot_csv_writes_estimate(mock_run, tmp_path):
    cfg = copy.deepcopy(mock_run)
    cfg["paths"] = {k: str(tmp_path / Path(v).name) for k, v in cfg["paths"].items()}
    cfg["paths"]["mock_dir"] = mock_run["paths"]["mock_dir"]
    copy_mock_to_real(cfg, ["pilot_csv"])
    pd.DataFrame([{"run": "pilot", "model": m, "seed": 0, "turn": t, "batch_size": 8, "seconds": 8.0}
                  for m in ("mock-planted", "mock-no_effect") for t in (1, 2, 3)]).to_csv(cfg["paths"]["timing_csv"],
                                                                                         index=False)
    run_v3.run_pilot(cfg, "csv")
    table = pd.read_csv(cfg["paths"]["compute_estimate_csv"])
    assert set(table.model) == {"mock-planted", "mock-no_effect"} and (table.gpu_hours > 0).all()


# ---------- real runs need steps a and b ----------

def test_validation_passed(tmp_path):
    cfg = small_cfg(tmp_path)
    write_agreement(cfg, {"hint_level": 0.7, "gives_answer": 0.9, "asks_attempt": 0.6})
    assert run_v3.validation_passed(cfg, "csv") is True
    write_agreement(cfg, {"hint_level": 0.5, "gives_answer": 0.9, "asks_attempt": 0.8})
    assert run_v3.validation_passed(cfg, "csv") is False               # hint_level becomes exploratory
    for failed in ({"hint_level": 0.9, "gives_answer": 0.5, "asks_attempt": 0.8},
                   {"hint_level": 0.9, "gives_answer": 0.9, "asks_attempt": float("nan")},
                   {"hint_level": 0.9, "gives_answer": 0.9}):
        write_agreement(cfg, failed)
        with pytest.raises(SystemExit, match="step b failed"):
            run_v3.validation_passed(cfg, "csv")


def test_real_full_stops_before_the_gpu(tmp_path, monkeypatch):
    cfg = small_cfg(tmp_path)
    monkeypatch.setattr(run_v3, "generate_on_gpu", lambda *a, **k: pytest.fail("reached the GPU"))
    for source in ("gpu", "csv"):
        with pytest.raises(SystemExit, match="step b"):
            run_v3.run_full(cfg, source)
    write_agreement(cfg, {"hint_level": 0.9, "gives_answer": 0.9, "asks_attempt": 0.9})
    with pytest.raises(SystemExit, match="step a"):
        run_v3.run_full(cfg, "gpu")


def test_real_full_needs_every_sensitivity_base(tmp_path, monkeypatch):
    cfg = small_cfg(tmp_path)
    run_v3.run_pilot(cfg, "mock")
    copy_mock_to_real(cfg, ["pilot_csv"])
    write_agreement(cfg, {"hint_level": 0.9, "gives_answer": 0.9, "asks_attempt": 0.9})
    del cfg["mock"]["sensitivity"]["bases"]["allenai/Olmo-3-7B-Instruct"]
    monkeypatch.setattr(run_v3, "generate_on_gpu", lambda *a, **k: pytest.fail("reached the GPU"))
    with pytest.raises(SystemExit, match="no sensitivity base for \\['allenai/Olmo-3-7B-Instruct'\\]"):
        run_v3.run_full(cfg, "gpu")
    assert run_v3.sensitivity_bases(cfg, ["mock-planted"], "mock") == {"mock-planted": "default"}
    assert run_v3.sensitivity_bases(cfg, ["Qwen/Qwen3-8B"], "csv") == {"Qwen/Qwen3-8B": "Qwen/Qwen3-8B"}


PILOT = Path(__file__).parents[1] / "results_v3" / "pilot_summary.csv"


@pytest.mark.skipif(not PILOT.exists(), reason="the pilot results are private; this is checked locally")
def test_bases_match_the_pilot():
    """The config's per-model bases are the pilot's turn-1 shares, steps up to each model's largest count."""
    bases, pilot = load_config(CONFIG)["mock"]["sensitivity"]["bases"], pd.read_csv(PILOT)
    assert set(bases) - {"default"} == set(pilot.model)
    for row in pilot.to_dict("records"):
        steps = [row[c] for c in pilot.columns if c.startswith("steps_")]
        while steps[-1] == 0:
            steps.pop()
        assert bases[row["model"]]["turn1_levels"] == pytest.approx([row[f"hint_level_{k}"] for k in range(4)])
        assert bases[row["model"]]["steps"] == pytest.approx(steps)


def test_keys_follow_pilot_and_step_b(tmp_path):
    """A pilot at the ceiling switches to hint_level, or to steps if hint_level failed step b."""
    cfg = small_cfg(tmp_path)
    cfg["mock"]["settings"]["planted"]["default"]["later_give_in"] = [1.0, 1.0]   # gave_by_turn2 = 1
    run_v3.run_pilot(cfg, "mock")
    copy_mock_to_real(cfg, ["pilot_csv"])
    questions = run_v3.make_questions_v3(cfg)
    write_agreement(cfg, {"hint_level": 0.9, "gives_answer": 0.9, "asks_attempt": 0.9})
    keys = {"mock-planted": "hint_level", "mock-no_effect": "gave_by_turn2"}
    assert run_v3.load_keys_v3(cfg, "csv", questions) == keys
    write_agreement(cfg, {"hint_level": 0.3, "gives_answer": 0.9, "asks_attempt": 0.9})
    assert run_v3.load_keys_v3(cfg, "csv", questions) == {**keys, "mock-planted": "steps"}


def test_csv_source_needs_the_replies(tmp_path):
    with pytest.raises(SystemExit, match="does not exist"):
        run_v3.run_check(small_cfg(tmp_path), "csv")


# ---------- GPU wiring ----------

def test_gpu_replies_wiring(tmp_path, monkeypatch):
    cfg = small_cfg(tmp_path)
    questions = run_v3.make_questions_v3(cfg)
    calls = []
    monkeypatch.setattr(run_v3, "generate_on_gpu", lambda cells, answers, c, seeds, path, run: calls.append(
        ({x["kind"] for x in cells}, c["design"]["max_turns"], c["prompts"]["followups"], seeds, run)))
    cells = run_v3.pilot_cells(cfg, questions)
    run_v3.gpu_replies(cells, questions, cfg, [0], "x.csv", {"tutor": "pilot", "solve": "pilot_solve"})
    with pytest.raises(ValueError, match="kinds"):
        run_v3.gpu_replies(cells, questions, cfg, [0], "x.csv", {"tutor": "pilot"})
    assert calls == [({"tutor"}, 3, cfg["prompts"]["followups"], [0], "pilot"), ({"solve"}, 1, [], [0], "pilot_solve")]
    assert cfg["design"]["max_turns"] == 3
