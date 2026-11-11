"""v2 must stay reproducible while v3 is added (docs/protocol_v3.md, Definition of done).

The hashes were computed from a clean checkout of tag v2-results and match the current code.
They depend on the pandas and numpy versions that format the CSV text; if an upgrade changes
them, recheck against the tag before updating them.
"""
import hashlib
from pathlib import Path

import pandas as pd
import pytest

from ds import load_config
from ds.analyze import check_report, diagnostics, pilot_report, smallest_detectable, summarize
from ds.mock import run_mock, run_mock_check, run_mock_solve, setting_probs
from ds.score import score_check_rows, score_rows, score_solve_rows
from ds.stimuli import build_cells, build_check_cells, build_solve_cells, make_questions

ROOT = Path(__file__).parents[1]
PINNED = {
    "scores": "e8b7b3216c10585e89a78841d9313c7933c3ea77c4d3fa644da96b8b5f8084d8",
    "summary": "d859e76adf936f58b3ce20ca8f5153f0dc6d3d4fa764e22b71e0b02abe552652",
    "diagnostics": "8a77717538663b99d03c2c6020c74af90f0463a22390caa2dde687c379032a9c",
    "pilot": "34ffd300bc246b8955e709ce29f07bca902f42e2865ec3b370b5581fde457727",
    "check": "a835aefb4d8a1dbf83a184b275f942bba2f688ead68740f06ce981e65af0f4f2",
}


@pytest.fixture(scope="module")
def cfg():
    return load_config(ROOT / "configs" / "experiment.yaml")


def digest(df):
    return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()


@pytest.fixture(scope="module")
def mock_outputs(cfg):
    qs, cells = make_questions(cfg), build_cells(cfg)
    rows = []
    for s in ("planted", "no_effect"):
        rows += score_rows(run_mock(cells, qs, cfg, setting_probs(cfg, s), f"mock-{s}"), qs, cfg["scoring"])
    scores = pd.DataFrame(rows)
    solve = pd.DataFrame(score_solve_rows(run_mock_solve(build_solve_cells(cfg), qs, cfg, "mock-planted", [0]), qs))
    check = pd.DataFrame(score_check_rows(run_mock_check(build_check_cells(cfg), cfg, "mock-planted", [0]),
                                          cfg["check"]["expected"]))
    return {"scores": scores, "summary": summarize(scores, cfg), "diagnostics": diagnostics(scores),
            "pilot": pilot_report(scores[scores.condition == "neutral"], solve, cfg), "check": check_report(check)}


@pytest.mark.parametrize("name", sorted(PINNED))
def test_v2_mock_outputs_unchanged(mock_outputs, name):
    assert digest(mock_outputs[name]) == PINNED[name]


@pytest.mark.skipif(not (ROOT / "results" / "replies.csv").exists(),
                    reason="the real v2 replies are private; unpack v2-results.tar.gz to run this check")
def test_v2_real_analysis_reproduced(cfg):
    # The v2 sensitivity table comes from the mock and is covered above; it is read back, not rerun.
    results = ROOT / "results"
    replies = pd.read_csv(results / "replies.csv", keep_default_na=False)
    scores = pd.DataFrame(score_rows(replies.to_dict("records"), make_questions(cfg), cfg["scoring"]))
    keys = dict(pd.read_csv(results / "pilot_summary.csv")[["model", "key_measure"]].values)
    min_diff = smallest_detectable(pd.read_csv(results / "sensitivity.csv"), cfg["mock"]["sensitivity"]["power"])
    rebuilt = {"scores": scores, "diagnostics": diagnostics(scores),
               "summary": summarize(scores, cfg, keys=keys, min_detectable=min_diff)}
    for name, df in rebuilt.items():
        assert df.to_csv(index=False) == (results / f"{name}.csv").read_text(), name
