"""v3 must stay reproducible while v4 is added (docs/protocol_v4.md, Definition of done).

Rebuilds v3's analysis from the private v3 replies and compares it, byte for byte, with the files
in results_v3/. The summary must also have the hash fixed in protocol v4. As in v2's check, the
sensitivity table comes from the mock and is read back, not rerun. Runs only where results_v3/
is present (the author's machine); the data are private.
"""
import hashlib
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

from ds import load_config
from ds.analyze import (answer_turn_shares, diagnostics_v3, dialogues_v3, hint_level_shares,
                        min_detectable_by_model, replication_v3, smallest_detectable_v3, summarize_v3)
from ds.stimuli_v3 import make_questions_v3

ROOT = Path(__file__).parents[1]
RESULTS = ROOT / "results_v3"
SUMMARY_SHA256 = "fd6e03ad9daa8ccc26f1bcd537bf20282eb920a0d7c8217d30829234e3beb6d9"   # protocol v4: Why v4
sys.path.insert(0, str(ROOT / "scripts"))
import run_v3  # noqa: E402

pytestmark = pytest.mark.skipif(not (RESULTS / "replies.csv").exists(),
                                reason="the real v3 replies are private; unpack v3-results.tar.gz to run this check")


@pytest.fixture(scope="module")
def rebuilt():
    cfg, here = load_config(ROOT / "configs" / "experiment_v3.yaml"), Path.cwd()
    try:
        os.chdir(ROOT)                                 # the v3 config's paths are relative to the repo root
        questions = make_questions_v3(cfg)
        replies = run_v3.read_replies(RESULTS / "replies.csv")
        d = dialogues_v3(run_v3.score_tutor(replies, questions, cfg))
        keys = run_v3.load_keys_v3(cfg, "csv", questions)
        base_of = run_v3.sensitivity_bases(cfg, sorted(set(d["model"])), "csv")
        found = smallest_detectable_v3(pd.read_csv(RESULTS / "sensitivity.csv"), cfg["mock"]["sensitivity"]["power"])
        summary = summarize_v3(d, cfg, keys, min_detectable_by_model(found, base_of))
    finally:
        os.chdir(here)
    return {"dialogues": d, "diagnostics": diagnostics_v3(replies.to_dict("records")),
            "answer_turns": answer_turn_shares(d), "hint_levels": hint_level_shares(d),
            "summary": summary, "replication": replication_v3(d, cfg)}


@pytest.mark.parametrize("name", ["dialogues", "diagnostics", "answer_turns", "hint_levels", "summary", "replication"])
def test_v3_real_analysis_reproduced(rebuilt, name):
    assert rebuilt[name].to_csv(index=False) == (RESULTS / f"{name}.csv").read_text(), name


def test_v3_summary_has_the_fixed_hash(rebuilt):
    assert hashlib.sha256(rebuilt["summary"].to_csv(index=False).encode()).hexdigest() == SUMMARY_SHA256
