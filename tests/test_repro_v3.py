"""scripts/repro_v3.py: the pure parts of the environment reproduction check (the GPU run is on the pod)."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from ds import load_config
from ds.dialogue import run_dialogues
from ds.stimuli_v3 import build_cells_v3, make_questions_v3

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import repro_v3 as r  # noqa: E402

V3 = load_config(ROOT / "configs" / "experiment_v3.yaml")


class FirstBatch(Exception):
    pass


def test_first_batch_is_v3s():
    """The states v3's own run_dialogues asks for first, in its order, for seed 0."""
    seen = []

    def record(batch):
        seen.extend(batch)
        raise FirstBatch
    questions = make_questions_v3(V3)
    with pytest.raises(FirstBatch):
        run_dialogues(build_cells_v3(V3, questions), 0, record, {q["id"]: q["answer"] for q in questions}, V3)
    states = r.first_batch_states(V3)
    assert len(states) == V3["generation"]["batch_size"] == 16 and states == seen


def test_first_difference():
    assert r.first_difference([1, 2, 3], [1, 2, 3]) is None
    assert r.first_difference([1, 2, 3], [1, 9, 3]) == 1
    assert r.first_difference([1, 2], [1, 2, 3]) == 2


def fake_replies(tmp_path, states, model="m", change=()):
    rows = [{"model": model, "seed": 0, "turn": 1, **{k: s[k] for k in r.V3_KEY}, "reply": f"reply number {i}"}
            for i, s in enumerate(states)]
    rows.append({**rows[0], "seed": 1, "reply": "another seed"})
    path = tmp_path / "replies.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path, [f"reply number {i}" if i not in change else f"reply changed {i}" for i in range(len(states))]


def test_compare_keeps_counts_and_positions_only(tmp_path):
    states = r.first_batch_states(V3)
    path, new = fake_replies(tmp_path, states, change=(3,))
    archived = r.archived_first_batch(path, "m", states)
    result = r.compare(new, archived, states, encode=lambda text: text.split())
    assert result["n"] == 16 and result["identical"] == 15 and not result["passes"]
    assert result["differences"] == [{"cell": {k: states[3][k] for k in r.V3_KEY}, "first_token_differs": 1}]
    assert "reply" not in json.dumps(result)                                     # no reply text kept
    same = r.compare(archived, archived, states, encode=str.split)
    assert same["passes"] and same["share_identical"] == 1.0
    assert len(r.file_sha256(path)) == 64


def test_missing_first_batch_cells_stop_the_check(tmp_path):
    states = r.first_batch_states(V3)
    path, _ = fake_replies(tmp_path, states[:10])
    with pytest.raises(SystemExit, match="not in"):
        r.archived_first_batch(path, "m", states)


V4 = load_config(ROOT / "configs" / "experiment_v4.yaml")
MOCK_REPLY = "Let us think about which operation links the numbers."


def recording_parts(seen):
    """Mock parts that record the settings make_generate receives and when the seed is set."""
    from ds.generate_v4 import mock_parts

    def make(spec, cfg):
        parts = mock_parts(spec, cfg)
        make_generate = parts["make_generate"]
        return {**parts, "make_generate": lambda gen, stash: seen.append(("gen", gen)) or make_generate(gen, stash),
                "set_seed": lambda seed: seen.append(("seed", seed)), "gpu": {"name": "mock"}}
    return make


def test_rerun_uses_arm3s_pipeline_and_settings():
    seen, spec = [], next(s for s in V4["models"] if s["group"] == "v3")
    states, new, gpu = r.rerun_first_batch(spec, V4, V3, recording_parts(seen))
    gens = [g for kind, g in seen if kind == "gen"]
    assert gens == [V4["arm3"]["generation"]] and gens[0]["max_new_tokens"] == 512 and gens[0]["batch_size"] == 16
    assert [k for k, _ in seen] == ["gen", "seed"] and ("seed", 0) in seen      # seed set after loading
    assert states == r.first_batch_states(V3) and new == [MOCK_REPLY] * 16 and gpu == {"name": "mock"}


def test_check_model_on_the_mock(tmp_path):
    spec = next(s for s in V4["models"] if s["group"] == "v3")
    states = r.first_batch_states(V3)
    rows = [{"model": spec["id"], "seed": 0, "turn": 1, **{k: s[k] for k in r.V3_KEY}, "reply": MOCK_REPLY} for s in states]
    path = tmp_path / "replies.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    encode = lambda spec: str.split
    result = r.check_model(spec, V4, V3, path, recording_parts([]), encode)
    assert result["passes"] and result["identical"] == 16 and result["gpu"] == {"name": "mock"}
    rows[5]["reply"] = "Let us think about which number comes next."
    pd.DataFrame(rows).to_csv(path, index=False)
    result = r.check_model(spec, V4, V3, path, recording_parts([]), encode)
    assert not result["passes"] and result["identical"] == 15
    assert result["differences"][0]["first_token_differs"] == 5                  # "operation" vs "number"
