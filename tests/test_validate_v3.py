"""scripts/validate_v3.py (step b) on the mock pilot: the stratified blind sheet, labels, kappa, disagreements."""
import copy
import sys
from pathlib import Path

import pandas as pd
import pytest

from ds import load_config

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import run_v3  # noqa: E402
import validate_v3 as v3  # noqa: E402

CONFIG = Path(__file__).parents[1] / "configs" / "experiment_v3.yaml"
MEASURES = ["hint_level", "gives_answer", "asks_attempt"]


@pytest.fixture
def cfg(tmp_path):
    cfg = copy.deepcopy(load_config(CONFIG))
    cfg["paths"] = {k: str(tmp_path / "real" / Path(v).name) for k, v in cfg["paths"].items()}
    cfg["paths"]["mock_dir"] = str(tmp_path / "mock")
    run_v3.run_pilot(cfg, "mock")
    return cfg


def sheet_path(cfg, ai=False):
    return v3.paths(cfg, "mock", ai)[0]


def label_like_scorer(cfg, flips=(), ai=False):
    """Fill the sheet with the scorer's own labels, except hint_level on the first `flips` rows."""
    sheet = pd.read_csv(sheet_path(cfg), keep_default_na=False)
    scored = v3.scored_pilot(cfg, "mock").loc[sheet.row_id]
    for m in MEASURES:
        sheet[f"label_{m}"] = scored[m].values
    for i in flips:
        sheet.loc[i, "label_hint_level"] = (sheet.loc[i, "label_hint_level"] + 1) % 4
    sheet.to_csv(sheet_path(cfg, ai), index=False)
    return sheet


def test_sheet_is_blind_and_stratified(cfg):
    v3.sample(cfg, "mock")
    sheet = pd.read_csv(sheet_path(cfg), keep_default_na=False)
    assert list(sheet.columns) == ["row_id", "question", "reply"] + [f"label_{m}" for m in MEASURES]
    assert len(sheet) == 100 and sheet.row_id.is_unique and (sheet[[f"label_{m}" for m in MEASURES]] == "").all().all()
    scored = v3.scored_pilot(cfg, "mock")
    assert set(scored.loc[sheet.row_id, "condition"]) == {"neutral"}            # tutor replies only, no solve
    picked = scored.loc[sheet.row_id]
    for m in MEASURES:
        for value in v3.VALUES[m]:
            available = (scored[m] == value).sum()
            assert (picked[m] == value).sum() >= min(10, available)
    assert picked.turn.nunique() > 1                                           # every turn can be drawn
    assert not sheet.row_id.is_monotonic_increasing                            # one random order


def test_sample_is_reproducible(cfg):
    v3.sample(cfg, "mock")
    first = pd.read_csv(sheet_path(cfg))
    v3.sample(cfg, "mock")
    assert pd.read_csv(sheet_path(cfg)).equals(first)


def test_strata_are_met_beyond_random_draws():
    """A rare value is filled up to the minimum, which a plain random draw would miss."""
    scored = pd.DataFrame({"hint_level": [2] * 12 + [0] * 988, "gives_answer": 0, "asks_attempt": 0})
    v = {"measures": MEASURES, "n": 100, "min_per_stratum": 10}
    ids = v3.stratified_sample(scored, v, 7)
    assert len(ids) == 100 and (scored.loc[ids, "hint_level"] == 2).sum() >= 10


def test_labelled_sheet_is_never_overwritten(cfg):
    v3.sample(cfg, "mock")
    label_like_scorer(cfg)
    with pytest.raises(SystemExit, match="already has labels"):
        v3.sample(cfg, "mock")


def test_exclude_gives_a_fresh_sample(cfg, tmp_path):
    v3.sample(cfg, "mock")
    old = tmp_path / "round1.csv"
    Path(sheet_path(cfg)).rename(old)
    v3.sample(cfg, "mock", exclude=[old])
    new = pd.read_csv(sheet_path(cfg), keep_default_na=False)
    assert not set(new.reply) & set(pd.read_csv(old, keep_default_na=False).reply)


def test_agree_perfect_and_with_disagreements(cfg):
    v3.sample(cfg, "mock")
    label_like_scorer(cfg)
    v3.agree(cfg, "mock", "author", ai=False)
    _, agreement_path, disagreements_path = v3.paths(cfg, "mock", False)
    report = pd.read_csv(agreement_path).set_index("measure")
    assert (report.agreement == 1).all() and (report.kappa == 1).all() and report.passes.all()
    assert set(report.labeller) == {"author"} and set(report.labeller_kind) == {"human"}
    assert len(pd.read_csv(disagreements_path)) == 0

    label_like_scorer(cfg, flips=range(40))
    v3.agree(cfg, "mock", "author", ai=False)
    report = pd.read_csv(agreement_path).set_index("measure")
    assert report.loc["hint_level", "agreement"] == 0.6 and not report.loc["hint_level", "passes"]
    assert report.loc["hint_level", "kappa"] < 0.6                             # chance-corrected, not raw agreement
    assert report.loc["gives_answer", "passes"] and report.loc["asks_attempt", "passes"]
    wrong = pd.read_csv(disagreements_path, keep_default_na=False)
    assert len(wrong) == 40 and set(wrong.measure) == {"hint_level"} and (wrong.label != wrong.scorer).all()
    assert run_v3.validation_passed(cfg, "mock") is False                      # run_v3 reads this file


def test_ai_labels_kept_apart(cfg):
    v3.sample(cfg, "mock")
    label_like_scorer(cfg, ai=True)
    v3.agree(cfg, "mock", "assistant", ai=True)
    _, agreement_path, _ = v3.paths(cfg, "mock", False)
    assert not agreement_path.exists()                                          # the human file is untouched
    report = pd.read_csv(v3.paths(cfg, "mock", True)[1])
    assert set(report.labeller_kind) == {"ai"} and agreement_path.name.replace(".csv", "_ai.csv") == \
        v3.paths(cfg, "mock", True)[1].name


def test_bad_or_missing_labels_refused(cfg):
    v3.sample(cfg, "mock")
    sheet = label_like_scorer(cfg)
    for bad in ("", "4", "yes"):
        broken = sheet.copy()
        broken["label_hint_level"] = broken["label_hint_level"].astype(str)
        broken.loc[3, "label_hint_level"] = bad
        broken.to_csv(sheet_path(cfg), index=False)
        with pytest.raises(SystemExit, match="label_hint_level"):
            v3.agree(cfg, "mock", "author", ai=False)


def test_rerun_pilot_detected(cfg):
    v3.sample(cfg, "mock")
    label_like_scorer(cfg)
    pilot = run_v3.out_path(cfg, "pilot_csv", "mock")
    df = pd.read_csv(pilot, keep_default_na=False)
    df.loc[df.condition != "solve", "reply"] = "Let's look at it again."
    df.to_csv(pilot, index=False)
    with pytest.raises(SystemExit, match="pilot was rerun"):
        v3.agree(cfg, "mock", "author", ai=False)
