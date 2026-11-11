"""Run v3 end to end: get replies, score them, analyse them (docs/protocol_v3.md).

From the repo root, with PYTHONPATH=src:
    python scripts/run_v3.py pilot    # step a: neutral pilot and accuracy with working; step c from its timing
    python scripts/run_v3.py check    # step d: manipulation check, history cells, every question
    python scripts/run_v3.py full     # full design: dialogues, summary, distributions, sensitivity

--source mock (default) uses the mock model and writes to paths.mock_dir.
--source gpu generates with the real models (needs a CUDA GPU; run on the pod). Rows are appended
    after every batch, so a stopped run resumes when it is started again.
--source csv analyses replies already in results_v3/, e.g. copied off the pod.

Step b (scorer validation) is scripts/validate_v3.py. A real full run (gpu or csv) needs the pilot
replies and the step b agreement first, because they decide each model's key measure.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ds import load_config
from ds.analyze import (answer_turn_shares, check_report_v3, diagnostics_v3, dialogues_v3, hint_level_shares,
                        key_measures_v3, pilot_report_v3, replication_v3, sensitivity_v3, shares,
                        smallest_detectable_v3, summarize_v3)
from ds.generate_v3 import generate_on_gpu, single_turn   # torch is imported only inside generate_on_gpu
from ds.mock import run_mock_check_v3, run_mock_dialogues, run_mock_solve_v3, sensitivity_rows_v3, v3_setting
from ds.score import score_check_rows_v3, score_rows_v3, score_solve_rows_v3
from ds.stimuli_v3 import (build_cells_v3, build_check_cells_v3, build_solve_cells_v3, hint_ignore_numbers,
                           list_cells_v3, make_questions_v3, pilot_questions)

MOCK_SETTINGS = ("planted", "no_effect")


# ---------- files ----------

def out_path(cfg, key, source):
    # Mock results keep the real file names but live in their own folder, so they never mix.
    path = Path(cfg["paths"][key])
    return Path(cfg["paths"]["mock_dir"]) / path.name if source == "mock" else path


def write(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"wrote {path} ({len(df)} rows)")


def read_replies(path):
    # keep_default_na=False so a reply such as "None" stays text instead of becoming missing.
    return pd.read_csv(path, keep_default_na=False)


def require(path, source, step):
    """A real run needs the pre-freeze outputs. The mock may go without them, and says so."""
    if path.exists():
        return True
    if source != "mock":
        raise SystemExit(f"{path} is missing: {step} comes before a real full run")
    print(f"no {path}: skipped (mock only)")
    return False


# ---------- replies ----------

def of_kind(cells, kind):
    return [c for c in cells if c["kind"] == kind]


def mock_replies(cells, questions, cfg, seeds):
    """Both mock settings act as two 'models'. Each kind of cell has its own mock."""
    rows = []
    for setting in MOCK_SETTINGS:
        name = f"mock-{setting}"
        rows += run_mock_dialogues(of_kind(cells, "tutor"), questions, cfg, v3_setting(cfg, setting), name, seeds)
        rows += run_mock_solve_v3(of_kind(cells, "solve"), questions, cfg, name, seeds)
        rows += run_mock_check_v3(of_kind(cells, "check"), cfg, name, seeds)
    return rows


def gpu_replies(cells, questions, cfg, seeds, path, runs):
    """runs: cell kind -> its name in the timing log. Solve and check prompts are one-turn dialogues.

    Each kind is its own pass over the models, so the pilot loads each model twice.
    """
    if {c["kind"] for c in cells} != set(runs):
        raise ValueError(f"cells of kinds {sorted({c['kind'] for c in cells})} but runs for {sorted(runs)}")
    answers = {q["id"]: q["answer"] for q in questions}
    for kind, run in runs.items():
        generate_on_gpu(of_kind(cells, kind), answers, cfg if kind == "tutor" else single_turn(cfg), seeds,
                        path, run)


def get_replies(cells, questions, cfg, source, key, seeds, runs):
    path = out_path(cfg, key, source)
    if source == "mock":
        write(pd.DataFrame(mock_replies(cells, questions, cfg, seeds)), path)
    elif source == "gpu":
        gpu_replies(cells, questions, cfg, seeds, path, runs)
    elif not path.exists():
        raise SystemExit(f"--source csv: {path} does not exist")
    return read_replies(path)


def score_tutor(df, questions, cfg, cache=None):
    """Scored tutor-reply rows; solve rows in the same file are left out."""
    rows = df[df["condition"] != "solve"].to_dict("records")
    return score_rows_v3(rows, questions, cfg["scoring"], hint_ignore_numbers(cfg), cache)


# ---------- step a: pilot, and step c from its timing ----------

def pilot_cells(cfg, questions):
    """Neutral cells under the pilot prompt, and the solve prompt, on every second question."""
    picked, p = pilot_questions(cfg, questions), cfg["pilot"]
    tutor = [c for c in build_cells_v3(cfg, picked) if (c["prompt"], c["condition"]) == (p["prompt"], p["condition"])]
    return tutor + build_solve_cells_v3(cfg, picked)


def turn1_distributions(d):
    """Turn-1 hint_level and steps shares per model: the sensitivity base, copied to the config before the freeze."""
    steps = shares(d, "steps", list(range(int(d["steps"].max()) + 1)))
    out = hint_level_shares(d).merge(steps, on=["model", "prompt", "condition"])
    return out.drop(columns=["prompt", "condition"])


def estimate_row(model, reach, per_reply, n_dialogues):
    """Replies needed at each turn times seconds per reply. A turn reached without timing gives NaN hours."""
    row, hours = {"model": model, "n_dialogues": n_dialogues}, 0.0
    for turn, share in reach.items():
        seconds = per_reply.get(turn, np.nan)
        row[f"replies_turn{turn}"] = n_dialogues * share
        row[f"seconds_per_reply_turn{turn}"] = seconds
        hours += 0.0 if share == 0 else n_dialogues * share * seconds / 3600
    return {**row, "gpu_hours": hours}


def compute_estimate(timing, d, n_dialogues, max_turns):
    """Step c per model: pilot seconds per reply at each turn, times the replies the full design needs.

    The share of dialogues reaching turns 2 and 3 comes from the neutral pedagogical pilot. Under the
    plain prompt dialogues may stop sooner, so the estimate is more likely high than low.
    """
    t = timing[timing["run"] == "pilot"].groupby(["model", "turn"])[["seconds", "batch_size"]].sum()
    per_reply = (t["seconds"] / t["batch_size"]).unstack("turn")
    rows = []
    for model, g in d.groupby("model"):
        reach = {k: float((g["n_turns"] >= k).mean()) for k in range(1, max_turns + 1)}
        timed = per_reply.loc[model].to_dict() if model in per_reply.index else {}
        rows.append(estimate_row(model, reach, timed, n_dialogues))
    return pd.DataFrame(rows)


def write_estimate(cfg, source, d, questions):
    path = out_path(cfg, "timing_csv", source)
    if not path.exists():
        print(f"no {path}: step c needs the timing of a real pilot (--source gpu, or csv with the timing log)")
        return
    n = len(questions) * len(list_cells_v3(cfg)) * len(cfg["seeds"]["generation"])
    table = compute_estimate(pd.read_csv(path), d, n, cfg["design"]["max_turns"])
    write(table, out_path(cfg, "compute_estimate_csv", source))
    print(table.to_string(index=False))


def run_pilot(cfg, source):
    questions = make_questions_v3(cfg)
    df = get_replies(pilot_cells(cfg, questions), questions, cfg, source, "pilot_csv", [cfg["pilot"]["seed"]],
                     {"tutor": "pilot", "solve": "pilot_solve"})
    tutor = score_tutor(df, questions, cfg)
    solve = pd.DataFrame(score_solve_rows_v3(df[df["condition"] == "solve"].to_dict("records"), questions))
    d = dialogues_v3(tutor)
    report = pilot_report_v3(tutor, solve, cfg).merge(turn1_distributions(d), on="model")
    write(report, out_path(cfg, "pilot_summary_csv", source))
    print(report.to_string(index=False))
    write_estimate(cfg, source, d, questions)


# ---------- step d: manipulation check ----------

def run_check(cfg, source):
    questions = make_questions_v3(cfg)
    df = get_replies(build_check_cells_v3(cfg, questions), questions, cfg, source, "check_csv",
                     [cfg["check"]["seed"]], {"check": "check"})
    scored = score_check_rows_v3(df.to_dict("records"), cfg["check"]["expected"], cfg["scoring"]["check_parser"])
    report = check_report_v3(pd.DataFrame(scored))
    write(report, out_path(cfg, "check_summary_csv", source))
    print(report.to_string(index=False))


# ---------- full design ----------

def validation_passed(cfg, source):
    """Step b. Returns whether hint_level stays valid; stops if gives_answer or asks_attempt failed."""
    path, v = out_path(cfg, "validate_agreement_csv", source), cfg["validate"]
    if not require(path, source, "step b (scripts/validate_v3.py)"):
        return True
    kappa = pd.read_csv(path).set_index("measure")["kappa"]
    low = [m for m in v["measures"] if not kappa.get(m, np.nan) >= v["kappa_min"]]   # missing or NaN is low
    if set(low) - {"hint_level"}:
        raise SystemExit(f"step b failed for {sorted(set(low) - {'hint_level'})}: revise the rule and repeat step b")
    return "hint_level" not in low


def load_keys_v3(cfg, source, questions):
    """Key measure per model: the floor or ceiling rule on the neutral pilot, with step b deciding the fallback."""
    hint_valid = validation_passed(cfg, source)
    path = out_path(cfg, "pilot_csv", source)
    if not require(path, source, "step a (run_v3.py pilot)"):
        return {}
    keys = key_measures_v3(dialogues_v3(score_tutor(read_replies(path), questions, cfg)), cfg, hint_valid)
    print(f"key measures from the pilot: {keys}")
    return keys


def run_sensitivity(cells, questions, cfg, source):
    """Always the mock: it describes the design, not a model. One score cache serves every run."""
    cache = {}
    table = sensitivity_v3(lambda m, size, run: sensitivity_rows_v3(cells, questions, cfg, m, size, run, cache), cfg)
    write(table, out_path(cfg, "sensitivity_csv", source))
    return smallest_detectable_v3(table, cfg["mock"]["sensitivity"]["power"])


def write_tables(d, df, cfg, source):
    write(d, out_path(cfg, "dialogues_csv", source))
    write(diagnostics_v3(df.to_dict("records")), out_path(cfg, "diagnostics_csv", source))
    write(answer_turn_shares(d), out_path(cfg, "answer_turns_csv", source))
    write(hint_level_shares(d), out_path(cfg, "hint_levels_csv", source))


def run_full(cfg, source):
    questions, cells = make_questions_v3(cfg), build_cells_v3(cfg)
    keys = load_keys_v3(cfg, source, questions)   # first, so a real run without steps a and b stops before the GPU
    df = get_replies(cells, questions, cfg, source, "replies_csv", cfg["seeds"]["generation"], {"tutor": "full"})
    d = dialogues_v3(score_tutor(df, questions, cfg))
    if source != "mock" and set(d["model"]) - set(keys):
        raise SystemExit(f"no pilot for {sorted(set(d['model']) - set(keys))}: run step a for every model")
    write_tables(d, df, cfg, source)
    summary = summarize_v3(d, cfg, keys, run_sensitivity(cells, questions, cfg, source))
    write(summary, out_path(cfg, "summary_csv", source))
    replication = replication_v3(d, cfg)
    write(replication, out_path(cfg, "replication_csv", source))
    print(summary[summary["role"] != "exploratory"].to_string(index=False))
    print(replication.to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["pilot", "check", "full"])
    p.add_argument("--source", choices=["mock", "gpu", "csv"], default="mock")
    p.add_argument("--config", default="configs/experiment_v3.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    {"pilot": run_pilot, "check": run_check, "full": run_full}[args.mode](cfg, args.source)


if __name__ == "__main__":
    main()
