"""Run the pipeline end to end: get replies, score them, analyse them.

From the repo root:
    PYTHONPATH=src python scripts/run_all.py pilot     # pre-freeze checks a, b, d (neutral only)
    PYTHONPATH=src python scripts/run_all.py check     # pre-freeze check c (manipulation check)
    PYTHONPATH=src python scripts/run_all.py full      # full design: scores, summary, sensitivity

--source mock (default) uses the mock model and writes to paths.mock_dir.
--source gpu generates with the real models (needs a CUDA GPU; run on the pod).
--source csv analyses replies already in results/, e.g. copied off the pod.
"""
import argparse
from pathlib import Path

import pandas as pd

from ds import load_config
from ds.analyze import (check_report, diagnostics, pilot_report, sensitivity, smallest_detectable,
                        summarize)
from ds.mock import run_mock, run_mock_check, run_mock_solve, sensitivity_run, setting_probs
from ds.score import score_check_rows, score_rows, score_solve_rows
from ds.stimuli import build_cells, build_check_cells, build_solve_cells, make_questions

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


# ---------- replies ----------

def mock_replies(cells, questions, cfg, seeds):
    """Both mock settings act as two 'models'."""
    rows = []
    for setting in MOCK_SETTINGS:
        name = f"mock-{setting}"
        of_kind = lambda kind: [c for c in cells if c["kind"] == kind]
        rows += run_mock(of_kind("tutor"), questions, cfg, setting_probs(cfg, setting), name, seeds=seeds)
        rows += run_mock_solve(of_kind("solve"), questions, cfg, name, seeds)
        rows += run_mock_check(of_kind("check"), cfg, name, seeds)
    return rows


def generate_on_gpu(cells, cfg, seeds, path):
    from ds import generate  # torch is needed only here, so mock runs work without it
    generate.check_generation(cfg["generation"], seeds)
    for spec in cfg["models"]:
        generate.run_model(spec, cells, seeds, cfg, path)


def get_replies(cells, questions, cfg, source, key, seeds):
    path = out_path(cfg, key, source)
    if source == "mock":
        write(pd.DataFrame(mock_replies(cells, questions, cfg, seeds)), path)
    elif source == "gpu":
        generate_on_gpu(cells, cfg, seeds, path)
    elif not path.exists():
        raise SystemExit(f"--source csv: {path} does not exist")
    # keep_default_na=False so a reply such as "None" stays text instead of becoming missing.
    return pd.read_csv(path, keep_default_na=False)


# ---------- modes ----------

def run_pilot(cfg, source):
    questions = make_questions(cfg)
    cells = [c for c in build_cells(cfg) if c["condition"] == "neutral"] + build_solve_cells(cfg)
    df = get_replies(cells, questions, cfg, source, "pilot_csv", [cfg["pilot"]["seed"]])
    neutral = score_rows(df[df.condition == "neutral"].to_dict("records"), questions, cfg["scoring"])
    solve = score_solve_rows(df[df.condition == "solve"].to_dict("records"), questions)
    report = pilot_report(pd.DataFrame(neutral), pd.DataFrame(solve), cfg)
    write(report, out_path(cfg, "pilot_summary_csv", source))
    print(report.to_string(index=False))


def run_check(cfg, source):
    df = get_replies(build_check_cells(cfg), make_questions(cfg), cfg, source, "check_csv",
                     [cfg["check"]["seed"]])
    report = check_report(pd.DataFrame(score_check_rows(df.to_dict("records"), cfg["check"]["expected"])))
    write(report, out_path(cfg, "check_summary_csv", source))
    print(report.to_string(index=False))


def load_keys(cfg, source):
    """Key measure per model from the pilot (ceiling/floor rule); configured measure if no pilot."""
    path = out_path(cfg, "pilot_summary_csv", source)
    if not path.exists():
        print(f"no pilot summary at {path}: every model uses the configured measures")
        return {}
    return dict(pd.read_csv(path)[["model", "key_measure"]].values)


def run_full(cfg, source):
    questions, cells = make_questions(cfg), build_cells(cfg)
    df = get_replies(cells, questions, cfg, source, "replies_csv", cfg["seeds"]["generation"])
    scored = pd.DataFrame(score_rows(df.to_dict("records"), questions, cfg["scoring"]))
    write(scored, out_path(cfg, "scores_csv", source))
    write(diagnostics(scored), out_path(cfg, "diagnostics_csv", source))
    # Sensitivity always uses the mock: it describes the design, not a model.
    table = sensitivity(lambda diff, run: sensitivity_run(cells, questions, cfg, diff, run), cfg)
    write(table, out_path(cfg, "sensitivity_csv", source))
    min_diff = smallest_detectable(table, cfg["mock"]["sensitivity"]["power"])
    summary = summarize(scored, cfg, keys=load_keys(cfg, source), min_detectable=min_diff)
    write(summary, out_path(cfg, "summary_csv", source))
    print(summary[summary.role != "exploratory"].to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["pilot", "check", "full"])
    p.add_argument("--source", choices=["mock", "gpu", "csv"], default="mock")
    p.add_argument("--config", default="configs/experiment.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    {"pilot": run_pilot, "check": run_check, "full": run_full}[args.mode](cfg, args.source)


if __name__ == "__main__":
    main()
