"""Step b: validate the v3 scorer against hand labels (docs/protocol_v3.md, Pre-freeze steps).

From the repo root, with PYTHONPATH=src, after scripts/run_v3.py pilot:
    python scripts/validate_v3.py sample [--source mock|csv] [--exclude OLD_SHEET ...]
        Draws 100 tutor replies from the pilot, stratified by the scorer's own output, and writes
        them to paths.validate_csv with empty label columns. The sheet shows the question and the
        reply only: no model, condition, turn or scorer label. --exclude leaves out every reply in
        an earlier sheet, for the fresh sample after a rule is revised.
    Fill label_hint_level (0 to 3), label_gives_answer and label_asks_attempt (0 or 1), then:
    python scripts/validate_v3.py agree --labeller NAME [--ai] [--source mock|csv]
        Writes raw agreement and Cohen's kappa per measure, and every disagreement.

Labels from an AI assistant are reported separately (--ai): label a copy of the sheet named
validation_sheet_ai.csv; its outputs get the same _ai suffix. Only the human agreement file is
read by run_v3.py full. Agreement on a stratified sample describes the scorer on that sample,
not the base rate.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ds import load_config
from ds.analyze import cohen_kappa
from ds.score import score_rows_v3
from ds.stimuli_v3 import hint_ignore_numbers, make_questions_v3
from run_v3 import out_path, read_replies, write

VALUES = {"hint_level": [0, 1, 2, 3], "gives_answer": [0, 1], "asks_attempt": [0, 1]}


# ---------- files ----------

def ai_path(path):
    return path.with_name(f"{path.stem}_ai{path.suffix}")


def paths(cfg, source, ai):
    keys = ("validate_csv", "validate_agreement_csv", "validate_disagreements_csv")
    found = [out_path(cfg, key, source) for key in keys]
    return [ai_path(p) for p in found] if ai else found


def read_csv(path):
    if not path.exists():
        raise SystemExit(f"{path} does not exist")
    return pd.read_csv(path, keep_default_na=False)


def label_column(measure):
    return f"label_{measure}"


def scored_pilot(cfg, source):
    """The scorer's output on every tutor reply of the pilot, indexed by its row in the pilot CSV."""
    df = read_csv(out_path(cfg, "pilot_csv", source))
    tutor = df[df["condition"] != "solve"]
    rows = score_rows_v3(tutor.to_dict("records"), make_questions_v3(cfg), cfg["scoring"], hint_ignore_numbers(cfg))
    return pd.DataFrame(rows, index=tutor.index)


# ---------- sample ----------

def stratified_sample(scored, v, seed):
    """Row ids: every value of every measure at least min_per_stratum times where the pilot has
    that many, then random rows up to n. Returned in one random order, so strata are not grouped."""
    order = np.random.default_rng(seed).permutation(scored.index.to_numpy())
    chosen = set()
    for measure in v["measures"]:
        for value in VALUES[measure]:
            have = sum(scored.at[i, measure] == value for i in chosen)
            more = [i for i in order if scored.at[i, measure] == value and i not in chosen]
            chosen.update(more[:max(0, v["min_per_stratum"] - have)])
    if len(chosen) > v["n"]:
        raise ValueError(f"the strata need {len(chosen)} replies, more than n = {v['n']}")
    rest = [i for i in order if i not in chosen]
    chosen.update(rest[:v["n"] - len(chosen)])
    return [i for i in order if i in chosen]


def has_labels(sheet, measures):
    return any((sheet[label_column(m)].astype(str).str.strip() != "").any() for m in measures)


def sample(cfg, source, exclude=()):
    sheet_path, v = paths(cfg, source, False)[0], cfg["validate"]
    if sheet_path.exists() and has_labels(read_csv(sheet_path), v["measures"]):
        raise SystemExit(f"{sheet_path} already has labels; move it away before drawing a new sample")
    scored = scored_pilot(cfg, source)
    seen = set().union(*(set(read_csv(Path(p))["reply"]) for p in exclude))
    scored = scored[~scored["reply"].isin(seen)]
    ids = stratified_sample(scored, v, cfg["seeds"]["validate"])
    texts = {q["id"]: q["text"] for q in make_questions_v3(cfg)}
    sheet = pd.DataFrame({"row_id": ids, "question": [texts[q] for q in scored.loc[ids, "question_id"]],
                          "reply": scored.loc[ids, "reply"].values})
    for m in v["measures"]:
        sheet[label_column(m)] = ""
    write(sheet, sheet_path)
    for m in v["measures"]:      # counts per value only; the sheet itself shows no scorer label
        counts = scored.loc[ids, m].value_counts().sort_index()
        print(f"{m}: " + ", ".join(f"{value}={n}" for value, n in counts.items()))


# ---------- agree ----------

def read_labels(sheet, measures):
    """Integer labels per measure. Every cell must hold one of the measure's values."""
    out = {}
    for m in measures:
        col = sheet[label_column(m)].astype(str).str.strip()
        bad = ~col.isin([str(x) for x in VALUES[m]])
        if bad.any():
            raise SystemExit(f"{label_column(m)} needs one of {VALUES[m]} on rows with row_id {list(sheet.row_id[bad])}")
        out[m] = col.astype(int).values
    return out


def matched_scores(sheet, scored):
    # Row ids point into the pilot CSV; if the pilot was rerun, they may point elsewhere.
    if not sheet.row_id.isin(scored.index).all() or \
            not (scored.loc[sheet.row_id, "reply"].values == sheet.reply.values).all():
        raise SystemExit("the pilot replies no longer match the sheet; the pilot was rerun after sampling")
    return scored.loc[sheet.row_id]


def agreement_rows(labels, rule, labeller, kind, kappa_min):
    rows = []
    for m, human in labels.items():
        kappa = cohen_kappa(human, rule[m].values)
        rows.append({"measure": m, "labeller": labeller, "labeller_kind": kind, "n": len(human),
                     "agreement": float(np.mean(human == rule[m].values)), "kappa": kappa,
                     "kappa_min": kappa_min, "passes": bool(kappa >= kappa_min)})
    return pd.DataFrame(rows)


def disagreement_rows(sheet, labels, rule):
    """Every reply where the label and the scorer differ, one row per measure. The turn is shown now."""
    rows = []
    for m, human in labels.items():
        for i in np.flatnonzero(human != rule[m].values):
            rows.append({"measure": m, "row_id": sheet.row_id.iloc[i], "model": rule["model"].iloc[i],
                         "turn": rule["turn"].iloc[i], "label": human[i], "scorer": rule[m].iloc[i],
                         "question": sheet.question.iloc[i], "reply": sheet.reply.iloc[i]})
    return pd.DataFrame(rows, columns=["measure", "row_id", "model", "turn", "label", "scorer", "question", "reply"])


def print_rules(report):
    """The two step b rules (protocol: Pre-freeze steps, b)."""
    failed = set(report.loc[~report.passes, "measure"])
    if "hint_level" in failed:
        print("hint_level kappa is below the minimum: hint_level becomes exploratory; the fallback is steps")
    for m in sorted(failed - {"hint_level"}):
        print(f"{m} kappa is below the minimum: revise the rule and repeat step b on a fresh sample")
    if "gives_answer" in failed:
        print("gives_answer decides when a dialogue stops: rerun the pilot (step a) after revising it")


def agree(cfg, source, labeller, ai):
    sheet_path, agreement_path, disagreements_path = paths(cfg, source, ai)
    sheet, measures = read_csv(sheet_path), cfg["validate"]["measures"]
    labels = read_labels(sheet, measures)
    rule = matched_scores(sheet, scored_pilot(cfg, source))
    report = agreement_rows(labels, rule, labeller, "ai" if ai else "human", cfg["validate"]["kappa_min"])
    write(report, agreement_path)
    write(disagreement_rows(sheet, labels, rule), disagreements_path)
    print(report.to_string(index=False))
    print_rules(report)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["sample", "agree"])
    p.add_argument("--source", choices=["mock", "csv"], default="mock")
    p.add_argument("--config", default="configs/experiment_v3.yaml")
    p.add_argument("--exclude", nargs="*", default=[], help="earlier sheets whose replies are left out")
    p.add_argument("--labeller", help="who labelled the sheet (agree)")
    p.add_argument("--ai", action="store_true", help="labels from an AI assistant, reported separately")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.step == "sample":
        sample(cfg, args.source, args.exclude)
    elif not args.labeller:
        raise SystemExit("agree needs --labeller")
    else:
        agree(cfg, args.source, args.labeller, args.ai)


if __name__ == "__main__":
    main()
