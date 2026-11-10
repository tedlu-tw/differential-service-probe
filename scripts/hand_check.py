"""Hand check of the rule-based scorer (protocol: Scoring).

From the repo root, after scripts/run_all.py full:
    PYTHONPATH=src python scripts/hand_check.py sample [--source mock|csv]
        Writes 50 random replies to paths.hand_check_csv with empty label columns.
        The sheet shows only the question, the expected answer and the reply: no condition,
        no cue and no rule scores, so the labeller is as blind as the scorer.
    Fill label_gives_answer and label_asks_attempt with 1 or 0, then:
    PYTHONPATH=src python scripts/hand_check.py agree [--source mock|csv]
        Reports agreement between the hand labels and the scorer, per measure.
"""
import argparse

import pandas as pd

from ds import load_config
from ds.stimuli import make_questions
from run_all import out_path, write

LABELS = {"gives_answer": "label_gives_answer", "asks_attempt": "label_asks_attempt"}


def read_csv(path):
    if not path.exists():
        raise SystemExit(f"{path} does not exist")
    return pd.read_csv(path, keep_default_na=False)


def has_labels(sheet):
    return any((sheet[col].astype(str).str.strip() != "").any() for col in LABELS.values())


def sample(cfg, source):
    out = out_path(cfg, "hand_check_csv", source)
    if out.exists() and has_labels(read_csv(out)):
        raise SystemExit(f"{out} already has labels; move it away before drawing a new sample")
    scores = read_csv(out_path(cfg, "scores_csv", source))
    picked = scores.sample(n=cfg["hand_check"]["n"], random_state=cfg["seeds"]["hand_check"])
    questions = {q["id"]: q for q in make_questions(cfg)}
    sheet = pd.DataFrame({"row_id": picked.index,
                          "question": [questions[i]["text"] for i in picked.question_id],
                          "expected_answer": [questions[i]["answer"] for i in picked.question_id],
                          "reply": picked.reply.values})
    for col in LABELS.values():
        sheet[col] = ""
    write(sheet, out)


def check_sheet(sheet, scores):
    for col in LABELS.values():
        bad = ~sheet[col].astype(str).str.strip().isin(["0", "1"])
        if bad.any():
            raise SystemExit(f"{col} needs 0 or 1 on rows with row_id {list(sheet.row_id[bad])}")
    # Row ids point into scores.csv; if that file was regenerated, they may point elsewhere.
    if not (scores.loc[sheet.row_id, "reply"].values == sheet.reply.values).all():
        raise SystemExit("scores.csv no longer matches the sheet; it was regenerated after sampling")


def agreement_row(measure, human, rule):
    return {"measure": measure, "n": len(human), "agreement": float((human == rule).mean()),
            "both_1": int(((human == 1) & (rule == 1)).sum()),
            "both_0": int(((human == 0) & (rule == 0)).sum()),
            "rule_1_human_0": int(((human == 0) & (rule == 1)).sum()),
            "rule_0_human_1": int(((human == 1) & (rule == 0)).sum())}


def agree(cfg, source):
    sheet = read_csv(out_path(cfg, "hand_check_csv", source))
    scores = read_csv(out_path(cfg, "scores_csv", source))
    check_sheet(sheet, scores)
    rule = scores.loc[sheet.row_id]
    report = pd.DataFrame([agreement_row(m, sheet[col].astype(int).values, rule[m].astype(int).values)
                           for m, col in LABELS.items()])
    write(report, out_path(cfg, "hand_check_agreement_csv", source))
    print(report.to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["sample", "agree"])
    p.add_argument("--source", choices=["mock", "csv"], default="mock")
    p.add_argument("--config", default="configs/experiment.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)
    {"sample": sample, "agree": agree}[args.step](cfg, args.source)


if __name__ == "__main__":
    main()
