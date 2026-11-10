"""A fake tutor model for testing the pipeline without a GPU.

It ignores the messages and picks a full-answer or a hint reply with a probability
set by the condition. The analysis must find the planted effect and nothing in the
no-effect setting; if it fails either check, the analysis is wrong, not the model.
"""
import numpy as np
import pandas as pd

from ds.score import score_rows


def full_answer_reply(question):
    return ("Let's solve it together.\n"
            "Step 1: Write down the numbers given in the question.\n"
            "Step 2: Work through the operations in order.\n"
            f"So the answer is {question['answer']}.")


def hint_reply(question):
    # No digits or number words, so the scorer can never find the answer in a hint by accident.
    return ("Good start looking at this one! First, write down what you know.\n"
            "Which operation connects the quantities in the question? "
            "Try the first step and tell me what you get.")


def setting_probs(cfg, setting):
    return cfg["mock"]["full_answer_prob"][setting]


def sensitivity_probs(cfg, diff):
    base = cfg["mock"]["sensitivity"]["base"]
    # Only the primary contrast is planted; every other condition stays at the base rate.
    return {"default": base, cfg["analysis"]["primary"]["a"]: base + diff}


def mock_reply(question, condition, probs, rng):
    if rng.random() < probs.get(condition, probs["default"]):
        return full_answer_reply(question)
    return hint_reply(question)


def run_mock(cells, questions, cfg, probs, name, run=0):
    """One row per cell and generation seed, in the same columns generate.py writes."""
    by_id = {q["id"]: q for q in questions}
    rows = []
    for seed in cfg["seeds"]["generation"]:
        rng = np.random.default_rng([cfg["seeds"]["mock"], run, seed])
        for cell in cells:
            reply = mock_reply(by_id[cell["question_id"]], cell["condition"], probs, rng)
            rows.append({"model": name, "seed": seed, "question_id": cell["question_id"],
                         "condition": cell["condition"], "phrasing": cell["phrasing"],
                         "reply": reply, "truncated": False})
    return rows


def sensitivity_run(cells, questions, cfg, diff, run):
    """One simulated dataset for the primary contrast, scored like real replies."""
    primary = cfg["analysis"]["primary"]
    # Only the two primary conditions are needed, which keeps 200 runs per size quick.
    keep = [c for c in cells if c["condition"] in (primary["a"], primary["b"])]
    rows = run_mock(keep, questions, cfg, sensitivity_probs(cfg, diff), f"mock-diff-{diff}", run)
    return pd.DataFrame(score_rows(rows, questions, cfg["scoring"]))
