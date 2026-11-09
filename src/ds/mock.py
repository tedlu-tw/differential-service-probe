"""A fake tutor model for testing the pipeline without a GPU.

It ignores the messages and picks a full-answer or a hint reply with a probability
set by the condition. The analysis must find the planted effect and nothing in the
null setting; if it fails either check, the analysis is wrong, not the model.
"""
import numpy as np


def full_answer_reply(question):
    return ("Let's solve it together.\n"
            "Step 1: Write down the numbers given in the question.\n"
            "Step 2: Work through the operations in order.\n"
            f"So the answer is {question['answer']}.")


def hint_reply(question):
    # No digits at all, so the scorer can never find the answer in a hint by accident.
    return ("Good start looking at this one! First, write down what you know.\n"
            "Which operation connects the quantities in the question? "
            "Try the first step and tell me what you get.")


def full_answer_prob(condition, setting, cfg):
    probs = cfg["mock"]["full_answer_prob"][setting]
    return probs.get(condition, probs["default"])


def mock_reply(question, condition, setting, cfg, rng):
    if rng.random() < full_answer_prob(condition, setting, cfg):
        return full_answer_reply(question)
    return hint_reply(question)


def run_mock(cells, questions, cfg, setting):
    """One row per cell and generation seed, in the same columns generate.py writes."""
    by_id = {q["id"]: q for q in questions}
    rows = []
    for seed in cfg["seeds"]["generation"]:
        rng = np.random.default_rng([cfg["seeds"]["mock"], seed])
        for cell in cells:
            reply = mock_reply(by_id[cell["question_id"]], cell["condition"], setting, cfg, rng)
            rows.append({"model": f"mock-{setting}", "seed": seed,
                         "question_id": cell["question_id"], "condition": cell["condition"],
                         "phrasing": cell["phrasing"], "reply": reply})
    return rows
