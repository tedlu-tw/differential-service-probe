"""A fake tutor model for testing the pipeline without a GPU.

It ignores the messages and picks a full-answer or a hint reply with a probability
set by the condition. The analysis must find the planted effect and nothing in the
no-effect setting; if it fails either check, the analysis is wrong, not the model.
It also answers the solve-only and manipulation-check prompts of the pre-freeze checks.
"""
import zlib

import numpy as np
import pandas as pd

from ds.dialogue import run_dialogues
from ds.score import score_rows, score_rows_v3
from ds.stimuli_v3 import hint_ignore_numbers


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


def solve_reply(question, accuracy, rng):
    return str(question["answer"] if rng.random() < accuracy else question["answer"] + 1)


def check_reply(expected, accuracy, rng):
    other = "no" if expected == "yes" else "yes"
    first, second = (expected if rng.random() < accuracy else other for _ in range(2))
    return f"1. {first.capitalize()}\n2. {second.capitalize()}"


def mock_rows(cells, cfg, name, seeds, reply_fn, run=0):
    """One row per cell and seed, in the same columns generate.py writes."""
    rows = []
    for seed in seeds:
        rng = np.random.default_rng([cfg["seeds"]["mock"], run, seed])
        for cell in cells:
            rows.append({"model": name, "seed": seed, "question_id": cell["question_id"],
                         "condition": cell["condition"], "phrasing": cell["phrasing"],
                         "reply": reply_fn(cell, rng), "truncated": False, "system_mode": "mock"})
    return rows


def run_mock(cells, questions, cfg, probs, name, run=0, seeds=None):
    by_id = {q["id"]: q for q in questions}
    fn = lambda cell, rng: mock_reply(by_id[cell["question_id"]], cell["condition"], probs, rng)
    return mock_rows(cells, cfg, name, seeds or cfg["seeds"]["generation"], fn, run)


def run_mock_solve(cells, questions, cfg, name, seeds):
    by_id = {q["id"]: q for q in questions}
    fn = lambda cell, rng: solve_reply(by_id[cell["question_id"]], cfg["mock"]["solve_accuracy"], rng)
    return mock_rows(cells, cfg, name, seeds, fn)


def run_mock_check(cells, cfg, name, seeds):
    expected = cfg["check"]["expected"]
    fn = lambda cell, rng: check_reply(expected[cell["condition"]], cfg["mock"]["check_accuracy"], rng)
    return mock_rows(cells, cfg, name, seeds, fn)


def sensitivity_run(cells, questions, cfg, diff, run):
    """One simulated dataset for the primary contrast, scored like real replies."""
    primary = cfg["analysis"]["primary"]
    # Only the two primary conditions are needed, which keeps 200 runs per size quick.
    keep = [c for c in cells if c["condition"] in (primary["a"], primary["b"])]
    rows = run_mock(keep, questions, cfg, sensitivity_probs(cfg, diff), f"mock-diff-{diff}", run)
    return pd.DataFrame(score_rows(rows, questions, cfg["scoring"]))


# =====================================================================================
# v3 multi-turn mock (docs/protocol_v3.md). New functions only; the v2 mock above is unchanged.
#
# Each dialogue draws its plan once: the turn-1 hint level (3 = the answer), the number of step
# lines at turn 1, and whether it gives in at turn 2 or 3. The plan comes from a random stream
# seeded by the dialogue's own key, so batch order and resuming never change a dialogue.
# The mock ignores the system prompt; parameters are set per condition only.
# =====================================================================================

def v3_setting(cfg, setting):
    return cfg["mock"]["settings"][setting]


def dialogue_rng(cfg, state, run):
    key = "|".join(str(state[k]) for k in ("question_id", "prompt", "condition", "phrasing"))
    return np.random.default_rng([cfg["seeds"]["mock"], run, int(state["seed"]), zlib.crc32(key.encode())])


def dialogue_plan(state, params, cfg, run):
    p = params.get(state["condition"], params["default"])
    rng = dialogue_rng(cfg, state, run)
    level, steps, later = int(rng.choice(4, p=p["turn1_levels"])), int(rng.choice(4, p=p["steps"])), rng.random(2)
    if level == 3:
        answer_turn = 1
    elif later[0] < p["later_give_in"][0]:
        answer_turn = 2
    elif later[1] < p["later_give_in"][1]:
        answer_turn = 3
    else:
        answer_turn = None
    return {"level": level, "steps": steps, "answer_turn": answer_turn}


# The texts below are built so the v3 scorer reads exactly the planned level and step count:
# no digits or method words except where a level needs them, and step lines without a number.
LEVEL_TEXT = {
    0: "Let's think about what the question is asking. What do you notice?",
    1: "Work out which operation links the quantities: add, subtract, multiply or divide?",
}


def turn1_reply(question, plan):
    if plan["level"] == 3:
        body = f"So the answer is {question['answer']}."
    elif plan["level"] == 2:
        # Larger than any number in a question, the history pool or a cue, and never the answer.
        body = f"So far the running value is {question['answer'] + 1000}. What comes next?"
    else:
        body = LEVEL_TEXT[plan["level"]]
    return "\n".join(["Step: read the question again."] * plan["steps"] + [body])


def later_reply(question, plan, turn):
    if plan["answer_turn"] == turn:
        return f"Alright. The answer is {question['answer']}."
    return LEVEL_TEXT[0]


def mock_reply_fn(questions, params, cfg, run=0):
    """A reply_fn for dialogue.run_dialogues."""
    by_id = {q["id"]: q for q in questions}

    def reply_fn(batch):
        out = []
        for state in batch:
            question, plan = by_id[state["question_id"]], dialogue_plan(state, params, cfg, run)
            text = turn1_reply(question, plan) if state["turn"] == 1 else later_reply(question, plan, state["turn"])
            out.append((text, False))
        return out
    return reply_fn


def run_mock_dialogues(cells, questions, cfg, params, name, seeds, run=0):
    answers, rows = {q["id"]: q["answer"] for q in questions}, []
    fn = mock_reply_fn(questions, params, cfg, run)
    for seed in seeds:
        rows += run_dialogues(cells, seed, fn, answers, cfg, extra={"model": name, "system_mode": "mock"})
    return rows


# ---------- solve and check (v3 formats) ----------

def solve_reply_v3(question, accuracy, rng):
    value = question["answer"] if rng.random() < accuracy else question["answer"] + 1
    return f"Work through the problem one operation at a time.\n{value}"


def check_reply_v3(expected, cfg, rng):
    """Two items, each right with check_accuracy; equal items are sometimes written combined."""
    other = "no" if expected == "yes" else "yes"
    first, second = (expected if rng.random() < cfg["mock"]["check_accuracy"] else other for _ in range(2))
    if first == second and rng.random() < cfg["mock"]["check_combined_share"]:
        return "Yes, both of your answers were correct." if first == "yes" else "No, neither answer was correct."
    return f"1. {first.capitalize()}\n2. {second.capitalize()}"


def mock_rows_v3(cells, cfg, name, seeds, reply_fn):
    rows = []
    for seed in seeds:
        rng = np.random.default_rng([cfg["seeds"]["mock"], seed])
        rows += [{"model": name, "seed": seed, "question_id": c["question_id"], "prompt": c["prompt"],
                  "condition": c["condition"], "phrasing": c["phrasing"], "turn": 1, "reply": reply_fn(c, rng),
                  "truncated": False, "system_mode": "mock"} for c in cells]
    return rows


def run_mock_solve_v3(cells, questions, cfg, name, seeds):
    by_id = {q["id"]: q for q in questions}
    fn = lambda cell, rng: solve_reply_v3(by_id[cell["question_id"]], cfg["mock"]["solve_accuracy"], rng)
    return mock_rows_v3(cells, cfg, name, seeds, fn)


def run_mock_check_v3(cells, cfg, name, seeds):
    fn = lambda cell, rng: check_reply_v3(cfg["check"]["expected"][cell["condition"]], cfg, rng)
    return mock_rows_v3(cells, cfg, name, seeds, fn)


# ---------- sensitivity parameters ----------

def shift_up(probs, shift):
    """Move each value below the top up by one with probability q, so the mean rises by exactly shift."""
    p, q = list(probs), shift / sum(probs[:-1])
    if not 0 <= q <= 1:
        raise ValueError(f"a shift of {shift} is too large for {probs}")
    return [p[0] * (1 - q)] + [p[k] * (1 - q) + p[k - 1] * q for k in range(1, len(p) - 1)] + [p[-1] + p[-2] * q]


def later_for_rate(turn1_levels, rate, last):
    """later_give_in for turns 2 and 3 so that gave_by_turn2 equals rate."""
    p1 = turn1_levels[3]
    if not p1 <= rate <= 1:
        raise ValueError(f"gave_by_turn2 of {rate} is below the turn-1 answer rate {p1}")
    return [(rate - p1) / (1 - p1), last]


def sensitivity_params(cfg, measure, size):
    """Mock parameters that plant one difference of one size in one measure, primary contrast only."""
    sens = cfg["mock"]["sensitivity"]
    levels, steps, base = sens["base_turn1_levels"], sens["base_steps"], sens["gave_by_turn2"]["base"]
    default = {"turn1_levels": levels, "steps": steps, "later_give_in": later_for_rate(levels, base, 0.5)}
    planted = dict(default)
    if measure == "gave_by_turn2":
        planted["later_give_in"] = later_for_rate(levels, base + size, 0.5)
    elif measure == "hint_level":
        # Raising some turn-1 replies to level 3 also changes gave_by_turn2; only hint_level is analysed here.
        planted["turn1_levels"] = shift_up(levels, size)
    elif measure == "steps":
        planted["steps"] = shift_up(steps, size)
    else:
        raise ValueError(f"no sensitivity for {measure}")
    return {"default": default, cfg["analysis"]["primary"]["a"]: planted}


def sensitivity_rows_v3(cells, questions, cfg, measure, size, run, cache=None):
    """One simulated dataset of the primary contrast, every tutor reply scored like a real one.

    cache: passed to score_rows_v3, so the same mock texts are scored once across all runs.
    """
    primary = cfg["analysis"]["primary"]
    keep = [c for c in cells if c["prompt"] == primary["prompt"] and c["condition"] in (primary["a"], primary["b"])]
    rows = run_mock_dialogues(keep, questions, cfg, sensitivity_params(cfg, measure, size),
                              f"mock-{measure}-{size}", cfg["seeds"]["generation"], run)
    return score_rows_v3(rows, questions, cfg["scoring"], hint_ignore_numbers(cfg), cache)
