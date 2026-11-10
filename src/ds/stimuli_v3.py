"""v3 stimuli: questions, cells per system prompt, solve and check prompts (docs/protocol_v3.md).

Built on the v2 functions in stimuli.py, which stay unchanged so v2 stays reproducible.
The cue and the system prompt live only in the messages built here; the scorer never sees them.
"""
from pathlib import Path

import numpy as np

from ds import load_config
from ds.stimuli import (FAMILIES, history_numbers, history_setup, history_turns, make_questions,
                        numbers_in, student_message)

ROOT = Path(__file__).resolve().parents[2]
MAX_TRIES = 100_000


# ---------- questions ----------

def make_linear_v3(rng, low, high):
    # Same wording as v2; only the range of Maya's number changes (protocol: minimum answer).
    x, a, b = int(rng.integers(low, high + 1)), int(rng.integers(2, 10)), int(rng.integers(1, 41))
    text = (f"Maya thinks of a number. She multiplies it by {a} and then adds {b}. "
            f"The result is {a * x + b}. What is Maya's number?")
    return text, x


def templates(cfg):
    low, high = cfg["questions"]["linear_answer_range"]
    return {**FAMILIES, "linear_equation": lambda rng: make_linear_v3(rng, low, high)}


def cue_numbers(cfg):
    return numbers_in(" ".join(text for phrasings in cfg["cues"].values() for text in phrasings.values()))


def v2_question_texts(cfg):
    return {q["text"] for q in make_questions(load_config(ROOT / cfg["questions"]["exclude_config"]))}


def is_valid_v3(text, answer, seen, blocked, min_answer):
    # seen holds v2 texts too, so no v3 question repeats a v2 question.
    # blocked holds the cue numbers: a tutor echoing "40%" must not count as giving the answer.
    return (text not in seen and answer >= min_answer
            and float(answer) not in numbers_in(text) and float(answer) not in blocked)


def make_family_v3(name, make, n, rng, seen, blocked, min_answer):
    items, tries = [], 0
    while len(items) < n:
        tries += 1
        if tries > MAX_TRIES:
            raise ValueError(f"could not draw {n} valid {name} questions")
        text, answer = make(rng)
        if is_valid_v3(text, answer, seen, blocked, min_answer):
            seen.add(text)
            items.append({"id": f"v3_{name}_{len(items):02d}", "family": name, "text": text, "answer": answer})
    return items


def make_questions_v3(cfg):
    q, rng = cfg["questions"], np.random.default_rng(cfg["seeds"]["questions"])
    make, seen, blocked = templates(cfg), v2_question_texts(cfg), cue_numbers(cfg)
    return [item for name in q["families"]
            for item in make_family_v3(name, make[name], q["per_family"], rng, seen, blocked, q["min_answer"])]


def pilot_questions(cfg, questions):
    # Every second question; questions are ordered by family, so each family keeps half.
    return questions[::cfg["pilot"]["question_step"]]


# ---------- hint_level ignore list ----------

def hint_ignore_numbers(cfg):
    """Cue numbers and every history-pool number: the same list in every condition."""
    pool, _ = history_setup(cfg, [])
    return cue_numbers(cfg).union(*(history_numbers(item) for item in pool))


# ---------- chat messages ----------

def system_text(cfg, prompt):
    return cfg["prompts"]["system"][prompt]


def build_messages_v3(question, condition, phrasing, prompt, cfg, pool, assignment):
    """Turn-1 messages for one cell. dialogue.py adds the scripted turns 2 and 3."""
    messages = [{"role": "system", "content": system_text(cfg, prompt)}]
    if condition.startswith("history_"):
        items = [pool[i] for i in assignment[question["id"]][phrasing]]
        messages += history_turns(items, condition == "history_success", cfg["prompts"]["tutor_ack"])
    prefix = cfg["cues"][condition][phrasing]
    messages.append({"role": "user", "content": student_message(question["text"], prefix)})
    return messages


def list_cells_v3(cfg):
    return [(prompt, cond, ph) for prompt, conds in cfg["design"]["conditions"].items()
            for cond in conds for ph in cfg["cues"][cond]]


def build_cells_v3(cfg, questions=None):
    """One row per question x prompt x cell. Pass questions to build a subset (e.g. the pilot)."""
    every = make_questions_v3(cfg)
    pool, assignment = history_setup(cfg, every)
    rows = []
    for q in questions or every:
        for prompt, condition, phrasing in list_cells_v3(cfg):
            rows.append({"kind": "tutor", "question_id": q["id"], "prompt": prompt, "condition": condition,
                         "phrasing": phrasing,
                         "messages": build_messages_v3(q, condition, phrasing, prompt, cfg, pool, assignment)})
    return rows


# ---------- pre-freeze steps ----------

def build_solve_cells_v3(cfg, questions):
    """Accuracy with working: the plain prompt, so a withholding tutor cannot hide the answer."""
    prompt = cfg["prompts"]["solve_system"]
    return [{"kind": "solve", "question_id": q["id"], "prompt": prompt, "condition": "solve", "phrasing": "A",
             "messages": [{"role": "system", "content": system_text(cfg, prompt)},
                          {"role": "user", "content": f"{q['text']}\n\n{cfg['prompts']['solve_instruction']}"}]}
            for q in questions]


def build_check_cells_v3(cfg, questions):
    """Manipulation check: the history turns under the check prompt, then the check question."""
    every = make_questions_v3(cfg)
    pool, assignment = history_setup(cfg, every)
    prompt, rows = cfg["check"]["prompt"], []
    for q in questions:
        for condition in cfg["check"]["expected"]:
            for phrasing in cfg["cues"][condition]:
                messages = build_messages_v3(q, condition, phrasing, prompt, cfg, pool, assignment)[:-1]
                messages.append({"role": "user", "content": cfg["prompts"]["check_question"]})
                rows.append({"kind": "check", "question_id": q["id"], "prompt": prompt,
                             "condition": condition, "phrasing": phrasing, "messages": messages})
    return rows
