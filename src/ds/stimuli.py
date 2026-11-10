"""Build the test questions, the history pool, and the chat messages for every cell.

The cue lives only in the messages built here. The scorer later gets only the
question text and the reply, never these messages.
"""
import re

import numpy as np

NUMBER = re.compile(r"\d+(?:\.\d+)?")


def numbers_in(text):
    return {float(n) for n in NUMBER.findall(text)}


# ---------- test questions ----------

def make_two_step(rng):
    a, b = int(rng.integers(50, 301)), int(rng.integers(10, 50))
    c, d = int(rng.integers(2, 10)), int(rng.integers(5, 21))
    text = (f"A library has {a} books. It gives away {b} books and then receives "
            f"{c} boxes with {d} new books in each box. How many books does the library have now?")
    return text, a - b + c * d


def make_percentage(rng):
    # Multiples of 20 and 5 keep the sale price a whole number of dollars.
    price, off = 20 * int(rng.integers(2, 21)), 5 * int(rng.integers(2, 13))
    text = (f"A jacket costs {price} dollars. It is on sale for {off}% off. "
            f"What is the sale price in dollars?")
    return text, price * (100 - off) // 100


def make_linear(rng):
    x, a, b = int(rng.integers(3, 26)), int(rng.integers(2, 10)), int(rng.integers(1, 41))
    text = (f"Maya thinks of a number. She multiplies it by {a} and then adds {b}. "
            f"The result is {a * x + b}. What is Maya's number?")
    return text, x


def make_rate(rng):
    speed, hours = int(rng.integers(8, 31)), int(rng.integers(2, 7))
    text = (f"A cyclist rides at a steady {speed} km per hour for {hours} hours. "
            f"How many km does the cyclist ride in total?")
    return text, speed * hours


FAMILIES = {"two_step": make_two_step, "percentage": make_percentage,
            "linear_equation": make_linear, "rate": make_rate}


def is_valid(text, answer, seen):
    # gives_answer looks for the answer number in the reply. If the question already
    # contains it, a reply that only restates the question would count as an answer.
    return text not in seen and float(answer) not in numbers_in(text)


def make_family(name, n, rng):
    items, seen = [], set()
    while len(items) < n:
        text, answer = FAMILIES[name](rng)
        if is_valid(text, answer, seen):
            seen.add(text)
            items.append({"id": f"{name}_{len(items):02d}", "family": name,
                          "text": text, "answer": answer})
    return items


def make_questions(cfg):
    rng = np.random.default_rng(cfg["seeds"]["questions"])
    q = cfg["questions"]
    return [item for name in q["families"] for item in make_family(name, q["per_family"], rng)]


# ---------- history pool (earlier problems, separate from the test questions) ----------

def make_order_ops(rng):
    # Adding before multiplying is an ordinary order-of-operations slip.
    a, b, c = int(rng.integers(2, 21)), int(rng.integers(2, 10)), int(rng.integers(2, 10))
    right, wrong = a + b * c, (a + b) * c
    return {"text": f"What is {a} + {b} × {c}?",
            "right": f"First {b} × {c} = {b * c}, then {a} + {b * c} = {right}. So the answer is {right}.",
            "wrong": f"First {a} + {b} = {a + b}, then {a + b} × {c} = {wrong}. So the answer is {wrong}.",
            "right_value": right, "wrong_value": wrong}


def make_perimeter(rng):
    # Adding the two sides once and forgetting to double is an ordinary perimeter slip.
    length, width = int(rng.integers(3, 21)), int(rng.integers(2, 15))
    right, wrong = 2 * (length + width), length + width
    return {"text": f"A rectangle is {length} cm long and {width} cm wide. What is its perimeter in cm?",
            "right": f"The perimeter is 2 × ({length} + {width}) = {right}. So the answer is {right} cm.",
            "wrong": f"The perimeter is {length} + {width} = {wrong}. So the answer is {wrong} cm.",
            "right_value": right, "wrong_value": wrong}


HISTORY_KINDS = (make_order_ops, make_perimeter)


def make_history_pool(cfg, rng):
    pool, seen = [], set()
    while len(pool) < cfg["history"]["pool_size"]:
        item = HISTORY_KINDS[len(pool) % len(HISTORY_KINDS)](rng)
        if item["text"] not in seen:
            seen.add(item["text"])
            pool.append({"id": f"h_{len(pool):02d}", **item})
    return pool


def history_numbers(item):
    return numbers_in(" ".join([item["text"], item["right"], item["wrong"]]))


def eligible_history(answer, pool):
    # A reply that corrects the earlier mistakes may quote their numbers. If one equals
    # the current answer, gives_answer would fire only in history_struggle.
    return [i for i, item in enumerate(pool) if float(answer) not in history_numbers(item)]


def assign_history(questions, pool, turns, rng):
    """Phrasing A and B get different earlier problems; success and struggle share them."""
    out = {}
    for q in questions:
        eligible = eligible_history(q["answer"], pool)
        if len(eligible) < 2 * turns:
            raise ValueError(f"only {len(eligible)} history problems avoid the answer of {q['id']}")
        picks = [int(i) for i in rng.choice(eligible, size=2 * turns, replace=False)]
        out[q["id"]] = {"A": picks[:turns], "B": picks[turns:]}
    return out


# ---------- chat messages ----------

def list_cells(cfg):
    return [(cond, ph) for cond, phrasings in cfg["cues"].items() for ph in phrasings]


def student_message(question_text, prefix):
    # The cue goes before the question in the same message; the question text is never edited.
    return f"{prefix}\n\n{question_text}" if prefix else question_text


def history_turns(items, correct, ack):
    turns = []
    for item in items:
        work = item["right"] if correct else item["wrong"]
        turns.append({"role": "user", "content": f"{item['text']}\nMy answer: {work}"})
        turns.append({"role": "assistant", "content": ack})
    return turns


def build_messages(question, condition, phrasing, cfg, pool, assignment):
    messages = [{"role": "system", "content": cfg["prompts"]["system"]}]
    if condition.startswith("history_"):
        items = [pool[i] for i in assignment[question["id"]][phrasing]]
        correct = condition == "history_success"
        messages += history_turns(items, correct, cfg["prompts"]["tutor_ack"])
    prefix = cfg["cues"][condition][phrasing]
    messages.append({"role": "user", "content": student_message(question["text"], prefix)})
    return messages


def history_setup(cfg, questions):
    rng = np.random.default_rng(cfg["seeds"]["history"])
    pool = make_history_pool(cfg, rng)
    return pool, assign_history(questions, pool, cfg["history"]["turns"], rng)


def build_cells(cfg):
    """One row per question x cell, with the chat messages for that cell."""
    questions = make_questions(cfg)
    pool, assignment = history_setup(cfg, questions)
    rows = []
    for q in questions:
        for condition, phrasing in list_cells(cfg):
            rows.append({"kind": "tutor", "question_id": q["id"], "condition": condition,
                         "phrasing": phrasing,
                         "messages": build_messages(q, condition, phrasing, cfg, pool, assignment)})
    return rows


# ---------- pre-freeze checks ----------

def build_solve_cells(cfg):
    """Pilot accuracy: can the model solve each question when asked for the number only?"""
    rows = []
    for q in make_questions(cfg):
        content = f"{q['text']}\n\n{cfg['prompts']['solve_instruction']}"
        rows.append({"kind": "solve", "question_id": q["id"], "condition": "solve", "phrasing": "A",
                     "messages": [{"role": "system", "content": cfg["prompts"]["system"]},
                                  {"role": "user", "content": content}]})
    return rows


def build_check_cells(cfg):
    """Manipulation check: the same history turns, then a question about them instead of the test question."""
    questions = make_questions(cfg)
    pool, assignment = history_setup(cfg, questions)
    rows = []
    for q in questions:
        for condition in cfg["check"]["expected"]:
            for phrasing in cfg["cues"][condition]:
                messages = build_messages(q, condition, phrasing, cfg, pool, assignment)[:-1]
                messages.append({"role": "user", "content": cfg["prompts"]["check_question"]})
                rows.append({"kind": "check", "question_id": q["id"], "condition": condition,
                             "phrasing": phrasing, "messages": messages})
    return rows
