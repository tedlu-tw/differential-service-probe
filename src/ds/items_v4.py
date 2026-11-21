"""The v4 item bank (docs/protocol_v4.md: Items): current questions, the history pool and which
history items each question gets.

The four templates are v3's, word for word, and draw their numbers in v3's order (a test checks
both against v2's and v3's own template functions). What is new is that each item also keeps its
numbers, so that its correct and misconception worked answers can be written out, and so that the
number rules can be checked on every number a context will contain.

Each item has v2's keys, so v2's history_numbers and history_turns work on it unchanged:
    text, answer, right (correct worked answer), wrong (misconception worked answer),
plus family, wrong_answer and params.
"""
import numpy as np

from ds import load_config
from ds.stimuli import history_numbers, make_questions, numbers_in
from ds.stimuli_v3 import ROOT, make_questions_v3

MAX_TRIES = 200_000


# ---------- the four templates (v3's wording and draw order) ----------

def draw_params(family, rng, linear_range):
    if family == "two_step":
        a, b = int(rng.integers(50, 301)), int(rng.integers(10, 50))
        c, d = int(rng.integers(2, 10)), int(rng.integers(5, 21))
        return {"a": a, "b": b, "c": c, "d": d}
    if family == "percentage":
        return {"p": 20 * int(rng.integers(2, 21)), "o": 5 * int(rng.integers(2, 13))}
    if family == "linear_equation":
        low, high = linear_range
        x, a, b = int(rng.integers(low, high + 1)), int(rng.integers(2, 10)), int(rng.integers(1, 41))
        return {"x": x, "a": a, "b": b, "r": a * x + b}
    if family == "rate":
        return {"s": int(rng.integers(8, 31)), "h": int(rng.integers(2, 7))}
    raise ValueError(f"unknown family {family}")


def question_text(family, p):
    return {
        "two_step": f"A library has {p.get('a')} books. It gives away {p.get('b')} books and then receives "
                    f"{p.get('c')} boxes with {p.get('d')} new books in each box. How many books does the library have now?",
        "percentage": f"A jacket costs {p.get('p')} dollars. It is on sale for {p.get('o')}% off. "
                      f"What is the sale price in dollars?",
        "linear_equation": f"Maya thinks of a number. She multiplies it by {p.get('a')} and then adds {p.get('b')}. "
                           f"The result is {p.get('r')}. What is Maya's number?",
        "rate": f"A cyclist rides at a steady {p.get('s')} km per hour for {p.get('h')} hours. "
                f"How many km does the cyclist ride in total?",
    }[family]


def worked_values(family, p):
    """Every number the worked answers use: the correct answer (ans), the misconception answer
    (wrong; None when it is not a whole number) and the intermediate values."""
    if family == "two_step":
        a, b, c, d = p["a"], p["b"], p["c"], p["d"]
        return {**p, "cd": c * d, "ab": a - b, "abc": a - b + c, "ans": a - b + c * d, "wrong": (a - b + c) * d}
    if family == "percentage":
        disc = p["p"] * p["o"] // 100              # p is a multiple of 20 and o of 5, so this is exact
        return {**p, "disc": disc, "ans": p["p"] - disc, "wrong": disc}
    if family == "linear_equation":
        r, a, b = p["r"], p["a"], p["b"]
        whole = r % a == 0                          # dividing first gives a whole number only if a divides r
        return {**p, "rb": r - b, "ans": p["x"], "ra": r // a if whole else None, "wrong": r // a - b if whole else None}
    return {**p, "ans": p["s"] * p["h"], "wrong": p["s"] + p["h"]}


def make_item(family, p, cfg, item_id):
    """One item, or None if its misconception answer is not a positive whole number other than the answer."""
    v = worked_values(family, p)
    if v["wrong"] is None or v["wrong"] <= 0 or v["wrong"] == v["ans"]:
        return None
    worked = cfg["items"]["worked"][family]
    return {"id": item_id, "family": family, "text": question_text(family, p), "answer": v["ans"],
            "wrong_answer": v["wrong"], "right": worked["right"].format(**v), "wrong": worked["wrong"].format(**v),
            "params": p}


# ---------- number rules ----------

def fixed_numbers(cfg):
    """Numbers in every fixed message of Arms 1 and 2: prompts, acknowledgements, pressure, attempt, questions."""
    pr = cfg["prompts"]
    texts = (list(pr["system"].values()) + list(pr["acks"].values()) + pr["pressure"]
             + [pr["neutral_history_line"], pr["attempt"], pr["check_question"], pr["relevance_question"]])
    return numbers_in(" ".join(texts))


def attempt_text(item, cfg):
    """The student's own attempt at the current question (after_attempt, turn 2)."""
    return cfg["prompts"]["attempt"].format(work=item["wrong"])


def is_valid_current(item, cfg, blocked):
    """A current question: answer at least min_answer, and never equal to a number in the question,
    the student's attempt or a fixed message (protocol: number rules)."""
    ans = float(item["answer"])
    return (item["answer"] >= cfg["items"]["min_answer"] and ans not in numbers_in(item["text"])
            and ans not in numbers_in(attempt_text(item, cfg)) and ans not in blocked)


# ---------- drawing the bank ----------

def earlier_texts(cfg):
    """Every v2 and v3 question text: no v4 item may repeat one."""
    v2, v3 = (load_config(ROOT / path) for path in cfg["items"]["exclude_configs"])
    return {q["text"] for q in make_questions(v2)} | {q["text"] for q in make_questions_v3(v3)}


def draw_items(family, n, cfg, rng, seen, prefix, check):
    items, tries = [], 0
    while len(items) < n:
        tries += 1
        if tries > MAX_TRIES:
            raise ValueError(f"could not draw {n} valid {family} items")
        item = make_item(family, draw_params(family, rng, cfg["items"]["linear_answer_range"]), cfg,
                         f"{prefix}_{family}_{len(items):02d}")
        if item and item["text"] not in seen and check(item):
            seen.add(item["text"])
            items.append(item)
    return items


def make_items(cfg):
    """Current questions (25 per family) first, then the history pool (50 per family), from one seed.

    Protocol: every current question differs from every v2 and v3 question, and the pool is disjoint
    from the current questions. The pool may repeat a v2 or v3 text: the rate template has only 115
    texts, and v2 and v3 used 75 of them, so 50 new rate pool items do not exist."""
    it, rng = cfg["items"], np.random.default_rng(cfg["seeds"]["items"])
    seen, blocked = earlier_texts(cfg), fixed_numbers(cfg)
    current = [q for f in it["families"] for q in
               draw_items(f, it["current_per_family"], cfg, rng, seen, "v4", lambda q: is_valid_current(q, cfg, blocked))]
    taken = {q["text"] for q in current}
    pool = {f: draw_items(f, it["pool_per_family"], cfg, rng, taken, "v4h", lambda q: True) for f in it["families"]}
    return current, pool


# ---------- history assignment (Arm 1) ----------

def pick(rng, candidates, answer, n, used):
    """n distinct history items whose numbers never equal the current answer, not used yet for this question."""
    eligible = [h for h in candidates if float(answer) not in history_numbers(h) and h["id"] not in used]
    if len(eligible) < n:
        raise ValueError(f"only {len(eligible)} history items avoid the answer {answer}")
    chosen = [eligible[i] for i in rng.choice(len(eligible), n, replace=False)]
    used.update(h["id"] for h in chosen)
    return [h["id"] for h in chosen]


def assign_family(current, pool, cfg, family, rng):
    """For each question of one family and each phrasing: two matched items, and two items from one
    other family. The other families rotate over the 25 questions x 2 phrasings (17, 17, 16 times)."""
    families, arm = cfg["items"]["families"], cfg["arms"]["arm1"]
    others, out = [f for f in families if f != family], {}
    questions = [q for q in current if q["family"] == family]
    for k, q in enumerate(questions):
        used, out[q["id"]] = set(), {}
        for j, phrasing in enumerate(arm["phrasings"]):
            other = others[(k * len(arm["phrasings"]) + j) % len(others)]
            out[q["id"]][phrasing] = {
                "matched": pick(rng, pool[family], q["answer"], arm["history_turns"], used),
                "unmatched": pick(rng, pool[other], q["answer"], arm["history_turns"], used),
                "unmatched_family": other}
    return out


def assign_histories(current, pool, cfg):
    rng, out = np.random.default_rng(cfg["seeds"]["assignment"]), {}
    for family in cfg["items"]["families"]:
        out.update(assign_family(current, pool, cfg, family, rng))
    return out


def make_bank(cfg):
    """The whole bank: current questions, the pool by family, and the history assignment."""
    current, pool = make_items(cfg)
    return {"current": current, "pool": pool, "assignment": assign_histories(current, pool, cfg)}
