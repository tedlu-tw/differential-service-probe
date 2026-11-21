"""v4 cells and messages for Arms 1 and 2, and the step e checks (docs/protocol_v4.md: Arms and cells).

Every cell is one dialogue's turn-1 messages plus its own student messages for turns 2 to 5
("followups"), so that an after_attempt cell can carry the student's attempt at its own question.
The v4 dialogue loop reads them from the cell; v3's run_dialogues, which takes one list from the
config, still runs Arm 3.

Reused unchanged: v2's history_turns (worked answers after "My answer:", then the acknowledgement)
and student_message, and v3's build_solve_cells_v3 (accuracy with working), dry_run_questions and
pilot_questions. The cue, prompt and condition live only in the messages; the scorer and the judge
never see them.
"""
from ds.items_v4 import attempt_text, make_bank
from ds.stimuli import history_turns, student_message
from ds.stimuli_v3 import build_solve_cells_v3

KEY = ["arm", "question_id", "prompt", "condition", "phrasing"]


# ---------- pieces ----------

def system(cfg, prompt):
    return {"role": "system", "content": cfg["prompts"]["system"][prompt]}


def question_turn(question):
    # The current question alone, as in v3's neutral cells: no cue, the text never edited.
    return {"role": "user", "content": student_message(question["text"], "")}


def neutral_turns(items, line, ack):
    """history_neutral: the same turns as a history, with the line in place of a worked answer."""
    turns = []
    for item in items:
        turns += [{"role": "user", "content": f"{item['text']}\n{line}"}, {"role": "assistant", "content": ack}]
    return turns


def history_for(condition, question, phrasing, bank, cfg):
    """The two earlier problems of one Arm 1 cell, each with the student's work and the acknowledgement."""
    pool = {h["id"]: h for items in bank["pool"].values() for h in items}
    assigned, ack = bank["assignment"][question["id"]][phrasing], cfg["prompts"]["acks"][phrasing]
    if condition == "history_neutral":
        return neutral_turns([pool[h] for h in assigned["matched"]], cfg["prompts"]["neutral_history_line"], ack)
    kind = "matched" if condition.startswith("history_matched_") else "unmatched"
    return history_turns([pool[h] for h in assigned[kind]], condition.endswith("_success"), ack)


def cell(arm, question, prompt, condition, phrasing, messages, followups):
    return {"kind": "tutor", "arm": arm, "question_id": question["id"], "family": question["family"],
            "prompt": prompt, "condition": condition, "phrasing": phrasing,
            "messages": messages, "followups": followups}


# ---------- Arms 1 and 2 ----------

def build_arm1_cells(cfg, bank, questions=None):
    """Arm 1 (RQ1): prompt A, five history conditions x two phrasings, demand pressure."""
    arm = cfg["arms"]["arm1"]
    rows = []
    for q in questions or bank["current"]:
        for condition in arm["conditions"]:
            for phrasing in arm["phrasings"]:
                messages = ([system(cfg, arm["prompt"])] + history_for(condition, q, phrasing, bank, cfg)
                            + [question_turn(q)])
                rows.append(cell("arm1", q, arm["prompt"], condition, phrasing, messages, list(cfg["prompts"]["pressure"])))
    return rows


def followups_for(pressure, question, cfg):
    """Student turns 2 to 5. after_attempt replaces turn 2 with the student's own attempt."""
    pressure_msgs = list(cfg["prompts"]["pressure"])
    if pressure == "demand":
        return pressure_msgs
    return [attempt_text(question, cfg)] + pressure_msgs[1:]


def build_arm2_cells(cfg, bank, questions=None):
    """Arm 2 (RQ2): no history; prompt A and B x demand and after_attempt; one phrasing."""
    arm = cfg["arms"]["arm2"]
    return [cell("arm2", q, prompt, pressure, "A", [system(cfg, prompt), question_turn(q)],
                 followups_for(pressure, q, cfg))
            for q in questions or bank["current"] for prompt in arm["prompts"] for pressure in arm["pressures"]]


def build_cells_v4(cfg, bank=None, questions=None):
    """Every Arm 1 and Arm 2 cell. Pass questions to build a subset (the dry run, the pilot)."""
    bank = bank or make_bank(cfg)
    return build_arm1_cells(cfg, bank, questions) + build_arm2_cells(cfg, bank, questions)


def build_solve_cells_v4(cfg, questions):
    """Accuracy with working (step c): v3's solve cells, unchanged, on the v4 current questions."""
    return [{**c, "arm": "solve"} for c in build_solve_cells_v3(cfg, questions)]


# ---------- step e: manipulation checks ----------

def build_check_cells_v4(cfg, bank, questions=None):
    """Correctness: after each Arm 1 success or struggle history, v3's check question. Relevance: after
    each struggle history and the current problem, the relevance question. Both under the check prompt."""
    ch, rows = cfg["check"], []
    for c in build_arm1_cells(cfg, bank, questions):
        if c["condition"] == "history_neutral":
            continue
        history = c["messages"][:-1]
        q_text = c["messages"][-1]["content"]
        rows.append({**c, "kind": "check_correctness", "followups": [],
                     "messages": history + [{"role": "user", "content": cfg["prompts"]["check_question"]}]})
        if c["condition"] in ch["relevance_expected"]:
            rows.append({**c, "kind": "check_relevance", "followups": [],
                         "messages": history + [{"role": "user",
                                                 "content": f"{q_text}\n\n{cfg['prompts']['relevance_question']}"}]})
    return rows
