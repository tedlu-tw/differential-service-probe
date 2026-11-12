"""Multi-turn dialogues for v3 (docs/protocol_v3.md, Dialogue).

Turn 1 is the cell's own messages. Turns 2 and 3 add the scripted student messages from the config.
A dialogue stops at the first tutor reply that gives the answer (the v2 gives_answer rule,
unchanged) or after max_turns.

Dialogues advance turn by turn in batches, so a GPU sees full batches; one cell is a batch of one.
reply_fn(states) -> [(reply, truncated), ...] takes a batch of states, each with the cell key,
"seed", "turn" and "messages". The real model and the mock both fit this shape.

Every tutor reply becomes one row (protocol: Logging). The stop decision is recomputed from the
reply text, so rows read back from a CSV resume a dialogue exactly where it stopped.
"""
from ds.score import gives_answer

KEY = ["question_id", "prompt", "condition", "phrasing", "seed"]


def dialogue_key(row):
    return tuple(row[k] for k in KEY)


def check_design(cfg):
    # Each turn after the first needs one scripted student message.
    turns, followups = cfg["design"]["max_turns"], cfg["prompts"]["followups"]
    if len(followups) != turns - 1:
        raise ValueError(f"max_turns {turns} needs {turns - 1} followups, got {len(followups)}")


def messages_for_turn(cell, replies, followups):
    """Turn-1 messages, then each earlier tutor reply and the scripted student message after it."""
    messages = list(cell["messages"])
    for reply, followup in zip(replies, followups):
        messages += [{"role": "assistant", "content": reply}, {"role": "user", "content": followup}]
    return messages


def is_finished(replies, answer, max_turns):
    return len(replies) >= max_turns or any(gives_answer(r, answer) for r in replies)


def group_replies(rows):
    """Replies per dialogue in turn order. A missing turn means the log is damaged."""
    out = {}
    for row in sorted(rows, key=lambda r: int(r["turn"])):
        replies = out.setdefault(dialogue_key(row), [])
        if int(row["turn"]) != len(replies) + 1:
            raise ValueError(f"dialogue {dialogue_key(row)} has turn {row['turn']} after {len(replies)} turns")
        replies.append(row["reply"])
    return out


def pending_states(cells, seed, replies_by_key, answers, cfg):
    max_turns, followups, states = cfg["design"]["max_turns"], cfg["prompts"]["followups"], []
    for cell in cells:
        state = {**{k: cell[k] for k in KEY if k != "seed"}, "seed": seed}
        replies = replies_by_key.get(dialogue_key(state), [])
        if not is_finished(replies, answers[cell["question_id"]], max_turns):
            states.append({**state, "turn": len(replies) + 1,
                           "messages": messages_for_turn(cell, replies, followups)})
    return states


def by_length(states):
    # Similar lengths in one batch waste less padding. The sort is stable, so the order is reproducible.
    return sorted(states, key=lambda s: sum(len(m["content"]) for m in s["messages"]))


def reply_rows(batch, outputs, answers, extra):
    if len(outputs) != len(batch):
        raise ValueError(f"reply_fn returned {len(outputs)} replies for {len(batch)} dialogues")
    return [{**extra, **{k: s[k] for k in KEY}, "turn": s["turn"], "reply": reply,
             "truncated": bool(truncated), "gives_answer": gives_answer(reply, answers[s["question_id"]])}
            for s, (reply, truncated) in zip(batch, outputs)]


def run_dialogues(cells, seed, reply_fn, answers, cfg, done_rows=(), on_rows=None, extra=None, progress=None):
    """Run every dialogue of one seed to its end and return the new rows.

    done_rows: rows already logged for this model (any seed); finished turns are not redone.
    on_rows: called after every batch, e.g. to append to the CSV, so a stopped run loses one batch.
    extra: fields added to every row, e.g. the model id and system_mode.
    progress: progress(turn, n_dialogues) at the start of each turn, returning an object with
        update(n) and close(), e.g. a tqdm bar. Display only; it never changes a reply.
    """
    check_design(cfg)
    rows, new, size = list(done_rows), [], cfg["generation"]["batch_size"]
    while True:
        states = pending_states(cells, seed, group_replies(rows), answers, cfg)
        if not states:
            return new
        turn = min(s["turn"] for s in states)
        todo = by_length([s for s in states if s["turn"] == turn])
        bar = progress(turn, len(todo)) if progress else None
        for i in range(0, len(todo), size):
            batch = todo[i:i + size]
            batch_rows = reply_rows(batch, reply_fn(batch), answers, extra or {})
            rows += batch_rows
            new += batch_rows
            if on_rows:
                on_rows(batch_rows)
            if bar:
                bar.update(len(batch))
        if bar:
            bar.close()
