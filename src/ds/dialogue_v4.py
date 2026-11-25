"""Multi-turn dialogues for v4 (docs/protocol_v4.md: Dialogue, Code).

Arms 1 and 2 run through run_dialogues_v4: v3's loop, with the same batching by length, turn order,
resume and stop rule, except that each cell's student messages for turns 2 to 5 come from the
cell's own "followups", and that the dialogue key includes the arm. v3's by_length, is_finished
and messages_for_turn are used unchanged.

Arm 3 runs through v3's run_dialogues itself (run_arm3); only the raw reply and the arm are added to
each row, in the on_rows callback that v3's loop calls right after each batch.

reply_fn(states) -> [(reply, truncated, raw_reply), ...] for Arms 1 and 2. with_raw turns v3's
make_reply_fn, which gives (reply, truncated), into that shape, from the stash that
models_v4.make_generate fills for the same batch.
"""
import json
from pathlib import Path

from ds.dialogue import by_length, is_finished, messages_for_turn, run_dialogues
from ds.models_v4 import RUN_SETTINGS, GenerationFailure
from ds.score import gives_answer

KEY = ["arm", "question_id", "prompt", "condition", "phrasing", "seed"]


def dialogue_key(row):
    return tuple(str(row[k]) for k in KEY)


def check_cells(cells, max_turns):
    # Each turn after the first needs one student message, from the cell itself.
    for c in cells:
        if len(c["followups"]) != max_turns - 1:
            raise ValueError(f"cell {c['arm']} {c['question_id']} {c['condition']} has {len(c['followups'])} "
                             f"followups, max_turns {max_turns} needs {max_turns - 1}")


def group_replies(rows):
    """Replies per dialogue in turn order. A missing turn means the log is damaged."""
    out = {}
    for row in sorted(rows, key=lambda r: int(r["turn"])):
        replies = out.setdefault(dialogue_key(row), [])
        if int(row["turn"]) != len(replies) + 1:
            raise ValueError(f"dialogue {dialogue_key(row)} has turn {row['turn']} after {len(replies)} turns")
        replies.append(row["reply"])
    return out


def pending_states(cells, seed, replies_by_key, answers, max_turns):
    states = []
    for cell in cells:
        state = {**{k: cell[k] for k in KEY if k != "seed"}, "seed": seed}
        replies = replies_by_key.get(dialogue_key(state), [])
        if not is_finished(replies, answers[cell["question_id"]], max_turns):
            states.append({**state, "turn": len(replies) + 1,
                           "messages": messages_for_turn(cell, replies, cell["followups"])})
    return states


def reply_rows(batch, outputs, answers, extra):
    if len(outputs) != len(batch):
        raise ValueError(f"reply_fn returned {len(outputs)} replies for {len(batch)} dialogues")
    return [{**extra, **{k: s[k] for k in KEY}, "turn": s["turn"], "reply": reply, "raw_reply": raw,
             "truncated": bool(truncated), "gives_answer": gives_answer(reply, answers[s["question_id"]])}
            for s, (reply, truncated, raw) in zip(batch, outputs)]


def run_dialogues_v4(cells, seed, reply_fn, answers, max_turns, batch_size, *, failure_log, done_rows=(),
                     on_rows=None, extra=None, progress=None):
    """Run every dialogue of one seed to its end and return the new rows: Arms 1 and 2, the pilot, and
    the one-turn solve and check runs. done_rows: rows already logged for this model; finished turns
    are not redone. failure_log = (path, {"model", "run", ...}) is required: every batch goes through
    log_failures. Otherwise as v3's run_dialogues."""
    check_cells(cells, max_turns)
    extra = {**(extra or {}), "run": run_name(failure_log)}       # every row says which run wrote it
    reply_fn = log_failures(reply_fn, *failure_log)
    rows, new = list(done_rows), []
    while True:
        states = pending_states(cells, seed, group_replies(rows), answers, max_turns)
        if not states:
            return new
        turn = min(s["turn"] for s in states)
        todo = by_length([s for s in states if s["turn"] == turn])
        bar = progress(turn, len(todo)) if progress else None
        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]
            batch_rows = reply_rows(batch, reply_fn(batch), answers, extra)
            rows += batch_rows
            new += batch_rows
            if on_rows:
                on_rows(batch_rows)
            if bar:
                bar.update(len(batch))
        if bar:
            bar.close()


# ---------- run names ----------

RUNS = set(RUN_SETTINGS) | {"dry_run"}


def run_name(failure_log):
    """The run named in failure_log's details, which every loop requires, so no row lacks it."""
    run = failure_log[1].get("run")
    if run not in RUNS:
        raise ValueError(f"failure_log names run {run!r}; expected one of {sorted(RUNS)}")
    return run


# ---------- failures ----------

def failure_lines(batch, raw, error, extra):
    """One JSON line per reply of a failed batch: its dialogue key, turn, raw output and the error."""
    keys = [k for k in KEY if k in batch[0]] or ["question_id", "prompt", "condition", "phrasing", "seed"]
    return [json.dumps({**extra, **{k: s.get(k) for k in keys}, "turn": s["turn"], "raw_reply": r, "error": error},
                       ensure_ascii=False) for s, r in zip(batch, raw)]


def log_failures(reply_fn, path, extra=None):
    """Wrap a reply_fn: if a batch fails a check after generation, append its keys and raw outputs to
    path before the error stops the run. A restart regenerates that batch from another random state,
    so without this the failed reply would vanish unrecorded (protocol: Logging)."""
    def fn(batch):
        try:
            return reply_fn(batch)
        except GenerationFailure as e:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write("\n".join(failure_lines(batch, e.raw, str(e), extra or {})) + "\n")
            raise
    return fn


# ---------- raw replies ----------

def take_raw(stash, n):
    """The raw replies of the batch just generated, removed from the stash so they are used once."""
    raw = stash.pop("raw", None)
    if raw is None or len(raw) != n:
        raise RuntimeError(f"expected {n} raw replies for this batch, found {None if raw is None else len(raw)}")
    return raw


def with_raw(reply_fn, stash):
    """v3's (reply, truncated) reply_fn -> (reply, truncated, raw_reply), for run_dialogues_v4."""
    def fn(batch):
        out = reply_fn(batch)
        return [(reply, truncated, raw) for (reply, truncated), raw in zip(out, take_raw(stash, len(out)))]
    return fn


ARM3_RUNS = ("arm3", "dry_run")     # runs whose v3-format cells are Arm 3 cells; v3_pilot and v3_solve are not


def run_arm3(cells, seed, reply_fn, answers, arm3_cfg, stash, *, failure_log, done_rows=(), on_rows=None,
             extra=None, progress=None):
    """Arm 3, and v3's pilot and solve cells for the new models, through v3's run_dialogues, unchanged.
    The reply_fn goes through log_failures. on_rows adds each row's raw reply (from the batch just
    generated), the run, and the arm: "arm3" only for Arm 3 cells (run arm3, or the dry run), so that
    v3_pilot and v3_solve rows never read as Arm 3. The analysis selects Arm 3 by run."""
    path, meta = failure_log
    run = run_name(failure_log)
    arm = "arm3" if run in ARM3_RUNS else None
    if arm and any(c["condition"] == "solve" for c in cells):         # the label must not be wrong
        raise ValueError(f"run {run} labels its rows arm3, but solve cells were passed; use run v3_solve")
    reply_fn = log_failures(reply_fn, path, {**({"arm": arm} if arm else {}), **meta})

    def add_raw(rows):
        for row, raw in zip(rows, take_raw(stash, len(rows))):
            row["raw_reply"], row["run"], row["arm"] = raw, run, arm
        if on_rows:
            on_rows(rows)
    return run_dialogues(cells, seed, reply_fn, answers, arm3_cfg, done_rows=done_rows, on_rows=add_raw,
                         extra=extra, progress=progress)
