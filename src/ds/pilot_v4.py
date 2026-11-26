"""Step c, the neutral pilot (docs/protocol_v4.md: Pre-freeze steps, c).

Arm 2 cells (prompts A and B x demand and after_attempt), 1 seed, every second current question (50).
Every row records the max_new_tokens it was generated with, so a model whose pilot truncation is 1%
or more reruns at the next candidate (768, 1024, 1536) into the same file without mixing the two.

The report (paths.pilot_summary_csv) is per model and max_new_tokens:
- every pilot reply: the truncation rate, which sets max_new_tokens;
- prompt A demand dialogues only: gave_by_turn2, the give-in turn distribution, the mean give-in turn
  (never = 6), and whether RQ2's floor or ceiling rule applies.
after_attempt and prompt B replies count only towards truncation, so neither the RQ2 contrast nor the
prompt B - A contrast is seen before the freeze.
"""
import pandas as pd

from ds.dialogue_v4 import KEY, group_replies, pending_states, run_dialogues_v4, with_raw
from ds.generate_v3 import append_rows, load_done_rows, make_reply_fn, progress_bar, timing_rows
from ds.generate_v4 import (append_jsonl, context_rows, contexts_written, out_path, read_rows, run_info_row,
                            seeds_for)
from ds.models_v4 import resolved_generation
from ds.stimuli_v3 import pilot_questions
from ds.stimuli_v4 import build_arm2_cells

RUN = "pilot"


# ---------- cells and max_new_tokens ----------

def pilot_cells(cfg, bank):
    """Arm 2 cells on every second current question, and the answers of all current questions."""
    return build_arm2_cells(cfg, bank, pilot_questions(cfg, bank["current"])), {q["id"]: q["answer"] for q in bank["current"]}


def candidate(cfg, n_tokens=None):
    """The max_new_tokens a pilot runs at: one of the candidates, the first by default."""
    candidates = cfg["generation"]["max_new_tokens_candidates"]
    n = candidates[0] if n_tokens is None else n_tokens
    if n not in candidates:
        raise SystemExit(f"max_new_tokens {n} is not one of the candidates {candidates}")
    return n


def rows_at(rows, model_id, n):
    return [r for r in rows if r["model"] == model_id and int(r["max_new_tokens"]) == n]


def truncation_rate(rows):
    return sum(str(r["truncated"]) == "True" for r in rows) / len(rows)


def finished(cells, rows, answers, cfg):
    """Every pilot dialogue of these rows has ended: the answer given, or the last turn reached."""
    seed, max_turns = cfg["pilot"]["seed"], cfg["arms"]["arm2"]["max_turns"]
    return bool(rows) and not pending_states(cells, seed, group_replies(rows), answers, max_turns)


def check_previous(cfg, cells, answers, rows, model_id, n):
    """A model reruns at the next candidate only after its pilot at the previous one finished with
    truncation at or above the limit."""
    candidates, limit = cfg["generation"]["max_new_tokens_candidates"], cfg["generation"]["max_truncation"]
    i = candidates.index(n)
    if i == 0:
        return
    prev = rows_at(rows, model_id, candidates[i - 1])
    if not finished(cells, prev, answers, cfg):
        raise SystemExit(f"{model_id}: the pilot at {candidates[i - 1]} has not finished; finish it first")
    rate = truncation_rate(prev)
    if rate < limit:
        raise SystemExit(f"{model_id}: truncation at {candidates[i - 1]} was {rate:.4f}, below {limit}; no rerun")


# ---------- running ----------

def pilot_writer(path, timing_path, pending):
    """on_rows: the batch's replies, then its timing, so a crash never leaves timing without replies."""
    def write(rows):
        append_rows(path, rows)
        if pending:
            append_rows(timing_path, pending)
            pending.clear()
    return write


def run_pilot_model(parts, cfg, cells, answers, source, n):
    """One model's pilot at max_new_tokens n: run info, turn-1 contexts once, then every dialogue."""
    spec, path, pending, stash = parts["spec"], out_path(cfg, "pilot_csv", source), [], {}
    gen = {**resolved_generation(cfg, spec["id"], RUN), "max_new_tokens": n}
    append_jsonl(out_path(cfg, "run_info_jsonl", source), run_info_row(parts, RUN, "arm2", gen))
    contexts = out_path(cfg, "contexts_csv", source)
    if not contexts_written(contexts, spec["id"], RUN, "arm2"):        # the same at every candidate
        append_rows(contexts, context_rows(parts, cells, RUN))
    log = lambda batch, seconds: pending.extend(timing_rows(spec["id"], RUN, batch, seconds))
    reply_fn = with_raw(make_reply_fn(parts["prepare"], parts["render"], parts["make_generate"](gen, stash), log), stash)
    seed = seeds_for(cfg, spec, RUN)[0]
    parts["set_seed"](seed)
    run_dialogues_v4(cells, seed, reply_fn, answers, cfg["arms"]["arm2"]["max_turns"], gen["batch_size"],
                     failure_log=(out_path(cfg, "failures_jsonl", source), {"model": spec["id"], "run": RUN}),
                     done_rows=rows_at(load_done_rows(path, spec["id"]), spec["id"], n),
                     on_rows=pilot_writer(path, out_path(cfg, "timing_csv", source), pending),
                     extra={"model": spec["id"], "revision": spec["revision"], "precision": parts["info"]["precision"],
                            "system_mode": parts["mode"], "max_new_tokens": n},
                     progress=progress_bar(spec["id"], f"{RUN} max_new_tokens={n}", seed) if source == "gpu" else None)


def run_pilot(cfg, bank, make_parts, source, model_id=None, n_tokens=None):
    """The pilot for every model (or one) at one candidate, then the report, rebuilt from the file.
    Every model's rerun precondition is checked before any model is loaded."""
    n = candidate(cfg, n_tokens)
    specs = [s for s in cfg["models"] if model_id in (None, s["id"])]
    if not specs:
        raise SystemExit(f"{model_id} is not a model id in the config")
    cells, answers = pilot_cells(cfg, bank)
    rows = read_rows(out_path(cfg, "pilot_csv", source)).to_dict("records")
    for spec in specs:
        check_previous(cfg, cells, answers, rows, spec["id"], n)
    for spec in specs:
        parts = make_parts(spec, cfg)
        run_pilot_model(parts, cfg, cells, answers, source, n)
        parts["close"]()
    return write_pilot_report(cfg, bank, source)


# ---------- the report ----------

def dialogue_table(df, never):
    """One row per dialogue: answer_turn (the first turn that gives the answer, else `never`) and gave_by_turn2."""
    keys = ["model", "max_new_tokens"] + KEY
    gave = df[df["gives_answer"].astype(int) == 1].groupby(keys)["turn"].min().rename("answer_turn")
    d = df.groupby(keys).size().rename("n_turns").to_frame().join(gave)
    d["answer_turn"] = d["answer_turn"].fillna(never).astype(int)
    d["gave_by_turn2"] = (d["answer_turn"] <= 2).astype(int)
    return d.reset_index()


def demand_a_table(df, cfg):
    """gave_by_turn2, the give-in turn shares and the mean give-in turn, from prompt A demand dialogues only."""
    p, never = cfg["pilot"], cfg["analysis"]["rq2"]["never_turn"]
    d = dialogue_table(df[(df["prompt"] == p["reported_prompt"]) & (df["condition"] == p["reported_pressure"])], never)
    g = d.groupby(["model", "max_new_tokens"], sort=False)["answer_turn"]
    out = pd.DataFrame({"demand_a_dialogues": g.size(), "gave_by_turn2": g.apply(lambda s: (s <= 2).mean()),
                        "mean_give_in_turn": g.mean()})
    for t in range(1, never + 1):
        out[f"share_turn_{t}" if t < never else "share_never"] = g.apply(lambda s, t=t: (s == t).mean())
    return out


def pilot_report(rows, cfg, done):
    """Per model and max_new_tokens. done: {(model, max_new_tokens): every dialogue ended}. No reply
    count over all cells: it would carry the after_attempt and prompt B dialogues' lengths."""
    df, fc = pd.DataFrame(rows), cfg["analysis"]["floor_ceiling"]
    df["truncated"] = df["truncated"].astype(str) == "True"
    keys = ["model", "max_new_tokens"]
    out = df.groupby(keys, sort=False).agg(truncation_rate=("truncated", "mean"))
    out = out.join(demand_a_table(df, cfg))
    out["finished"] = [done[k] for k in out.index]
    out["truncation_ok"] = out["truncation_rate"] < cfg["generation"]["max_truncation"]
    out["rq2_floor_or_ceiling"] = (out["gave_by_turn2"] > fc["rq2_ceiling"]) | (out["gave_by_turn2"] < fc["rq2_floor"])
    report = out.reset_index()
    chosen = chosen_tokens(report, cfg)
    report["chosen"] = [chosen[m] == n for m, n in zip(report["model"], report["max_new_tokens"])]
    return report


def chosen_tokens(report, cfg):
    """Per model: the first candidate, in order, whose finished pilot is below the truncation limit, with
    every earlier candidate finished at or above it. None while that is not settled."""
    out = {}
    for model, g in report.groupby("model", sort=False):
        by, out[model] = g.set_index("max_new_tokens"), None
        for n in cfg["generation"]["max_new_tokens_candidates"]:
            if n not in by.index or not by.loc[n, "finished"]:
                break
            if by.loc[n, "truncation_ok"]:
                out[model] = n
                break
    return out


def write_pilot_report(cfg, bank, source):
    """Rebuild the report from paths.pilot_csv and write it; no model is loaded."""
    rows = read_rows(out_path(cfg, "pilot_csv", source)).to_dict("records")
    if not rows:
        raise SystemExit("no pilot replies yet")
    cells, answers = pilot_cells(cfg, bank)
    done = {(m, n): finished(cells, rows_at(rows, m, n), answers, cfg)
            for m, n in {(r["model"], int(r["max_new_tokens"])) for r in rows}}
    report = pilot_report(rows, cfg, done)
    report.to_csv(out_path(cfg, "pilot_summary_csv", source), index=False)
    return report
