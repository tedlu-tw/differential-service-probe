"""v4 generation on the GPU: the dry run (step b) now; the pilots and full run use the same pieces.

A model is handled through "parts", a dict of functions built once per model:
    spec, info (load_model_v4's record), mode (system_mode), prepare, render, encode (text -> ids),
    make_generate(gen, stash) -> generate(prompts), set_seed(seed), roundtrip ({name: tokenizer}),
    gpu (GPU name, driver, CUDA), close().
gpu_parts builds them from the real model; tests pass fakes, so everything below runs without a GPU.

Reused unchanged: v3's make_reply_fn, append_rows, load_done_rows, timing_rows and dry_run_questions,
and v3's build_cells_v3 for Arm 3. Every run goes through run_dialogues_v4 or run_arm3, so through
log_failures, and every row names its run.

The dry-run report is per model only (protocol: step b): how replies end, the round trip, empty
replies and timing. It holds no gives_answer, answer turn or other outcome by condition or arm.
"""
import csv
import datetime
import json
import subprocess
from pathlib import Path

import pandas as pd

from ds.dialogue_v4 import run_arm3, run_dialogues_v4, with_raw
from ds.generate_v3 import append_rows, dry_run_questions, load_done_rows, make_reply_fn, timing_rows
from ds.models_v4 import (effective_settings, load_model_v4, load_tokenizer_v4, make_generate, render_parts,
                          resolved_generation, roundtrip_differs)
from ds.stimuli_v3 import build_cells_v3, make_questions_v3
from ds.stimuli_v4 import build_arm1_cells, build_arm2_cells


# ---------- paths ----------

def out_path(cfg, key, source):
    """Mock runs keep the real file names in their own folder, so they never mix with real runs."""
    path = Path(cfg["paths"][key])
    return Path(cfg["paths"]["mock_dir"]) / path.name if source == "mock" else path


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


# ---------- which questions, cells and seeds ----------

def dry_run_questions_v4(cfg, bank, v3_questions):
    """Arm 1: 3 per family at fixed positions (12). Arms 2 and 3: v3's every-nth rule (5 each)."""
    families, positions = cfg["items"]["families"], cfg["dry_run"]["arm1_positions"]
    by_family = {f: [q for q in bank["current"] if q["family"] == f] for f in families}
    return {"arm1": [by_family[f][i] for f in families for i in positions],
            "arm2": dry_run_questions(cfg, bank["current"]),
            "arm3": dry_run_questions(cfg, v3_questions)}


def dry_run_cells(cfg, bank):
    v3_questions = make_questions_v3(cfg["arm3"])
    picked = dry_run_questions_v4(cfg, bank, v3_questions)
    cells = {"arm1": build_arm1_cells(cfg, bank, picked["arm1"]), "arm2": build_arm2_cells(cfg, bank, picked["arm2"]),
             "arm3": build_cells_v3(cfg["arm3"], picked["arm3"])}
    answers = {**{q["id"]: q["answer"] for q in bank["current"]}, **{q["id"]: q["answer"] for q in v3_questions}}
    return cells, answers


def seeds_for(cfg, spec, run):
    """Generation seeds per run. Arm 3: seeds 2 and 3 for the v3 models (test-retest), 0 and 1 for the new."""
    if run == "dry_run":
        return list(cfg["dry_run"]["seeds"])
    if run in ("arm1", "arm2"):
        return list(cfg["seeds"]["generation"])
    if run == "arm3":
        return list(cfg["arm3"]["seeds"]["generation"][spec["group"]])
    single = {"pilot": cfg["pilot"]["seed"], "solve": cfg["solve"]["seed"], "check": cfg["check"]["seed"],
              "v3_pilot": cfg["arm3"]["pilot"]["seed"], "v3_solve": cfg["arm3"]["pilot"]["seed"]}
    if run not in single:
        raise ValueError(f"unknown run {run!r}")
    return [single[run]]


# ---------- records ----------

def run_info_row(parts, run, arm, gen):
    """One line of run_info.jsonl per model, run and arm: the load record, the resolved settings, the
    effective settings, the GPU, system_mode and the forced text."""
    return {**parts["info"], "run": run, "arm": arm, "settings": gen,
            "effective": effective_settings(gen, parts["info"]["generation_config_diff"]),
            "gpu": parts["gpu"], "system_mode": parts["mode"], "forced_text": parts["spec"]["forced_text"],
            "written": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}


def context_rows(parts, cells, run):
    """Probe addition: every turn-1 context as rendered, with its token ids and the cell keys."""
    spec, rows = parts["spec"], []
    for c in cells:
        text = parts["render"](parts["prepare"](c["messages"]))
        rows.append({"model": spec["id"], "revision": spec["revision"], "run": run, "arm": c.get("arm", "arm3"),
                     "question_id": c["question_id"], "prompt": c["prompt"], "condition": c["condition"],
                     "phrasing": c["phrasing"], "text": text, "token_ids": json.dumps(parts["encode"](text))})
    return rows


def contexts_written(path, model_id, run, arm):
    if not Path(path).exists():
        return False
    df = pd.read_csv(path, usecols=["model", "run", "arm"], keep_default_na=False)
    return bool(((df["model"] == model_id) & (df["run"] == run) & (df["arm"] == arm)).any())


DIAG_BASE = ["model", "run", "turn", "n_tokens", "ended_on_stop", "truncated", "empty", "roundtrip_differs_own"]
DIAG_BOOL = {"True": 1.0, "False": 0.0, "": float("nan")}


def roundtrip_columns(cfg, model_id=None):
    """The extra round-trip columns of dry_run.roundtrip_also, in config order, for one model or all."""
    also = cfg["dry_run"].get("roundtrip_also", {})
    return [f"roundtrip_differs_{roundtrip_name(o)}" for m, o in also.items() if model_id in (None, m)]


def diagnostics_columns(cfg):
    """Fixed for every model, so the file's header holds whichever model writes first."""
    return DIAG_BASE + roundtrip_columns(cfg)


def with_diagnostics(fn, stash, parts, sink, meta):
    """Per-reply dry-run diagnostics, kept apart from the replies: model, run and turn only, no cell keys,
    so the report cannot be broken down by condition or arm."""
    def wrapped(batch):
        out = fn(batch)
        for s, o, ids, ended in zip(batch, out, stash["ids"], stash["ended"]):
            row = {**meta, "turn": s["turn"], "n_tokens": len(ids), "ended_on_stop": bool(ended),
                   "truncated": bool(o[1]), "empty": not str(o[0]).strip()}
            for name, tok in parts["roundtrip"].items():
                row[f"roundtrip_differs_{name}"] = roundtrip_differs(ids, tok)
            sink.append(row)
        return out
    return wrapped


def dry_run_report(diagnostics, timing):
    """Per model only: replies, share ending on a stop token, share cut off, empty replies, round trip
    per tokenizer, seconds per reply. diagnostics and timing are tables (or lists of rows)."""
    d, t, rows = pd.DataFrame(diagnostics), pd.DataFrame(timing), []
    if not len(d):
        return pd.DataFrame(columns=["model", "replies", "share_ended_on_stop", "share_truncated", "empty_replies"])
    if not len(t):
        t = pd.DataFrame(columns=["model", "run", "seconds"])
    for model, g in d.groupby("model", sort=False):
        row = {"model": model, "replies": len(g), "share_ended_on_stop": g["ended_on_stop"].mean(),
               "share_truncated": g["truncated"].mean(), "empty_replies": int(g["empty"].sum())}
        for col in sorted(c for c in g.columns if c.startswith("roundtrip_differs_")):
            row[col.replace("roundtrip_differs_", "share_roundtrip_differs_")] = g[col].mean()
        seconds = t[(t["model"] == model) & (t["run"] == "dry_run")]["seconds"].sum()
        rows.append({**row, "seconds_per_reply": seconds / len(g)})
    return pd.DataFrame(rows)


def read_rows(path):
    return pd.read_csv(path, keep_default_na=False) if Path(path).exists() else pd.DataFrame()


def diagnostics_row(fields, header, cfg):
    """One line of the diagnostics file as a dict. A file begun before the columns were fixed has the
    base header only; a model with extra round trips wrote them after the base fields, in config order."""
    own = DIAG_BASE + roundtrip_columns(cfg, fields[0])
    if len(fields) == len(header):
        return dict(zip(header, fields))
    if len(fields) == len(own):
        return dict(zip(own, fields))
    raise SystemExit(f"diagnostics row of {fields[0]}: {len(fields)} fields, expected {len(header)} or {len(own)}")


def read_diagnostics(path, cfg):
    """dry_run_diagnostics.csv against the fixed columns: turn and n_tokens as int, the flags as 1.0,
    0.0, or NaN where a model has no such round trip."""
    cols = diagnostics_columns(cfg)
    if not Path(path).exists():
        return pd.DataFrame(columns=cols)
    with open(path, newline="") as f:
        header, *lines = list(csv.reader(f))
    if header != cols[:len(header)]:
        raise SystemExit(f"{path}: header {header} is not a prefix of {cols}")
    df = pd.DataFrame([diagnostics_row(fields, header, cfg) for fields in lines], columns=cols).fillna("")
    for c in ("turn", "n_tokens"):
        df[c] = df[c].astype(int)
    for c in cols[4:]:
        unknown = set(df[c]) - set(DIAG_BOOL)
        if unknown:
            raise SystemExit(f"{path}: column {c} has values {sorted(unknown)}")
        df[c] = df[c].map(DIAG_BOOL)
    return df


def check_report_counts(report, cfg, source):
    """Per model, the report's reply count must equal that model's rows in the two dry-run reply files."""
    replies = pd.concat([read_rows(out_path(cfg, k, source)) for k in ("dry_run_csv", "dry_run_arm3_csv")])
    logged = replies.groupby("model").size().to_dict() if len(replies) else {}
    reported = dict(zip(report["model"], report["replies"]))
    wrong = {m: (reported.get(m, 0), logged.get(m, 0)) for m in set(logged) | set(reported)
             if reported.get(m, 0) != logged.get(m, 0)}
    if wrong:
        raise SystemExit(f"dry-run report and reply files differ (report, replies) for: {wrong}")


def report_from_files(cfg, source):
    """The dry-run report from dry_run_diagnostics.csv and timing.csv on disk, across every model there."""
    timing = read_rows(out_path(cfg, "timing_csv", source))
    report = dry_run_report(read_diagnostics(out_path(cfg, "dry_run_diagnostics_csv", source), cfg),
                            timing[timing["run"] == "dry_run"] if len(timing) else timing)
    check_report_counts(report, cfg, source)
    return report


def write_report(cfg, source):
    """Rebuild the report from the files and write it; no model is loaded."""
    report = report_from_files(cfg, source)
    report.to_csv(out_path(cfg, "dry_run_report_csv", source), index=False)
    return report


# ---------- running ----------

def batch_writer(cfg, source, path, pending):
    """on_rows: append a batch's replies, then its diagnostics and timing, so the three files grow
    together and a crash loses at most the batch in progress. Diagnostics keep the fixed columns."""
    cols = diagnostics_columns(cfg)

    def write(rows):
        append_rows(path, rows)
        unknown = {k for row in pending["diagnostics"] for k in row} - set(cols)
        if unknown:
            raise ValueError(f"diagnostics columns not in the fixed list: {sorted(unknown)}")
        for key, name in (("dry_run_diagnostics_csv", "diagnostics"), ("timing_csv", "timing")):
            if pending[name]:
                table = pd.DataFrame(pending[name])
                append_rows(out_path(cfg, key, source), table.reindex(columns=cols) if name == "diagnostics" else table)
                pending[name].clear()
    return write


def run_one_arm(parts, cfg, arm, cells, answers, source):
    """One arm of the dry run for one model: its own settings, contexts, run info, every seed."""
    spec, run = parts["spec"], "dry_run"
    gen, stash, pending = resolved_generation(cfg, spec["id"], arm), {}, {"diagnostics": [], "timing": []}
    append_jsonl(out_path(cfg, "run_info_jsonl", source), run_info_row(parts, run, arm, gen))
    contexts = out_path(cfg, "contexts_csv", source)
    if not contexts_written(contexts, spec["id"], run, arm):          # a resumed run does not repeat them
        append_rows(contexts, context_rows(parts, cells, run))
    log = lambda batch, seconds: pending["timing"].extend(timing_rows(spec["id"], run, batch, seconds))
    base = make_reply_fn(parts["prepare"], parts["render"], parts["make_generate"](gen, stash), log)
    diag = lambda fn: with_diagnostics(fn, stash, parts, pending["diagnostics"], {"model": spec["id"], "run": run})
    common = {"failure_log": (out_path(cfg, "failures_jsonl", source), {"model": spec["id"], "run": run}),
              "extra": {"model": spec["id"], "revision": spec["revision"], "precision": parts["info"]["precision"],
                        "system_mode": parts["mode"]}}
    path = out_path(cfg, "dry_run_arm3_csv" if arm == "arm3" else "dry_run_csv", source)
    for seed in seeds_for(cfg, spec, run):
        parts["set_seed"](seed)
        common.update(done_rows=load_done_rows(path, spec["id"]), on_rows=batch_writer(cfg, source, path, pending))
        if arm == "arm3":
            run_arm3(cells, seed, diag(base), answers, cfg["arm3"], stash, **common)
        else:
            run_dialogues_v4(cells, seed, diag(with_raw(base, stash)), answers, cfg["arms"][arm]["max_turns"],
                             gen["batch_size"], **common)


def run_dry_run(cfg, bank, make_parts, source, model_id=None):
    """Step b: every arm for every model (or one), then the per-model report, rebuilt from the files."""
    specs = [s for s in cfg["models"] if model_id in (None, s["id"])]
    if not specs:
        raise SystemExit(f"{model_id} is not a model id in the config")
    cells, answers = dry_run_cells(cfg, bank)
    for spec in specs:
        parts = make_parts(spec, cfg)
        for arm in ("arm1", "arm2", "arm3"):
            run_one_arm(parts, cfg, arm, cells[arm], answers, source)
        parts["close"]()
    return write_report(cfg, source)


# ---------- the real model ----------

def gpu_info():
    """GPU name, driver and CUDA version, for run_info.jsonl (protocol: reproduction check)."""
    import torch
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, check=True).stdout.strip().splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        driver = None
    return {"name": torch.cuda.get_device_name(0), "count": torch.cuda.device_count(), "driver": driver,
            "cuda": torch.version.cuda}


def roundtrip_name(options):
    return "_".join(f"{k}_{v}" for k, v in options.items()).lower()      # e.g. fix_mistral_regex_false


def roundtrip_tokenizers(spec, cfg, tokenizer):
    """The model's own tokenizer, plus one with the options in dry_run.roundtrip_also, if listed
    (Ministral without fix_mistral_regex)."""
    out = {"own": tokenizer}
    options = cfg["dry_run"].get("roundtrip_also", {}).get(spec["id"])
    if options:
        out[roundtrip_name(options)] = load_tokenizer_v4({**spec, "tokenizer_kwargs": options})
    return out


def gpu_parts(spec, cfg):
    """Load one model (load_model_v4: revision, class, lock, allowlist) and wrap it as parts."""
    import gc

    from ds import generate as g
    model, tokenizer, info = load_model_v4(spec, cfg)
    mode, prepare, render = render_parts(tokenizer, spec)
    held = {"model": model}

    def close():
        held.clear()
        gc.collect()
        g.torch.cuda.empty_cache()
    return {"spec": spec, "info": info, "mode": mode, "prepare": prepare, "render": render,
            "encode": lambda text: tokenizer(text, add_special_tokens=False)["input_ids"],
            "make_generate": lambda gen, stash: make_generate(held["model"], tokenizer, spec, gen,
                                                              cfg["thinking_markers"], stash),
            "set_seed": g.set_seed, "roundtrip": roundtrip_tokenizers(spec, cfg, tokenizer), "gpu": gpu_info(),
            "close": close}


# ---------- the mock (no GPU): same orchestration, canned replies ----------

class MockTokenizer:
    """Words are tokens; nothing is special. Re-encoding always gives the same tokens."""
    all_special_ids = []

    def __init__(self):
        self.vocab = {}

    def encode(self, text, add_special_tokens=False):
        return [self.vocab.setdefault(w, len(self.vocab) + 10) for w in text.split()]

    def decode(self, ids, skip_special_tokens=True):
        words = {i: w for w, i in self.vocab.items()}
        return " ".join(words[i] for i in ids)


def mock_parts(spec, cfg):
    """Parts without a model: every reply is the same hint, so no dialogue stops early. Round trips as
    in the config (Ministral's extra one too), so the mock writes the same diagnostics columns."""
    tok = MockTokenizer()
    also = cfg["dry_run"].get("roundtrip_also", {}).get(spec["id"])
    roundtrip = {"own": tok, **({roundtrip_name(also): tok} if also else {})}

    def make(gen, stash):
        def generate(prompts):
            replies = ["Let us think about which operation links the numbers." for _ in prompts]
            stash["raw"], stash["ids"] = list(replies), [tok.encode(r) for r in replies]
            stash["ended"] = [True] * len(prompts)
            return replies, [False] * len(prompts)
        return generate
    return {"spec": spec, "mode": "mock", "prepare": lambda messages: messages,
            "render": lambda messages: "\n".join(m["content"] for m in messages), "encode": tok.encode,
            "info": {"model": spec["id"], "revision": spec["revision"], "precision": "mock", "packages": {},
                     "lock_checked": False, "generation_config_diff": {}},
            "make_generate": make, "set_seed": lambda seed: None, "roundtrip": roundtrip, "gpu": None,
            "close": lambda: None}
