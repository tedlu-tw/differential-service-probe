"""Environment reproduction check (docs/protocol_v4.md: step b). Descriptive, not a gate.

For each v3 model, rerun v3's first turn-1 batch of seed 0 through the code Arm 3 uses: load_model_v4
at the revision recorded for v3, render_parts, make_generate and v3's make_reply_fn, with Arm 3's
settings (batch size 16, 512 new tokens), set_seed(0) after loading, and v3's full cell list and order.
Compare each reply with the one v3 logged. A pass shows that Arm 3's own pipeline reproduces v3.

On the pod, in the v4 environment, on the GPU type v3 used (NVIDIA A40), from the repo root:

    PYTHONPATH=src python scripts/repro_v3.py --v3-replies /workspace/private/v3_replies.csv [--model MODEL_ID]

v3's replies are private: they are read from the path given, never copied into the repo, and the
output (results_v4/repro_v3.jsonl, one line per run) keeps only counts, cell keys, first differing
token positions, the GPU and the file's SHA-256, never reply text.

Pass: every reply of the first batch is identical to v3's. Otherwise the share identical and, for
each differing reply, the first token position where it differs (both texts re-encoded with the
model's tokenizer). A mismatch does not stop the run; log a deviations row.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import pandas as pd

from ds import load_config
from ds.dialogue import by_length, pending_states
from ds.stimuli_v3 import ROOT, build_cells_v3, make_questions_v3

V3_KEY = ["question_id", "prompt", "condition", "phrasing"]


# ---------- pure parts ----------

def first_batch_states(v3cfg):
    """The states of v3's first turn-1 batch of seed 0: every v3 cell, pending at turn 1, sorted by
    length as v3's run_dialogues sorts them, the first batch_size of them."""
    questions = make_questions_v3(v3cfg)
    cells = build_cells_v3(v3cfg, questions)
    states = pending_states(cells, 0, {}, {q["id"]: q["answer"] for q in questions}, v3cfg)
    return by_length(states)[:v3cfg["generation"]["batch_size"]]


def first_difference(a, b):
    """The first position where two token lists differ, or None if they are identical."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def archived_first_batch(path, model_id, states):
    """v3's logged turn-1 replies of seed 0 for these states, in their order."""
    df = pd.read_csv(path, keep_default_na=False)
    df = df[(df["model"] == model_id) & (df["seed"].astype(str) == "0") & (df["turn"].astype(str) == "1")]
    by_key = {tuple(str(r[k]) for k in V3_KEY): r["reply"] for r in df.to_dict("records")}
    missing = [s for s in states if tuple(str(s[k]) for k in V3_KEY) not in by_key]
    if missing:
        raise SystemExit(f"{len(missing)} first-batch cells of {model_id} are not in {path}")
    return [by_key[tuple(str(s[k]) for k in V3_KEY)] for s in states]


def compare(new, archived, states, encode):
    """Counts and first differing token positions only: no reply text."""
    diffs = []
    for s, a, b in zip(states, new, archived):
        if a != b:
            diffs.append({"cell": {k: s[k] for k in V3_KEY}, "first_token_differs": first_difference(encode(a), encode(b))})
    n = len(states)
    return {"n": n, "identical": n - len(diffs), "share_identical": (n - len(diffs)) / n, "passes": not diffs,
            "differences": diffs}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------- the run: v3's first batch through Arm 3's own pipeline ----------

def rerun_first_batch(spec, cfg, v3cfg, make_parts):
    """Load the model as Arm 3 does (load_model_v4 at the pinned revision), render and generate with
    Arm 3's code and settings (batch size 16, 512 new tokens), set_seed(0) after loading, and return
    the replies to v3's first batch, in its order."""
    from ds.generate_v3 import make_reply_fn
    from ds.models_v4 import resolved_generation
    states = first_batch_states(v3cfg)
    gen = resolved_generation(cfg, spec["id"], "arm3")
    if gen["batch_size"] != len(states):
        raise SystemExit(f"Arm 3's batch size {gen['batch_size']} differs from v3's first batch ({len(states)})")
    parts, stash = make_parts(spec, cfg), {}
    reply_fn = make_reply_fn(parts["prepare"], parts["render"], parts["make_generate"](gen, stash), lambda b, s: None)
    parts["set_seed"](0)
    new = [reply for reply, _ in reply_fn(states)]
    gpu = parts["gpu"]
    parts["close"]()
    return states, new, gpu


def check_model(spec, cfg, v3cfg, replies_path, make_parts, encode):
    states, new, gpu = rerun_first_batch(spec, cfg, v3cfg, make_parts)
    return {**compare(new, archived_first_batch(replies_path, spec["id"], states), states, encode(spec)), "gpu": gpu}


def tokenizer_encoder(spec):
    from ds.models_v4 import load_tokenizer_v4
    tokenizer = load_tokenizer_v4(spec)
    return lambda text: tokenizer(text, add_special_tokens=False)["input_ids"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--v3-replies", required=True, help="local path to v3's private replies.csv (never committed)")
    p.add_argument("--model", help="one v3 model id (default: all three)")
    p.add_argument("--config", default=str(ROOT / "configs" / "experiment_v4.yaml"))
    args = p.parse_args()
    from ds.generate_v4 import append_jsonl, gpu_parts
    from ds.models_v4 import package_versions
    cfg, v3cfg = load_config(args.config), load_config(ROOT / "configs" / "experiment_v3.yaml")
    specs = [s for s in cfg["models"] if s["group"] == "v3" and args.model in (None, s["id"])]
    out = {"replies_file_sha256": file_sha256(args.v3_replies), "packages": package_versions(),
           "written": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "models": {s["id"]: check_model(s, cfg, v3cfg, args.v3_replies, gpu_parts, tokenizer_encoder) for s in specs}}
    append_jsonl(ROOT / cfg["paths"]["repro_v3_jsonl"], out)
    print(json.dumps({m: {k: r[k] for k in ("n", "identical", "share_identical", "passes")}
                      for m, r in out["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
