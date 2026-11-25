"""Environment reproduction check (docs/protocol_v4.md: step b). Descriptive, not a gate.

For each v3 model, rerun v3's first turn-1 batch of seed 0 exactly as v3 ran it, through v3's own
code (generate_v3.gpu_parts): v3's full cell list and order, set_seed(0), batch size 16, 512 new
tokens. Compare each reply with the one v3 logged.

On the pod, in the v4 environment, on the GPU type v3 used, from the repo root:

    PYTHONPATH=src python scripts/repro_v3.py --v3-replies /local/path/to/v3/replies.csv [--model MODEL_ID]

v3's replies are private: they are read from the path given, never copied into the repo, and the
output (results_v4/repro_v3.jsonl, one line per run) keeps only counts, cell keys, first differing token positions and
the file's SHA-256, never reply text.

Pass: every reply of the first batch is identical to v3's. Otherwise the share identical and, for
each differing reply, the first token position where it differs (both texts re-encoded with the
model's tokenizer). A mismatch does not stop the run; log a deviations row.
"""
import argparse
import datetime
import gc
import hashlib
import json
import os
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


def cached_revision(model_id):
    """The commit the local Hugging Face cache's main ref points to (what v3's load_model loads)."""
    from huggingface_hub import scan_cache_dir
    for repo in scan_cache_dir().repos:
        if repo.repo_id == model_id:
            return next((r.commit_hash for r in repo.revisions if "main" in r.refs), None)
    return None


# ---------- the GPU run ----------

def check_model(spec, v3cfg, replies_path):
    """v3's first batch for one model through v3's own gpu_parts, compared with v3's logged replies."""
    from ds import generate as g
    from ds.generate_v3 import gpu_parts as v3_gpu_parts
    from ds.models_v4 import load_tokenizer_v4
    found = cached_revision(spec["id"])
    if found != spec["revision"]:
        raise SystemExit(f"{spec['id']}: the cache loads {found}, not v3's revision {spec['revision']}")
    states = first_batch_states(v3cfg)
    v3_spec = next(m for m in v3cfg["models"] if m["id"] == spec["id"])
    _, prepare, render, generate = v3_gpu_parts(v3_spec, v3cfg)
    g.set_seed(0)
    new, _ = generate([render(prepare(s["messages"])) for s in states])
    del generate, prepare, render                      # the model lives in these closures; free it
    gc.collect()
    g.torch.cuda.empty_cache()
    tokenizer = load_tokenizer_v4(spec)
    encode = lambda text: tokenizer(text, add_special_tokens=False)["input_ids"]
    return compare(new, archived_first_batch(replies_path, spec["id"], states), states, encode)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--v3-replies", required=True, help="local path to v3's private replies.csv (never committed)")
    p.add_argument("--model", help="one v3 model id (default: all three)")
    p.add_argument("--config", default=str(ROOT / "configs" / "experiment_v4.yaml"))
    args = p.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"                 # load only what is cached: v3's snapshots
    from ds.generate_v4 import append_jsonl, gpu_info
    from ds.models_v4 import package_versions
    cfg, v3cfg = load_config(args.config), load_config(ROOT / "configs" / "experiment_v3.yaml")
    specs = [s for s in cfg["models"] if s["group"] == "v3" and args.model in (None, s["id"])]
    out = {"replies_file_sha256": file_sha256(args.v3_replies), "gpu": gpu_info(), "packages": package_versions(),
           "written": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "models": {s["id"]: check_model(s, v3cfg, args.v3_replies) for s in specs}}
    append_jsonl(ROOT / cfg["paths"]["repro_v3_jsonl"], out)
    print(json.dumps({m: {k: r[k] for k in ("n", "identical", "share_identical", "passes")}
                      for m, r in out["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
