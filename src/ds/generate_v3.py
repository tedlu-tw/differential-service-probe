"""Generate v3 dialogues with the real models. Needs a CUDA GPU; not run in the sandbox.

On the pod, from the repo root:
    PYTHONPATH=src python -m ds.generate_v3 --dry-run                  # 5 questions, 1 seed, every model
    PYTHONPATH=src python -m ds.generate_v3 --model Qwen/Qwen3-8B      # full design, one model

The model code is v2's generate.py, unchanged. It is the only module that imports torch and
transformers, and it is imported inside the GPU functions below, so the rest of this file and its
tests run without them. The reply_fn is assembled from four small parts (prepare, render,
generate, log the time), which are the real model on the pod and fakes in the tests.

Rows are appended after every batch and finished turns are skipped, so a stopped run can be
started again. A resumed run batches dialogues differently, so record it in docs/deviations_v3.md.
"""
import argparse
import gc
import time
from pathlib import Path

import pandas as pd

from ds import load_config
from ds.dialogue import run_dialogues
from ds.stimuli_v3 import build_cells_v3, make_questions_v3


# ---------- pieces that need no GPU ----------

def single_turn(cfg):
    """The solve and check prompts get one reply each: a dialogue of one turn, with the same logging."""
    return {**cfg, "design": {**cfg["design"], "max_turns": 1}, "prompts": {**cfg["prompts"], "followups": []}}


def dry_run_questions(cfg, questions):
    # Spread over all families instead of taking the first few.
    n = cfg["dry_run"]["n_questions"]
    return questions[:: len(questions) // n][:n]


def append_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def load_done_rows(path, model_id):
    """Rows already logged for this model. keep_default_na so a reply such as "None" stays text."""
    if not Path(path).exists():
        return []
    df = pd.read_csv(path, keep_default_na=False)
    return df[df["model"] == model_id].to_dict("records")


def timing_rows(model_id, run, batch, seconds):
    return [{"run": run, "model": model_id, "seed": batch[0]["seed"], "turn": batch[0]["turn"],
             "batch_size": len(batch), "seconds": round(seconds, 3)}]


def make_reply_fn(prepare, render, generate, log_timing):
    """A reply_fn for run_dialogues.

    prepare(messages) -> messages for the chat template; render(messages) -> prompt text;
    generate(prompts) -> (replies, truncated); log_timing(batch, seconds) records the batch.
    """
    def reply_fn(batch):
        start = time.time()
        replies, truncated = generate([render(prepare(s["messages"])) for s in batch])
        log_timing(batch, time.time() - start)
        return list(zip(replies, truncated))
    return reply_fn


# ---------- GPU ----------

def gpu_parts(spec, cfg):
    """Load one model and wrap the v2 generation code as the four reply_fn parts."""
    from ds import generate as g
    model, tokenizer = g.load_model(spec["id"], cfg["generation"]["dtype"])
    kwargs, gen = spec.get("chat_template_kwargs", {}), cfg["generation"]
    mode, stops = g.system_mode(tokenizer, kwargs), g.stop_ids(model, tokenizer)
    prepare = (lambda messages: messages) if mode == "system_role" else g.prepend_system
    render = lambda messages: g.render(tokenizer, messages, kwargs)

    def generate(prompts):
        new = g.generate_batch(model, tokenizer, prompts, gen)
        return g.decode(tokenizer, new, stops, gen["max_new_tokens"])
    return mode, prepare, render, generate


def progress_bar(model_id, run, seed):
    """One tqdm bar per turn, counting dialogues. tqdm comes with transformers, so it is imported here only."""
    from tqdm import tqdm
    name = model_id.split("/")[-1]
    return lambda turn, total: tqdm(total=total, desc=f"{name} {run} seed={seed} turn={turn}", unit="dialogue")


def run_model_v3(spec, cells, seeds, answers, cfg, out_path, run):
    from ds import generate as g
    mode, prepare, render, generate = gpu_parts(spec, cfg)
    print(f"{spec['id']} system_mode={mode}", flush=True)

    def log_timing(batch, seconds):
        # Seconds per batch go to the timing log (step c); the bar shows progress on screen.
        append_rows(cfg["paths"]["timing_csv"], timing_rows(spec["id"], run, batch, seconds))

    fn = make_reply_fn(prepare, render, generate, log_timing)
    for seed in seeds:
        g.set_seed(seed)
        run_dialogues(cells, seed, fn, answers, cfg, done_rows=load_done_rows(out_path, spec["id"]),
                      on_rows=lambda rows: append_rows(out_path, rows),
                      extra={"model": spec["id"], "system_mode": mode},
                      progress=progress_bar(spec["id"], run, seed))


def generate_on_gpu(cells, answers, cfg, seeds, out_path, run, model_id=None):
    """Every configured model (or one), one after another, into one CSV of tutor-reply rows."""
    from ds import generate as g
    g.check_generation(cfg["generation"], seeds)
    specs = [m for m in cfg["models"] if model_id in (None, m["id"])]
    if not specs:
        raise SystemExit(f"{model_id} is not a model id in the config")
    for spec in specs:
        run_model_v3(spec, cells, seeds, answers, cfg, out_path, run)
        gc.collect()                       # the model lived in run_model_v3's closures; free it first
        g.torch.cuda.empty_cache()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/experiment_v3.yaml")
    p.add_argument("--model", help="one model id from the config (default: all)")
    p.add_argument("--dry-run", action="store_true", help="few questions, one seed, separate CSV")
    return p.parse_args()


def main():
    args = parse_args()
    from ds import generate as g
    if not g.torch.cuda.is_available():
        raise SystemExit("generate_v3 needs a CUDA GPU. Use scripts/run_v3.py for the mock run.")
    cfg = load_config(args.config)
    questions = make_questions_v3(cfg)
    chosen = dry_run_questions(cfg, questions) if args.dry_run else questions
    seeds = cfg["dry_run"]["seeds"] if args.dry_run else cfg["seeds"]["generation"]
    out_path = cfg["paths"]["dry_run_csv" if args.dry_run else "replies_csv"]
    generate_on_gpu(build_cells_v3(cfg, chosen), {q["id"]: q["answer"] for q in questions}, cfg, seeds, out_path,
                    "dry_run" if args.dry_run else "full", args.model)


if __name__ == "__main__":
    main()
