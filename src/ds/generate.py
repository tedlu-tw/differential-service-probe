"""Generate tutor replies with the real models. Needs a CUDA GPU; not run in the sandbox.

On the pod, from the repo root:
    PYTHONPATH=src python -m ds.generate --dry-run                    # 5 questions, 1 seed, every model
    PYTHONPATH=src python -m ds.generate --model Qwen/Qwen3-8B        # full design, one model

Rows are appended after every batch, and cells already in the output CSV are skipped,
so a run that stops part-way can simply be started again.
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from ds import load_config
from ds.stimuli import build_cells, make_questions

KEY = ["model", "seed", "question_id", "condition", "phrasing"]


# ---------- choosing cells ----------

def select_cells(cfg, dry_run):
    cells = build_cells(cfg)
    if not dry_run:
        return cells
    questions, n = make_questions(cfg), cfg["dry_run"]["n_questions"]
    # Spread the dry-run questions over all families instead of taking the first few.
    keep = {q["id"] for q in questions[:: len(questions) // n][:n]}
    return [c for c in cells if c["question_id"] in keep]


def cell_key(model_id, seed, cell):
    return (model_id, str(seed), cell["question_id"], cell["condition"], cell["phrasing"])


def load_done(path):
    if not Path(path).exists():
        return set()
    return set(map(tuple, pd.read_csv(path, usecols=KEY)[KEY].astype(str).values))


def append_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


# ---------- model ----------

def load_model(model_id, dtype):
    # Left padding so every prompt in a batch ends where generation starts.
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype))
    return model.to("cuda").eval(), tokenizer


def render(tokenizer, messages, template_kwargs):
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         **template_kwargs)


def stop_ids(model, tokenizer):
    eos = model.generation_config.eos_token_id
    ids = set(eos) if isinstance(eos, list) else {eos}
    return (ids | {tokenizer.pad_token_id}) - {None}


@torch.no_grad()
def generate_batch(model, tokenizer, prompts, gen):
    # The chat template already adds BOS; adding special tokens again would double it.
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    out = model.generate(**inputs, do_sample=True, temperature=gen["temperature"], top_p=gen["top_p"],
                         top_k=gen["top_k"], repetition_penalty=gen["repetition_penalty"],
                         max_new_tokens=gen["max_new_tokens"], pad_token_id=tokenizer.pad_token_id)
    return out[:, inputs["input_ids"].shape[1]:]


def decode(tokenizer, new_tokens, stops, max_new):
    """Reply text, and whether it was cut off at max_new_tokens instead of ending by itself."""
    replies, truncated = [], []
    for row in new_tokens:
        replies.append(tokenizer.decode(row, skip_special_tokens=True).strip())
        truncated.append(len(row) == max_new and int(row[-1]) not in stops)
    return replies, truncated


# ---------- running ----------

def run_seed(model, tokenizer, model_id, seed, cells, cfg, out_path):
    gen, done = cfg["generation"], load_done(out_path)
    todo = [c for c in cells if cell_key(model_id, seed, c) not in done]
    set_seed(seed)
    stops, start = stop_ids(model, tokenizer), time.time()
    for i in range(0, len(todo), gen["batch_size"]):
        batch = todo[i:i + gen["batch_size"]]
        new = generate_batch(model, tokenizer, [c["prompt"] for c in batch], gen)
        replies, truncated = decode(tokenizer, new, stops, gen["max_new_tokens"])
        append_rows(out_path, [dict(zip(KEY, cell_key(model_id, seed, c)), reply=r, truncated=t)
                               for c, r, t in zip(batch, replies, truncated)])
        print(f"{model_id} seed={seed} {i + len(batch)}/{len(todo)} {time.time() - start:.0f}s", flush=True)


def run_model(spec, cells, seeds, cfg, out_path):
    model, tokenizer = load_model(spec["id"], cfg["generation"]["dtype"])
    kwargs = spec.get("chat_template_kwargs", {})
    # Prompts of similar length in one batch waste less padding.
    prompted = sorted(({**c, "prompt": render(tokenizer, c["messages"], kwargs)} for c in cells),
                      key=lambda c: len(c["prompt"]))
    for seed in seeds:
        run_seed(model, tokenizer, spec["id"], seed, prompted, cfg, out_path)
    del model
    torch.cuda.empty_cache()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/experiment.yaml")
    p.add_argument("--model", help="one model id from the config (default: all)")
    p.add_argument("--dry-run", action="store_true", help="few questions, one seed, separate CSV")
    return p.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("generate.py needs a CUDA GPU. Use scripts/run_all.py for the mock run.")
    cfg = load_config(args.config)
    specs = [m for m in cfg["models"] if args.model in (None, m["id"])]
    if not specs:
        raise SystemExit(f"{args.model} is not a model id in {args.config}")
    seeds = cfg["dry_run"]["seeds"] if args.dry_run else cfg["seeds"]["generation"]
    out_path = cfg["paths"]["dry_run_csv" if args.dry_run else "replies_csv"]
    cells = select_cells(cfg, args.dry_run)
    for spec in specs:
        run_model(spec, cells, seeds, cfg, out_path)


if __name__ == "__main__":
    main()
