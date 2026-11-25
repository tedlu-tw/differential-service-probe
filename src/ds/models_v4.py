"""Model handling for v4 (docs/protocol_v4.md: Environment, Model-specific handling).

Only the model-specific parts live here: loading at the fixed revision and class, the version
check, the generation settings, rendering with the forced-text check, and the thinking-block
stripping. Generation, decoding, padding, batching and seeding are v2's and v3's code, unchanged
(ds.generate, ds.generate_v3, ds.dialogue).

torch and transformers are imported only inside the GPU functions at the bottom, so the rest of
this module, and its tests, run without them.
"""
import platform
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # the repo root, as stimuli_v3.ROOT
PACKAGES = ["torch", "transformers", "tokenizers", "huggingface_hub", "safetensors", "accelerate",
            "numpy", "pandas", "pyyaml"]


# ---------- environment ----------

def package_versions(packages=PACKAGES):
    """Installed version of each package, or None if it is not installed, plus Python's."""
    out = {"python": platform.python_version()}
    for name in packages:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def check_versions(pinned, installed):
    """Refuse to run unless every pinned package has exactly its pinned version."""
    wrong = {k: (v, installed.get(k)) for k, v in pinned.items() if installed.get(k) != v}
    if wrong:
        raise RuntimeError(f"package versions differ from the pins (pinned, installed): {wrong}")


def normalise(name):
    return name.strip().lower().replace("-", "_")


def parse_lock(text):
    """Package -> version from a pip freeze file ("name==version" lines); other lines are skipped."""
    out = {}
    for line in text.splitlines():
        if "==" in line and not line.lstrip().startswith("#"):
            name, version = line.split("==", 1)
            out[normalise(name)] = version.strip()
    return out


def check_against_lock(lock, installed, packages=PACKAGES):
    """Every package in PACKAGES must have the lock's version, or be absent if the lock lacks it."""
    wrong = {p: (lock.get(normalise(p)), installed.get(p)) for p in packages
             if lock.get(normalise(p)) != installed.get(p)}
    if wrong:
        raise RuntimeError(f"package versions differ from the lock file (locked, installed): {wrong}")


def check_environment(env, installed, root=None, require_lock=False):
    """Before any model loads: the pins (torch, transformers), and every package in PACKAGES against
    the lock file, found relative to the repo root, not the working directory. Returns whether the
    lock was checked. require_lock (every model load): a missing lock file stops the run, since the
    lock is frozen at the start of the first pod session, before any model loads."""
    check_versions({k: env[k] for k in ("torch", "transformers")}, installed)
    lock = Path(root or ROOT) / env["lock_file"]
    if lock.exists():
        check_against_lock(parse_lock(lock.read_text()), installed)
        return True
    if require_lock:
        raise RuntimeError(f"lock file {lock} is missing; no model loads without it (protocol: Environment)")
    return False


# ---------- generation settings ----------

def config_diff(loaded, defaults):
    """Every field of a loaded generation config (as a dict) that differs from transformers' defaults."""
    return {k: v for k, v in loaded.items() if k not in defaults or defaults[k] != v}


def allowed_value(value, allowed):
    """Exact match, type included, so that 0 never passes for False or 1 for True."""
    return any(type(value) is type(a) and value == a for a in allowed)


def check_generation_config(loaded, defaults, recorded_suppress, allowed):
    """Allowlist: a loaded generation config may differ from the defaults only in token ids, the
    recorded suppress_tokens, the settings every generate call passes explicitly, metadata, and the
    listed values of a few load-time fields that cannot change token probabilities (no_effect).
    Any other non-default field (num_beams, penalty_alpha, min_p, ...) stops the run.
    Returns the full difference, for the run log."""
    diff = config_diff(loaded, defaults)
    ok = set(allowed["token_ids"]) | set(allowed["explicit"]) | set(allowed["metadata"]) | {"suppress_tokens"}
    no_effect = allowed["no_effect"]              # field -> the non-default values allowed
    other = {k: v for k, v in diff.items()
             if k not in ok and not (k in no_effect and allowed_value(v, no_effect[k]))}
    if other:
        raise RuntimeError(f"generation config fields or values not on the allowlist: {other}")
    shipped = sorted(diff.get("suppress_tokens") or [])
    if shipped != sorted(recorded_suppress):
        raise RuntimeError(f"suppress_tokens shipped {shipped}, recorded {sorted(recorded_suppress)}")
    return diff


def effective_settings(gen, diff):
    """What is in effect for one run: the explicit settings (resolved_generation) and the loaded
    generation config's difference from the defaults, whose explicit fields the call overrides."""
    explicit = {k: gen[k] for k in ("temperature", "top_p", "top_k", "repetition_penalty", "max_new_tokens")}
    return {**explicit, "do_sample": True, "generation_config_diff": diff}


def max_new_tokens(gen, model_id):
    """Arms 1 and 2: the value fixed for this model after the pilot, else the first candidate."""
    return gen["max_new_tokens"].get(model_id, gen["max_new_tokens_candidates"][0])


SETTINGS = ("temperature", "top_p", "top_k", "repetition_penalty", "batch_size", "dtype")

# Which settings each run uses (protocol: Generation settings). Dry-run cells use their own arm's.
#   v4: Arms 1 and 2, the neutral pilot (Arm 2 cells), v4 accuracy with working, the step e checks.
#   v3: Arm 3, and v3's step a pilot and its accuracy with working, run for the new models in step c.
RUN_SETTINGS = {"arm1": "v4", "arm2": "v4", "pilot": "v4", "solve": "v4", "check": "v4",
                "arm3": "v3", "v3_pilot": "v3", "v3_solve": "v3"}


def resolved_generation(cfg, model_id, run):
    """One settings dict per model and run, with a whole number for max_new_tokens. It is what v2's
    generate_batch and decode, effective_settings and the run log all receive.
    v4 runs: the model's pilot value, or the first candidate before the pilot. v3 runs: v3's own
    settings (512 new tokens). Any other run name is refused."""
    if run not in RUN_SETTINGS:
        raise ValueError(f"unknown run {run!r}; expected one of {sorted(RUN_SETTINGS)}")
    if RUN_SETTINGS[run] == "v3":
        gen = dict(cfg["arm3"]["generation"])
    else:
        source = cfg["generation"]
        gen = {**{k: source[k] for k in SETTINGS}, "max_new_tokens": max_new_tokens(source, model_id)}
    if not isinstance(gen["max_new_tokens"], int) or isinstance(gen["max_new_tokens"], bool):
        raise TypeError(f"max_new_tokens for {model_id} in {run} is {gen['max_new_tokens']!r}, not a whole number")
    return gen


# ---------- rendering ----------

def check_forced_text(prompt, forced, expected):
    """Forced text, if any, must appear in every rendered context exactly as often as the config
    records (measured on the real template in 5-turn dialogues)."""
    if forced and prompt.count(forced) != expected:
        raise RuntimeError(f"forced text {forced!r} appears {prompt.count(forced)} times in a rendered "
                           f"context, expected {expected}")
    return prompt


# ---------- thinking blocks ----------

def find_blocks(ids, open_id, close_id):
    """(start, end) spans of thinking blocks in a list of token ids, end exclusive.
    An opening marker without a closing one is an error."""
    spans, start = [], None
    for i, tok in enumerate(ids):
        if tok == open_id and start is None:
            start = i
        elif tok == close_id and start is not None:
            spans.append((start, i + 1))
            start = None
    if start is not None:
        raise RuntimeError("a thinking block is opened and never closed")
    return spans


def strip_empty_blocks(ids, open_id, close_id, empty_text, decode_special):
    """Remove empty thinking blocks; raise on a block that is not empty.

    decode_special(ids) -> text with special tokens kept. A block is empty when its text is
    exactly empty_text, the model's open and close markers with nothing between them."""
    spans = find_blocks(ids, open_id, close_id)
    for start, end in spans:
        text = decode_special(ids[start:end])
        if text != empty_text:
            raise RuntimeError(f"non-empty thinking block in a reply: {text[:200]!r}")
    keep = set(range(len(ids))) - {i for s, e in spans for i in range(s, e)}
    return [ids[i] for i in sorted(keep)]


def check_no_thinking(text, markers):
    """After stripping, no thinking marker may remain in the reply (decoded with special tokens)."""
    found = [m for m in markers if m in text]
    if found:
        raise RuntimeError(f"thinking markers {found} in a reply: {text[:200]!r}")


def trim_padding(ids, pad_id):
    """Generated rows of a batch are padded at the end; the raw reply does not include the padding."""
    end = len(ids)
    while end and ids[end - 1] == pad_id:
        end -= 1
    return ids[:end]


# ---------- dry-run diagnostic: does a reply's text re-encode to its own tokens? ----------

def roundtrip_differs(ids, tokenizer):
    """True if a reply's tokens, special tokens left out, differ from re-encoding their decoded
    text. A reply is fed back as text, so a difference means the model sees other tokens next turn."""
    special = set(tokenizer.all_special_ids)
    content = [i for i in ids if i not in special]
    text = tokenizer.decode(content, skip_special_tokens=True)
    return tokenizer.encode(text, add_special_tokens=False) != content


def roundtrip_share(rows, tokenizer):
    """Share of replies (lists of token ids) whose text does not re-encode to the same tokens."""
    return sum(roundtrip_differs(r, tokenizer) for r in rows) / len(rows) if rows else float("nan")


# ---------- GPU: load, render, generate ----------

def load_tokenizer_v4(spec):
    """The tokenizer at the fixed revision, left padding as in v3, with any tokenizer options the
    config records (Ministral: transformers' fix_mistral_regex). Also used by the template tests."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"], padding_side="left",
                                              **spec.get("tokenizer_kwargs", {}))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model_v4(spec, cfg, root=None):
    """Load one model at its fixed revision and class, as v3's load_model otherwise does it:
    left padding, the end-of-sequence token as pad if there is none, bf16, cuda, eval mode.
    Refuses to load without the lock file, on a package mismatch, or on a generation config field
    off the allowlist. The environment is checked before torch is imported."""
    installed = package_versions()
    lock_checked = check_environment(cfg["environment"], installed, root, require_lock=True)
    import torch
    import transformers
    rev = spec["revision"]
    tokenizer = load_tokenizer_v4(spec)
    model_class = getattr(transformers, spec["auto_class"])
    model = model_class.from_pretrained(spec["id"], revision=rev, dtype=getattr(torch, cfg["generation"]["dtype"]))
    model = model.to("cuda").eval()
    diff = check_generation_config(model.generation_config.to_dict(), transformers.GenerationConfig().to_dict(),
                                   spec["suppress_tokens"], cfg["generation"]["allowed_fields"])
    return model, tokenizer, {"model": spec["id"], "revision": rev, "precision": str(model.dtype),
                              "packages": installed, "lock_checked": lock_checked, "generation_config_diff": diff}


def render_parts(tokenizer, spec):
    """v2's system_mode, prepend_system and render, unchanged, plus the forced-text check."""
    from ds import generate as g
    kwargs = spec.get("chat_template_kwargs", {})
    mode = g.system_mode(tokenizer, kwargs)
    prepare = (lambda messages: messages) if mode == "system_role" else g.prepend_system
    render = lambda messages: check_forced_text(g.render(tokenizer, messages, kwargs), spec["forced_text"],
                                                spec["forced_text_count"])
    return mode, prepare, render


class GenerationFailure(RuntimeError):
    """A batch failed a check after generation (a non-empty or unclosed thinking block, or a leftover
    marker). Carries the batch's raw outputs, so they can be logged before the run stops."""
    def __init__(self, message, raw):
        super().__init__(message)
        self.raw = raw


def block_ids(tokenizer, block):
    """Token ids of a thinking block's opening and closing markers (each one special token)."""
    open_id = tokenizer.encode(block["open"], add_special_tokens=False)[0]
    close = tokenizer.encode(block["close"], add_special_tokens=False)
    if len(close) != 1:
        raise RuntimeError(f"the closing marker {block['close']!r} is not one token")
    return open_id, close[0]


def make_generate(model, tokenizer, spec, gen, markers, stash, g=None):
    """generate(prompts) -> (replies, truncated) for make_reply_fn, using v2's generate_batch and
    decode unchanged. gen is the resolved settings dict (resolved_generation). For each batch,
    stash["raw"] holds the raw replies (special tokens kept), stash["ids"] the reply tokens after
    stripping, and stash["ended"] whether each reply ended on a stop token, all in reply order. g is v2's generate module; tests pass a fake."""
    if g is None:
        from ds import generate as g
    stops, block, pad = g.stop_ids(model, tokenizer), spec.get("thinking_block"), tokenizer.pad_token_id
    special = lambda ids: tokenizer.decode(ids, skip_special_tokens=False)
    ids_of_block = block_ids(tokenizer, block) if block else None

    def checked(rows, raw):
        try:
            kept = [strip_empty_blocks(r, *ids_of_block, block["open"] + block["close"], special)
                    if block else r for r in rows]
            for r in kept:
                check_no_thinking(special(trim_padding(r, pad)), markers)
        except RuntimeError as e:                 # logged by the caller with the batch's keys, then raised
            raise GenerationFailure(str(e), raw) from e
        return kept

    def generate(prompts):
        rows = [list(map(int, r)) for r in g.generate_batch(model, tokenizer, prompts, gen)]
        raw = [special(trim_padding(r, pad)) for r in rows]
        kept = checked(rows, raw)
        _, truncated = g.decode(tokenizer, rows, stops, gen["max_new_tokens"])      # on the tokens as generated
        replies, _ = g.decode(tokenizer, kept, stops, gen["max_new_tokens"])       # text without empty blocks
        stash["raw"], stash["ids"] = raw, [trim_padding(r, pad) for r in kept]
        stash["ended"] = [any(tok in stops for tok in r) for r in rows]          # ended on a stop token
        return replies, truncated
    return generate
