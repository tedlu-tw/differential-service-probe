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


# ---------- generation settings ----------

def apply_off_fields(gen_config, off_fields):
    """Set every listed field to its off value on the model's generation config.
    Returns the fields whose shipped value was changed, as {field: [shipped, off]}."""
    changed = {}
    for field, off in off_fields.items():
        shipped = getattr(gen_config, field, None)
        if shipped != off:
            changed[field] = [shipped, off]
        setattr(gen_config, field, off)
    return changed


def check_suppress_tokens(gen_config, recorded):
    """The shipped suppress_tokens list must equal the one recorded in the config (protocol)."""
    shipped = sorted(getattr(gen_config, "suppress_tokens", None) or [])
    if shipped != sorted(recorded):
        raise RuntimeError(f"suppress_tokens shipped {shipped}, recorded {sorted(recorded)}")


def effective_settings(gen_config, gen, off_fields):
    """What is in effect for one run: the explicit sampling settings, then every off field,
    the suppressed tokens and the end-of-turn tokens, as the loaded model holds them."""
    explicit = {k: gen[k] for k in ("temperature", "top_p", "top_k", "repetition_penalty", "max_new_tokens")}
    fields = {k: getattr(gen_config, k, None) for k in off_fields}
    return {**explicit, "off_fields": fields,
            "suppress_tokens": getattr(gen_config, "suppress_tokens", None),
            "eos_token_id": getattr(gen_config, "eos_token_id", None)}


def max_new_tokens(gen, model_id):
    """Arms 1 and 2: the value fixed for this model after the pilot, else the first candidate."""
    return gen["max_new_tokens"].get(model_id, gen["max_new_tokens_candidates"][0])


SETTINGS = ("temperature", "top_p", "top_k", "repetition_penalty", "batch_size", "dtype")


def resolved_generation(cfg, model_id, arm):
    """One settings dict per model and arm, with a whole number for max_new_tokens. It is what v2's
    generate_batch and decode, effective_settings and the run log all receive.
    Arms 1 and 2: the model's pilot value, or the first candidate before the pilot. Arm 3: v3's own
    settings (512 new tokens)."""
    if arm == "arm3":
        gen = dict(cfg["arm3"]["generation"])
    else:
        source = cfg["generation"]
        gen = {**{k: source[k] for k in SETTINGS}, "max_new_tokens": max_new_tokens(source, model_id)}
    if not isinstance(gen["max_new_tokens"], int) or isinstance(gen["max_new_tokens"], bool):
        raise TypeError(f"max_new_tokens for {model_id} in {arm} is {gen['max_new_tokens']!r}, not a whole number")
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


def load_model_v4(spec, cfg):
    """Load one model at its fixed revision and class, as v3's load_model otherwise does it:
    left padding, the end-of-sequence token as pad if there is none, bf16, cuda, eval mode."""
    import torch
    import transformers
    check_versions(cfg["environment"], package_versions())
    rev = spec["revision"]
    tokenizer = load_tokenizer_v4(spec)
    model_class = getattr(transformers, spec["auto_class"])
    model = model_class.from_pretrained(spec["id"], revision=rev, dtype=getattr(torch, cfg["generation"]["dtype"]))
    model = model.to("cuda").eval()
    check_suppress_tokens(model.generation_config, spec["suppress_tokens"])
    changed = apply_off_fields(model.generation_config, cfg["generation"]["off_fields"])
    return model, tokenizer, {"model": spec["id"], "revision": rev, "precision": str(model.dtype),
                              "packages": package_versions(), "off_fields_changed": changed}


def render_parts(tokenizer, spec):
    """v2's system_mode, prepend_system and render, unchanged, plus the forced-text check."""
    from ds import generate as g
    kwargs = spec.get("chat_template_kwargs", {})
    mode = g.system_mode(tokenizer, kwargs)
    prepare = (lambda messages: messages) if mode == "system_role" else g.prepend_system
    render = lambda messages: check_forced_text(g.render(tokenizer, messages, kwargs), spec["forced_text"],
                                                spec["forced_text_count"])
    return mode, prepare, render


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
    stash["raw"] holds the raw replies (special tokens kept) and stash["ids"] the reply tokens
    after stripping, both in the order of the replies. g is v2's generate module; tests pass a fake."""
    if g is None:
        from ds import generate as g
    stops, block, pad = g.stop_ids(model, tokenizer), spec.get("thinking_block"), tokenizer.pad_token_id
    special = lambda ids: tokenizer.decode(ids, skip_special_tokens=False)
    ids_of_block = block_ids(tokenizer, block) if block else None

    def generate(prompts):
        rows = [list(map(int, r)) for r in g.generate_batch(model, tokenizer, prompts, gen)]
        kept = [strip_empty_blocks(r, *ids_of_block, block["open"] + block["close"], special)
                if block else r for r in rows]
        for r in kept:
            check_no_thinking(special(trim_padding(r, pad)), markers)
        _, truncated = g.decode(tokenizer, rows, stops, gen["max_new_tokens"])      # on the tokens as generated
        replies, _ = g.decode(tokenizer, kept, stops, gen["max_new_tokens"])       # text without empty blocks
        stash["raw"] = [special(trim_padding(r, pad)) for r in rows]
        stash["ids"] = [trim_padding(r, pad) for r in kept]
        return replies, truncated
    return generate
