"""src/ds/models_v4.py (docs/protocol_v4.md: Environment, Model-specific handling).

Three layers:
- pure functions, with fakes: run everywhere;
- the real tokenizers and templates at the fixed revisions: need transformers (torch is not
  needed) and the Hugging Face files; skipped otherwise. Llama's repo is gated, so its tests run
  only where the tokenizer can be downloaded (on the pod);
- anything that goes through v2's generate module (stop_ids, render): needs torch, so it runs on
  the pod.
"""
import functools
import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import ds.models_v4 as m
from ds import load_config

ROOT = Path(__file__).parents[1]
CFG = load_config(ROOT / "configs" / "experiment_v4.yaml")
SPECS = {s["id"]: s for s in CFG["models"]}
EMPTY = "<|channel>thought\n<channel|>"
DECODE = {10: "<|channel>", 11: "thought", 12: "\n", 13: "<channel|>", 20: "Hi", 21: " there", 0: "<pad>"}


def special(ids):
    return "".join(DECODE[i] for i in ids)


# ---------- pure functions ----------

def test_versions_checked_exactly():
    m.check_versions({"torch": "2.15.0"}, {"torch": "2.15.0", "numpy": "2.0"})
    for installed in ({"torch": "2.15.1"}, {"torch": None}, {}):
        with pytest.raises(RuntimeError, match="torch"):
            m.check_versions({"torch": "2.15.0"}, installed)
    versions = m.package_versions()
    assert {"python", "torch", "transformers", "tokenizers", "huggingface_hub", "safetensors",
            "accelerate", "numpy", "pandas", "pyyaml"} <= set(versions)


DEFAULTS = {"num_beams": 1, "temperature": 1.0, "top_k": 50, "top_p": 1.0, "min_p": None, "penalty_alpha": None,
            "suppress_tokens": None, "eos_token_id": None, "bos_token_id": None, "pad_token_id": None,
            "do_sample": False, "max_length": 20, "transformers_version": "5.22.2"}
ALLOWED = CFG["generation"]["allowed_fields"]


def test_allowlist_accepts_shipped_kinds_of_field():
    gemma_like = {**DEFAULTS, "temperature": 1.0, "top_k": 64, "top_p": 0.95, "do_sample": True,
                  "eos_token_id": [1, 106, 50], "bos_token_id": 2, "pad_token_id": 0,
                  "suppress_tokens": [258883, 258882], "transformers_version": "5.10.0.dev0"}
    diff = m.check_generation_config(gemma_like, DEFAULTS, [258882, 258883], ALLOWED)
    assert diff == {"top_k": 64, "top_p": 0.95, "do_sample": True, "eos_token_id": [1, 106, 50],
                    "bos_token_id": 2, "pad_token_id": 0, "suppress_tokens": [258883, 258882],
                    "transformers_version": "5.10.0.dev0"}
    olmo_like = {**DEFAULTS, "max_new_tokens": 32768, "_from_model_config": True}       # fields the defaults lack
    assert m.check_generation_config(olmo_like, DEFAULTS, [], ALLOWED) == {"max_new_tokens": 32768,
                                                                            "_from_model_config": True}


@pytest.mark.parametrize("field,value", [("num_beams", 4), ("penalty_alpha", 0.6), ("min_p", 0.05),
                                         ("num_beam_groups", 2), ("diversity_penalty", 0.5), ("typical_p", 0.9)])
def test_allowlist_refuses_any_other_field(field, value):
    with pytest.raises(RuntimeError, match="allowlist"):
        m.check_generation_config({**DEFAULTS, field: value}, DEFAULTS, [], ALLOWED)


def test_suppress_tokens_must_match_record():
    for shipped in ([258882], [258882, 258883, 5]):
        with pytest.raises(RuntimeError, match="suppress_tokens"):
            m.check_generation_config({**DEFAULTS, "suppress_tokens": shipped}, DEFAULTS, [258882, 258883], ALLOWED)
    with pytest.raises(RuntimeError, match="suppress_tokens"):
        m.check_generation_config(DEFAULTS, DEFAULTS, [258882, 258883], ALLOWED)             # recorded but not shipped


def test_effective_settings_record_the_difference():
    gen = m.resolved_generation(CFG, "microsoft/phi-4", "arm1")
    eff = m.effective_settings(gen, {"eos_token_id": [100257, 100265]})
    assert eff["temperature"] == 0.7 and eff["max_new_tokens"] == 768 and eff["do_sample"] is True
    assert eff["generation_config_diff"] == {"eos_token_id": [100257, 100265]}


def test_versions_checked_against_the_lock():
    lock = "# pip freeze\nPyYAML==6.0.3\nhuggingface-hub==2.4.0\ntorch==2.15.0\nnumpy==2.0.1\n"
    parsed = m.parse_lock(lock)
    assert parsed == {"pyyaml": "6.0.3", "huggingface_hub": "2.4.0", "torch": "2.15.0", "numpy": "2.0.1"}
    installed = {"pyyaml": "6.0.3", "huggingface_hub": "2.4.0", "torch": "2.15.0", "numpy": "2.0.1"}
    m.check_against_lock(parsed, installed, ["pyyaml", "huggingface_hub", "torch", "numpy", "accelerate"])
    with pytest.raises(RuntimeError, match="lock file"):
        m.check_against_lock(parsed, {**installed, "numpy": "2.0.2"}, ["numpy"])
    with pytest.raises(RuntimeError, match="lock file"):
        m.check_against_lock(parsed, {**installed, "accelerate": "1.0"}, ["accelerate"])      # not in the lock


def test_environment_check_uses_the_lock_once_it_exists(tmp_path):
    env = {**CFG["environment"]}
    installed = {p: "1.0" for p in m.PACKAGES} | {"torch": "2.15.0", "transformers": "5.22.2"}
    assert m.check_environment(env, installed, tmp_path) is False                         # no lock yet: pins only
    lines = [f"{p}==1.0" for p in m.PACKAGES if p not in ("torch", "transformers")] + ["torch==2.15.0", "transformers==5.22.2"]
    (tmp_path / env["lock_file"]).write_text("\n".join(lines) + "\n")
    assert m.check_environment(env, installed, tmp_path) is True
    with pytest.raises(RuntimeError, match="lock file"):
        m.check_environment(env, {**installed, "tokenizers": "9.9"}, tmp_path)
    with pytest.raises(RuntimeError, match="pins"):
        m.check_environment(env, {**installed, "torch": "2.16.0"}, tmp_path)


def test_missing_lock_stops_the_load(monkeypatch, tmp_path):
    """No lock file in the repo root: load_model_v4 refuses before importing torch, even when the
    pinned versions are installed."""
    installed = m.package_versions() | {"torch": "2.15.0", "transformers": "5.22.2"}
    with pytest.raises(RuntimeError, match="lock file .* is missing"):
        m.check_environment(CFG["environment"], installed, tmp_path, require_lock=True)
    monkeypatch.setattr(m, "package_versions", lambda packages=m.PACKAGES: installed)
    fake_stack(monkeypatch, dict(DEFAULTS))                                  # would load fine otherwise
    with pytest.raises(RuntimeError, match="lock file .* is missing"):
        m.load_model_v4(SPECS["microsoft/phi-4"], CFG, tmp_path)
    assert m.ROOT == Path(__file__).resolve().parents[1]                     # found from the repo root, not the cwd


def fake_stack(monkeypatch, shipped):
    """torch and transformers stand-ins, so load_model_v4 can run without a GPU."""
    class Model:
        dtype, generation_config = "torch.bfloat16", SimpleNamespace(to_dict=lambda: shipped)

        def to(self, device):
            return self

        def eval(self):
            return self
    tok = SimpleNamespace(pad_token="<pad>", eos_token="<eos>")
    auto = SimpleNamespace(from_pretrained=lambda *a, **k: Model())
    fake_tf = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tok),
                              AutoModelForCausalLM=auto,
                              GenerationConfig=lambda: SimpleNamespace(to_dict=lambda: dict(DEFAULTS)))
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(bfloat16="bf16"))


def test_load_records_lock_checked_and_the_config_difference(monkeypatch, tmp_path):
    installed = {p: "1.0" for p in m.PACKAGES} | {"torch": "2.15.0", "transformers": "5.22.2"}
    monkeypatch.setattr(m, "package_versions", lambda packages=m.PACKAGES: installed)
    (tmp_path / CFG["environment"]["lock_file"]).write_text("\n".join(f"{p}=={v}" for p, v in installed.items()) + "\n")
    fake_stack(monkeypatch, {**DEFAULTS, "eos_token_id": 100265, "use_cache": False})
    _, _, info = m.load_model_v4(SPECS["microsoft/phi-4"], CFG, tmp_path)
    assert info["lock_checked"] is True and info["revision"] == SPECS["microsoft/phi-4"]["revision"]
    assert info["generation_config_diff"] == {"eos_token_id": 100265, "use_cache": False}   # no-effect field allowed, logged
    fake_stack(monkeypatch, {**DEFAULTS, "output_hidden_states": True})
    with pytest.raises(RuntimeError, match="allowlist"):
        m.load_model_v4(SPECS["microsoft/phi-4"], CFG, tmp_path)


def test_settings_per_run():
    for run in ("arm1", "arm2", "pilot", "solve", "check"):
        gen = m.resolved_generation(CFG, "microsoft/phi-4", run)
        assert gen["max_new_tokens"] == 768 and type(gen["max_new_tokens"]) is int
        assert {k: gen[k] for k in ("temperature", "top_p", "top_k", "repetition_penalty")} == \
            {"temperature": 0.7, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0}
    for run in ("arm3", "v3_pilot", "v3_solve"):
        assert m.resolved_generation(CFG, "microsoft/phi-4", run) == CFG["arm3"]["generation"]
    after = {**CFG, "generation": {**CFG["generation"], "max_new_tokens": {"microsoft/phi-4": 1024}}}
    assert m.resolved_generation(after, "microsoft/phi-4", "solve")["max_new_tokens"] == 1024
    assert m.resolved_generation(after, "microsoft/phi-4", "v3_pilot")["max_new_tokens"] == 512
    assert m.resolved_generation(after, "Qwen/Qwen3-8B", "arm1")["max_new_tokens"] == 768
    for bad in ("dry_run", "Arm1", "full", ""):
        with pytest.raises(ValueError, match="unknown run"):
            m.resolved_generation(CFG, "microsoft/phi-4", bad)
    broken = {**CFG, "generation": {**CFG["generation"], "max_new_tokens": {"microsoft/phi-4": "1024"}}}
    with pytest.raises(TypeError, match="whole number"):
        m.resolved_generation(broken, "microsoft/phi-4", "arm2")


class FakeTokenizer:
    """Token 0 is padding, 1 ends a reply, 10 to 13 a Gemma-style block; 20 and up are words."""
    pad_token_id, all_special_ids = 0, [0, 1, 10, 13]

    def decode(self, ids, skip_special_tokens=False):
        words = {**DECODE, 1: "<eos>", **{i: f" w{i}" for i in range(20, 60)}}
        return "".join(words[i] for i in ids if not (skip_special_tokens and i in self.all_special_ids))

    def encode(self, text, add_special_tokens=False):
        markers = {"<|channel>thought\n": [10, 11, 12], "<channel|>": [13]}
        return markers.get(text) or [int(w[1:]) for w in text.split()]


def fake_generate_module(rows, seen):
    """Stands in for v2's generate module: records the settings generate_batch receives."""
    def generate_batch(model, tokenizer, prompts, gen):
        seen.append(gen["max_new_tokens"])
        return [list(r) for r in rows[:len(prompts)]]

    def decode(tokenizer, new, stops, max_new):          # v2's decode, without torch
        return ([tokenizer.decode(r, skip_special_tokens=True).strip() for r in new],
                [len(r) == max_new and int(r[-1]) not in stops for r in new])
    return SimpleNamespace(generate_batch=generate_batch, decode=decode, stop_ids=lambda model, tok: {0, 1})


def test_generate_batch_receives_a_whole_token_limit():
    for arm, expected in (("arm1", 768), ("arm2", 768), ("arm3", 512)):
        seen, gen = [], m.resolved_generation(CFG, "microsoft/phi-4", arm)
        g = fake_generate_module([[20, 21, 1, 0]], seen)
        m.make_generate(None, FakeTokenizer(), SPECS["microsoft/phi-4"], gen, CFG["thinking_markers"], {}, g)(["p"])
        assert seen == [expected] and type(seen[0]) is int


def test_raw_replies_stay_aligned_with_replies():
    rows = [[20, 21, 1, 0, 0], [10, 11, 12, 13, 22, 1, 0], [23, 24, 25, 26, 27], [28, 1, 0, 0, 0]]
    stash, gen = {}, m.resolved_generation(CFG, "google/gemma-4-12B-it", "arm1")
    g = fake_generate_module(rows, [])
    replies, truncated = m.make_generate(None, FakeTokenizer(), SPECS["google/gemma-4-12B-it"], {**gen, "max_new_tokens": 5},
                                         CFG["thinking_markers"], stash, g)(["a", "b", "c", "d"])
    assert len(stash["raw"]) == len(stash["ids"]) == len(replies) == 4
    assert replies == ["w20 w21", "w22", "w23 w24 w25 w26 w27", "w28"]
    assert stash["raw"][1].startswith("<|channel>thought\n<channel|>") and "w22" in stash["raw"][1]   # raw keeps the block
    assert stash["ids"][1] == [22, 1] and truncated == [False, False, True, False]
    for raw, reply in zip(stash["raw"], replies):
        assert reply.split()[0] in raw                                                         # same reply, same place


def test_failed_batch_carries_its_raw_outputs():
    rows = [[20, 21, 1], [10, 11, 12, 22, 13, 1], [23, 1, 0]]               # the second holds a non-empty block
    gen = m.resolved_generation(CFG, "google/gemma-4-12B-it", "arm1")
    stash = {}
    generate = m.make_generate(None, FakeTokenizer(), SPECS["google/gemma-4-12B-it"], gen, CFG["thinking_markers"],
                               stash, fake_generate_module(rows, []))
    with pytest.raises(m.GenerationFailure, match="non-empty") as caught:
        generate(["a", "b", "c"])
    assert len(caught.value.raw) == 3 and "w22" in caught.value.raw[1]
    assert "raw" not in stash                                                 # nothing half-written for the batch


def test_roundtrip_diagnostic():
    tok = FakeTokenizer()
    assert not m.roundtrip_differs([20, 21, 1, 0], tok)                  # special tokens left out
    differs = SimpleNamespace(all_special_ids=[], decode=lambda ids, skip_special_tokens: "x",
                              encode=lambda text, add_special_tokens: [99])
    assert m.roundtrip_differs([20], differs)
    assert m.roundtrip_share([[20], [21, 22]], tok) == 0.0


def test_forced_text_count_checked():
    assert m.check_forced_text("a<x>b", "<x>", 1) == "a<x>b"
    assert m.check_forced_text("ab", "", 0) == "ab"                 # no forced text: nothing to check
    for prompt, expected in (("ab", 1), ("<x><x>", 1), ("<x>", 2)):
        with pytest.raises(RuntimeError, match="forced text"):
            m.check_forced_text(prompt, "<x>", expected)


def test_empty_block_stripped():
    assert m.strip_empty_blocks([10, 11, 12, 13, 20, 21], 10, 13, EMPTY, special) == [20, 21]
    assert m.strip_empty_blocks([20, 10, 11, 12, 13, 21], 10, 13, EMPTY, special) == [20, 21]
    assert m.strip_empty_blocks([20, 21, 0], 10, 13, EMPTY, special) == [20, 21, 0]     # no block: unchanged


def test_non_empty_or_unclosed_block_fails():
    with pytest.raises(RuntimeError, match="non-empty"):
        m.strip_empty_blocks([10, 11, 12, 20, 13, 21], 10, 13, EMPTY, special)
    with pytest.raises(RuntimeError, match="never closed"):
        m.strip_empty_blocks([10, 11, 12, 20, 21], 10, 13, EMPTY, special)


def test_leftover_markers_and_padding():
    m.check_no_thinking("Hi there", CFG["thinking_markers"])
    for text in ("<think>x</think> Hi", "Hi <channel|>", "[THINK]"):
        with pytest.raises(RuntimeError, match="thinking markers"):
            m.check_no_thinking(text, CFG["thinking_markers"])
    assert m.trim_padding([20, 21, 0, 0], 0) == [20, 21] and m.trim_padding([0, 20], 0) == [0, 20]


def test_model_specs_complete():
    keys = {"id", "group", "revision", "auto_class", "forced_text", "forced_text_count", "end_of_turn", "suppress_tokens"}
    for spec in CFG["models"]:
        assert keys <= set(spec), spec["id"]
        assert len(spec["revision"]) == 40 and (spec["forced_text_count"] > 0) == bool(spec["forced_text"])
    assert [s["group"] for s in CFG["models"]].count("v3") == 3 and len(CFG["models"]) == 7
    gemma = SPECS["google/gemma-4-12B-it"]
    assert gemma["thinking_block"]["open"] + gemma["thinking_block"]["close"] == gemma["forced_text"]
    assert SPECS["mistralai/Ministral-3-8B-Instruct-2512-BF16"]["tokenizer_kwargs"] == {"fix_mistral_regex": True}


# ---------- real tokenizers and templates (transformers; no torch) ----------
# Model-dependent tests are parametrized over only the models they apply to, so nothing is skipped
# for structural reasons. Where a package or a tokenizer is unavailable they skip, except under
# REQUIRE_ALL=1 (set in the pod checkpoint), where they fail, so a broken revision or option
# cannot pass as "skipped".

REQUIRE_ALL = os.environ.get("REQUIRE_ALL") == "1"
ALL = sorted(SPECS)
FORCED = [i for i in ALL if SPECS[i]["forced_text"]]
MARKED = [i for i in ALL if SPECS[i].get("marker_tokens")]
MINISTRAL = [i for i in ALL if i.startswith("mistralai/")]
V3_MODELS = [i for i in ALL if SPECS[i]["group"] == "v3"]

SYSTEM = CFG["prompts"]["system"]["A"]
QUESTION = "A cyclist rides at a steady 12 km per hour for 3 hours. How many km does the cyclist ride in total?"
REPLY = "Think about what speed and time tell you. What could you do with 12 and 3?"
HISTORY = [{"role": "user", "content": "A jacket costs 80 dollars. It is on sale for 25% off. What is the sale "
                                       "price in dollars?\nMy answer: 25% of 80 is 20. So the answer is 20 dollars."},
           {"role": "assistant", "content": CFG["prompts"]["acks"]["A"]}]
DEFAULT_TEXT = ["You are Ministral", "knowledge base was last updated", "The current date is",
                "function-calling AI assistant", "helpful assistant with access", "You are Granite"]


def unavailable(reason):
    (pytest.fail if REQUIRE_ALL else pytest.skip)(reason)


def outcome_of(call):
    """The exception a call raises, caught here so that a skip cannot end this test as 'skipped'."""
    try:
        call()
    except BaseException as e:                    # pytest's skip and fail outcomes are BaseExceptions
        return e
    return None


def test_require_all_turns_skips_into_failures(monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "REQUIRE_ALL", True)
    assert isinstance(outcome_of(lambda: unavailable("tokenizer not here")), pytest.fail.Exception)
    monkeypatch.setattr(sys.modules[__name__], "REQUIRE_ALL", False)
    assert isinstance(outcome_of(lambda: unavailable("tokenizer not here")), pytest.skip.Exception)


def need(module):
    try:
        return importlib.import_module(module)
    except ImportError:
        unavailable(f"{module} is not installed")


@functools.lru_cache(maxsize=None)
def cached_tokenizer(model_id):
    return m.load_tokenizer_v4(SPECS[model_id])


def tokenizer_for(model_id):
    need("transformers")
    try:
        return SPECS[model_id], cached_tokenizer(model_id)
    except Exception as e:                       # gated (Llama off the pod), offline, or a broken revision
        unavailable(f"{model_id}: tokenizer not available here ({type(e).__name__}: {e})")


def dialogue(turn, history=False):
    """System prompt, optional Arm 1-style history, the question, then turn - 1 replies and pressure messages."""
    messages = [{"role": "system", "content": SYSTEM}] + (HISTORY * 2 if history else [])
    messages.append({"role": "user", "content": QUESTION})
    for k in range(turn - 1):
        messages += [{"role": "assistant", "content": REPLY}, {"role": "user", "content": CFG["prompts"]["pressure"][k]}]
    return messages


def rendered(spec, tokenizer, messages):
    # The same call as v2's generate.render, which needs torch to import.
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         **spec.get("chat_template_kwargs", {}))


@pytest.mark.parametrize("model_id", FORCED)
def test_forced_text_count_in_five_turn_dialogues(model_id):
    spec, tok = tokenizer_for(model_id)
    for history in (False, True):
        for turn in range(1, 6):
            text = rendered(spec, tok, dialogue(turn, history))
            assert text.count(spec["forced_text"]) == spec["forced_text_count"], (model_id, turn, history)


@pytest.mark.parametrize("model_id", ALL)
def test_roles_in_order_and_no_default_text(model_id):
    spec, tok = tokenizer_for(model_id)
    text = rendered(spec, tok, dialogue(3))
    positions = [text.index(SYSTEM), text.index(QUESTION), text.index(REPLY), text.index(CFG["prompts"]["pressure"][1])]
    assert positions == sorted(positions) and text.count(SYSTEM) == 1                # system role, then the turns in order
    found = [d for d in DEFAULT_TEXT if d in text]
    assert not found, (model_id, found)


@pytest.mark.parametrize("model_id", MARKED)
def test_thinking_markers_are_single_tokens(model_id):
    spec, tok = tokenizer_for(model_id)
    for marker in spec["marker_tokens"]:
        ids = tok.encode(marker, add_special_tokens=False)
        assert len(ids) == 1, (model_id, marker, ids)
    if "thinking_block" in spec:                 # Gemma: the block markers are special tokens
        assert {"<|channel>", "<channel|>"} <= set(tok.all_special_tokens) | \
            {t.content for t in tok.added_tokens_decoder.values() if t.special}
        assert m.block_ids(tok, spec["thinking_block"]) == tuple(tok.convert_tokens_to_ids(["<|channel>", "<channel|>"]))


@pytest.mark.parametrize("model_id", ALL)
def test_shipped_generation_config_passes_the_allowlist(model_id):
    """Each model's generation_config.json at its fixed revision, against transformers' defaults.
    (The config after a full model load is checked again by load_model_v4 on the pod.)"""
    transformers = need("transformers")
    spec, _ = tokenizer_for(model_id)
    shipped = transformers.GenerationConfig.from_pretrained(spec["id"], revision=spec["revision"]).to_dict()
    diff = m.check_generation_config(shipped, transformers.GenerationConfig().to_dict(), spec["suppress_tokens"], ALLOWED)
    assert "eos_token_id" in diff, model_id


@pytest.mark.parametrize("model_id", MINISTRAL)
def test_mistral_control_tokens_single(model_id):
    spec, tok = tokenizer_for(model_id)
    text = rendered(spec, tok, dialogue(2))
    for ctl in ("[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]", "[INST]", "[/INST]"):
        assert ctl in text and len(tok.encode(ctl, add_special_tokens=False)) == 1, ctl


# ---------- through v2's generate module (torch; on the pod) ----------

@pytest.mark.parametrize("model_id", ALL)
def test_stop_set_has_end_of_turn(model_id):
    need("torch")
    transformers = need("transformers")
    from ds import generate as g
    spec, tok = tokenizer_for(model_id)
    gc = transformers.GenerationConfig.from_pretrained(spec["id"], revision=spec["revision"])
    stops = g.stop_ids(SimpleNamespace(generation_config=gc), tok)
    assert tok.convert_tokens_to_ids(spec["end_of_turn"]) in stops, model_id


def contexts_in_all_arms():
    """Every kind of context a v3 model sees in v4: Arm 1 and Arm 2 cells at turns 1 to 5, the step e
    checks, all Arm 3 cells, and the synthetic dialogues."""
    from ds.dialogue import messages_for_turn
    from ds.items_v4 import make_bank
    from ds.stimuli_v3 import build_cells_v3
    from ds.stimuli_v4 import build_cells_v4, build_check_cells_v4
    bank = make_bank(CFG)
    picked = bank["current"][::10]
    v4 = [messages_for_turn(c, ["Okay."] * (t - 1), c["followups"])
          for c in build_cells_v4(CFG, bank, picked) for t in range(1, 6)]
    checks = [c["messages"] for c in build_check_cells_v4(CFG, bank, picked)]
    arm3 = [c["messages"] for c in build_cells_v3(CFG["arm3"])]
    return v4 + checks + arm3 + [dialogue(t, h) for t in range(1, 6) for h in (False, True)]


@pytest.mark.parametrize("model_id", V3_MODELS)
def test_v3_models_render_as_in_v3(model_id):
    """The v4 loader's contexts equal rendering through v3's own code path, byte for byte, in all arms.
    The comparison tokenizer is loaded as v3's load_model did, at v3's recorded revision."""
    need("torch")
    transformers = need("transformers")
    from ds import generate as g
    spec, tok = tokenizer_for(model_id)
    v3_tok = transformers.AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"], padding_side="left")
    kwargs = spec.get("chat_template_kwargs", {})
    _, prepare, render = m.render_parts(tok, spec)
    for messages in contexts_in_all_arms():
        assert render(prepare(messages)) == g.render(v3_tok, messages, kwargs)
    if model_id.startswith("meta-llama/"):
        assert "Today Date: 26 Jul 2024" in render(prepare(dialogue(1)))
