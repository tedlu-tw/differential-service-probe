"""src/ds/models_v4.py (docs/protocol_v4.md: Environment, Model-specific handling).

Three layers:
- pure functions, with fakes: run everywhere;
- the real tokenizers and templates at the fixed revisions: need transformers (torch is not
  needed) and the Hugging Face files; skipped otherwise. Llama's repo is gated, so its tests run
  only where the tokenizer can be downloaded (on the pod);
- anything that goes through v2's generate module (stop_ids, render): needs torch, so it runs on
  the pod.
"""
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


def test_off_fields_switched_off_and_recorded():
    gc = SimpleNamespace(top_k=64, min_p=0.05, typical_p=1.0, temperature=1.0)
    changed = m.apply_off_fields(gc, CFG["generation"]["off_fields"])
    assert gc.top_k == 0 and gc.min_p is None and gc.repetition_penalty == 1.0
    assert changed["top_k"] == [64, 0] and changed["min_p"] == [0.05, None]
    assert "typical_p" not in changed                              # already off: nothing to record
    assert gc.temperature == 1.0                                   # passed explicitly at generation instead


def test_required_off_fields_listed():
    required = {"repetition_penalty", "no_repeat_ngram_size", "min_p", "typical_p", "epsilon_cutoff",
                "eta_cutoff", "top_k"}
    assert required <= set(CFG["generation"]["off_fields"])


def test_suppress_tokens_must_match_record():
    m.check_suppress_tokens(SimpleNamespace(suppress_tokens=[258883, 258882]), [258882, 258883])
    m.check_suppress_tokens(SimpleNamespace(suppress_tokens=None), [])
    m.check_suppress_tokens(SimpleNamespace(), [])
    for shipped in ([258882], [258882, 258883, 5], None):
        with pytest.raises(RuntimeError, match="suppress_tokens"):
            m.check_suppress_tokens(SimpleNamespace(suppress_tokens=shipped), [258882, 258883])


def test_effective_settings_and_token_limit():
    gc = SimpleNamespace(min_p=None, suppress_tokens=[1], eos_token_id=[1, 106])
    gen = {**CFG["generation"], "max_new_tokens": 768}
    eff = m.effective_settings(gc, gen, {"min_p": None})
    assert eff["temperature"] == 0.7 and eff["max_new_tokens"] == 768 and eff["eos_token_id"] == [1, 106]
    assert m.max_new_tokens(CFG["generation"], "microsoft/phi-4") == 768
    assert m.max_new_tokens({**CFG["generation"], "max_new_tokens": {"microsoft/phi-4": 1024}}, "microsoft/phi-4") == 1024


def test_resolved_settings_have_a_whole_token_limit():
    for model in ("microsoft/phi-4", "Qwen/Qwen3-8B"):
        for arm in ("arm1", "arm2"):
            gen = m.resolved_generation(CFG, model, arm)
            assert gen["max_new_tokens"] == 768 and type(gen["max_new_tokens"]) is int      # before the pilot
            assert {k: gen[k] for k in ("temperature", "top_p", "top_k", "repetition_penalty")} == \
                {"temperature": 0.7, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0}
        assert m.resolved_generation(CFG, model, "arm3")["max_new_tokens"] == 512
    after = {**CFG, "generation": {**CFG["generation"], "max_new_tokens": {"microsoft/phi-4": 1024}}}
    assert m.resolved_generation(after, "microsoft/phi-4", "arm1")["max_new_tokens"] == 1024
    assert m.resolved_generation(after, "Qwen/Qwen3-8B", "arm1")["max_new_tokens"] == 768
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

SYSTEM = CFG["prompts"]["system"]["A"]
QUESTION = "A cyclist rides at a steady 12 km per hour for 3 hours. How many km does the cyclist ride in total?"
REPLY = "Think about what speed and time tell you. What could you do with 12 and 3?"
HISTORY = [{"role": "user", "content": "A jacket costs 80 dollars. It is on sale for 25% off. What is the sale "
                                       "price in dollars?\nMy answer: 25% of 80 is 20. So the answer is 20 dollars."},
           {"role": "assistant", "content": CFG["prompts"]["acks"]["A"]}]
DEFAULT_TEXT = ["You are Ministral", "knowledge base was last updated", "The current date is",
                "function-calling AI assistant", "helpful assistant with access", "You are Granite"]


def dialogue(turn, history=False):
    """System prompt, optional Arm 1-style history, the question, then turn - 1 replies and pressure messages."""
    messages = [{"role": "system", "content": SYSTEM}] + (HISTORY * 2 if history else [])
    messages.append({"role": "user", "content": QUESTION})
    for k in range(turn - 1):
        messages += [{"role": "assistant", "content": REPLY}, {"role": "user", "content": CFG["prompts"]["pressure"][k]}]
    return messages


@pytest.fixture(scope="module", params=sorted(SPECS))
def loaded(request):
    pytest.importorskip("transformers")
    spec = SPECS[request.param]
    try:
        return spec, m.load_tokenizer_v4(spec)
    except Exception as e:                       # gated (Llama off the pod) or offline
        pytest.skip(f"{spec['id']}: tokenizer not available here ({type(e).__name__})")


def rendered(spec, tokenizer, messages):
    # The same call as v2's generate.render, which needs torch to import.
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         **spec.get("chat_template_kwargs", {}))


def test_forced_text_count_in_five_turn_dialogues(loaded):
    spec, tok = loaded
    if not spec["forced_text"]:
        pytest.skip("no forced text")
    for history in (False, True):
        for turn in range(1, 6):
            text = rendered(spec, tok, dialogue(turn, history))
            assert text.count(spec["forced_text"]) == spec["forced_text_count"], (spec["id"], turn, history)


def test_roles_in_order_and_no_default_text(loaded):
    spec, tok = loaded
    text = rendered(spec, tok, dialogue(3))
    positions = [text.index(SYSTEM), text.index(QUESTION), text.index(REPLY), text.index(CFG["prompts"]["pressure"][1])]
    assert positions == sorted(positions) and text.count(SYSTEM) == 1                # system role, then the turns in order
    found = [d for d in DEFAULT_TEXT if d in text]
    assert not found, (spec["id"], found)


def test_thinking_markers_are_single_tokens(loaded):
    spec, tok = loaded
    for marker in spec.get("marker_tokens", []):
        ids = tok.encode(marker, add_special_tokens=False)
        assert len(ids) == 1, (spec["id"], marker, ids)
    if "thinking_block" in spec:                 # Gemma: the block markers are special tokens
        assert {"<|channel>", "<channel|>"} <= set(tok.all_special_tokens) | \
            {t.content for t in tok.added_tokens_decoder.values() if t.special}
        assert m.block_ids(tok, spec["thinking_block"]) == tuple(tok.convert_tokens_to_ids(["<|channel>", "<channel|>"]))


def test_mistral_control_tokens_single(loaded):
    spec, tok = loaded
    if not spec["id"].startswith("mistralai/"):
        pytest.skip("Ministral only")
    text = rendered(spec, tok, dialogue(2))
    for ctl in ("[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]", "[INST]", "[/INST]"):
        assert ctl in text and len(tok.encode(ctl, add_special_tokens=False)) == 1, ctl


# ---------- through v2's generate module (torch; on the pod) ----------

def test_stop_set_has_end_of_turn(loaded):
    pytest.importorskip("torch")
    from transformers import GenerationConfig
    from ds import generate as g
    spec, tok = loaded
    gc = GenerationConfig.from_pretrained(spec["id"], revision=spec["revision"])
    stops = g.stop_ids(SimpleNamespace(generation_config=gc), tok)
    assert tok.convert_tokens_to_ids(spec["end_of_turn"]) in stops, spec["id"]


def test_v3_models_render_as_in_v3(loaded):
    """The v4 loader's contexts equal rendering through v3's own code path, byte for byte."""
    pytest.importorskip("torch")
    from transformers import AutoTokenizer
    from ds import generate as g
    from ds.stimuli_v3 import build_cells_v3
    spec, tok = loaded
    if spec["group"] != "v3":
        pytest.skip("v3 models only")
    v3_tok = AutoTokenizer.from_pretrained(spec["id"], padding_side="left")          # as v3's load_model
    kwargs = spec.get("chat_template_kwargs", {})
    _, prepare, render = m.render_parts(tok, spec)
    contexts = [c["messages"] for c in build_cells_v3(CFG["arm3"])] + [dialogue(t, h) for t in range(1, 6) for h in (0, 1)]
    for messages in contexts:
        assert render(prepare(messages)) == g.render(v3_tok, messages, kwargs)
    if spec["id"].startswith("meta-llama/"):
        assert "Today Date: 26 Jul 2024" in render(prepare(dialogue(1)))
