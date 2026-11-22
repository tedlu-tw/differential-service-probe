# Deviations and run log (protocol v4)

Kept from the start of v4, as required by `docs/protocol_v4.md`.
- Add a row to the run log for every real-model run and every judge run, pre-freeze steps included.
- Add a row to the deviations table for every departure from the protocol, together with what data had been seen when the change was made.

## Data seen before v4

When the v4 protocol was drafted (2026-11-21), these had been seen:
- **v3:** the full results, held privately. Their hashes are in the protocol's "Why v4" section. v4's design is motivated by two design limits of v3, as the protocol states. The author had seen v3's results when v4 was designed.
- **v3b:** a replication of v3 on the four new models was drafted and withdrawn before any model run. No v3b data exist.
- **v4:** no data. No v4 model or judge had been run.
- **Model choice:** made from model cards, chat templates, generation configs and the Hugging Face model API only.

## Run log

| Date | Step | Commit or tag | Model | Revision | Precision | GPU (type × count, memory) | Batch size | Wall time | Packages (Python, torch, transformers, CUDA; lock file) | Notes |
|---|---|---|---|---|---|---|---|---|---|---|

_No v4 run yet._

## Deviations

| Date | Protocol section | What changed | Why | Data seen at that point | Effect on results |
|---|---|---|---|---|---|
| 2026-11-21 | Model-specific handling: chat templates (draft, before the freeze) | The three v3 models are rendered exactly as in v3 in all three arms. Text their templates add when a system message is passed is kept as forced text, not removed. This means Llama 3.1's "Cutting Knowledge Date" and "Today Date" lines, and Qwen3's empty thinking block. The rule to remove default date and cutoff lines now applies to the four new models only. | Removing Llama's lines would change v3's rendering in Arm 3, or render Llama two ways within v4. | No v4 data. Templates and model cards only. | None on any contrast: the text is identical in every cell of a model. |
| 2026-11-21 | Model-specific handling: Ministral tokenizer (draft, before the freeze) | Ministral's tokenizer is loaded with transformers' `fix_mistral_regex=True`. The decision to use the repo's own tokenizer and template is unchanged, and no file is edited. | transformers warns that the repo's `tokenizer.json` regex leads to incorrect tokenization. The model's own replies are re-tokenized when fed back. | No v4 data. On all 2,000 Arm 3 contexts and every fixed v4 message, token IDs are identical with and without the option. | None on the protocol's own text; protects fed-back replies. |
| 2026-11-22 | Items: history pool (code, before the freeze) | The item bank first also kept pool items away from v2 and v3 question texts. That extra rule was removed, so the code matches the protocol as approved: only current questions must differ from v2 and v3, and the pool must differ from the current questions. With the v4 seed, 42 of the 50 rate and 19 of the 50 percentage pool items repeat a v2 or v3 question. | The extra rule made a 50-item rate pool impossible: the rate template has 115 texts, and v2 and v3 used 75. | No data. | History items in Arm 1 may be problems that v2 or v3 used as current questions. No v4 current question repeats one. |
| 2026-11-22 | Items: drawing the bank (code, before the freeze) | The item bank is redrawn with one random stream per family and part, instead of one shared stream. Current questions are drawn up to the cap (38 per family) and the first 25 used. The pool is drawn after excluding all 38, and each question's history assignment has its own stream. Every current question, pool item and assignment differs from the earlier draw. With the new draw, 48 of the 50 rate and 18 of the 50 percentage pool items repeat a v2 or v3 question (earlier draw: 42 and 19). | With one shared stream, raising `current_per_family` from 25 to 26 changed two families' current questions, the whole pool and 94 of 100 assignments. A step f increase would no longer have matched the dry run, the pilot or the step d sample. | No data: no model had been run on the bank. | None on any contrast. Adding questions in step f now only appends them. |
