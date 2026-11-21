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

_None so far._
