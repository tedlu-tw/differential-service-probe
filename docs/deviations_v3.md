# Deviations and run log (protocol v3)

Kept from the start of v3, as required by `docs/protocol_v3.md`. Add a row to the run log for every real-model run, pre-freeze steps included. Add a row to the deviations table for every departure from the protocol, together with what data had been seen when the change was made.

## Run log

| Date | Step | Commit or tag | Models | GPU (type × count, memory) | Batch size | Wall time | Notes |
|---|---|---|---|---|---|---|---|
| 2026-11-12 | Dry run | f16d18e | Llama 3.1 8B, Qwen3 8B, Olmo 3 7B | NVIDIA A40 × 1, 46 GB | 8 | 11.7 min of generation (sum of batch times; model loading not included) | 549 replies, none empty, no `<think>`, all `system_role`; dialogues reach turns 2 and 3. Run in the clone kept from v2, after a Runpod volume migration showed spurious local changes; no code was edited. |
| 2026-11-12 00:25 | a (pilot), c (estimate) | f16d18e | Llama 3.1 8B, Qwen3 8B, Olmo 3 7B | NVIDIA A40 × 1, 46 GB | 8 | 17.4 min of generation (dialogues 13.3, solve 4.2; model loading not included) | Same session, after the dry run. Not resumed. **Step c:** estimated GPU hours for the full design at batch size 8: Llama 5.2, Qwen 6.2, Olmo 6.2, about 17.7 in all (`compute_estimate.csv`). Rebuilt locally from the copied replies: `pilot_summary.csv` and `compute_estimate.csv` byte-identical. |

## Deviations

| Date | Protocol section | What changed | Why | Data seen at that point | Effect on results |
|---|---|---|---|---|---|

_None so far._
