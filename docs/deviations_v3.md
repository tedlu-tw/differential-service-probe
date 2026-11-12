# Deviations and run log (protocol v3)

Kept from the start of v3, as required by `docs/protocol_v3.md`. Add a row to the run log for every real-model run, pre-freeze steps included. Add a row to the deviations table for every departure from the protocol, together with what data had been seen when the change was made.

## Run log

| Date | Step | Commit or tag | Models | GPU (type × count, memory) | Batch size | Wall time | Notes |
|---|---|---|---|---|---|---|---|
| 2026-11-12 | Dry run | f16d18e | Llama 3.1 8B, Qwen3 8B, Olmo 3 7B | NVIDIA A40 × 1, 46 GB | 8 | 11.7 min of generation (sum of batch times; model loading not included) | 549 replies, none empty, no `<think>`, all `system_role`; dialogues reach turns 2 and 3. Run in the clone kept from v2, after a Runpod volume migration showed spurious local changes; no code was edited. |
| 2026-11-12 00:25 | a (pilot), c (estimate) | f16d18e | Llama 3.1 8B, Qwen3 8B, Olmo 3 7B | NVIDIA A40 × 1, 46 GB | 8 | 17.4 min of generation (dialogues 13.3, solve 4.2; model loading not included) | Same session, after the dry run. Not resumed. **Step c:** estimated GPU hours for the full design at batch size 8: Llama 5.2, Qwen 6.2, Olmo 6.2, about 17.7 in all (`compute_estimate.csv`). Rebuilt locally from the copied replies: `pilot_summary.csv` and `compute_estimate.csv` byte-identical. |
| 2026-11-12 | Dry run at batch size 16 | d6f5062 or later (to confirm) | Llama 3.1 8B, Qwen3 8B, Olmo 3 7B | NVIDIA A40 × 1, 46 GB | 16 | 9.2 min of generation (sum of batch times; model loading not included) | Repeated after raising the batch size. No out-of-memory error. 569 replies, none empty, no `<think>`, all `system_role`, 3 truncated. Seconds per reply against the batch-8 dry run: Llama 0.62 vs 0.83, Qwen 1.08 vs 1.50, Olmo 1.43 vs 1.82. Scaling the step c estimate by these ratios gives about 13 GPU hours for the full design (rough: 5 questions). The batch-8 dry run is kept as `replies_dry_run_bs8.csv`. |

## Deviations

| Date | Protocol section | What changed | Why | Data seen at that point | Effect on results |
|---|---|---|---|---|---|
| 2026-11-12 | Analysis: Sensitivity (draft, before the freeze) | The `hint_level` and `steps` sweeps draw from each model's own pilot distribution instead of one shared one; sizes a distribution cannot reach are reported as not plantable; `min_detectable_diff` is per model. The `gave_by_turn2` sweep is unchanged and shared. | The pilot's turn-1 distributions differ too much between models for one base (Llama `hint_level` 88% at level 1, Olmo 81% at level 3). | Step a pilot summary (neutral, pedagogical, 100 questions, 1 seed). No contrast data. | Changes only the sensitivity table and `min_detectable_diff`; no reply or contrast. |
| 2026-11-12 | Dialogue: models and sampling (config, before the freeze) | `generation.batch_size` 8 → 16 for the full run and later steps. The dry run and pilot used 8. | Speed: the step c estimate was about 17.7 GPU hours at batch size 8. | Step a pilot summary and timing. | Batching and padding differ from the pilot, which can change sampled tokens slightly; sampling settings are unchanged. The dry run is repeated at 16 before the freeze. |
| 2026-11-12 | Analysis: floor or ceiling rule | Not changed. Recorded because the pilot shows Olmo's fallback measure `hint_level` near its own ceiling (81% at level 3). | Kept as written before any data. | Step a pilot summary. | Olmo's primary and control contrasts use `hint_level`; struggle can raise its mean by at most about 0.19. |
