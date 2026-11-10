# Differential-Service Probe

A small pilot experiment. It asks whether open-weight tutoring models give different help to the same math question when the student carries a different learner cue. The cue is stated outright (explicit), implied by two earlier exchanges (history) or implied by writing style (style).

The frozen design is in [docs/protocol.md](docs/protocol.md) (version 2). All questions and replies are synthetic or model-generated. There is no student data anywhere.

## Layout

| Path | What it does |
|---|---|
| `configs/experiment.yaml` | Every path, seed, cue phrasing, model id and setting |
| `docs/protocol.md` | The frozen design |
| `src/ds/stimuli.py` | 100 questions, the history pool, and the chat messages for every cell |
| `src/ds/mock.py` | A fake model for testing without a GPU |
| `src/ds/score.py` | Rule-based scoring. Tutoring replies are scored from the reply and the expected answer only |
| `src/ds/analyze.py` | Paired contrasts per model with a bootstrap over questions, the pre-freeze reports and the sensitivity runs |
| `src/ds/generate.py` | Generation with the real models (GPU only) |
| `scripts/run_all.py` | Runs the pipeline in three modes: `pilot`, `check` and `full` |
| `scripts/hand_check.py` | Blind hand check of 50 replies |
| `tests/` | Tests for stimuli, scoring and analysis |

## Setup

Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas pyyaml pytest   # enough for tests and the mock
export PYTHONPATH=src                     # every command below assumes this
pytest
```

`torch` and `transformers` (in `requirements.txt`) are needed only on the GPU machine.

## Mock run (no GPU)

The mock model plays two "models". `mock-planted` gives more full answers under `history_struggle` and `low_score`. `mock-no_effect` gives the same rate everywhere. Mock outputs go to `results/mock/` and never mix with real results.

```bash
python scripts/run_all.py pilot    # pre-freeze checks a, b, d: neutral only, no contrasts
python scripts/run_all.py check    # pre-freeze check c: manipulation check
python scripts/run_all.py full     # full design: scores, summary, sensitivity, diagnostics
python scripts/hand_check.py sample
```

The analysis is working if all three of these hold:
- `mock-planted` shows the planted effect in both the primary and the control contrast.
- `mock-no_effect` has a primary mean near zero.
- `sensitivity.csv` reaches 80% at a difference of 0.10.

In a single run, any one interval can exclude zero by chance about 1 time in 20. The tests therefore check the no-effect false-positive rate over 100 runs.

## Running on Runpod

Check each model's license first. The configured models are:
- `meta-llama/Llama-3.1-8B-Instruct`: Llama 3.1 Community License, gated. Accept the license on Hugging Face first.
- `Qwen/Qwen3-8B`: Apache 2.0. Its thinking mode is turned off in the config.
- `allenai/Olmo-3-7B-Instruct`: Apache 2.0.

**1. Tag the frozen protocol before any real model run.**
```bash
git tag -a protocol-v2 -m "Protocol frozen before the first real model run"
git push origin main --tags
```

**2. Start a pod** with a PyTorch template and at least 24 GB of GPU memory.
```bash
git clone <repo-url> differential-service-probe && cd differential-service-probe
git describe --tags   # should print protocol-v2: main is at the tagged commit
pip install -r requirements.txt
export HF_TOKEN=hf_...          # needed for the gated Llama model
export HF_HOME=/workspace/hf    # keep model downloads on the persistent volume
export PYTHONPATH=src
```

**3. Small test.** This makes 5 questions × 15 cells × 1 seed, or 75 replies per model, written to `results/replies_dry_run.csv`.
```bash
python -m ds.generate --dry-run
```
Check the output:
- No reply is empty.
- No Qwen reply contains `<think>`.
- The log shows the `system_mode=` line for each model.
- The log shows the seconds per batch. Use them to estimate the full run.

**4. Pre-freeze checks** (protocol section "Pre-freeze checks").
```bash
python scripts/run_all.py pilot --source gpu   # results/pilot_summary.csv
python scripts/run_all.py check --source gpu   # results/check_summary.csv
```
- **Truncation (rule d).** Every model needs `truncation_ok` to be True, meaning fewer than 1% of neutral replies were cut off. If any model fails:
  1. Raise `generation.max_new_tokens` for **all** models, since the settings must be the same for every model.
  2. Delete `results/pilot_replies.csv`.
  3. Rerun the pilot.
- **Ceiling or floor rule (rule b).** The `key_measure` column records it. `full` reads it from the pilot summary, so leave that file in place.
- **Commit and tag the settled config:**
  ```bash
  git commit -am "chore(config): set max_new_tokens from pilot"
  git tag -a full-run -m "Config for the first full run"
  ```

**5. Full run, one model at a time.** This is 3,000 replies per model. Rows are appended after every batch, so if the pod stops part-way, rerun the same command.
```bash
nohup python -m ds.generate --model meta-llama/Llama-3.1-8B-Instruct > gen_llama.log 2>&1 &
tail -f gen_llama.log
```
Check `results/replies.csv` after the first model, then run the others.

**6. Copy `results/` off the pod before stopping it.** For example, `scp -P <port> -r root@<pod-ip>:/workspace/differential-service-probe/results .`

**7. Stop the pod.**

## Analysis of the real run

On your own machine, with `results/` copied off the pod:
```bash
python scripts/run_all.py full --source csv       # scores, summary, sensitivity, diagnostics
python scripts/hand_check.py sample --source csv  # then fill the two label columns with 1 or 0
python scripts/hand_check.py agree --source csv
```

| File | Contents |
|---|---|
| `results/summary.csv` | One row per model × contrast × measure, with the mean difference and the 95% interval. `role` is `primary`, `control` or `exploratory`. `min_detectable_diff` is on the primary `gives_answer` row |
| `results/pilot_summary.csv` | Per model: `gives_answer` rate, mean steps, truncation rate, accuracy and key measure |
| `results/check_summary.csv` | Manipulation check accuracy per model × condition, and the number of unreadable replies |
| `results/diagnostics.csv` | Per model × condition: truncation rate, and the share of cells where both seeds gave the same reply |
| `results/sensitivity.csv` | Share of mock runs whose primary interval excludes zero, at planted differences of 0.05, 0.10 and 0.20 |
| `results/hand_check_agreement.csv` | Agreement between the hand labels and the scorer |

Report every contrast for every model. Do not average, rank or highlight models.

## Hand-check agreement

_Not yet run. Fill in from `results/hand_check_agreement.csv` after the real run._

| Measure | n | Agreement | Rule 1, hand 0 | Rule 0, hand 1 |
|---|---|---|---|---|
| gives_answer | 50 | | | |
| asks_attempt | 50 | | | |

## Deviations from the protocol

_None so far._ Report here any change made after the first real run to the rubric, the cues, the primary contrast or the exclusion rules.

## Limits

**What the study can and cannot show**
- This is a small pilot. It can show that help differs when the cue differs. It cannot show that a model holds an internal estimate of the learner.
- A difference is evidence of differential service, not by itself of unfairness. Adapting to a struggling student can be appropriate.
- The control contrast shows only that models respond to a stated level. The mock model, not the control contrast, validates the pipeline.

**Questions and cues**
- The questions come from four templates, with one wording each. They are less natural than real word problems.
- History cues use two fixed kinds of earlier problem: order of operations and rectangle perimeter.

**Scoring**
- **A wrong final answer counts as no answer.** `gives_answer` looks only for the correct answer.
- **`gives_answer` fires when a step mentions the answer number,** even if the reply does not hand over the answer.
- **Number words can cause false matches when the answer is small.** "zero" to "twenty" are read as numbers, and 19 of the 100 answers are 20 or below. A reply saying "four steps" counts as giving the answer 4.
- **`asks_attempt` counts an invitation phrase anywhere in the reply.** For example, "Let's try…" at the start of a full answer counts.
- **`encourages` also fires on phrases like "you can see that…".**
- **The hand check covers only 50 random replies.** If a model rarely gives the answer, the sample contains few positive cases.

**Analysis**
- **The "vs neutral" rows are exploratory and not corrected for multiple comparisons.** They share one neutral baseline, which has half the replies of the other conditions, so they move together.
- **Any single 95% interval excludes zero about 5% of the time when there is no effect.**
- **The sensitivity runs use the mock.** Its replies are independent yes/no outcomes, so real replies may be noisier.
- **No exclusion rules are defined.** Truncated replies are kept, and their rate is reported per condition.

**Pre-freeze checks**
- **The manipulation check reads the first two "yes"/"no" words.** Unusual phrasing can be misread, or counted as unreadable.
- **The mock's "identical across seeds" share is high (about 57%)** because it has only two reply texts. For real models at temperature 0.7, it should be near zero.
