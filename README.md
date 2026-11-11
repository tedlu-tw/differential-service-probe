# Differential-Service Probe

A small pilot experiment. It asks whether open-weight tutoring models give different help to the same math question when the student carries a different learner cue. The cue is stated outright (explicit), implied by two earlier exchanges (history) or implied by writing style (style).

The frozen design is in [docs/protocol.md](docs/protocol.md) (version 2). All questions and replies are synthetic or model-generated. There is no student data anywhere.

Version 3 is a new study with multi-turn dialogues. Its protocol is a draft and is not frozen yet. See [v3: multi-turn dialogues](#v3-multi-turn-dialogues). Everything above that section is about v2.

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

From `results/hand_check_agreement.csv`: 50 random replies from the real run, labelled by hand by the author, blind to condition and to the rule scores.

| Measure | n | Agreement | Rule 1, hand 0 | Rule 0, hand 1 |
|---|---|---|---|---|
| gives_answer | 50 | 1.00 | 0 | 0 |
| asks_attempt | 50 | 0.92 | 4 | 0 |

All 50 replies gave the answer, by both the hand labels and the rule, so this sample cannot test `gives_answer` on replies that withhold it. Every `asks_attempt` disagreement is the rule saying yes where the hand label said no.

## Deviations from the protocol

_None so far._ Report here any change made after the first real run to the rubric, the cues, the primary contrast or the exclusion rules.

## Data availability

The raw v2 replies, scores and summaries are held privately until the planned paper is published, and are available on request. The archive below holds the `results/` folder from the real run. Its checksum fixes which data the reported numbers came from.

`v2-results.tar.gz` sha256: `3aba09bccea9b8031d781892ee066b3419343995a4ccd8a7b862a2b70cce0e26`

To check a copy:
```bash
echo "3aba09bccea9b8031d781892ee066b3419343995a4ccd8a7b862a2b70cce0e26  v2-results.tar.gz" | shasum -a 256 -c
```

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

## v3: multi-turn dialogues

**Status: draft, not frozen.** The design is in [docs/protocol_v3.md](docs/protocol_v3.md). Two kinds of entry go in [docs/deviations_v3.md](docs/deviations_v3.md):
- every real-model run, the pre-freeze steps included,
- every departure from the protocol.

v2 stays frozen. Its config, code paths and results are unchanged, and tests pin the v2 mock outputs.

**What changes from v2:**
- **A pedagogical system prompt** asks the tutor not to give the final answer right away. v2's plain prompt is kept as a replication baseline.
- **Each cell is a dialogue of up to three tutor replies.** In turns 2 and 3, the student asks for the answer again with fixed messages. The dialogue stops at the first reply that gives the answer.
- **The primary measure is `gave_by_turn2`:** did the tutor give the answer by its second reply. The primary contrast is `history_struggle` − `history_success` under the pedagogical prompt.
- **There are 200 new questions,** none equal to a v2 question. Every answer is at least 21.

### v3 files

| Path | What it does |
|---|---|
| `configs/experiment_v3.yaml` | Every v3 path, seed, prompt, setting and word list. Outputs go to `results_v3/` |
| `src/ds/stimuli_v3.py` | 200 questions, and the turn-1 messages for every cell, solve prompt and check prompt |
| `src/ds/dialogue.py` | Runs dialogues turn by turn in batches, and resumes from logged rows |
| `src/ds/score.py`, `mock.py`, `analyze.py` | v3 functions added below the v2 code, which is unchanged |
| `src/ds/generate_v3.py` | Real-model dialogues on the GPU, with v2's model code and a per-batch timing log |
| `scripts/run_v3.py` | Modes `pilot` (steps a and c), `check` (step d) and `full` |
| `scripts/validate_v3.py` | Step b: a blind, stratified sheet of 100 replies, then agreement and kappa |

### v3 mock run (no GPU)

```bash
python scripts/run_v3.py pilot      # neutral pilot; no timing on the mock, so no compute estimate
python scripts/run_v3.py check
python scripts/run_v3.py full       # about 4 minutes, mostly the 200-run sensitivity
python scripts/validate_v3.py sample
```

Mock outputs go to `results_v3/mock/`. `mock-planted` raises `gave_by_turn2`, `hint_level` and `steps` under `history_struggle` and `low_score`. `mock-no_effect` changes nothing. The analysis is working if:
- `mock-planted` shows the planted effect in the primary and control contrasts.
- `mock-no_effect` has a primary interval around zero.

The sensitivity bases for `hint_level` and `steps` are placeholders until step a.

### Pre-freeze steps (protocol: Pre-freeze steps)

These run at the draft protocol, before `protocol-v3` is tagged. Add a run-log row for each GPU session with the date, step, commit, GPU type, count and memory, batch size and wall time.

**1. Start a pod** as in v2 step 2, and record `git rev-parse --short HEAD`. Run long commands under `nohup` or `tmux`, so a closed connection does not stop them.

**2. Dry run.** 5 questions × 20 cells × 1 seed, or 100 dialogues per model, written to `results_v3/replies_dry_run.csv`.
```bash
python -m ds.generate_v3 --dry-run
```
Check the output:
- No reply is empty.
- No Qwen reply contains `<think>`.
- The log shows the `system_mode=` line for each model.
- Some dialogues reach turns 2 and 3.
- `results_v3/timing.csv` has rows.

**3. Steps a and c: neutral pilot.**
```bash
nohup python scripts/run_v3.py pilot --source gpu > pilot.log 2>&1 &
```
It writes four files:
- `pilot_replies.csv`,
- `pilot_summary.csv`, with per model `gave_by_turn2`, the `answer_turn` shares, accuracy with working, the truncation rate, whether the floor or ceiling rule applies, and the turn-1 `hint_level` and `steps` shares,
- `timing.csv`,
- `compute_estimate.csv`: the GPU hours per model for the full design, for the run log.

Copy `results_v3/` off the pod before stopping it.

**4. Step b: scorer validation,** on your own machine.
```bash
python scripts/validate_v3.py sample --source csv
# label results_v3/validation_sheet.csv with the rubric in protocol_v3.md, section Scoring
python scripts/validate_v3.py agree --source csv --labeller "Your Name"
```
- **AI labels:** label a copy named `validation_sheet_ai.csv` and run `agree --ai`. These are reported separately and never decide a rule.
- **If `gives_answer` or `asks_attempt` falls below kappa 0.6:** revise that rule. Then move the old sheet away and draw a fresh sample with `sample --source csv --exclude <old sheet>`. A revised `gives_answer` means rerunning step a first.

**5. Step d: manipulation check,** on the GPU. This is 200 questions × 4 history cells, or 800 replies per model.
```bash
nohup python scripts/run_v3.py check --source gpu > check.log 2>&1 &
```

**6. Before the freeze:**
- Copy the pilot's turn-1 `hint_level` and `steps` shares into `mock.sensitivity.base_turn1_levels` and `base_steps`.
- Report steps a to d.
- Approve the protocol, then tag it:
```bash
git tag -a protocol-v3 -m "Protocol v3 frozen before the first full run"
git push origin main --tags
```

### v3 full run (only after the `protocol-v3` tag)

**Before you start:** copy `results_v3/pilot_replies.csv` and `results_v3/validation_agreement.csv` onto the pod. They decide each model's key measure, and `full` stops before using the GPU without them.
```bash
nohup python scripts/run_v3.py full --source gpu > full.log 2>&1 &
```
- **Resuming:** rows are appended after every batch, so if the pod stops part-way, rerun the same command. A resumed run batches dialogues differently, so record it in the run log.
- **One model at a time:** `python -m ds.generate_v3 --model <id>` generates without the analysis.

Copy `results_v3/` off the pod, then analyse on your own machine:
```bash
python scripts/run_v3.py full --source csv
```

| File | Contents |
|---|---|
| `results_v3/summary.csv` | One row per model × contrast × measure, all within one system prompt: mean difference, 95% interval and role. `precision` (the interval half-width) is on `hint_level`, `steps` and `words`. `min_detectable_diff` is on the primary row |
| `results_v3/replication.csv` | Turn-1 answer rate per model, plain prompt, neutral cells |
| `results_v3/dialogues.csv` | One row per dialogue: `answer_turn`, `gave_by_turn2` and the turn-1 measures |
| `results_v3/answer_turns.csv`, `hint_levels.csv` | Distributions per model × prompt × condition |
| `results_v3/diagnostics.csv` | Truncation, and identical turn-1 replies across seeds |
| `results_v3/sensitivity.csv` | Mock power per measure and planted size |
| `results_v3/validation_agreement.csv` | Step b: agreement and Cohen's kappa per measure, with the labeller |
| `results_v3/check_summary.csv` | Step d: accuracy, the unparsed rate and the parser rule per model × condition |

Report every contrast for every model. Do not average, rank or highlight models. A difference is evidence of differential service, not by itself of unfairness. Adapting to a struggling student can be appropriate.

### v3 scorer validation

_Not run yet._ Results will be reported here per labeller, with human and AI labels kept separate.

### v3 data availability

The raw v3 replies will be held privately, as v2's are, and their archive checksum will be published here.

### v3 limits

- **The scripted student never attempts the problem.** The dialogue measures how soon a tutor gives in to repeated requests, not the quality of the tutoring.
- **`gives_answer` stops the dialogue.** A scoring error therefore changes the dialogue itself, not only its score. Step b checks this rule.
- **`hint_level` and `asks_attempt` are rule-based.** Agreement on the stratified step b sample describes the scorer on that sample, not on the full run.
- **The sensitivity runs use the mock,** with turn-1 distributions taken from the pilot. The mock writes at most three step lines per reply.
- **The compute estimate is likely high.** It assumes every condition continues to turns 2 and 3 as often as the neutral pedagogical pilot does.
