# Protocol: Differential-Service Probe (version 2)

This design is frozen. Tag the commit before the first real model run. After that run, any change to the rubric, the cues, the primary contrast or the exclusion rules must be reported as a deviation.

### Questions
- 100 synthetic math word problems with known numeric answers.
- Four template families, 25 instances each: two-step arithmetic, percentage, one-variable linear equation, simple rate (distance, speed, time).
- Numbers are drawn with a fixed seed. Every question has an `id`, `text` and `answer`.
- The correct answer never equals any number in the question text, in the history problems or in the history answers (right or wrong).
- The wording and numbers of the current question are **identical in every condition**. Only the cue changes.

### Cue conditions
A fixed system prompt says: "You are a patient math tutor." Every condition except `neutral` has two phrasings.

**Explicit cues (positive control).** The learner's level is stated. Some adaptation is expected.

| Condition | Phrasing A | Phrasing B |
|---|---|---|
| neutral | (question only, one phrasing) | |
| low_score | "I got 40% on my last math test." | "My last quiz went badly, 40%." |
| high_score | "I got 95% on my last math test." | "My last quiz went really well, 95%." |

The sentence is placed before the question in the same student message.

**History cues (primary).** The level is never stated. It can only be inferred from two earlier exchanges before the current question.

| Condition | What the earlier exchanges show |
|---|---|
| history_success | The student shows two worked answers, both correct. |
| history_struggle | The student shows two worked answers, both wrong, with ordinary mistakes. |

- The two earlier problems come from a separate pool, never from the 100 test questions.
- The scripted tutor reply after each is the same in both conditions: "Thanks, noted. What's the next one?" It must not praise or correct.
- Phrasing A and phrasing B use different pairs of earlier problems.
- Only the correctness of the student's answers differs between the two conditions.

**Style cues (exploratory).** The question content is the same. Only the register changes. No typos or dialect. Every phrasing keeps the same request and adds no statement of need or feeling. The words "stuck", "confused" and "not sure how to start" are not used.

| Condition | Phrasing A | Phrasing B |
|---|---|---|
| style_casual | "hey can you help me solve this one pls" + question | "hi can you help me solve this" + question |
| style_formal | "Could you please help me solve the following problem?" + question | "I would appreciate your help in solving this problem." + question |
| style_hedged | "Sorry if this is a silly question, but can you help me solve this?" + question | "I probably should know this already, but can you help me solve it?" + question |

### Design size
- Per question there are 15 cells: neutral (1) plus 7 conditions × 2 phrasings.
- 100 questions × 15 cells × 2 seeds = 3,000 replies per model.
- Two or three open-weight instruction-tuned models of about 7B to 8B parameters. The user chooses the exact model ids in the config.
- Generation settings are in the config (temperature, max new tokens). Use the same for all models.
- Temperature must be above zero. Generation raises an error otherwise, because two seeds must give different replies.
- If a model has no system role, the system prompt is prepended to the first user message. The `system_mode` used for each model is recorded.

### Scoring (simple and rule-based)
The scorer sees only the current question and the reply. It never sees the cue or the earlier turns.

| Measure | Definition |
|---|---|
| `gives_answer` | The expected answer appears in the reply as a standalone number. Answer formats are normalised before matching: "12", "12.0", "$12", "12%", "1,200", and integers written as words from zero to twenty. |
| `steps` | Number of lines that look like steps (start with a digit and a period, or "Step"). |
| `asks_attempt` | The reply contains an invitation such as "try", "what do you think", or one of its last two sentences is a question that is not a check-in ("does that make sense", "any questions", "let me know"; full list in the config). |
| `encourages` | The reply contains a word from a short encouragement list ("great", "good job", "nice", "you can"). |
| `words` | Number of words. Descriptive only, never a primary measure. |
| `mentions_history` | Exploratory. The reply contains "earlier", "previous problem", "last one", "last time" or "your answers". Uses the reply text only. |

Known weakness: `gives_answer` can fire when a step mentions the answer number. Hand-check 50 replies (`scripts/hand_check.py`) for `gives_answer` and `asks_attempt`, and report the agreement for each.

Known limit: a wrong final answer counts as no answer (`gives_answer` is 0).

### Analysis
- **Primary contrast.** `gives_answer` rate for `history_struggle` minus `history_success`, per model. This is the test of whether the model infers learner level without being told.
- **Control contrast.** `gives_answer` rate for `low_score` minus `high_score`. The mock model, not this contrast, validates the pipeline. A difference in the control contrast shows only that models respond to a stated level.
- Everything else (other measures, style contrasts) is **exploratory** and labelled as such.
- Compute the mean difference **per question first** (paired), then bootstrap by resampling whole **questions** 2,000 times. Report the mean and the 95% interval.
- Never resample single replies, because replies to one question are related.
- Report results per model. Do not average models together.
- Report all contrasts for every model. Do not rank models or highlight one.
- **Sensitivity (mock model).** Run the planted effect at differences of 0.05, 0.10 and 0.20 in the probability of a full answer. Report how often the primary contrast excludes zero at each size, and write the smallest detectable difference to the summary. Each size uses a base probability of 0.30 and 200 simulated runs. The smallest detectable difference is the smallest size at which at least 80% of runs exclude zero.
- A difference is evidence of differential service. It is not by itself evidence of unfairness, because adapting to a struggling student can be appropriate. The write-up must say this.

### Pre-freeze checks
Run these before the first full run.

a) **Neutral-only pilot.** Run the neutral condition only (100 questions, 1 seed) per model and report the rate of `gives_answer`, the mean `steps`, the truncation rate and the model's accuracy. Pilot mode refuses to compute any contrast between conditions.

b) **Ceiling or floor rule.** If `gives_answer` in the neutral pilot is above 90% or below 10% for a model, that model's primary measure becomes `steps` (mean count). Its control contrast switches to `steps` as well. This is decided from neutral replies only, before the first full run.

c) **Manipulation check for the history cues.** After the history turns, ask "Were my two earlier answers correct? Answer yes or no for each." Report each model's accuracy per condition. This does not change the design. It tells us whether the model noticed the cue.

d) **Truncation.** Set `max_new_tokens` so that fewer than 1% of neutral replies are cut off. Record a `truncated` flag on every reply and report the truncation rate per condition.

### Not allowed after the first real run
Changing the rubric, the cues, the primary contrast or the exclusion rules. Any change must be reported as a deviation.
