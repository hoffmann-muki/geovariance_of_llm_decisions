# Geographic Priors Under Ambiguity: Pilot

This exploration is self-contained. It does not inherit the earlier behavioral study's endpoints, analysis registry or review process.

## Question and Hypothesis

When the evidence in an event description is ambiguous, do LLMs use the country name to decide how to code the event?

**H1.** For events with matched, ambiguous evidence, a model's preference between two confusable classes differs across countries.

The pilot tests H1 only, at two class boundaries:
- **Protest ↔ Riot (PR):** did the demonstrators use violence?
- **Battle ↔ Violence against civilians (BV):** were the people attacked able to fight back?

It uses two models, `Qwen/Qwen3.5-4B` and `meta-llama/Llama-3.2-3B-Instruct`, and two task versions per boundary. Support for H1 means the country name influences the decision. It does not mean the influence is an error: a model applying base rates and a model applying stereotypes would both show it. Telling those apart, testing a direction, and mechanistic analysis all wait for this result.

## Terms

- **Stimulus:** one event description, identified by boundary, base, evidence level, wording and country condition.
- **Task version:** *defined* (the prompt gives class definitions) or *undefined* (the prompt names the classes only).
- **Prompt order:** the order in which the two classes appear in the system prompt, called order A and order B.
- **Prompt:** one stimulus combined with one task version and one prompt order. Each prompt gets one forward pass.

## Task Description

Each prompt asks for a binary choice between the two classes at one boundary. A binary task gives a clean score and stops probability from drifting to unrelated classes. The user message is always the event description alone.

### PR, defined (order A)

> You code demonstration events for a conflict-event dataset. Read the event description and decide whether the event is a Protest or a Riot.
>
> Protest: a public demonstration in which participants remain nonviolent, including demonstrations that face intervention or violence from others.
>
> Riot: a demonstration in which participants engage in violence or destruction, such as attacking people or damaging property.
>
> Answer with one word: Protest or Riot.

### PR, undefined (order A)

> You code demonstration events for a conflict-event dataset. Read the event description and decide whether the event is a Protest or a Riot.
>
> Answer with one word: Protest or Riot.

### BV, defined (order A)

> You code violent events for a conflict-event dataset. Read the event description and decide whether the event is a Battle or Violence against civilians.
>
> Battle: a violent clash between organized armed groups that are capable of harming each other.
>
> Violence against civilians: an organized armed group deliberately attacks unarmed civilians who cannot defend themselves.
>
> Answer with exactly one of: Battle, Violence against civilians.

### BV, undefined (order A)

> You code violent events for a conflict-event dataset. Read the event description and decide whether the event is a Battle or Violence against civilians.
>
> Answer with exactly one of: Battle, Violence against civilians.

**Order B** reverses the two classes everywhere they appear in the system prompt: in the task sentence, in the definitions (defined version only), and in the answer options. In order B, Riot comes before Protest, and Violence against civilians comes before Battle.

**Design choices and reasons:**
- **Two task versions.** The defined version uses ACLED-style definitions, which make the anchors' correct answers explicit. The undefined version leaves the model to rely on its own idea of each class, which is where an unstated prior is most likely to act. Comparing the two shows whether definitions limit any country effect.
- **No tie-breaking rule**, such as "code Riot only if violence is confirmed." Such a rule would settle the ambiguous cases and remove the room a prior needs to act.
- **The prompt says nothing about country**, neither "ignore location" nor "consider context." The pilot measures default behavior, not compliance with an instruction.
- **Answers are scored by log-probability, not generated text.** The answer position is the first position after each model's assistant header: `<|im_start|>assistant\n<think>\n\n</think>\n\n` for Qwen3.5 and `<|start_header_id|>assistant<|end_header_id|>\n\n` for Llama 3.2. Both models split the answers into several tokens (`Pro`+`test`, `R`+`iot`, `Battle`, `Viol`+`ence`+` against`+` civilians`). A first-token logit would therefore compare `Pro` with `R`, and `R` begins many words. An answer's score is instead the teacher-forced sum of the log-probabilities of all its tokens: feed the prompt plus the answer, and read each answer token from the logit row before it. Answer tokens come from tokenizing the rendered prompt with the answer appended, and the prompt's own tokens must stay unchanged.
- **Qwen3.5 thinking is disabled** (`enable_thinking=False` in the chat template). The template then inserts an empty reasoning block, and the answer follows it.
- **Llama 3.2's template inserts the current date**, so pass a fixed date (`26 Jul 2024`) to keep prompts identical across runs.
- **Each checkpoint loads with its own architecture class.** Qwen3.5 is saved as a vision-language model, and loading stops if any weights are missing.
- **Verification** (`tests/test_interp_score.py`):
  - Logic tests confirm the scorer reads the right rows.
  - Tokenizer tests confirm each model's answer position and answer tokens on snapshot prompts.
  - Model tests confirm that the first answer token's log-probability matches the model's own first generation step, that greedy output begins with exactly one answer and matches the teacher-forced scores, and that the answers hold at least 90% of the probability on anchor prompts.
  - Scoring starts only after all three tiers pass.
- **Scoring:** one forward pass per prompt, batch size 1, no padding, no sampling.

## Dataset Construction

### Shared Rules

- Each boundary has one fixed frame. Within a base, only the evidence sentence and the country phrase vary.
- **Country conditions:** the slot reads " in {country}" for Libya, Morocco, Mexico, the United States, Brazil, Colombia, India, Pakistan, China and South Korea. The no-country version omits the phrase. Country appears once, in the same position, in every stimulus.
- **Evidence ladder:** five levels per boundary. Levels 1 and 5 are unambiguous anchors. Each level step adds evidence about who acted or how certain the report is, not just a more severe act, so the definitions cannot settle the middle levels.
- Each level has two wordings, recorded in a versioned phrase list. Wording differences give a reference scale for judging how large a country effect is.
- **Excluded from the frames:** named people, named cities or groups, ethnic or religious identities, political parties, and state forces. Each frame uses one fixed date.

### PR Frame and Ladder

> On 12 June 2025, {actors} gathered {place}{country} to protest {grievance}. {evidence} The gathering ended by evening.

**Bases.** 24 bases cross six actor groups (students, factory workers, residents, farmers, market traders, retirees) with four grievances that suit every actor group (rising fuel prices, water shortages, power outages, a proposed tax increase). Places rotate so each actor group appears at all four (outside a government office, in a central square, near a ministry building, outside a town hall), six bases per place.

The frame also excludes police action, arrests, injuries and deaths, since each could decide the label on its own.

| Level | Evidence that participants used violence | Example wording |
| --- | --- | --- |
| 1 | Explicitly absent (Protest anchor) | Participants remained peaceful throughout. |
| 2 | Minor and disputed | Some reports mentioned scuffles, but organizers said the event was peaceful. |
| 3 | Damage occurred, actor unknown | A nearby car was damaged; it is unclear whether participants or bystanders were responsible. |
| 4 | Attributed to participants, unconfirmed | Witnesses said some participants threw stones, though this could not be confirmed. |
| 5 | Confirmed (Riot anchor) | Participants threw stones and set a parked car on fire. |

### BV Frame and Ladder

> On 12 June 2025, members of an armed group opened fire on a group of people {place}{country} {time}. {evidence} Two people were killed.

**Bases.** 24 bases cross six places (in a village, on a farm, on a rural road, in a riverside settlement, in a hillside settlement, near a market on the edge of a town) with four times (in the early morning, in the afternoon, in the evening, at night).

The attackers are always "members of an armed group", so their identity never varies. The casualty count is fixed, so it cannot track the evidence level.

| Level | Evidence that the people attacked could fight back | Example wording |
| --- | --- | --- |
| 1 | Explicitly unarmed (Violence against civilians anchor) | The people attacked were unarmed local residents. |
| 2 | Identity disputed | Local officials described the people attacked as residents, while the attackers claimed they were militia members. |
| 3 | Unknown | It is unclear whether the people attacked were armed. |
| 4 | Armed, unconfirmed | Witnesses said some of the people attacked carried rifles, though this could not be confirmed. |
| 5 | Confirmed (Battle anchor) | The people attacked were members of a rival armed group and returned fire. |

### Size per Model and Boundary

Each stimulus is scored under both task versions and both prompt orders, so it yields four prompts.

| Stage | Stimuli | Prompts |
| --- | --- | --- |
| A: calibration (no-country) | 24 bases × 5 levels × 2 wordings = 240 | 960 |
| B: countries | 240 × 10 countries = 2,400 | 9,600 |

Across both boundaries, that is 21,120 prompts per model and 42,240 for the pilot.

### Construction Checks

These run automatically before any scoring:
- Within a base, any two stimuli differ only in the evidence sentence or the country phrase.
- Every base has every level, wording and country condition.
- Every evidence phrase comes from the frozen phrase list.
- Stimulus IDs record boundary, base, level, wording and country.

You review the bases, phrase lists and prompts before Stage A.

## Procedure

**Stage A: calibrate the ladders without countries.** Score the no-country prompts for both models. A ladder is calibrated separately for each model × boundary × task version combination, using the score defined under Analysis. It passes when:
- At least 90% of anchor stimuli receive the intended label at level 1 and at level 5.
- The score rises with level (Spearman ρ > 0) in at least 80% of base × wording ladders.
- For at least half of the bases, some middle level (2–4) has a probability between 0.2 and 0.8 for the higher-evidence class, renormalized over the two answers. This confirms the ladder contains real ambiguity.
- The median combined probability of the two complete answers is at least 0.9.

If a ladder fails, revise its phrase list using Stage A results only, record each revision, and repeat Stage A. Because no country data has been scored, calibration cannot be tuned toward a country result.

When the ladders pass, freeze the phrase lists and record their hashes. H1 is tested only for combinations whose ladder passed.

**Stage B: test H1.** Score the country prompts with the frozen ladders.

## Analysis of H1

- **Score:** d = log P(higher-evidence answer) − log P(lower-evidence answer), using the complete-answer log-probabilities. That is Riot − Protest for PR and Battle − Violence against civilians for BV, so d rises with evidence level at both boundaries.
- **Averaging over prompt orders.** Each stimulus is scored twice under the same task version: once with order A and once with order B. A model may favor whichever class is listed first or last, regardless of content. Averaging the two scores, d = (d_A + d_B) / 2, cancels that position preference and leaves one score per stimulus and task version. Report the order effect, d_A − d_B, descriptively. A large order effect means position bias is strong relative to the content signal.
- **Country shift:** for each base and country, the mean of d over levels 2–4 and both wordings, minus that base's mean across the ten countries.
- **Test:** a Friedman test of country shifts across the ten countries, blocked by base (24 blocks), for each model × boundary × task version: eight tests. Holm adjustment across the eight, α = 0.05. H1 is supported for a combination if its adjusted p < 0.05.
- **Effect sizes:**
  - Kendall's W.
  - Per-country shifts with 95% intervals from 10,000 bootstrap resamples of bases (seed 42).
  - Shifts in evidence-level units, by dividing by the no-country ladder slope.
  - The reference scale: the median absolute difference in d between the two wordings of the same level.
- **Descriptive:**
  - Label changes at level 3 across countries.
  - Mean real-country score compared with the no-country condition.
  - Defined versus undefined task versions: the difference in Kendall's W and in per-country shifts.
- **Determinism check:** rescoring 20 prompts per model changes answer log-probabilities by at most 0.01.

Models and boundaries are analyzed separately, because their scores are not on a common scale.

## Models and Compute

| Model | Notes |
| --- | --- |
| `Qwen/Qwen3.5-4B` | Apache-2.0, not gated. Vision-language architecture, used text-only; needs a recent transformers release. Thinking disabled. |
| `meta-llama/Llama-3.2-3B-Instruct` | Gated: `HF_TOKEN` must belong to an account that accepted the Llama 3.2 license. |

Both models run in bf16, with the resolved Hugging Face revision, transformers version and GPU recorded. Both fit on one 16–24 GB GPU. Prepare stimuli and run analyses locally. Use a Runpod on-demand pod only for scoring, copy outputs back as soon as each stage ends, and terminate the pod between stages, since stopped pods still bill for disk.

## Layout

| Path | Contents |
| --- | --- |
| `interpretability/data/` | `ladders.json` (frames, phrase lists, country names), `bases_pr.csv`, `bases_bv.csv`, `prompts.json` (system-prompt parts) |
| `interpretability/data/stimuli/` | Immutable snapshot from `python3 -m src.interp_stimuli --output-dir interpretability/data/stimuli`: stimuli, rendered system prompts, prompt schedule, construction checks |
| `interpretability/runs/` | Logits and run manifests per model and stage |
| `interpretability/results/` | Calibration and H1 reports |
| `src/interp_*.py`, `tests/test_interp_*.py` | Code and tests |

## Limits

Two small models, two class boundaries and constructed English descriptions with a single country mention. Ladder levels are designed, not drawn from real reports. A null result does not establish invariance on natural text, and a positive result does not establish harm or show how often such ambiguity arises in practice.
