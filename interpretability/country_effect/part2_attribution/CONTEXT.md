# Part 2 Context Brief: Everything Needed to Plan the Internals Experiments

Written 21 November 2026 as a briefing for planning Part 2 (why the country matters, using the model's internals). It is meant to be read on its own, without repository access, by a model or person helping to design the experiments and to write an agent spec that runs them.

Sources: the Part 1 design ([DESIGN.md](../part1_behavioral/DESIGN.md)), materials ([MATERIALS.md](../../MATERIALS.md)), report ([preliminary_report.md](../part1_behavioral/report/preliminary_report.md), [main.tex](../main.tex)), proposed construction work ([CONSTRUCTION.md](../../CONSTRUCTION.md)) and the released dataset. Facts in § 6 marked *(derived)* were computed for this brief from the released `dataset/` files and are not in the report.

## 1. The Project in One Paragraph

We test whether an LLM used to code conflict events gives the same label to the same event regardless of the country it happened in. A description is held fixed and only the phrase `" in {country}"` changes, across 37 real countries, 20 invented placebo names and no country. Llama-3.2-3B-Instruct answers a two-option question (A/B) in both option orders, and we read the log-probabilities of the two letters. **Part 1 (behavioural, done)** found a strong, geographically ordered country effect for Battle vs Violence against civilians (BV) and essentially none beyond placebo level for Protest vs Riot (PR). **Part 2 (not started)** asks *why*: where in the model the country information enters the decision, and whether it acts as evidence about the deciding question.

## 2. Task

| Boundary | Lower class | Higher class | Deciding question |
| --- | --- | --- | --- |
| PR | Protest | Riot | Did the demonstrators themselves use violence? |
| BV | Violence against civilians (VAC) | Battle | Could the targeted group fight back? |

The sign convention is the same throughout: **positive = higher class** (Riot, Battle).

### 2.1 Prompt Structure (Exact)

Chat template of Llama-3.2-3B-Instruct, date header fixed at `Today Date: 26 Jul 2024`:

- **System turn:** the task prompt (BV order A shown; order B lists VAC first with its definition first and labels it A):

  ```
  Below is a short report of a violent incident. Assign it to one of the two event types used in conflict monitoring that are listed below.

  Battle: a fight between organized armed sides, each capable of hurting the other.

  Violence against civilians: an organized armed group purposely harms unarmed civilians who are unable to protect themselves.

  A. Battle
  B. Violence against civilians

  Give your answer as one letter: A or B.
  ```

  PR order A: "You code demonstration events for a conflict-event dataset. Read the event description and decide which of the two event types below it is." / Protest: "a public demonstration in which participants remain nonviolent, including demonstrations that face intervention or violence from others." / Riot: "a demonstration in which participants engage in violence or destruction, such as attacking people or damaging property." / `A. Protest`, `B. Riot` / "Answer with one letter: A or B."

- **User turn:** the description (one or two sentences).
- **Read-out position:** the first assistant token, after `<|start_header_id|>assistant<|end_header_id|>\n\n`. "A" and "B" are each exactly one token there (checked by code).

### 2.2 Descriptions

Each description = frame (scene sentence with context slots) + one evidence sentence. The country slot sits directly after the place:

> BV f1: Members of an armed group and members of another group were involved in a shooting **on a coastal road in Kenya at dawn**. *The other group were farmers who carried no weapons.*
>
> BV f2: A shooting {time} {place}{country} involved members of an armed group and members of another group. {evidence}
>
> PR f1: A group of airport staff rallied **near a train depot in Kenya** to oppose late scholarship payments. *Some participants built barricades across the road.*
>
> PR f2: A crowd of {actors} assembled {place}{country} to call attention to {grievance}. {evidence}

Consequences for internals work:
- **The country comes before the evidence sentence** in every frame. A causal model reads the country first and then the evidence, so any interaction must be computed at evidence tokens or later.
- **Names differ in token count.** 32 of the 37 real names are one token. "the United States" and "the United Kingdom" are three; "South Africa", "South Korea" and "the Philippines" two. **No placebo is a single token:** the 20 placebos have 2 (8 names), 3 (6), 4 (3) or 5 (3) tokens. Real and placebo names are therefore not token-matched, and token positions are not aligned across names. This matters for patching and attribution (aggregate over the name span, or align on the last name token), and a real-vs-placebo contrast inside the model partly contrasts one-token vs multi-token names.
- **The rendered text differs from the no-country text only by `" in {country}"`.** This is checked for every description and name.

## 3. Design of the Dataset

| Factor | PR | BV |
| --- | --- | --- |
| Evidence item | 29 items + 2 anchors | 22 items + 2 anchors |
| Frame | f1, f2 | f1, f2 |
| Wording | W1 (canonical), W2 (fixed subject swap + one verb substitution) | same |
| Context | 16: actors × grievance Latin square, place balanced | 16: place × time, 4 × 4 crossing |
| Country | none + 37 real + 20 placebo | same |

- **3,520 descriptions** (1,984 PR, 1,536 BV), × 58 name conditions = 204,160 texts, × 2 orders = 408,320 scored prompts.
- **Evidence scale:** every item is an act attested in ≥ 20 ACLED event notes. Its position is **P(higher | act)**, the share of those notes that human coders put in the higher class.
- **Anchors:** one clear sentence at each end of each boundary (competence check).
- **Sentence rules (V0):** 5–16 words, no hedges or modals, no word sharing a 5-letter stem with a class name or definition, no named entities, no outcome or casualty words.
- **Halves:** each context belongs to half 0 or 1, for split-half replication.
- **Held out:** context-value indices 12–19 have never been scored and are reserved for a held-out replication. Indices 0–7 were used for prompt selection; 8–11 are the study contexts.
- **Placebos:** 20 names from a fixed syllable inventory and seed (e.g. Gaor, Nilia, Sunderia, Krunstamiren), length-matched to real names, rejecting real places and single vocabulary tokens. They control for "a name was inserted", **not** for familiarity or frequency.

**Real countries (37):** Libya, Morocco, Egypt, Nigeria, Kenya, Ethiopia, South Africa, Sudan, Mexico, the United States, Canada, Brazil, Colombia, Venezuela, Argentina, India, Pakistan, Bangladesh, Afghanistan, Iran, Iraq, Syria, Yemen, China, South Korea, Japan, Indonesia, the Philippines, Myanmar, Russia, Ukraine, France, Germany, the United Kingdom, Sweden, Norway, Australia.

## 4. Measurement

- **Model:** `meta-llama/Llama-3.2-3B-Instruct`, revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`, bf16, one prompt per forward pass, no batching (batched bf16 moved log-probabilities by up to 0.6 nats in testing). bf16 scores move in steps of about 0.06 nats.
- **Score:** d = log P(letter of higher class) − log P(letter of lower class), per description i, name c, order o.
- **Order-averaged score:** s = (d_A + d_B) / 2. This cancels an additive letter preference and is the quantity country effects are measured on.
- **Order effect:** b = (d_A − d_B) / 2.
- **Labels, margin m = 0.5 nats:**
  - strict: same class in both orders, each beyond m; otherwise "order-dependent";
  - averaged: s > m higher, s < −m lower, otherwise "near boundary".
- **Country effect:** Δ_ic = s_ic − mean of s over the 20 placebos for the same description. A country's effect is the mean of Δ over descriptions.
- **Flip description:** among the 37 real countries, at least one gives the higher and one the lower label.

## 5. Part 1 Results

### 5.1 Validity

- The top token is A or B for every prompt. The 5th percentile of P(A) + P(B) is 0.9999 (PR) and 0.9994 (BV).
- Averaged-label anchor accuracy is 1.0 everywhere. Strict BV VAC-anchor accuracy with real countries is 0.98; the misses are mainly the UK (28) and the US (8).
- **Letter preference is large:** PR favours A by 1.06 nats, BV favours B by 1.19 nats (98% of pairs each). The strict label is order-dependent for 41% of PR and 82% of BV pairs. This letter preference is larger than any country effect.

### 5.2 Country Effects

| | PR | BV |
| --- | --- | --- |
| SD of country effects, real (placebo) | 0.112 (0.088) | 0.233 (0.065) |
| Within-description variance explained by name, real (placebo) | 27% (28%) | 65% (28%) |
| Real countries outside the placebo range | 10 / 37 | 25 / 37 |
| Largest shift up | Australia +0.10 | United States +0.59 |
| Largest shift down | Ukraine −0.33 | Bangladesh −0.44 |
| Split-half r, real (placebo) | 0.98 (0.98) | 0.9998 (0.996) |

- **BV toward Battle:** United States +0.59, United Kingdom +0.42, Canada +0.24, Norway +0.22, Australia +0.22.
- **BV toward VAC:** Bangladesh −0.44, Nigeria −0.39, Sudan −0.38, Indonesia −0.35, Pakistan −0.33, Afghanistan −0.30, Iran −0.29.
- **Token count:** BV effects correlate with token count (r = 0.55), driven by the two three-token names. Among the 32 single-token real names the SD is still 0.19, three times the placebo SD. "The Philippines" shifts toward VAC (−0.23), so the article alone does not produce the Battle shift.
- **PR:** real names on average move the score 0.10 nats toward Protest relative to placebos; no regional order.

### 5.3 Flips (Averaged Label)

| | PR | BV |
| --- | --- | --- |
| 37 real countries | 0.86% (17) | 8.14% (125 of 1,536) |
| 20 real, size-matched | 0.45% | 3.86% |
| 20 placebos | 0.05% | 0% |
| 37 real, US left out | 0.66% | 1.76% |

- **Strict flips: zero** in both boundaries, because the letter preference exceeds every country effect.
- **The US is on the Battle side of 124 of the 125 BV flips.** The VAC side is spread: Nigeria (84), Pakistan (68), Bangladesh (65), Colombia (60), Indonesia (59).
- **At m = 0** (sign changes), 46% of BV descriptions change sign across countries (placebo 20%); PR 11% (placebo 7%).
- **BV flip items:** security guard 67%, miners 36%, drivers 30%, armed residents 30%, traders 11%. Items that state the decisive fact (unarmed, returned fire, anchors) never flip.
- **BV flip factors:** W2 10.5% vs W1 5.7%; place "near a sugarcane plantation" 0.8% vs 10–11% for the other three; frames and times similar.
- **PR flip items:** deface with paint (5), fireworks (5), throw water (3), barricades (2), lock-in (1), block rail (1).

## 6. What the Data Looks Like Item by Item *(derived)*

Mean no-country score s_none, mean within-description SD of s across the 37 real countries, and averaged flip rate, over the 64 descriptions of each item.

### 6.1 BV (positive = Battle)

| P(Battle \| act) | Item | Sentence (W1) | s_none | SD real | Flip |
| --- | --- | --- | --- | --- | --- |
| 0.09 | fishermen | The other group were fishermen returning from the river. | −0.35 | 0.29 | 0.03 |
| 0.10 | miners | The other group were miners leaving a nearby mine. | 0.31 | 0.28 | 0.36 |
| 0.11 | anchor_lo | The other group were farmers who carried no weapons. | −2.96 | 0.56 | 0 |
| 0.11 | farmers | The other group were farmers working in nearby fields. | −0.67 | 0.34 | 0 |
| 0.14 | students | The other group were students walking home. | −1.52 | 0.41 | 0 |
| 0.15 | former_members | The other group were former members of a rival militia. | 0.72 | 0.24 | 0 |
| 0.15 | villagers | The other group were villagers from a nearby settlement. | −0.44 | 0.25 | 0 |
| 0.17 | traders | The other group were traders from a nearby market. | −0.18 | 0.24 | 0.11 |
| 0.17 | drivers | The other group were truck drivers waiting by the road. | 0.25 | 0.27 | 0.30 |
| 0.24 | sticks_machetes | The other group carried sticks and machetes. | 0.08 | 0.20 | 0.03 |
| 0.26 | unarmed | The other group carried no weapons. | −2.13 | 0.41 | 0 |
| 0.31 | security_guard | The other group included a night watchman. | 0.04 | 0.26 | 0.67 |
| 0.34 | fled | The other group fled when the shooting started. | 1.05 | 0.24 | 0 |
| 0.40 | self_defense_group | The other group was a village night patrol. | 0.33 | 0.22 | 0.03 |
| 0.51 | bodyguards | The other group were bodyguards escorting a local businessman. | 0.21 | 0.24 | 0.05 |
| 0.56 | stones | The other group threw stones at the armed group. | 0.80 | 0.16 | 0 |
| 0.62 | militia | The other group was a local militia. | 0.83 | 0.35 | 0.05 |
| 0.76 | rifle | One member of the other group carried a rifle. | 0.97 | 0.21 | 0 |
| 0.79 | shotgun | One member of the other group carried a shotgun. | 0.58 | 0.21 | 0 |
| 0.79 | returned_fire | One member of the other group fired back at the armed group. | 1.25 | 0.20 | 0 |
| 0.81 | armed_residents | The other group were local residents who carried guns. | 0.06 | 0.27 | 0.30 |
| 0.86 | pistol | One member of the other group carried a pistol. | 0.59 | 0.21 | 0 |
| 0.92 | grenade | One member of the other group threw a grenade at the armed group. | 0.47 | 0.24 | 0.03 |
| 0.99 | anchor_hi | The two groups exchanged fire for an hour. | 3.13 | 0.26 | 0 |

### 6.2 PR (positive = Riot)

| P(Riot \| act) | Item | s_none | SD real | Flip |
| --- | --- | --- | --- | --- |
| 0.01 | burn_effigy | 2.54 | 0.24 | 0 |
| 0.02 | anchor_lo (chanted slogans, held placards) | −6.37 | 0.14 | 0 |
| 0.05 | occupy | −1.34 | 0.21 | 0 |
| 0.05 | burn_documents | 1.76 | 0.24 | 0 |
| 0.06 | lock_in | −1.13 | 0.22 | 0.02 |
| 0.09 | block_entrance | −2.49 | 0.20 | 0 |
| 0.10 | burn_flag | 2.70 | 0.21 | 0 |
| 0.10 | block_rail | −1.02 | 0.22 | 0.02 |
| 0.12 | block_road | −2.41 | 0.20 | 0 |
| 0.17 | scuffle | −0.38 | 0.21 | 0 |
| 0.30 | throw_eggs | 1.92 | 0.23 | 0 |
| 0.30 | throw_bottles | 1.83 | 0.20 | 0 |
| 0.30 | throw_paint_ink | 1.30 | 0.20 | 0 |
| 0.35 | assault | 2.80 | 0.22 | 0 |
| 0.35 | deface_paint | −1.60 | 0.21 | 0.08 |
| 0.37 | storm | 1.09 | 0.17 | 0 |
| 0.38 | push_barrier | −0.65 | 0.16 | 0 |
| 0.52 | tear_down | 0.33 | 0.17 | 0 |
| 0.53 | barricade | −0.63 | 0.22 | 0.03 |
| 0.60 | set_off_fireworks | 0.52 | 0.27 | 0.08 |
| 0.61 | throw_water | −0.12 | 0.22 | 0.05 |
| 0.68 | vandalise | 1.97 | 0.21 | 0 |
| 0.80 | break_gate | 0.05 | 0.15 | 0 |
| 0.87 | throw_objects | 2.46 | 0.18 | 0 |
| 0.91 | break_windows | 2.45 | 0.18 | 0 |
| 0.93 | throw_stones | 2.37 | 0.20 | 0 |
| 0.96 | anchor_hi | 5.81 | 0.16 | 0 |
| 0.96 | burn_vehicle | 3.81 | 0.19 | 0 |
| 0.99 | burn_tires | 2.72 | 0.28 | 0 |
| 1.00 | throw_fireworks | 3.00 | 0.22 | 0 |
| 1.00 | throw_petrol_bomb | 5.68 | 0.30 | 0 |

### 6.3 Observations Relevant to Planning *(derived)*

1. **The model's evidence reading only partly follows human coding.** The correlation between P(higher | act) and s_none over item descriptions is 0.50 (PR) and 0.49 (BV). In PR, symbolic burning (effigy, flag, documents) is read as Riot. In BV, most scores are compressed into about ±1.3 nats, and items such as "grenade" (0.47) score lower than "fled" (1.05). "Near the boundary" for the model is not the same set as "ambiguous for human coders".
2. **The BV country effect is a near-constant shift, not confined to ambiguous items.** The US shift relative to placebos is positive on every BV item, from +0.31 (stones) to **+0.97 on the VAC anchor** (unarmed farmers). The UK is +1.22 on the VAC anchor. Bangladesh is negative on every item, −0.22 to −0.65. Flips happen only where s_none is near zero simply because that is where a shift of this size crosses the margin. **The largest country sensitivity is on the clearest VAC evidence**, the opposite of a "country fills in missing evidence" story. This bears directly on the prediction in § 8.1.
3. **The VAC anchor has the largest within-description country SD in BV** (0.56, against 0.16–0.41 for other items).
4. **No country vs placebo:** s_none is on average 0.12–0.13 nats above the placebo mean in both boundaries, so inserting any name shifts the score slightly toward the lower class.
5. **Context modulates the US effect:** +0.40 near a sugarcane plantation against +0.65 to +0.67 for the other places, and +0.54 (W1) vs +0.65 (W2), +0.54 (f1) vs +0.64 (f2).
6. **PR effects are item-dependent but not regional.** Ukraine is negative on nearly every PR item (−0.16 to −0.75); Australia's sign varies by item.
7. **Placebos are never single tokens** (2–5 tokens), while 32 of 37 real names are (§ 2.2). The single-token robustness check in § 5.2 compares single-token real names against multi-token placebos, so a token-matched null does not yet exist.

## 7. Released Data: Files and How to Join Them

All under `interpretability/country_effect/part1_behavioral/`.

| File | Rows | Key | Main columns |
| --- | --- | --- | --- |
| `dataset/descriptions.csv` | 3,520 | `description_id` | boundary, item, role (item / anchor_lower / anchor_higher), acled_acts, acled_n, acled_p_higher, frame, wording, context_id, actors, grievance, place, time, half, evidence_sentence, `template` (text with `{COUNTRY}`), text_none |
| `dataset/labels.csv.gz` | 204,160 | `description_id` × `name` | kind (none / real / fictional), tokens, d_A, d_B, mass_A, mass_B, top1_A, top1_B, s, b, strict, avg, strict_m0, avg_m0 (labels coded −1 / 0 / 1) |
| `dataset/description_summary.csv` | 3,520 | `description_id` | label counts by name kind, strict_flip, avg_flip, placebo flips, names on each side, s_none, s_real_mean, s_real_sd, most_higher_country, most_lower_country |
| `report/results.json` | — | — | every number in the report, by boundary: validity, none_baseline, strict, avg, *_m0, country, posthoc |
| `inputs/prompts.json` | — | — | the four exact system prompts |
| `inputs/names.json`, `inputs/placebos.json` | — | — | rendered name forms and token counts |

- **Description ID format:** `BOUNDARY|item|frame|wording|context_id`, e.g. `BV|pistol|f1|W1|BV_c00`.
- **Rendering a text:** replace `{COUNTRY}` in `template` with `" in " + rendered name` (e.g. `" in the United States"`), or with `""` for no country.
- **Placebo names have kind `fictional`** in `labels.csv.gz`.

**Not released and not stored:** activations, attention patterns, hidden states, or anything beyond the two letter log-probabilities and top-5 tokens. Every internals experiment must re-run the model. The Part 1 score cache (JSONL, keyed by sha256 of system prompt + text + letters) is on the lab cluster at `/fs/clip-scratch/pramu/country_effect/work/` (override with `CE_WORK`), not in the repo.

## 8. Plans and Ideas Already on Record

### 8.1 Part 2 as Planned in the Midterm Report

- **Method:** integrated gradients on the order-averaged logit difference, for near-boundary BV descriptions, with each description attributed once with a real country and once with a placebo, in flip and non-flip descriptions.
- **Pre-stated prediction:** if the model reads the country as evidence about whether the other group could fight back, attribution to the country should be **largest where the evidence sentence is least informative** (security guard, miners, drivers) and **smallest where it is decisive** (unarmed, returned fire).
- **Requirement:** predictions must be fixed in advance, so the analysis tests the evidence taxonomy rather than restating it.
- **Tension with the data:** § 6.3 item 2 shows the behavioural shift is, if anything, largest on the decisive VAC evidence. The prediction may already be partly contradicted at the behavioural level; the plan should say what each outcome would mean.

### 8.2 Other Next Steps in the Report

1. **Familiarity control:** add real but rarely mentioned places (small territories, sub-national regions) and familiar non-country names, to separate frequency from geography.
2. **Generality:** a second model and a second task paraphrase per boundary. The report names "Qwen3.5-4B"; the scoring code is configured for `Qwen/Qwen3-4B` (float32, thinking disabled). Which one is intended needs confirming.
3. **Held-out replication:** re-estimate country effects on the reserved contexts (indices 12–19) before any further design decision.
4. **Human reference:** human labels for a sample of descriptions.

### 8.3 Proposed Dataset Construction v4.1 ([CONSTRUCTION.md](../../CONSTRUCTION.md), not run)

- Lexicon precision audit, frames induced from real ACLED notes, human labels splitting items into **clear** and **ambiguous**.
- A second axis, **intensity**, carried by a party whose actions do not decide the class by definition (PR: state response such as tear gas; BV: the armed group's own weapons). This asks whether the model follows the definition or the human base rate.
- **Primary contrast:** flip rate on clear vs ambiguous items. This is the "override vs settling" distinction: a country that flips clear evidence *overrides* it; one that flips ambiguous evidence *settles* it.

## 9. Constraints Any Plan Must Respect

From the repository guidance and the Part 1 design:

- **Pre-specify.** Write analysis rules and predictions before looking at results; log any later change with its reason. Do not select items, countries or prompts on model scores or country effects.
- **Keep sample separation.** Pilot contexts (0–7), study contexts (8–11) and held-out contexts (12–19) stay separate. Do not use the held-out contexts for exploration.
- **Always compare to placebos.** A real-country finding counts only relative to what invented names do.
- **Use the order-averaged score** (or both orders) so that the letter preference does not masquerade as a country effect.
- **Scoring conventions:** one prompt per forward pass, no batching, the same model revision and chat template. Verify that the template and the answer tokens are as expected.
- **Preserve Part 1 artifacts.** Treat `part1_behavioral/` outputs as immutable. Write Part 2 outputs to new directories with manifests and input hashes.
- **Code layout:** shared code in `interpretability/geoflip/`, study code in `interpretability/country_effect/part2_attribution/`. Python 3.10+, standard library where practical, dependencies only for a concrete need, Black formatting, explicit failures, tests for invariants.
- **AI review is not human review.** Never describe AI judgements as human verification.

## 10. Practical Facts for an Agent Spec

- **Model size:** 3B parameters, 28 layers, hidden size 3072. It fits comfortably on one 16 GB GPU in bf16; gradient-based attribution needs more memory than forward passes.
- **Throughput reference:** Part 1 scored 408,320 single forward passes in about 4.3 h on one RTX A4000. Internals experiments with hooks, patching or integrated gradients (e.g. 50–300 steps per attribution) cost much more per prompt, so they need a subset chosen by a fixed rule.
- **Software used:** torch 2.14.1, transformers 5.22.1, huggingface_hub 2.4.0; the Llama model is gated (`HF_TOKEN`).
- **Reusable code:** `geoflip/scoring.py` renders and verifies prompts (`render_prompt`), checks single-token answers (`answer_token_ids`) and does one forward pass (`score_first_token`). `geoflip/render.py` renders descriptions with a country phrase. Hooks or attribution can be built on the same rendering so that internals runs reproduce the Part 1 scores exactly; reproducing d for a sample of cached prompts is a natural first check.
- **Natural units of analysis:** the 1,536 BV descriptions (and 1,984 PR as a near-null control), × names; matched pairs (real country vs placebo, or real vs no country) of the same description differ only in the name span.

## 11. Open Questions for the Planning Session

1. **What is the Part 2 claim?** For example: "the country is represented as evidence about the other side's capacity to fight back", or "the country adds a description-independent bias toward one class", or "the effect is familiarity, not geography". Each implies different experiments; § 6.3 item 2 favours the bias reading behaviourally.
2. **Which methods answer it?** Candidates: integrated gradients (planned); activation patching of the name span between real country and placebo, by layer and position; logit lens or tuned lens on the decision position; linear probes for a "can fight back" or "conflict-prone country" direction and a test of whether the country moves along it; steering along a derived direction to see whether it reproduces the country effect; attention from the decision position to name tokens.
3. **Which subset, chosen how?** All BV descriptions, flip descriptions, or items stratified by s_none, fixed before running? Which names: all 37, or the extremes (US, UK vs Bangladesh, Nigeria) plus placebos and none?
4. **How to handle multi-token names** (the US and UK, the two largest effects, are three tokens with an article)?
5. **What does PR contribute?** A within-model null control for whatever mechanism is found in BV.
6. **Does the familiarity control or held-out replication need to come first,** so that the internals work explains an effect known to be geographic and robust?
7. **Which outcomes would falsify each hypothesis,** stated before any internals run?
8. **Is a second model in scope for Part 2,** and if so Qwen3-4B or Qwen3.5-4B?
