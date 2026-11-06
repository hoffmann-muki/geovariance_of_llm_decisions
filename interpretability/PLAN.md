# Geographic Flips in LLM Event Coding: Dataset Construction (v4)

Version 4, 6 November 2026. **Status: design frozen up to the pilot; the pilot is partly scored; no country has been scored.** This document defines the task, the materials, the scoring and one pre-specified filtering procedure, so that the dataset can be rebuilt from `inputs/v4/` with `geoflip/`. Version 3 (an exploratory, stage-wise curation) and the review that replaced it are in `history/v3/` and `history/ITERATIONS.md`; § 10 summarises why.

## 1. Goal and Claims

The project builds event descriptions whose class, as coded by `meta-llama/Llama-3.2-3B-Instruct`, changes when only the name of the country where the event happened changes. A **flip** is a change in the model's **actual answer** on a concrete prompt, not in a score averaged over prompts.

Unlike v3, every description of a fixed, fully crossed population is scored and one rule decides inclusion, so the study supports two kinds of claim:
- **Existence:** flip descriptions exist beyond what invented placebo names produce.
- **Prevalence within the constructed population:** how often flips occur, by act, frame, task version and country. This is not a prevalence estimate for real event coding.

A flip shows that the decision depends on the country name, not that the dependence is wrong.

## 2. Task

Two pairs of confusable classes (**boundaries**), adapted from ACLED event coding:

| Boundary | Lower class | Higher class | Distinguishing question |
| --- | --- | --- | --- |
| **PR** | Protest | Riot | Did the demonstrators themselves use violence? |
| **BV** | Violence against civilians | Battle | Could the targeted group fight back? |

The higher class needs a specific fact (violence by demonstrators; an armed opposing side). Scores are signed so that positive favours the higher class.

## 3. Materials

### 3.1 Frames

No frame contains a date (v3's "On 12 June 2025" fell after the model's stated "today" and is a national day in Russia, the Philippines and Nigeria). The BV frames name the first group as an armed group, because both class definitions require one, and describe the other group neutrally without saying who acted against whom.

| Id | Frame |
| --- | --- |
| PR f1 | `A group of {actors} rallied {place}{country} to oppose {grievance}. {evidence}` |
| PR f2 | `A crowd of {actors} assembled {place}{country} to call attention to {grievance}. {evidence}` |
| BV f1 | `Members of an armed group and members of another group were involved in a shooting {place}{country} {time}. {evidence}` |
| BV f2 | `A shooting {time} {place}{country} involved members of an armed group and members of another group. {evidence}` |

### 3.2 Items: Where the Acts Come From

Items are not chosen by looking at model scores. They come from how human coders code real events:

1. **Source.** 137,947 ACLED events in `conflict_data/` (5 regional exports, 12 Oct 2024 – 10 Oct 2025) with free-text notes and sub-event types.
2. **Comparison groups.** PR: Violent demonstration (higher) vs Peaceful protest and Protest with intervention (lower); Mob violence is excluded because it is not a demonstration. BV: Armed clash (higher) vs Attack (lower).
3. **Act lexicon.** A fixed, versioned set of regular expressions (`geoflip/acled.py`, lexicon v1), each tagged with dimension levels derived from the class definitions: PR act type (projectile, fire, obstruction, property, entry, person) × target; BV the other side's means × use × organisation.
4. **Measurement.** For each act, the number of notes on each side that mention it, and **P(higher | act)**, the share coded as the higher class. Written to `inputs/v4/acled_acts.csv`; notes are never copied.
5. **Inclusion rule (fixed before any scoring).** Every concrete act (not a generic pattern such as "damage") mentioned in **at least 20 notes** becomes one item. An act that forms an anchor by itself is the anchor, not an item.

This gives **29 PR items and 22 BV items**, spanning the full range of human coding, for example:

| Act | Notes (higher / lower) | P(higher \| act) | Item W1 |
| --- | --- | --- | --- |
| burn_effigy | 13 / 952 | 0.01 | Some participants burned an effigy in the street. |
| throw_eggs | 8 / 19 | 0.30 | Some participants threw eggs at a nearby building. |
| barricade | 173 / 153 | 0.53 | Some participants built barricades across the road. |
| burn_tires | 523 / 5 | 0.99 | Some participants burned tires in the road. |
| fishermen | 2 / 21 | 0.09 | The other group were fishermen returning from the river. |
| stones | 48 / 37 | 0.56 | The other group threw stones at the armed group. |
| pistol | 581 / 94 | 0.86 | One member of the other group carried a pistol. |
| grenade | 293 / 27 | 0.92 | One member of the other group threw a grenade at the armed group. |

Acts excluded by the 20-note rule include shoes (8), tomatoes (5), vigilantes (6), hunters (5) and warning shots (6). The full list with counts is in `inputs/v4/items.json`.

**Anchors** are attested acts at the extremes. PR lower: "Participants chanted slogans and held placards the whole time." (P = 0.02). PR higher: "Participants threw stones and burned parked cars." BV lower: "The other group were farmers who carried no weapons." BV higher: "The two groups exchanged fire for an hour." (P = 0.99).

### 3.3 Rules for Evidence Sentences (Checked by V0)

`geoflip/study.py: static_findings` enforces every rule; the tests inject one violation of each.
1. One sentence of 5–16 words.
2. Stated as certain: none of a fixed list of hedges and modals (unclear, reportedly, allegedly, possibly, may, might, could, …).
3. **No word sharing a 5-letter stem with any content word of a class name or definition paraphrase.** The list is built automatically from `task_prompts.json`, after removing fixed role terms (group, members, participants, demonstrators, armed group). This caught "against" (class name), "gathered" (definition "gathering") and "committee" (definition "commit"); the frames and items were reworded.
4. No capitalised word except the first (a check for named entities); no state-force, ethnic, religious or party words.
5. No outcome or casualty words (killed, injured, arrested, …).
6. The country phrase appears once, directly after `{place}`, and is the only difference between conditions.

Two items use a stand-in where the ACLED term would break rule 3: a self-defence group is written as "a village night patrol", and "forced open" as "pried open".

### 3.4 Wordings

W1 is the canonical sentence. W2 applies fixed transformations (`items.W2_RULES`): the subject swap ("Some participants" → "A few demonstrators"; "The other group" → "Members of the other group"; "One member" → "A member") and at most one verb substitution from a fixed table (threw → hurled, burned → set fire to, carried → had, …). Wording is a factor of the population, analysed as such.

### 3.5 Bases

Fill-in values (PR: actors, grievance, place; BV: place, time) come from fixed lists in `geoflip/materials.py`, none of which appears in any v1–v3 base (checked against `history/`). The pilot (8 bases) and main (12 bases) splits use disjoint value lists, and each value appears once per split through a fixed permutation. BV times are generic and are shared across splits. Example: `pr4_main_01` = airport staff, late scholarship payments, near a train depot.

### 3.6 Names

- **37 real countries**, as in v1–v3: the 10 countries of the main behavioral study plus 27 for regional coverage. This is a convenience frame; token count and the "the" form are recorded as covariates.
- **20 placebo names** (Step 6, not yet generated): invented by `geoflip/placebo.py` from a fixed syllable inventory and seed, rejected if they are a real place (any ACLED country, admin area or location, or a world country) or an existing vocabulary token, and matched to the real names' character lengths by quantile. No token-count matching is needed.
- **No country.**

## 4. Task Prompts

Answers are **single tokens**: the options are listed as `A. <class>` and `B. <class>`, and the prompt asks for one letter. **Order** decides which class is A, so the letter is counterbalanced with order. 4 paraphrases (intro, definitions, answer instruction) × 2 orders × 2 versions (*defined*: with class definitions; *undefined*: without) per boundary. The intros do not name the classes. The definitions are the v3 paraphrases unchanged. Two layouts are candidates (§ 6.1): `options_last` (instruction, then options) and `options_first`.

Example, PR, defined, p0, order A, `options_last`:

```
You code demonstration events for a conflict-event dataset. Read the event description and decide which of the two event types below it is.

Protest: a public demonstration in which participants remain nonviolent, including demonstrations that face intervention or violence from others.

Riot: a demonstration in which participants engage in violence or destruction, such as attacking people or damaging property.

Answer with one letter: A or B.

A. Protest
B. Riot
```

The system prompt is followed by the description as the user turn. The Llama chat template always writes `Cutting Knowledge Date: December 2023 / Today Date: …`; it is fixed at `26 Jul 2024`, so the only date in any prompt is this constant header (tested).

## 5. Scoring

- Model revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`, bf16 (subject to the fp32 audit, § 6.1), one prompt per forward pass, no batching.
- `scoring.score_first_token` reads log P("A") and log P("B") at the first answer position, after checking that each letter is exactly one token there; it also records P(A) + P(B) (the **mass**) and the top-5 tokens.
- For one description and one concrete prompt k: **d_k** = log P(letter of the higher class) − log P(letter of the lower class). The model's answer is the higher class if d_k > 0.
- Scores are cached in `<work>/cache/llama3.2-3b_bfloat16_0cb88a4f76_first_token.jsonl`, keyed by the system prompt and description.

## 6. Procedure

### 6.1 Pilot (pilot bases only, then discarded)

All items and anchors × 2 frames × 8 pilot bases × 2 wordings, no country, all 32 system prompts, both layouts (56,320 prompts). Rules fixed in `study.py` before scoring:
- **Gates per concrete prompt** (boundary × version × paraphrase × order): anchor accuracy ≥ 0.9 at both ends in each frame; 5th percentile of mass ≥ 0.9.
- **First token:** at least 99% of pilot prompts have A or B as the most likely token.
- **Layout choice:** among layouts passing every gate, the one with the smallest mean |d(order A) − d(order B)|.
- **fp32 audit:** 200 random pilot prompts re-scored in fp32; bf16 is used for the main run if max |Δd| ≤ 0.05.
- **Throughput** fixes the number of main bases.
- If a frame or layout fails, it is revised, logged in `history/ITERATIONS.md`, and the pilot is rerun. The main population is untouched until then.

### 6.2 Pre-Registration (Step 6)

`inputs/v4/prereg.json` will hold the population definition, every threshold, the filtering rule and the analysis list, with sha256 hashes of all inputs, written before the main run; the pipeline will refuse to run if a hash differs.

### 6.3 Main Population, Scored Once

Items and anchors × 2 frames × 2 wordings × 12 main bases × 58 names (37 real, 20 placebo, none) × 2 versions × 8 prompts of the chosen layout: about 1.38M prompts for PR and 1.07M for BV. At the measured pilot rate (about 31 prompts/s on one RTX A4000) that is about 12 h and 10 h, in parallel on two GPUs.

### 6.4 One Filtering Rule

Applied within each version:
- An answer on prompt k is **confident** if |d_k| ≥ m, with **m = 0.5**.
- A pair of real countries **flips** on prompt k if both answers are confident and opposite.
- A description is a **flip description** if some real-country pair flips in the same direction on **at least K = 4 of the 8** prompts. The full curve over K = 1…8 is reported.
- **Null:** the same rule on placebo pairs and on real–placebo pairs. The excess of real-pair flips over placebo-pair flips is the main result.
- **Replication:** split-half correlation of country effects across base halves. No split is used to select items.

### 6.5 Release

`dataset/defined/` and `dataset/undefined/`, each with descriptions, cells (per-prompt d, mass, top-5 tokens), flip descriptions with their qualifying pairs and prompts, the population summary, the null comparison and a manifest. A second adversarial review checks the report against the data before release.

## 7. Status

| Step | State |
| --- | --- |
| 1. Archive v3 | Done: `history/v3/` |
| 2. Items from definitions and ACLED | Done: `inputs/v4/acled_acts.csv`, `inputs/v4/items.json` |
| 3. Frames, V0 rules, wordings, bases | Done: `inputs/v4/descriptions.json`, `inputs/v4/bases/`; V0 has no findings |
| 4. A/B task prompts, first-token scorer | Done: `inputs/v4/task_prompts.json`, `scoring.score_first_token`, `study.Scorer`; 11 tests pass |
| 5. Pilot | **Partly scored** when GPU access ended (see HANDOFF.md); no gate result yet |
| 6. Placebos, pre-registration, main run | Not started (`geoflip/placebo.py` written, not run) |
| 7. Filter and null | Not started |
| 8. Release, report, second review | Not started |

## 8. Reproduction

```bash
cd interpretability
export PY=python HF_HOME=/path/to/hf_cache GEOFLIP_WORK=/path/to/work   # HF_TOKEN in env or .env
$PY -m geoflip.acled --out inputs/v4/acled_acts.csv
$PY -m geoflip.items --acts inputs/v4/acled_acts.csv --out inputs/v4/items.json
$PY -m geoflip.materials --items inputs/v4/items.json --out inputs/v4
$PY -m geoflip.study check
CUDA_VISIBLE_DEVICES=0 $PY -m geoflip.study pilot
$PY -m geoflip.study fp32_audit --layout <chosen layout>
$PY -m unittest discover -s tests -t .
```

## 9. Known Limitations

- **The ACLED lexicon measures mentions, not who acted.** A note mentioning "threw stones" may describe police or bystanders; agent restriction is applied only to person-contact acts. Pattern precision has not been audited.
- **The ACLED notes cover five regions** (East Asia, North Africa, North America, South America, South Asia), so P(higher | act) reflects coding there.
- One fact per sentence: combinations of acts are not crossed.
- The 37-country set is a convenience frame.
- bf16 scores move in steps of about 0.06 nats unless the fp32 audit forces fp32.

## 10. Why v3 Was Replaced

An adversarial review of v3 (confirmed against the score cache) found:
- **Order bias.** Class order shifted single prompts by about 10 nats in BV, so under 7 of 8 BV prompts almost no released row changed its actual answer across countries. The v3 flips existed only in an order- and paraphrase-averaged score.
- **The robust-label rule.** Its 3-of-4 paraphrase vote failed as a robustness check, because one paraphrase was systematically offset (1,250 of 1,252 BV Battle labels had the pattern `++-+`).
- **Surface form.** Two invented names differed about as much as real countries in PR.
- **Selection on outcome.** Items, countries and caps were chosen after looking at results across five iterations.
- **Dates and versions.** The frame date was a confound, and versions were mixed across boundaries.
