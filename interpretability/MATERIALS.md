# Materials: How the Inputs Were Built

This document describes `inputs/v4/`, the materials from which `country_effect/part1_behavioral/build.py` assembles the study's descriptions: the task, the evidence items and where they come from, the sentence rules, the frames and context values, the names and the task prompts. Section numbers are cited from the code and from `country_effect/part1_behavioral/DESIGN.md`.

## 1. What the Materials Support

The materials are built so that a change in the model's label can be attributed to the country name alone. Every description has the same structure (frame, context, evidence sentence, country slot), the evidence comes from human coding practice, and the country is the only element that varies between the conditions of a description. A change in the label shows that the decision depends on the country name, not that the dependence is wrong.

## 2. Task

Two pairs of confusable classes (**boundaries**), adapted from ACLED event coding:

| Boundary | Lower class | Higher class | Distinguishing question |
| --- | --- | --- | --- |
| **PR** | Protest | Riot | Did the demonstrators themselves use violence? |
| **BV** | Violence against civilians | Battle | Could the targeted group fight back? |

The higher class needs a specific fact (violence by demonstrators; an armed opposing side). Scores are signed so that positive favours the higher class.

## 3. Materials

### 3.1 Frames

Frames contain no date, so the only date in a prompt is the chat template's fixed header. The BV frames name the first group as an armed group, because both class definitions require one, and describe the other group neutrally, without saying who acted against whom.

| Id | Frame |
| --- | --- |
| PR f1 | `A group of {actors} rallied {place}{country} to oppose {grievance}. {evidence}` |
| PR f2 | `A crowd of {actors} assembled {place}{country} to call attention to {grievance}. {evidence}` |
| BV f1 | `Members of an armed group and members of another group were involved in a shooting {place}{country} {time}. {evidence}` |
| BV f2 | `A shooting {time} {place}{country} involved members of an armed group and members of another group. {evidence}` |

### 3.2 Items: Where the Acts Come From

Items are derived from how human coders code real events, never from model scores.

1. **Source.** 137,947 ACLED events (Raleigh et al., 2010) in `../conflict_data/`: five regional exports (East Asia, North Africa, North America, South America, South Asia), 12 October 2024 to 10 October 2025, with free-text notes and sub-event types. The raw exports are not redistributed (ACLED terms of use).
2. **Comparison groups.** PR: Violent demonstration (higher) against Peaceful protest and Protest with intervention (lower). Mob violence is excluded because it is not a demonstration. BV: Armed clash (higher) against Attack (lower).
3. **Act lexicon.** A fixed, versioned set of regular expressions (`geoflip/acled.py`, lexicon v1). Each pattern is tagged with dimension levels derived from the class definitions: for PR, act type (projectile, fire, obstruction, property, entry, person) × target; for BV, the other side's means × use × organisation.
4. **Measurement.** For each act, the number of notes on each side that mention it, and **P(higher | act)**, the share coded as the higher class. These are written to `inputs/v4/acled_acts.csv`. Only counts are kept; notes are never copied.
5. **Inclusion rule.** Every concrete act (not a generic pattern such as "damage") mentioned in **at least 20 notes** becomes one item. An act that forms an anchor by itself is the anchor, not an item.

This gives **29 PR items and 22 BV items**, spanning the full range of human coding, for example:

| Act | Notes (higher / lower) | P(higher \| act) | Item sentence (W1) |
| --- | --- | --- | --- |
| burn_effigy | 13 / 952 | 0.01 | Some participants burned an effigy in the street. |
| throw_eggs | 8 / 19 | 0.30 | Some participants threw eggs at a nearby building. |
| barricade | 173 / 153 | 0.53 | Some participants built barricades across the road. |
| burn_tires | 523 / 5 | 0.99 | Some participants burned tires in the road. |
| fishermen | 2 / 21 | 0.09 | The other group were fishermen returning from the river. |
| stones | 48 / 37 | 0.56 | The other group threw stones at the armed group. |
| pistol | 581 / 94 | 0.86 | One member of the other group carried a pistol. |
| grenade | 293 / 27 | 0.92 | One member of the other group threw a grenade at the armed group. |

Acts below the 20-note threshold include shoes (8), tomatoes (5), vigilantes (6), hunters (5) and warning shots (6). The full list, with counts, is in `inputs/v4/items.json`.

**Anchors** are attested acts at the extremes and serve as competence checks:
- PR lower: "Participants chanted slogans and held placards the whole time." (P = 0.02)
- PR higher: "Participants threw stones and burned parked cars."
- BV lower: "The other group were farmers who carried no weapons."
- BV higher: "The two groups exchanged fire for an hour." (P = 0.99)

### 3.3 Rules for Evidence Sentences (V0)

`geoflip/study.py: static_findings` enforces every rule, and `build.py` refuses to write the dataset if any rule is violated.

1. One sentence of 5–16 words.
2. Stated as certain: no word from a fixed list of hedges and modals (unclear, reportedly, allegedly, possibly, may, might, could, …).
3. **No word sharing a 5-letter stem with any content word of a class name or definition.** The banned list is built automatically from `task_prompts.json`, after removing fixed role terms (group, members, participants, demonstrators, armed group).
4. No capitalised word except the first (a check for named entities), and no state-force, ethnic, religious or party words.
5. No outcome or casualty words (killed, injured, arrested, …).
6. The country phrase appears once, directly after `{place}`, and is the only difference between conditions.

Two items use a stand-in where the ACLED term would break rule 3: a self-defence group is written as "a village night patrol", and "forced open" as "pried open".

### 3.4 Wordings

W1 is the canonical sentence. W2 applies fixed transformations (`items.W2_RULES`):
- **Subject swap:** "Some participants" → "A few demonstrators"; "The other group" → "Members of the other group"; "One member" → "A member".
- **Verb substitution:** at most one, from a fixed table (threw → hurled, burned → set fire to, carried → had, …).

### 3.5 Context Values

The fill-in values (PR: actors, grievance, place; BV: place, time) come from fixed lists in `geoflip/materials.py` (`PR_VALUES`, `BV_VALUES`). The PR lists and the BV places have 20 entries each:
- **Indices 0–7** were used for the prompt-selection pilot (`inputs/v4/bases/*_pilot.csv`).
- **Indices 8–11** are the study contexts (`country_effect/part1_behavioral/build.py`).
- **Indices 12–19** are reserved for a held-out replication.

The 12 BV times are generic and shared across splits; the study uses indices 0, 3, 6 and 9.

### 3.6 Names

- **37 real countries:** the 10 countries of the authors' main behavioural study, plus 27 for regional coverage. Names are inserted in their natural English form ("the United States"); token count is recorded as a covariate.
- **20 placebo names**, generated by `geoflip/placebo.py` from a fixed syllable inventory and seed. A candidate is rejected if it is a real place (any ACLED country, admin area or location, or a world country) or an existing vocabulary token. The kept names match the real names' character lengths by quantile. The output is `country_effect/part1_behavioral/inputs/placebos.json`.
- **No country.**

## 4. Task Prompts

Answers are **single tokens**: the options are listed as `A. <class>` and `B. <class>`, and the prompt asks for one letter. **Order** decides which class is A, so the letter is counterbalanced with the order.

`inputs/v4/task_prompts.json` holds, for each boundary, 4 paraphrases (intro, definitions, answer instruction) × 2 orders × 2 versions:
- *defined*: with class definitions;
- *undefined*: without them.

The study uses the defined version with the options listed before the answer instruction (`options_first`). The four prompts it uses are written out in `country_effect/part1_behavioral/inputs/prompts.json`.

The system prompt is followed by the description as the user turn. The Llama chat template always writes `Today Date: …`, which is fixed at `26 Jul 2024`.

### 4.1 Prompt Selection

One paraphrase per boundary was selected by anchor competence alone, before any country was scored (`country_effect/part1_behavioral/inputs/prompt_selection.json`):
- **Pilot set:** every item and anchor × 2 frames × 8 pilot contexts × 2 wordings, with no country. The pilot contexts are disjoint from the study contexts.
- **Gate:** a prompt passes if anchor accuracy is ≥ 0.9 at both ends in each frame, and the 5th percentile of P(A) + P(B) is ≥ 0.9.
- **Eligibility:** a paraphrase is eligible if its prompts pass in both versions and both orders. Eligible paraphrases were PR 0, 1 and 2, and BV 3.
- **Choice:** the lowest-numbered eligible paraphrase, which gives **PR p0 and BV p3**. Both have anchor accuracy 1.0 in both orders and both frames.

## 5. Scoring

- **Model:** Llama-3.2-3B-Instruct, revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`, bf16.
- **One prompt per forward pass,** with no batching.
- **Read-out:** `scoring.score_first_token` reads log P("A") and log P("B") at the first answer position, after checking that each letter is exactly one token there. It also records P(A) + P(B) (the **mass**) and the top-5 tokens.
- **Score:** for one description and one prompt, **d** = log P(letter of the higher class) − log P(letter of the lower class).
- **Cache:** scores are cached as JSONL, keyed by sha256 of the system prompt, the description and the answer letters (`study.Scorer`).

## 6. Rebuilding the Materials

From `interpretability/` (raw ACLED exports in `../conflict_data/`):

```bash
python -m geoflip.acled --out inputs/v4/acled_acts.csv
python -m geoflip.items --acts inputs/v4/acled_acts.csv --out inputs/v4/items.json
python -m geoflip.materials --items inputs/v4/items.json --out inputs/v4
python -m geoflip.placebo --countries inputs/v4/names.json --out country_effect/part1_behavioral/inputs/placebos.json
python -m geoflip.study check          # V0 rules: prints "V0: no findings"
python -m geoflip.study pilot          # prompt-selection pilot (GPU)
```

## 7. Known Limitations

- **The ACLED lexicon measures mentions, not who acted.** A note mentioning "threw stones" may describe police or bystanders; the agent is restricted only for acts involving person contact. Pattern precision has not been audited (`CONSTRUCTION.md`, Step 2).
- **The ACLED notes cover five regions,** so P(higher | act) reflects coding there.
- **One fact per sentence:** combinations of acts are not crossed.
- **The 37-country set is a convenience frame.**
- **Score resolution:** bf16 scores move in steps of about 0.06 nats.
