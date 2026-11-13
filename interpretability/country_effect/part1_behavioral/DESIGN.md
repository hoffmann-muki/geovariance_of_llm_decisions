# Part 1 Design: Factorial Dataset and Preliminary Behavioural Analysis

Written 7 November 2026, **before any Part 1 result was looked at**. The only Part 1 scores seen when this was written are those of one description (`BV|pistol|f1|W1|BV_c00`), printed to check the pipeline (§ 4.3). Any later change to this document is logged in § 7 with its reason.

## 1. Question

When a conflict-event description is held fixed and only the name of the country where the event happened changes, does Llama-3.2-3B-Instruct's class label change? If so, how often, for which evidence, and in which direction for which countries? And is the effect larger than what invented names of similar length produce?

## 2. Task

Two boundaries adapted from ACLED event coding (`../../MATERIALS.md` § 2):

| Boundary | Lower class | Higher class | Deciding question |
| --- | --- | --- | --- |
| PR | Protest | Riot | Did the demonstrators themselves use violence? |
| BV | Violence against civilians | Battle | Could the targeted group fight back? |

## 3. Dataset

### 3.1 Factors

Every factor level comes from the v4 materials (`../../inputs/v4/`, built by `geoflip/`). The evidence for each item is ACLED attestation (`../../MATERIALS.md` § 3.2).

| Factor | What it varies | PR levels | BV levels | Source |
| --- | --- | --- | --- | --- |
| Item (evidence) | The decisive act | 29 items + 2 anchors | 22 items + 2 anchors | ACLED act lexicon, ≥ 20 attesting notes per act |
| Frame | Sentence structure (paraphrase) | f1 "rallied … to oppose", f2 "assembled … to call attention to" | f1 "were involved in a shooting", f2 "A shooting … involved" | v4 frames |
| Wording | Verbiage of the evidence sentence | W1, W2 | W1, W2 | Fixed W2 transformation rules |
| Actors | Who demonstrated | 4 | — | v4 main pool, indices 8–11 |
| Grievance | What they opposed | 4 | — | v4 main pool, indices 8–11 |
| Place | Where | 4 | 4 | v4 main pool, indices 8–11 |
| Time | When | — | 4 | v4 time list, indices 0, 3, 6, 9 |
| **Country** | The only manipulated variable | none + 37 real + 20 placebo | same | v4 names; `geoflip/placebo.py` |

**Context design:**
- **PR:** a 4 × 4 Latin square. Every actor × grievance pair occurs once, and the place is (actor index + grievance index) mod 4, so every actor × place and grievance × place pair also occurs exactly once. That gives 16 contexts.
- **BV:** a full 4 × 4 crossing of place × time, also 16 contexts.
- **Halves:** each context gets a half, (i + j) mod 2, for split-half replication.
- **Held out:** context values with indices 12–19 are not used, and stay reserved for a held-out replication in the final report.

**Size:**
- PR: 31 × 2 × 2 × 16 = 1,984 descriptions. BV: 24 × 2 × 2 × 16 = 1,536. In total **3,520 descriptions**.
- Each is rendered under 58 name conditions, giving **204,160 country-instantiated texts**.
- With 2 orders each, that is **408,320 scored prompts**.

### 3.2 Construction Rules

These are inherited from v4 and checked by `build.py`, which refuses to write the dataset on any finding:
- **Evidence sentences:** one per item, 5–16 words, stated as certain, with no hedges or modals.
- **No leakage:** no word sharing a 5-letter stem with a class name or definition, no named entities, no outcome or casualty words.
- **Country slot:** it is `" in {country}"` directly after the place. The rendered text differs from the no-country text only by that phrase. `build.py` asserts this for every description and name.
- **No dates in the frames.** The chat template's date header is fixed at 26 Jul 2024.

### 3.3 Placebo Names

`geoflip/placebo.py` (seed 20261106) builds names from a fixed syllable inventory, rejecting any ACLED place name, world country or single vocabulary token, and matches the character-length distribution of the real names by quantile. The 20 names are in `inputs/placebos.json`, with their token counts. Placebos estimate how much labels move because of the surface form of an inserted name alone.

## 4. Measurement

### 4.1 Prompts

- **Prompt:** the *defined* version, with both class definitions. One paraphrase per boundary: PR p0 and BV p3. Layout `options_first`.
- **Why these prompts:** the selection pilot (`../../MATERIALS.md` § 4.1; gates in `inputs/prompt_selection.json`) makes a paraphrase eligible if its prompts pass every anchor gate in both versions and both orders. The eligible paraphrases were PR 0–2 and BV 3; the lowest-numbered eligible paraphrase is used. Both selected prompts have anchor accuracy 1.0 in both orders and frames. The choice used anchors only, never country effects.
- **Order perturbation:** order A lists the classes as in `order_a` (PR: Protest, Riot; BV: Battle, Violence against civilians), and order B swaps them. The definitions follow the option order. The exact system prompts are in `inputs/prompts.json`.

### 4.2 Scoring

- **Scorer:** `geoflip.study.Scorer` → `scoring.score_first_token`, one prompt per forward pass, no batching, bf16, on one RTX A4000.
- **Per prompt:** log P("A") and log P("B") at the first answer position, the mass P(A) + P(B), and the top-5 tokens.
- **Score:** for description i, name c and order o, **d_ico** = log P(letter of higher class) − log P(letter of lower class).

### 4.3 Single-Example Check

Before the full run, one description was printed with every number (`score.py --show "BV|pistol|f1|W1|BV_c00"`). For all 58 names the top token was "B" in both orders: d < 0 in order A, d > 0 in order B. The strict label is therefore order-dependent for every name, while the order-averaged score is positive (Battle) for every name. This is why both label definitions below are specified.

## 5. Analysis (Specified Before Results)

### 5.1 Labels

| Quantity | Definition |
| --- | --- |
| **Strict label** L_ic | Higher if d_icA > m and d_icB > m; lower if both < −m; otherwise **order-dependent** |
| **Order-averaged score** s_ic | (d_icA + d_icB) / 2. If the letters carry an additive bias, averaging over the two orders cancels it |
| **Averaged label** A_ic | Higher if s_ic > m; lower if s_ic < −m; otherwise **near boundary** |
| **Order effect** b_ic | (d_icA − d_icB) / 2 |

The margin is **m = 0.5** nats. Every label count is also reported at m = 0.

### 5.2 Primary Outcomes

1. **Strict flip description:** among the 37 real countries, at least one gets the strict higher label and at least one the strict lower label, both at margin m.
2. **Averaged flip description:** the same with A_ic.
3. **Placebo null at the pair level.** The flip rate per pair of names is the share of (description, unordered pair) cells in which the two names get confident, opposite labels. Real–real pairs (666 per description) are compared with placebo–placebo pairs (190 per description).
4. **Placebo null at the description level, size-matched.** The flip-description rate using 20 real names, averaged over 500 random 20-name subsets (seed 20261107), is compared with the rate using the 20 placebos.

Uncertainty: 95% intervals by bootstrap over descriptions (2,000 resamples, seed 20261107), clustered by description.

### 5.3 Secondary Analyses

- **S1 Validity:**
  - anchor accuracy (strict and averaged) per boundary and name kind,
  - the share of prompts whose top token is A or B,
  - the 5th percentile of the mass,
  - the distribution of the order effect b.
- **S2 Country effects:**
  - Δ_ic = s_ic − (mean of s over the 20 placebos for description i).
  - A country's effect is the mean of Δ_ic over descriptions, per boundary, with a bootstrap 95% interval.
  - Split-half replication: the correlation of country effects estimated on context half 0 and half 1.
- **S3 Variance decomposition** within descriptions:
  - Fit s_ic = α_i + β_c + ε_ic separately for real and for placebo names.
  - Report the SD of β_c and the share of within-description variance explained by β_c.
- **S4 Where flips occur:** flip-description rates by item (ordered by ACLED P(higher | act)), frame, wording and context factor.
- **S5 Name covariates:** the correlation of β_c with token count, for real and placebo names.
- **S6 No-country baseline:** for each description, s with no country, and how often each real country's label differs from it.

### 5.4 Released Tables

- `dataset/descriptions.csv`: one row per description, with all factor levels and the text.
- `dataset/labels.csv.gz`: one row per description × name, with d_A, d_B, s, b, the strict label, the averaged label, and the mass in both orders.
- `dataset/description_summary.csv`: one row per description, with the number of real and placebo names in each label, both flip indicators, the highest- and lowest-s real countries, and the no-country label.

## 6. Known Limitations, Stated in Advance

- This is one model and one prompt per boundary. Effects of task paraphrase are not studied here; v4 found them large.
- The items carry v4's evidence base. The lexicon precision audit, frames derived from ACLED notes and human labels (`../../CONSTRUCTION.md` Steps 2–4) are not yet done.
- The 37 real countries are a convenience frame, as in v4.
- The placebo generator rejects ACLED place names and world countries only. One placebo, "Reine", is also a small Norwegian village and a French word; the generator's output is kept unedited.
- bf16 scores move in steps of about 0.06 nats.

## 7. Change Log

- 2026-11-07: written before the full run was analysed.
- 2026-11-07: `analyse.py --partial` was run once on the first 50 complete descriptions (in the fixed shuffled order), to test the code and check the figures for layout. Its outputs (`report/results_partial.json`, `report/figures_partial/`) are not results and are not reported. No analysis rule in § 5 changed afterwards.
- 2026-11-08: full run complete (3,520 descriptions, 408,320 prompts) and analysed with the § 5 rules unchanged. After the results were seen, four additions were made to `analyse.py`. Each is labelled post hoc in the report, and none changes any § 5 number (`results.json` was diffed before and after):
  - **Leave-one-out flip rate** (`posthoc.avg_flip_leave_one_out`). The United States sits on one side of almost every BV flip.
  - **SD of country effects among single-token real names.** This checks whether the token-count correlation explains the BV effect.
  - **Letter preference** (log P(A) − log P(B), averaged over orders). This replaces the order-effect histogram, whose axis label wrongly assumed the higher class is listed first in order A; in PR it is listed second.
  - **An order-averaged ladder figure.** The strict ladder is mostly order-dependent for BV.

  Figures were renumbered to follow the report.
- 2026-11-13: release clean-up. The partial-run outputs (`report/results_partial.json`, `report/figures_partial/`), which were code tests, are not part of the release. `geoflip.pipeline` was reduced to the four rendering helpers the study uses (`geoflip/render.py`). The pilot gates behind the prompt choice were copied into `inputs/prompt_selection.json`. Re-running `build.py` and `analyse.py` reproduces every released file (§ 5.4) unchanged.
