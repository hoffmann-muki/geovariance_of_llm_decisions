# Evidence-Based Dataset Construction (v4.1)

Plan written 7 November 2026. **Status: proposed; nothing in this document has been run except the exploratory work in § 7.** It extends [PLAN.md](PLAN.md) (v4) and does not replace it. The measurement design of v4 stays the same: A/B single-token answers, counterbalanced order, one prompt per forward pass, V0 rules, placebo names, a flip rule against a placebo null, and no selection on outcomes. What v4.1 changes is the **inputs**: every item, frame, label and intensity level must trace back to a named source of evidence, so that each design choice can be defended against a specific objection.

## 1. Claim the Dataset Must Support

> For descriptions built only from attested, definition-relevant evidence, Llama-3.2-3B's *actual* class answer changes when only the country name changes, more often than with invented names. These flips cluster in evidence that human coders also find ambiguous, rather than in evidence they find clear.

The second sentence separates two kinds of flip:
- **Override:** a flip on an item human coders agree on. The country overrides clear evidence.
- **Settling:** a flip on an item human coders split on. The country settles uncertain evidence.

A flip still shows only that the answer depends on the country name, not that the dependence is wrong (PLAN.md § 1).

## 2. Evidence Ledger: Objections and the Evidence That Answers Them

| # | Objection | v4 evidence | Gap | Closed by |
| --- | --- | --- | --- | --- |
| O1 | "The evidence sentences are invented" | ACLED attestation, ≥ 20 notes per act | The lexicon counts mentions, not who acted (PLAN.md § 9). Its precision has not been audited | Step 2 |
| O2 | "The frames don't look like real event reports" | None | The frames were written by hand. BV f2 resembles no ACLED note and holds most of the pilot's anchor failures | Step 3 |
| O3 | "The item labels are the authors' opinion" | P(higher \| act) over ACLED notes | That is a distribution over notes, not a judgement of the sentence as written. No human validation | Step 4 |
| O4 | "The model isn't doing the task" | Anchor gates | The gates failed on 6 of 32 (`options_first`) and 10 of 32 (`options_last`) prompts | Step 6 |
| O5 | "The effect is name surface form, not country" | Placebo design, token-count covariate | The placebos have not been generated | Step 6 |
| O6 | "Items were picked after seeing results" | Rules frozen before scoring | The pre-registration file is not yet written | Step 7 |
| O7 | "Templates don't generalise to real text" | None | No comparison with natural notes | Step 9 |
| O8 | "Why these 37 countries?" | "Convenience frame" | No stated sampling rule | Step 5 |
| O9 | "The model just reacts to how violent the text sounds" | None | Intensity is not varied apart from class evidence | § 3, Step 4b |

## 3. Two Axes of Evidence

Each description carries one sentence from each axis, and the two axes are crossed.

### 3.1 Axis 1: Class Evidence (Existing v4 Items)

This axis is what the definitionally relevant party did:
- **PR:** what the demonstrators did.
- **BV:** what means the other group had or used.

Its level is **P(higher | act)** from `inputs/v4/acled_acts.csv`. It runs from 0.01 (burned an effigy) to 0.99 (burned tires) for PR, and from 0.09 (fishermen) to 0.92 (grenade) for BV.

**Subset for crossing:** 6–8 items per boundary, chosen at fixed quantiles of P(higher | act) after the precision audit (Step 2) and before any scoring.

### 3.2 Axis 2: Intensity (New, Irrelevant to the Class by Definition)

This axis is how violent the event was, carried by a party whose actions **do not decide the class** under the definition:

| Boundary | Carrier | Example ordinal levels |
| --- | --- | --- |
| PR | State response to the demonstrators. The Protest definition includes "demonstrations that face intervention or violence from others" | none → water cannon → tear gas → rubber bullets → live fire |
| BV | The armed group's own means. The class depends only on whether the other side could fight back | knife → rifles → rocket-propelled grenades; set houses on fire |

Intensity is carried by the **means**, never by casualty or outcome words (V0 rule 5).

**How the level is grounded:**
- **BV:** the ACLED fatality rate, P(fatalities > 0), among notes that mention the intensity act.
- **PR:** an ordinal harm scale over state means (no contact < water < chemical < less-lethal projectiles < firearms), validated by human severity ratings (Step 4b). ACLED fatalities cannot be used for PR, because almost every PR act has P(fatalities > 0) ≤ 0.05 (§ 7.2).

**A known tension to report, not remove.** Intensity acts are irrelevant to the class by definition but correlated with it in human coding: ACLED notes that mention tear gas are coded Violent demonstration in 70% of cases (§ 7.2). A model that uses intensity is therefore following a real coder base rate, not inventing a stereotype. The design measures which one the model follows: **the definition or the base rate**.

### 3.3 Questions the Crossing Answers

1. Does intensity move the answer when the definition says it should not? For example, does "rifles" push BV toward Battle when the other side is fishermen?
2. Does the country interact with intensity? For example, does the same tear-gas description become Riot for some countries and stay Protest for others?
3. Does the country effect sit mainly in mid-ladder items (settling) or also in clear items (override)?

## 4. Steps, in Order

Each step finishes before the next one starts. Every change to the inputs is logged in `history/ITERATIONS.md`.

| Step | What | Deliverable | Gate or rule (fixed now) |
| --- | --- | --- | --- |
| 1 | Write the claim and the evidence ledger | This document | — |
| 2 | **Precision audit of the lexicon.** For each act, 30 sampled matching ACLED notes, coded by hand for whether the relevant party did the act. A model may pre-tag; a person decides | `inputs/v4.1/lexicon_audit.csv`; P(higher \| act) recomputed on true matches | Acts with precision < 0.7 are re-patterned or dropped |
| 3 | **Frames from the notes.** Code the structure of about 100 notes per comparison group (actor phrase, act verb, location phrase, purpose clause). Write frames from the commonest patterns, removing class-revealing verbs | `inputs/v4.1/frame_induction.md`; each frame component cites its attested pattern | V0 passes; departures from real notes are logged (§ 6) |
| 4a | **Human labels for class evidence.** 2–3 annotators label every item sentence (W1, W2) in a frame, with no country, in both versions | Gold class per item, Krippendorff's α, the share choosing the higher class per item | Pre-registered split into **clear** and **ambiguous** items (threshold fixed before scoring) |
| 4b | **Human ratings for intensity.** The same annotators rate the severity of each intensity sentence | Severity ordering per boundary | The ordinal levels are kept only if the human ordering agrees with § 3.2 |
| 4c | **Check across sources.** Does P(higher \| act) predict the human share choosing the higher class? | Correlation, reported | Reported whatever it shows |
| 5 | **Country sampling rule.** Stratify by ACLED region × event volume, and keep the 10 countries of the behavioural study | `inputs/v4.1/names.json` with its rule | Token count and the "the" form stay covariates |
| 6 | **Pilot, fixed by diagnosis.** Inspect failing prompts one example at a time (anchor, A vs B, both orders, top tokens). Attribute each failure to the frame, letter bias or the undefined version; revise; rerun. Then generate placebos | Pilot report | PLAN.md § 6.1 gates unchanged |
| 7 | **Pre-registration.** Extend `prereg.json` with the precision cut-off, the clear/ambiguous split, the intensity levels and the primary contrast | `inputs/v4.1/prereg.json` with input hashes | Pipeline refuses to run on a hash mismatch |
| 8 | **Main population and filter** | Scores; PLAN.md § 6.4 filter | Rule unchanged |
| 9 | **Bridge to natural notes (secondary).** About 200 real borderline ACLED notes: date removed, admin and location names replaced by a generic phrase plus "in {country}", scored with the same prompts | Correlation between template flips and natural-note flips per item | Exploratory; reported whatever it shows |
| 10 | **Release and second adversarial review** | `dataset/` | PLAN.md § 6.5 |

**Primary contrast (Step 7):** the flip rate on clear items versus ambiguous items, each compared with the placebo null. **Secondary:** the country × intensity interaction, and the effect of intensity on items where the definition says it should have none.

## 5. Rules Fixed Before Any New Scoring

- The pilot and main bases stay disjoint. Exploratory work (§ 7) uses **pilot bases only**.
- Items, intensity levels, frames and countries are chosen from ACLED counts, the definitions and human labels, **never from model scores**.
- Thresholds proposed here (precision 0.7, ≥ 20 notes, 6–8 items per boundary at fixed quantiles, m = 0.5, K = 4 of 8) are frozen in Step 7 before the main run.

## 6. Known Departures from Real ACLED Notes

These are logged as limitations, not fixed:
- **No dates.** Real notes always carry a date. Dates are removed because of the v3 date confound (PLAN.md § 10).
- **"In {country}".** Real notes locate events by admin areas (for example "Yunyang county (Yunyang, Chongqing)"), not by country name. The country slot is itself a departure.
- **Coverage.** The ACLED notes cover five regions only (PLAN.md § 9).

## 7. Exploratory Work So Far (7 November 2026)

Scripts and outputs are in `/fs/clip-scratch/pramu/interp_orderflip/`, with a resume log in `progress.md` there.

### 7.1 Order-Only Flips (Running)

The setting where option order is the only perturbation of the prompt:
- **Layout and prompts:** `options_first`, one paraphrase per boundary: PR p0, and BV p3, the only BV paraphrase that passed every pilot anchor gate.
- **Descriptions:** wording W1, 8 pilot bases, no country plus 37 real countries, both versions. About 133k prompts in total.
- **Flip rule:** two real countries each give the same class in both orders, and those classes are opposite.
- **Scripts:** `one_example.py`, `sweep.py` and `analyse.py`.

Single examples, checked before the sweep:
- **PR, "built barricades", defined:** Riot for every country in both orders. The country moves d by up to about 0.9 nats, with no flip.
- **BV, "threw stones", undefined:** the letter **B** is favoured in both orders. With no country, and for Norway, the US and France, the answer depends on the order. Sudan, Pakistan and China give Violence against civilians in both orders. The country changes whether the answer survives the order swap, but no pair is consistent in both orders with opposite classes.

### 7.2 ACLED Intensity Scan

Script: `intensity_scan.py`. For notes matching each pattern, it computes P(higher), the mean fatalities and P(fatalities > 0). The candidate intensity patterns are a first pass. They have not been audited for which party did the act, and some counts are too small (2 notes for "dozens of gunmen").

| Boundary | Act | n | P(higher) | P(fatalities > 0) |
| --- | --- | --- | --- | --- |
| BV | stones (item) | 85 | 0.56 | 0.41 |
| BV | grenade (item) | 320 | 0.92 | 0.59 |
| BV | farmers (item) | 259 | 0.11 | 0.76 |
| BV | traders (item) | 194 | 0.17 | 0.80 |
| BV | set houses or village on fire (candidate) | 58 | 0.09 | 0.50 (mean 4.4) |
| BV | rocket-propelled grenades (candidate) | 21 | 0.95 | 0.67 |
| PR | burned a vehicle (item; highest PR fatality rate) | 99 | 0.96 | 0.09 |
| PR | tear gas (candidate) | 416 | 0.70 | 0.04 |
| PR | water cannon (candidate) | 148 | 0.37 | 0.01 |
| PR | live fire at demonstrators (candidate) | 65 | 0.34 | 0.18 |

Findings:
- **In BV, intensity runs against the class.** The deadliest events are mostly one-sided attacks on civilians, so intensity is not a stand-in for Battle.
- **In PR, fatalities are near zero for every act,** so PR intensity needs the ordinal scale and human ratings (§ 3.2).
- **State-intervention acts correlate with Riot in ACLED coding** (tear gas 0.70; rubber bullets 0.68), although the definition makes them irrelevant (§ 3.2).

## 8. Open Decisions

1. **Annotators for Step 4.** Who labels about 100–200 sentences and the intensity ratings? This sets the cost and timeline of the step that answers O3 and O9.
2. **Version naming.** This plan is written as v4.1 (same measurement, new inputs in `inputs/v4.1/`). If the frames change substantially in Step 3, it may be clearer to call it v5.
