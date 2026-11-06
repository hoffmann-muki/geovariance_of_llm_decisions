# Geographic Flips in LLM Event Coding: Dataset Construction

Version 3, 6 November 2026. This document is self-contained: it defines the task, the materials, the scoring, the verification procedure and the resulting dataset, so that the dataset can be rebuilt from `inputs/` with the code in `geoflip/`. How the design evolved is recorded in [history/ITERATIONS.md](history/ITERATIONS.md); version 1 (ambiguity ladders) and the reason it was replaced are summarised in § 9.

## 1. Goal and Claims

This dataset is for an LLM interpretability project. It contains **event descriptions whose class, as coded by a language model, changes when only the name of the country where the event happened changes**. Each such change is verified so that it is not an artifact of one task wording, one class order, one sentence wording, or one set of event details.

**The claim the dataset supports is existence, not prevalence.** Descriptions were *selected because* they show country effects. The dataset shows that such effects exist, replicate on held-out descriptions and can be studied mechanistically. It does not estimate how often they occur in real event coding. A flip also does not show that the model is wrong: a model using base rates and a model using stereotypes would both produce flips. Any report that uses this dataset must say that its items were selected for the effect.

## 2. Task

The model codes a short event description into one of two **confusable classes**. Two such pairs (**boundaries**) are used, adapted from conflict-event coding:

| Boundary | Lower class (default) | Higher class | Distinguishing question |
| --- | --- | --- | --- |
| **PR** | Protest | Riot | Did the demonstrators themselves use violence? |
| **BV** | Violence against civilians | Battle | Could the targeted group fight back? |

The **higher class** is the one that needs a specific fact before it can be chosen: violence by demonstrators (Riot), or an armed opposing side (Battle). The **lower class** is what a coder chooses when that fact is missing. The sign convention throughout is that positive scores favour the higher class.

## 3. Materials

### 3.1 Descriptions

A description is rendered from four parts:

```
description = frame( base fields, country phrase, evidence sentence )
```

- **Frame:** the event template, fixed per boundary.
- **Base:** the fill-in values for the frame (actors, grievance and place for PR; place and time for BV).
- **Evidence sentence:** one sentence carrying the distinguishing fact, in one of **two wordings** (W1, W2) that state the same fact.
- **Country condition:** the phrase ` in {country}` placed directly after the place, or nothing.

**Frames.**

| Id | Frame |
| --- | --- |
| PR f1 | `On 12 June 2025, {actors} gathered {place}{country} to demonstrate against {grievance}. {evidence}` |
| PR f2 | `On 12 June 2025, a crowd of {actors} assembled {place}{country} to call attention to {grievance}. {evidence}` |
| BV f1 | `On 12 June 2025, members of an armed group and members of another group were involved in a shooting {place}{country} {time}. {evidence}` |

**Example bases.**

| Boundary | base_id | Fields |
| --- | --- | --- |
| PR | pr_01 | actors = students; grievance = rising fuel prices; place = outside a government office |
| PR | pru_13 | actors = artists; grievance = rising bread prices; place = near a regional court |
| BV | bv_01 | place = in a village; time = in the early morning |
| BV | bvt_07 | place = near a river crossing; time = late at night |

**Evidence sentences.** Each boundary has one **anchor** at each end and a set of **borderline** items. Anchors are clear cases used to check the task prompts. Borderline items state a fact *as certain* that lies near the boundary. Examples:

| Boundary | Item | Role | W1 | W2 |
| --- | --- | --- | --- | --- |
| PR | anchor_lo | anchor, lower | Participants stayed calm throughout the event. | The demonstrators remained calm for the whole gathering. |
| PR | anchor_hi | anchor, higher | Participants threw stones and set a parked car on fire. | Demonstrators hurled stones and set fire to a parked car. |
| PR | shoes | borderline | Some participants threw shoes at a nearby building. | A few demonstrators hurled their shoes at a nearby building. |
| PR | effigy | borderline | Some participants burned an effigy in front of a nearby building. | A few demonstrators set fire to an effigy outside a nearby building. |
| BV | anchor_lo | anchor, lower | The other group were farmers who carried no weapons. | Members of the other group were farmers and had no weapons. |
| BV | anchor_hi | anchor, higher | The other group was a rival armed group, and the two sides exchanged fire for an hour. | The other group was a rival armed group, and the two groups traded gunfire for an hour. |
| BV | watchman | borderline | The other group included a night watchman who carried a pistol. | One person in the other group was a night watchman with a pistol. |
| BV | stones | borderline | Members of the other group threw stones at the armed group. | People in the other group hurled stones at the armed group. |

There are 24 borderline items for PR and 22 for BV, listed in full in Appendix A. An **item** is a frame × evidence pair (for example `PR:f2:shoes`). Within an item, the base, the wording and the country vary.

**A complete description** (item `PR:f2:shoes`, base pru_13, W1, country United States):

> On 12 June 2025, a crowd of artists assembled near a regional court in the United States to call attention to rising bread prices. Some participants threw shoes at a nearby building.

Without a country, the same description reads "…assembled near a regional court to call attention to…". The country phrase is the only difference between conditions; a static check (V0) verifies this for every frame and country.

### 3.2 Rules for Borderline Evidence

The design rests on one finding from version 1 (§ 9): **hedged or missing facts do not create ambiguity, they resolve to the lower class.** A description such as "it is unclear whether participants threw anything" is coded as Protest with high confidence, because a coder (human or model) treats missing evidence as absence of evidence. Ambiguity comes from **vagueness**: a fact stated as certain whose category is borderline ("threw shoes at a building", "a night watchman who carried a pistol").

Every description must also satisfy these textual rules, which V0 enforces where they are mechanical:
1. **No class names, their stems, or the definitions' distinctive vocabulary.** The excluded terms are: protest, riot, battle, civilian, violen, attack, clash, destr, damag, harm, defend, deliberat, unarmed, peaceful, police, arrest, injur, wound, soldier, army, military, ethnic, relig, tribe, tribal, party, parties, killed, dead. They are checked in frames, base fields and evidence sentences.
2. **No presupposed class in the frame.** The BV frame says the two groups "were involved in a shooting" rather than that one group "opened fire on" the other, and it describes both groups in the same words.
3. **No outcome or casualty sentence.**
4. **No named people, cities, ethnic or religious identities, parties or state forces.** The country is the only geographic or identity cue.
5. **The country appears once, as the event location**, directly after `{place}`.
6. **Two different wordings per evidence item**, with no wording reused across items.

Frames, bases and evidence sentences were written for this project, drafted with an LLM assistant (Claude) under these rules. Draft evidence sentences were scored without a country on the explore bases before being added (the `probe` command), and the anchors were revised until they passed V2 (§ 5).

### 3.3 Task Prompts

The task instructions are given as the **system prompt**; the **user turn** is the description alone. Because the wording of the instructions is itself a possible confound, each boundary has **4 paraphrases** (p0–p3), each in **2 versions** and **2 class orders**, giving 16 task prompts per boundary:

- **Version:** *defined* (the class definitions are included) or *undefined* (class names only).
- **Order:** A lists the lower class first for PR (Protest, Riot) and Battle first for BV (Battle, Violence against civilians), as in `order_a`; B reverses the order everywhere, including the order of the definitions and of the answer options.
- **Paraphrase:** rewords the task sentence, the definitions and the answer instruction, keeping the class names and the meaning of each definition. No paraphrase mentions country, location or region, and none gives a rule for uncertain cases.

A system prompt is the intro sentence, then (defined version only) the two definitions in the chosen order, then the answer instruction, joined by blank lines. Example, BV, defined, p0, order A:

```
You code violent events for a conflict-event dataset. Read the event description and decide whether the event is a Battle or Violence against civilians.

Battle: a violent clash between organized armed groups that are capable of harming each other.

Violence against civilians: an organized armed group deliberately attacks unarmed civilians who cannot defend themselves.

Answer with exactly one of: Battle, Violence against civilians.
```

Example, PR, undefined, p1, order B:

```
You are an analyst labeling demonstration events for a dataset on political unrest. Classify the event described below as a Riot or a Protest.

Respond with a single word: Riot or Protest.
```

All paraphrases are listed in Appendix B; all 32 rendered system prompts are in `dataset/system_prompts.csv`.

### 3.4 Country Conditions

Each description is rendered under 40 country conditions:
- **37 real countries:** Libya, Morocco, Egypt, Nigeria, Kenya, Ethiopia, South Africa, Sudan, Mexico, the United States, Canada, Brazil, Colombia, Venezuela, Argentina, India, Pakistan, Bangladesh, Afghanistan, Iran, Iraq, Syria, Yemen, China, South Korea, Japan, Indonesia, the Philippines, Myanmar, Russia, Ukraine, France, Germany, the United Kingdom, Sweden, Norway, Australia. "The" is part of the phrase for the United States, the Philippines and the United Kingdom.
- **2 fictional countries:** Veloria and Tarvania. If invented names move the label as much as real ones, the effect comes from the surface form of a name rather than knowledge about the country.
- **No country:** the phrase is omitted.

### 3.5 Base Splits

Bases are split three ways per boundary, and no base appears in two splits:

| Split | Bases per boundary | Used for |
| --- | --- | --- |
| explore | 8 (`inputs/bases/<B>_explore.csv`); the first 4 are the **screen bases** | Choosing task prompts, items and countries (V2–V4) |
| confirm | 24 (`<B>_confirm.csv`) | Checking that the chosen country effects generalize (V5) |
| test | 16 (`<B>_test.csv`) | A final replication on new bases, scored once (V6) |

The confirm split includes 8 bases that served as the test split of an earlier iteration (it03). Since the it04 evidence items were modelled on what flipped in it03, those 8 bases are not fully independent of the design. The 16 test bases were written after it03 and were scored only once, at the end, so they are the clean held-out set.

## 4. Model and Scoring

**Model.** `meta-llama/Llama-3.2-3B-Instruct`, revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`, in bfloat16, on an NVIDIA RTX A4000 (16 GB), with torch 2.14.1 and transformers 5.22.1.

**Prompt rendering.** The chat template is applied with `add_generation_prompt=True` and a fixed `date_string="26 Jul 2024"`. The rendered prompt ends at the start of the assistant turn:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n{system prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{description}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
```

The code checks that encoding this text reproduces the template's own token ids, and that the system prompt and description each appear verbatim exactly once.

**Answer log-probability.** Class names span several tokens (`Pro`+`test`, `R`+`iot`, `Battle`, `Viol`+`ence`+` against`+` civilians`), so first-token logits are not used. For each class, the class name's tokens are appended to the prompt and the model is run once (teacher forcing); log P(class) is the **sum of the log-probabilities of all its tokens**, read from the rows that predict them. The code checks that appending the answer does not change the prompt's tokenization and that neither answer's tokens are a prefix of the other's. Each prompt is scored on its own, **one forward pass per answer, with no batching or padding**: in an early iteration, padded bf16 batches moved answer log-probabilities by up to 0.6 nats.

**Scores.** For one description and one task prompt:

- **d** = log P(higher class) − log P(lower class).
- **d̄_p** = the mean of d over the two class orders, for paraphrase p. This removes the bias towards whichever class is named first.
- **s** = the mean of d̄_p over the 4 paraphrases: the **task-robust score**. s > 0 leans towards Riot or Battle, s < 0 towards Protest or Violence against civilians. |s| = 0.5 corresponds to odds of about 1.65 to 1 between the two classes.
- **Answer mass** = P(lower) + P(higher), checked in V2 to make sure the model answers with one of the two classes.

So each description × version needs 8 prompts (4 paraphrases × 2 orders).

**Robust label.** A description under one country gets the label sign(s) if **|s| ≥ 0.5 and at least 3 of the 4 d̄_p share the sign of s**. Otherwise its label is **unstable**.

**Flip.** One description, under one task version, with a robust lower-class label for country c_lo and a robust higher-class label for country c_hi.

**Worked example** (row PR-012 of the dataset: `PR:f2:shoes`, undefined, base pru_13, W1; values from `dataset/cells.csv`):

| Country | d̄_p0 | d̄_p1 | d̄_p2 | d̄_p3 | s | Label |
| --- | --- | --- | --- | --- | --- | --- |
| United States | +2.88 | +0.25 | +0.06 | −0.06 | +0.78 | Riot (3 of 4 positive) |
| Iraq | +2.44 | +0.63 | −0.50 | +0.56 | +0.78 | Riot |
| Ukraine | +0.50 | −1.69 | −2.25 | −0.69 | −1.03 | Protest |
| Norway | +1.06 | −1.38 | −1.38 | −0.56 | −0.56 | Protest |
| Nigeria | +1.38 | −0.81 | −1.25 | −0.25 | −0.23 | unstable (\|s\| < 0.5) |
| no country | +1.88 | −0.81 | −1.06 | −0.38 | −0.09 | unstable |
| Veloria | +0.88 | −1.19 | −1.88 | −0.56 | −0.69 | Protest |

Paraphrase p0 leans towards Riot by 1–3 nats for every country here, which is why no single task wording can be trusted on its own.

## 5. Verification Procedure

The procedure is a sequence of stages, each with a pass criterion. Each stage writes a summary that records the sha256 of every input file and of the code; a later stage refuses to run if the inputs changed, and the test stage runs only once per iteration. Each stage scores only what it needs, and every score is cached by model, revision, dtype, system prompt, description and class pair, so stages that reuse prompts do not recompute them.

| Stage | What is scored | Criterion |
| --- | --- | --- |
| **V0 static** | Nothing: the input files | No findings: § 3.2 rules 1, 5 and 6; exactly one anchor at each end; the country phrase is the only difference between conditions; disjoint splits; no task prompt mentions country, countries, location, region, unclear or uncertain |
| **V1 scorer** | 8 stored reference prompts (`inputs/scorer_reference.jsonl`, from a version 1 run of the same model and revision), bypassing the cache | max \|Δ log P\| ≤ 0.01 |
| **V2 task prompts** | Anchors × explore bases × both wordings × every frame, no country, all 16 task prompts | For every paraphrase × version: anchor accuracy of d̄_p ≥ 0.9 at both ends; median answer mass ≥ 0.9 in each order |
| **V3 ambiguity** | Every evidence item × frame × explore base × both wordings, no country, both versions | An item × version is **ambiguous** if it is borderline and the median of s over its 16 descriptions lies within ±1.5 (p between 0.18 and 0.82) |
| **V4 screen** | Up to 16 ambiguous item × versions per boundary, those with median s closest to 0 first, × the 4 screen bases × W1 × all 40 conditions | Per base, a country's **shift** is its s minus the mean s over the 37 real countries. A country is **up** (or **down**) if its shift is positive (negative) on at least 3 of the 4 bases and on average; the 4 largest of each are kept. The item is **selected** if it has at least one up and one down country |
| **V5 confirm** | Up to 10 selected items per boundary, ordered by the number of screen bases on which some real country already flipped and then by the spread of mean shifts. Each is scored on the 24 confirm bases × both wordings (48 descriptions) for its up and down countries, no country and both fictional countries | Every (c_hi, c_lo) pair with c_hi up and c_lo down is **systematic** if s(c_hi) > s(c_lo) on at least 75% of the 48 descriptions **and** the mean of s(c_hi) − s(c_lo) exceeds the item's **wording reference**: the median over confirm bases of \|s(W1) − s(W2)\| with no country. Only flips from systematic pairs are kept |
| **V6 test** | Each confirmed item's pairs that were systematic and produced at least one flip in V5, on the 16 test bases × both wordings (32 descriptions), scored once | The same criterion, applied anew; only flips from pairs that are systematic on the test split are kept |
| **Matrix** | Every distinct description (item, version, split, base, wording) with at least one kept flip from V5 or V6, × all 40 country conditions | None: this is the released dataset. Rows are ordered test split first, then by the largest s(c_hi) − s(c_lo) among their flips; a cap of 250 rows per boundary was not reached |

The caps in V4 (16 items) and V5 (10 items) were set by the compute budget of one GPU for a few hours per boundary.

## 6. Results of Each Stage

These are the results of the final inputs (`inputs/`). PR was run as iteration it04 and BV as it05; the two iterations have identical PR inputs, and a rerun of both from the score cache with the current code reproduced every stage summary exactly (V1: max |Δ log P| = 0.000).

**V0–V2.** No static findings. V1 reproduced the 8 reference prompts exactly. For every paraphrase × version of both boundaries, anchor accuracy was 1.00 at both ends; the lowest median answer mass was 0.971 (BV, undefined, p3, order A). The weakest anchor was the BV Battle anchor in the undefined version (median s = +3.2); all other anchors had |median s| between 5.3 and 14.0.

**V3.** Ambiguous item × versions:

| Boundary | Version | Borderline item × versions | Ambiguous |
| --- | --- | --- | --- |
| PR | defined | 48 | 11 |
| PR | undefined | 48 | 20 |
| BV | defined | 22 | 15 |
| BV | undefined | 22 | 2 |

Without definitions, Llama codes 20 of the 22 BV borderline items as Violence against civilians (median s between −2.2 and −5.9); with definitions, 15 of them move close to the boundary. This is why every BV row in the dataset uses the defined version, and most PR rows the undefined one.

**V4.** All 16 screened item × versions per boundary were selected. In BV the same countries recur for every item: Norway, Russia, Ukraine and South Korea on the Battle side; Bangladesh, Nigeria, Myanmar, Iran and Australia on the Violence against civilians side. In PR the countries depend on the act: Iraq and the Philippines lead the Riot side for shoe-throwing, the United States and the United Kingdom for flag and effigy burning in frame f2, while Ukraine is on the Protest side for every item.

**V5 and V6.**

| Item | Version | Confirm: systematic pairs / 16 | Confirm flips | Test: systematic pairs / tested | Test flips |
| --- | --- | --- | --- | --- | --- |
| PR:f2:shoes | undefined | 16 | 80 | 10 / 12 | 22 |
| PR:f1:shoes | undefined | 16 | 83 | 13 / 16 | 4 |
| PR:f1:eggs | undefined | 16 | 21 | 12 / 12 | 0 |
| PR:f1:fireworks | defined | 16 | 12 | 10 / 10 | 24 |
| PR:f2:flag | undefined | 16 | 18 | 12 / 12 | 4 |
| PR:f2:effigy | undefined | 15 | 22 | 7 / 7 | 19 |
| PR:f1:flag | undefined | 8 | 14 | 2 / 6 | 2 |
| PR:f1:guards | defined | 15 | 10 | 7 / 8 | 0 |
| PR:f1:car_dent | undefined | 0 | 0 | – | – |
| PR:f2:firecrackers | undefined | 0 | 0 | – | – |
| BV:f1:watchman | defined | 16 | 242 | 16 / 16 | 124 |
| BV:f1:stones | defined | 16 | 123 | 16 / 16 | 59 |
| BV:f1:rifle_shots | defined | 16 | 92 | 13 / 13 | 72 |
| BV:f1:pistol | defined | 16 | 79 | 16 / 16 | 79 |
| BV:f1:warning_shot | defined | 16 | 61 | 14 / 14 | 41 |
| BV:f1:drivers_pistol | defined | 16 | 60 | 12 / 12 | 22 |
| BV:f1:former | defined | 13 | 53 | 8 / 13 | 35 |
| BV:f1:shotgun_once | defined | 16 | 43 | 11 / 13 | 25 |
| BV:f1:vigilantes | defined | 16 | 6 | 4 / 4 | 3 |
| BV:f1:sticks | defined | 0 | 0 | – | – |

Kept flips: PR 260 on confirm and 75 on test; BV 759 and 460. Items with no systematic pair (car_dent, firecrackers, sticks) had a wording reference above 1.6 nats: their two wordings differ by more than the country gaps.

**Replication on the test bases.** Of the tested pairs, every BV pair kept its order (s(c_hi) > s(c_lo)) on 100% of the 32 test descriptions, and every PR pair on at least 94%. The pairs that were tested but not systematic on test failed on the mean-gap criterion, not on order. For some items (PR:f1:eggs, PR:f1:shoes) the country order replicates on the test bases but yields few or no flips: a flip also needs the description to sit near the boundary, and in it03 the same pattern came from test bases whose no-country score was far from 0.

## 7. The Dataset

`dataset/` holds 302 descriptions, each scored under all 40 country conditions (12,080 cells). Its columns are described in [dataset/README.md](dataset/README.md).

| Boundary | Descriptions | From test bases | With ≥ 3 real countries on each side | Real-country cells with a robust label |
| --- | --- | --- | --- | --- |
| PR | 68 | 16 | 27 | 46% |
| BV | 234 | 98 | 64 | 28% |

Per item:

| Item | Version | Rows | Test rows | Rows ≥ 3 each side | Mean countries higher | Mean countries lower |
| --- | --- | --- | --- | --- | --- | --- |
| PR:f2:shoes | undefined | 23 | 5 | 6 | 3.3 | 11.4 |
| PR:f1:shoes | undefined | 11 | 1 | 7 | 5.1 | 15.0 |
| PR:f2:effigy | undefined | 10 | 4 | 4 | 2.9 | 16.6 |
| PR:f1:flag | undefined | 7 | 1 | 3 | 4.7 | 13.0 |
| PR:f1:fireworks | defined | 6 | 4 | 4 | 4.8 | 13.7 |
| PR:f1:eggs | undefined | 4 | 0 | 1 | 8.8 | 15.2 |
| PR:f2:flag | undefined | 4 | 1 | 2 | 2.8 | 12.8 |
| PR:f1:guards | defined | 3 | 0 | 0 | 3.7 | 2.3 |
| BV:f1:watchman | defined | 56 | 23 | 23 | 6.0 | 5.2 |
| BV:f1:stones | defined | 35 | 13 | 7 | 4.9 | 6.1 |
| BV:f1:rifle_shots | defined | 34 | 15 | 9 | 4.1 | 7.5 |
| BV:f1:warning_shot | defined | 24 | 11 | 2 | 5.2 | 3.8 |
| BV:f1:drivers_pistol | defined | 23 | 7 | 1 | 5.0 | 2.5 |
| BV:f1:pistol | defined | 20 | 11 | 10 | 6.9 | 6.9 |
| BV:f1:former | defined | 20 | 9 | 8 | 5.1 | 3.3 |
| BV:f1:shotgun_once | defined | 18 | 7 | 3 | 6.2 | 4.0 |
| BV:f1:vigilantes | defined | 4 | 2 | 1 | 3.8 | 2.2 |

**Example rows.** Each row is one description; the cells give the label under each country (37 real countries; countries not listed are unstable).

> **PR-012** (`PR:f2:shoes`, undefined, test, base pru_13, W1): On 12 June 2025, a crowd of artists assembled near a regional court in **[COUNTRY]** to call attention to rising bread prices. Some participants threw shoes at a nearby building.
> - **Riot:** the United States, Iraq, Japan
> - **Protest:** Ukraine, Colombia, France, Mexico, Ethiopia, Venezuela, Yemen, Sweden, Norway
> - Controls: no country −0.09 (unstable), Veloria −0.69 (Protest), Tarvania −0.38 (unstable)

> **BV-099** (`BV:f1:watchman`, defined, confirm, base bvt_07, W1): On 12 June 2025, members of an armed group and members of another group were involved in a shooting near a river crossing in **[COUNTRY]** late at night. The other group included a night watchman who carried a pistol.
> - **Battle (12):** the United States (+2.61), Norway, Russia, Ukraine, South Korea, Germany, the United Kingdom, Sweden, Canada, Mexico, Iraq, South Africa
> - **Violence against civilians (4):** Nigeria (−1.43), Bangladesh (−1.26), Indonesia, Myanmar
> - Controls: no country +1.16 (Battle), Veloria +0.56 (Battle), Tarvania +0.11 (unstable)

**Country effects across the dataset** (mean s over all rows of a boundary):

| Boundary | Most lower-leaning | Most higher-leaning | Median within-row SD of s (real countries) |
| --- | --- | --- | --- |
| PR | Ukraine −1.16, Colombia −0.91, Syria −0.77, Mexico −0.72, Norway −0.64 | China +0.31, the Philippines +0.27, Japan +0.25, the United States +0.16, Iraq +0.14 | 0.56 |
| BV | Bangladesh −1.19, Nigeria −1.04, Indonesia −0.82, India −0.74, Myanmar −0.73 | Norway +0.86, Ukraine +0.64, Russia +0.60, the United States +0.43, South Korea +0.42 | 0.53 |

**Controls.**
- **No country.** The median no-country s is +0.52 for PR and +0.40 for BV. In PR every real country's mean is below the no-country mean (+0.45): naming any real country pushes these descriptions towards Protest. In BV, the no-country condition (+0.41) sits between the two country groups.
- **Fictional countries.** Veloria and Tarvania are unstable on most rows (PR: 91 of 136 cells; BV: 395 of 468). Their scores sit between the no-country condition and the real countries: on average 0.54 below no country and 0.17 above the real-country mean in PR, and 0.25 below and 0.34 above in BV. Part of the shift from naming a country therefore comes from naming any place at all; the spread among real countries (median within-row SD above) is what a fictional name does not explain.

**Transfer to another model (exploratory).** Re-scoring the matrix rows with Qwen3-4B (revision `1cfa9a72`, bf16, which can shift Qwen's scores by up to 0.75 nats) showed that the flips are specific to Llama: 1 of 68 PR rows and 0 of 60 BV rows still flip, because Qwen sits far from its own boundary on these descriptions (mean s −12.6 for PR, +12.6 for BV). Qwen's country effects are as large or larger, but its country ordering agrees with Llama's only partly in PR (r = +0.34 across the 37 countries) and barely in BV (r = +0.13). A dataset for another model needs its own V3–V6 run so that its descriptions sit at that model's boundary. Results are in `history/transfer_qwen3-4b_bf16/`.

## 8. Reproducing the Dataset

The code is in `geoflip/` (`scoring.py`: prompt rendering and answer scoring; `pipeline.py`: the stages; `dataset.py`: the export). On a machine with a 16 GB GPU and the packages in `requirements.txt`:

```bash
cd interpretability
export PY=python HF_HOME=/path/to/hf_cache GEOFLIP_WORK=/path/to/work
# HF_TOKEN in the environment or in interpretability/.env (Llama 3.2 is gated)
scripts/run_loop.sh final PR 0      # iteration name, boundary, GPU index
scripts/run_loop.sh final BV 1
$PY -m geoflip.dataset --iteration final
$PY -m unittest discover -s tests -t .
```

`run_loop.sh` runs the stages `check ambiguity screen confirm test matrix` and writes a report to `results/<iteration>/` after each. Without a cache, one boundary takes about 2 hours on an RTX A4000; most of it is V5 (about 45 minutes) and V6 (about 35 minutes). With the score cache, the same commands finish in about 3 minutes. Because the test split is scored once per iteration, rerunning the test stage needs a new iteration name. Report the dataset's `manifest.json`, which records the input and code hashes, alongside any result.

To change the inputs, write them to a new directory, pass it with `--inputs`, and give the run a new iteration name, so that existing stage summaries and their hashes stay valid. Record the change and its result in `history/ITERATIONS.md`.

## 9. Development History

The dataset took five iterations. Each is logged in `history/ITERATIONS.md`, and its inputs, stage summaries and reports are kept in `history/`.

- **Version 1 (replaced).** 5-level evidence ladders per boundary, calibrated without a country, with a planned Friedman test of country effects. On Llama 3.2 every ladder failed its calibration. The middle levels were hedged facts, which resolve to the default class instead of the boundary; the PR frame contained the word "protest"; and the BV frame presupposed a one-sided attack. Version 1 also aimed at a population estimate, whereas the course needs a dataset in which the effect exists, so selecting items for the effect became part of the design.
- **it01.** The new frames and borderline items. Batched scoring failed V1 and was replaced by one prompt per forward pass. The BV Battle anchor failed V2 in the undefined version: the frame's "another group of people" signalled civilians.
- **it02.** A symmetric BV frame. The Battle anchor's second wording ("…a rival armed faction, and both sides fired at each other") was still weak; five candidate rewordings were scored with `probe`.
- **it03.** The Battle anchor's W2 became "…the two groups traded gunfire for an hour." All stages passed: 8 ambiguous item × versions per boundary, 57 (PR) and 183 (BV) confirm flips, 8 and 65 test flips.
- **it04.** 12 new borderline items per boundary, modelled on those that flipped in it03; a second frame per boundary; the it03 test bases moved into confirm; 16 new test bases. The new BV frame ("gunfire broke out … involving …") failed V2, so BV stopped. PR completed and is the final PR run.
- **it05.** it04 without the failing BV frame; the final BV run.

## 10. Limitations

- **One model, two boundaries, English only, constructed descriptions.** All descriptions share three frames, and the country is mentioned once, as the location.
- **Flips are selected.** Their number and size are not prevalence estimates, and the confirm split is only partly independent of the design choices (§ 3.5).
- **The agreement rule counts signs.** A robust label needs 3 of 4 paraphrases on the side of s, and values very close to 0 count. In PR-012 the United States' Riot label rests on paraphrase values of +0.25 and +0.06. Rows with at least 3 countries on each side, or with large |s|, are the most robust contrasts.
- **bf16 resolution.** Logits are in bf16, so log-probabilities move in steps of about 0.06 nats; many d̄ values in the dataset are multiples of 1/16.
- **A flip shows dependence, not its cause or its justification.** It shows that the model's decision depends on the country name for that description, not why, or whether the dependence is warranted.

## Appendix A. All Evidence Items

### A.1 PR (Protest vs Riot)

| Item | W1 | W2 |
| --- | --- | --- |
| anchor_lo | Participants stayed calm throughout the event. | The demonstrators remained calm for the whole gathering. |
| anchor_hi | Participants threw stones and set a parked car on fire. | Demonstrators hurled stones and set fire to a parked car. |
| barrier | Some participants pushed against a metal barrier until it fell over. | A few demonstrators shoved a metal barrier until it toppled. |
| graffiti | Some participants spray-painted slogans on the walls of a nearby building. | A few demonstrators painted slogans on the walls of a nearby building with spray paint. |
| eggs | Some participants threw eggs at a nearby building. | A few demonstrators hurled eggs at a nearby building. |
| tires | Some participants blocked the road with burning tires. | A few demonstrators set tires alight to block the road. |
| bottles | Some participants threw plastic water bottles toward a nearby building. | A few demonstrators tossed plastic water bottles at a nearby building. |
| door | Some participants kicked the front door of a nearby building until a glass panel cracked. | A few demonstrators kicked the front door of a nearby building until one of its glass panels cracked. |
| bins | Some participants knocked over rubbish bins along the street. | A few demonstrators tipped over trash bins on the street. |
| flag | Some participants burned a flag in front of a nearby building. | A few demonstrators set a flag on fire in front of a nearby building. |
| entrance | Some participants blocked the entrance of a nearby building and did not let staff leave. | A few demonstrators stood in the doorway of a nearby building and kept staff from leaving. |
| journalist | Some participants pushed a journalist who was filming the crowd. | A few demonstrators shoved a reporter who was filming the gathering. |
| fireworks | Some participants set off fireworks next to a nearby building. | A few demonstrators lit fireworks beside a nearby building. |
| window | Some participants broke a window of a nearby building. | A few demonstrators smashed one of the windows of a nearby building. |
| tomatoes | Some participants threw tomatoes at a nearby building. | A few demonstrators hurled tomatoes at a nearby building. |
| paint | Some participants threw paint at the entrance of a nearby building. | A few demonstrators splashed paint on the entrance of a nearby building. |
| shoes | Some participants threw shoes at a nearby building. | A few demonstrators hurled their shoes at a nearby building. |
| smoke | Some participants set off smoke bombs in front of a nearby building. | A few demonstrators lit smoke flares outside a nearby building. |
| effigy | Some participants burned an effigy in front of a nearby building. | A few demonstrators set fire to an effigy outside a nearby building. |
| sign | Some participants tore down a sign on a nearby building. | A few demonstrators pulled a sign off a nearby building. |
| planters | Some participants overturned flower planters along the street. | A few demonstrators knocked over flower planters on the street. |
| firecrackers | Some participants threw firecrackers toward a nearby building. | A few demonstrators tossed firecrackers at a nearby building. |
| guards | Some participants pushed past security guards at the entrance of a nearby building. | A few demonstrators shoved their way past security guards at the entrance of a nearby building. |
| bin_fire | Some participants set a rubbish bin on fire. | A few demonstrators lit a trash bin on fire. |
| car_dent | Some participants kicked a parked car, leaving dents in it. | A few demonstrators kicked a parked car and dented it. |
| rocks_ground | Some participants threw small rocks onto the road in front of a nearby building. | A few demonstrators tossed small rocks onto the road outside a nearby building. |

### A.2 BV (Battle vs Violence against civilians)

| Item | W1 | W2 |
| --- | --- | --- |
| anchor_lo | The other group were farmers who carried no weapons. | Members of the other group were farmers and had no weapons. |
| anchor_hi | The other group was a rival armed group, and the two sides exchanged fire for an hour. | The other group was a rival armed group, and the two groups traded gunfire for an hour. |
| selfdef | The other group belonged to a village self-defense committee and carried rifles. | Members of the other group were part of a local self-defense committee and were armed with rifles. |
| fired_fled | One member of the other group fired back with a hunting rifle before the group fled. | A member of the other group returned fire once with a hunting rifle, and then the group ran away. |
| air | One member of the other group fired a hunting rifle into the air. | A member of the other group shot a hunting rifle into the air. |
| guards | The other group was guarding the area, and some of its members carried rifles. | The other group was keeping watch over the area, and a few of its members had rifles. |
| hunters | The other group were hunters carrying shotguns. | Members of the other group were hunters who had shotguns with them. |
| patrol | The other group was a neighborhood patrol armed with machetes and sticks. | The other group was a local patrol whose members carried machetes and sticks. |
| stones | Members of the other group threw stones at the armed group. | People in the other group hurled stones at the armed group. |
| carried_nofire | Some members of the other group carried rifles, but none of them fired. | A few people in the other group had rifles but did not use them. |
| former | The other group included former fighters of a rival armed group who no longer carried weapons. | Some members of the other group had once fought for a rival armed group but no longer had weapons. |
| pistol | One member of the other group carried a pistol and fired it once. | A member of the other group had a pistol and fired a single shot. |
| sticks | Members of the other group fought back with sticks and machetes. | People in the other group hit back with sticks and machetes. |
| rifle_shots | One member of the other group carried a rifle and fired several shots. | A member of the other group had a rifle and fired a few shots. |
| shotgun_once | One member of the other group fired a shotgun once before the group scattered. | A member of the other group shot a shotgun one time, and then the group scattered. |
| petrol_bomb | Members of the other group threw a petrol bomb at the armed group. | People in the other group hurled a petrol bomb at the armed group. |
| watchman | The other group included a night watchman who carried a pistol. | One person in the other group was a night watchman with a pistol. |
| vigilantes | The other group was a vigilante group formed by local residents, some of whom carried shotguns. | The other group was a vigilante group of local residents, a few of whom had shotguns. |
| kept_rifles | The other group included men who had recently left a rival armed group and still had their rifles. | Some members of the other group had recently quit a rival armed group and kept their rifles. |
| warning_shot | One member of the other group fired a warning shot into the ground. | A member of the other group shot once into the ground as a warning. |
| hunters_back | The other group were hunters, and one of them fired back at the armed group. | Members of the other group were hunters, and one hunter returned fire at the armed group. |
| drivers_pistol | The other group were truck drivers, one of whom carried a pistol. | Members of the other group were truck drivers, and one of them had a pistol. |
| militia_unarmed | Members of the other group had joined a local militia a month earlier but carried no weapons that day. | Some members of the other group had joined a local militia a month before but had no weapons with them that day. |
| slingshots | Members of the other group fired slingshots at the armed group. | People in the other group used slingshots against the armed group. |

## Appendix B. Task Prompt Paraphrases

`{first}` and `{second}` are the class names in the chosen order; `{first_phrase}` and `{second_phrase}` are "a Protest", "a Riot", "a Battle" or "Violence against civilians". The definitions are included only in the defined version, in the chosen order.

### B.1 PR

| | Intro | Definition: Protest | Definition: Riot | Answer instruction |
| --- | --- | --- | --- | --- |
| p0 | You code demonstration events for a conflict-event dataset. Read the event description and decide whether the event is {first_phrase} or {second_phrase}. | Protest: a public demonstration in which participants remain nonviolent, including demonstrations that face intervention or violence from others. | Riot: a demonstration in which participants engage in violence or destruction, such as attacking people or damaging property. | Answer with one word: {first} or {second}. |
| p1 | You are an analyst labeling demonstration events for a dataset on political unrest. Classify the event described below as {first_phrase} or {second_phrase}. | Protest: a demonstration whose participants do not use violence, even if others use force against them. | Riot: a demonstration whose participants use violence or cause destruction, for example by attacking people or damaging property. | Respond with a single word: {first} or {second}. |
| p2 | Your job is to categorize reports of public demonstrations for a research database. Decide which category the reported event belongs to: {first_phrase} or {second_phrase}. | Protest: participants stay nonviolent, including events where others act violently toward them. | Riot: participants commit violent or destructive acts, such as hurting people or breaking property. | Reply with only the category name: {first} or {second}. |
| p3 | Below is a short report of a demonstration. Assign it to one of two event types used in conflict monitoring: {first_phrase} or {second_phrase}. | Protest: a public gathering in which the participants act without violence, including gatherings that are met with violence by others. | Riot: a gathering in which the participants act violently or destructively, such as assaulting people or wrecking property. | Give your answer as one word: {first} or {second}. |

### B.2 BV

| | Intro | Definition: Battle | Definition: Violence against civilians | Answer instruction |
| --- | --- | --- | --- | --- |
| p0 | You code violent events for a conflict-event dataset. Read the event description and decide whether the event is {first_phrase} or {second_phrase}. | Battle: a violent clash between organized armed groups that are capable of harming each other. | Violence against civilians: an organized armed group deliberately attacks unarmed civilians who cannot defend themselves. | Answer with exactly one of: {first}, {second}. |
| p1 | You are an analyst labeling violent events for a dataset on armed conflict. Classify the event described below as {first_phrase} or {second_phrase}. | Battle: organized armed groups fight each other, and each side is able to inflict harm on the other. | Violence against civilians: an organized armed group intentionally targets unarmed civilians who have no means of defending themselves. | Respond with exactly one label: {first} or {second}. |
| p2 | Your job is to categorize reports of violent incidents for a research database. Decide which category the reported event belongs to: {first_phrase} or {second_phrase}. | Battle: an armed confrontation between organized groups that can both use force against each other. | Violence against civilians: organized armed actors deliberately use force against civilians who are unarmed and defenseless. | Reply with only the category name: {first} or {second}. |
| p3 | Below is a short report of a violent incident. Assign it to one of two event types used in conflict monitoring: {first_phrase} or {second_phrase}. | Battle: a fight between organized armed sides, each capable of hurting the other. | Violence against civilians: an organized armed group purposely harms unarmed civilians who are unable to protect themselves. | Give your answer as exactly one of: {first}, {second}. |

## Appendix C. Explore Bases

The confirm and test bases have the same fields and are listed in `inputs/bases/`.

| PR base | Actors | Grievance | Place |
| --- | --- | --- | --- |
| pr_01 | students | rising fuel prices | outside a government office |
| pr_06 | factory workers | water shortages | near a ministry building |
| pr_11 | residents | power outages | outside a government office |
| pr_16 | farmers | a proposed tax increase | near a ministry building |
| pr_18 | market traders | water shortages | in a central square |
| pr_23 | retirees | power outages | outside a town hall |
| pr_03 | students | power outages | near a ministry building |
| pr_20 | market traders | a proposed tax increase | outside a town hall |

| BV base | Place | Time |
| --- | --- | --- |
| bv_01 | in a village | in the early morning |
| bv_06 | on a farm | in the afternoon |
| bv_11 | on a rural road | in the evening |
| bv_16 | in a riverside settlement | at night |
| bv_17 | in a hillside settlement | in the early morning |
| bv_22 | near a market on the edge of a town | in the afternoon |
| bv_03 | in a village | in the evening |
| bv_20 | in a hillside settlement | at night |
