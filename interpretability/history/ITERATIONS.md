# Dataset Iterations

Each iteration is one version of the inputs in this directory. Record what changed, why, and what the verifier loop showed. Reports are in `interpretability/results/geo/`.

## it01 (2026-11-05)

**Inputs.**
- Frames:
  - PR: "to demonstrate against", with no closing sentence.
  - BV: "a shooting involving" two groups, with no casualty sentence.
- Evidence: 12 borderline items (PR) and 10 (BV), each a fact stated as certain, plus one anchor at each end. The audit in `notebooks/ladder_basics.ipynb` § 4 supplied P6 (barrier), B5 (self-defense group) and B8 (fired back and fled); the rest are new.
- Task prompts: 4 paraphrases × 2 versions × 2 orders, where p0 is the v1 wording.
- Countries: 37 real and 2 fictional.

**Why.** Version 1 failed because its middle levels were hedges and its frames leaked labels (PLAN.md § Why Version 1 Was Replaced).

**Result: failed V1 and V2** (reports `results/geo/it01_PR.md`, `it01_BV.md`).
- **V1:** a batched screening scorer did not match the reference scorer. Mixing sequence lengths in one padded bf16 batch moved answer log-probabilities by up to 0.6 nats, and bf16 logit rounding differs between kernels. **Decision:** no batching. Every prompt is scored with `interp_score.score_prompt`, one per forward pass, as in Stage A, and V1 now checks that the scorer reproduces stored Stage A scores. All cached scores from the batched scorer were deleted.
- **V2:** all PR task prompts pass. In BV-undefined, the Battle anchor is correct for only 50–75% of cases under p0–p2. The frame's "another group of people" probably signals civilians.

## it02 (2026-11-05)

**Change.** The BV frame is now "members of an armed group and members of another group were involved in a shooting …". Both groups are described the same way, and "of people" is gone.

**Result: V0 passes. V1 passes: the scorer reproduces Stage A exactly (Δ = 0.0000). PR passes V2; BV fails V2** (report `results/geo/it02_BV.md`). The BV-undefined Battle anchor fails only in wording 2, "belonged to a rival armed faction, and both sides fired at each other". That wording scores about 3 nats below wording 1 on every base. The `probe` command compared five rewordings (explore bases, no country). "Exchanged fire" or "traded gunfire" carries the reciprocity; "fired at each other" and "faction" are weaker. Undefined-version medians, all paraphrases:

| Candidate | Median s | Min s |
| --- | --- | --- |
| it02 W1 "…exchanged fire for an hour." | +3.60 | +2.33 |
| it02 W2 "…belonged to a rival armed faction, and both sides fired at each other…" | +0.62 | −0.28 |
| "…was another armed group, and the two sides fired at each other…" | +2.04 | +0.81 |
| "Members of the other group were fighters from a rival armed group, and…exchanged fire…" | +2.42 | +1.56 |
| "…was a rival armed group, and the two groups traded gunfire for an hour." | +2.81 | +1.49 |

## it03 (2026-11-05)

**Change.** The BV Battle anchor's wording 2 is now "The other group was a rival armed group, and the two groups traded gunfire for an hour."

**Result: passed V0–V6** (reports `results/geo/it03_PR.md`, `it03_BV.md`).

| Stage | PR | BV |
| --- | --- | --- |
| V0–V2 checks | pass | pass |
| V3 ambiguous item × version (of 24) | 8 | 8 |
| V4 selected (opposite-direction countries) | 8 | 8 |
| V5 flips from systematic pairs (confirm, 16 bases × 2 wordings) | 57 | 183 |
| V6 flips from systematic pairs (test, 8 new bases × 2 wordings) | 8 | 65 |

- **Country ordering replicates on the test bases.** Nearly every confirmed pair keeps its order on 100% of test descriptions:
  - BV: Norway, Russia, Ukraine and South Korea score above Bangladesh, Nigeria, Myanmar and China by 1.0–2.3 nats, against a wording gap of 0.35.
  - PR: China, Philippines, Indonesia, Germany and Brazil score above Ukraine, Pakistan, Syria and Colombia.
- **Flips need the description near the boundary.** PR yielded few test flips (8) because the new test bases move the no-country baseline off 0, so every country lands on one side even though the country gaps hold. Yield therefore depends on having many bases and evidence variants near s = 0.
- **The model's sensitivity to the instructions is comparable to its sensitivity to the country.** In BV-defined, paraphrase p2 is about 2 nats below the others for every country. Averaging over paraphrases is necessary.
- **Matrix stage (added after V6):** every country condition is scored on each description with a verified flip. Output: `results/geo/it03_{PR,BV}_matrix_{labels,s}.csv`.

## it04 (2026-11-05)

Inputs are in `data/geo/it04/`. Each iteration from it04 on has its own input directory, so earlier iterations' inputs, and the hashes recorded in their stage summaries, stay unchanged.

**Changes (more description variety).**
- **Evidence:** keeps all it03 sentences and adds 12 borderline acts per boundary, modelled on the items that flipped in it03.
  - PR: tomatoes, paint, shoes, smoke bombs, an effigy, a sign, planters, firecrackers, pushing past security guards, a bin fire, a dented car, small rocks.
  - BV: sticks and machetes, several rifle shots, one shotgun shot, a petrol bomb, a watchman with a pistol, vigilantes with shotguns, ex-members who kept rifles, a warning shot, hunters who fired back, a truck driver with a pistol, unarmed militia members, slingshots.
- **Frames:** adds a second frame per boundary.
  - PR f2: "a crowd of {actors} assembled … to call attention to {grievance}."
  - BV f2: "gunfire broke out … involving members of an armed group and members of another group."
- **Bases:**
  - The it03 test bases (8 per boundary, already scored once) join the confirm split, which grows to 24 bases.
  - 16 new test bases per boundary bring new actors, grievances, places and times (`bases_*_test2.csv`).
- **Caps,** for the 4.5-hour compute budget: V4 screens up to 16 ambiguous item × version combinations per boundary, closest to s = 0 first. V5 confirms up to 10 per boundary, most flip bases first. The matrix holds up to 150 descriptions per boundary, test split first.

**Result (BV): failed V2** (report `results/geo/it04_BV.md`). The new BV frame f2 ("gunfire broke out … involving …") misses the Battle anchor in BV-undefined under paraphrase p2: wording 1 is correct for 6 of 8 bases, wording 2 for 4 of 8. Frame f1 passes every check. The BV it04 run stopped during V3. PR continues on the it04 inputs (V2 passed).

## it05 (2026-11-05, BV only)

**Change.** The inputs are identical to it04 (`data/geo/it05/`), except that BV frame f2 is removed because it failed V2. PR stays on it04.

**V0–V4 results, it04 (PR) and it05 (BV)** (reports `results/geo/it04_PR.md`, `it05_BV.md`):
- V0–V2 pass for both.
- **V3:** 31 ambiguous item × version combinations for PR (8 in it03), 17 for BV (8 in it03). The new acts and PR frame f2 add many descriptions near s = 0.
- **V4:** all 16 screened combinations per boundary were selected. New items lead the yield ranking.
  - BV: shotgun fired once (flips on 3 of 4 screen bases), several rifle shots, night watchman with a pistol, truck driver with a pistol, warning shot.
  - PR: shoes (f2 and f1), dented car.
- **The BV country axis is the same for all 16 items:**
  - Battle side: Norway, Russia, Ukraine, South Korea.
  - VAC side: Bangladesh, Nigeria, Myanmar, Iran, Australia.
- **The PR axis depends on the act:**
  - Iraq leads the Riot side for shoe-throwing but is on the Protest side for fireworks.
  - The US and UK are on the Riot side for flag burning and effigies.
  - Ukraine is on the Protest side for all 16 PR items.

**V5–V6 results, it04 (PR) and it05 (BV):**

| Stage | PR (it04) | BV (it05) |
| --- | --- | --- |
| V5 flips from systematic pairs (confirm, 24 bases × 2 wordings) | 260 | 759 |
| V6 flips from systematic pairs (test, 16 new bases × 2 wordings) | 75 | 460 |
| Matrix rows (distinct flip descriptions × 40 country conditions) | 68 | (running) |

- **Productive items in PR:** shoes (f1 and f2; 163 confirm flips), effigy (f2), eggs, flag (f2), fireworks, security guards. Dented car and firecrackers had no systematic pairs on the confirm bases.
- **Productive items in BV:**
  - Night watchman with a pistol: 242 confirm flips and 124 test flips.
  - Stones, several rifle shots, pistol, warning shot, truck driver with a pistol, former fighters, shotgun fired once.
  - Sticks and machetes had no systematic pairs.
- **Replication on the 16 new test bases:**
  - BV: every tested pair keeps its order on 100% of test descriptions, for all items.
  - PR: the lowest share for any pair is 0.94.
- **Transfer check (exploratory):** the it04 PR matrix rows are re-scored with Qwen3-4B in bf16 (stage `transfer`, work directory `/fs/clip-scratch/pramu/geo_loop_qwen`).
- **Matrix cap raised from 150 to 250 rows per boundary,** after the BV matrix hit 150 of its 234 flip descriptions (98 of the 150 from the test split). The extended matrix uses the same order (test split first, then largest gap), so its first 150 rows are the same descriptions as before, and their scores come from the cache.
- **Qwen3-4B transfer, PR (bf16, exploratory):** the 68 it04 PR matrix rows (Llama flip descriptions) were re-scored with Qwen.
  - Only 1 row still flips, label agreement is 0.38, and country deviations point the same way as Llama's in 54% of cases.
  - Qwen sits far from the boundary on these descriptions: mean s = −12.6 against −0.27 for Llama.
  - Its country effects within a row are larger, though (median country SD 1.57 against 0.56), and its average country deviations correlate with Llama's at r = +0.34 across the 37 countries. China is the most Riot-leaning country in both models, and Colombia is among the most Protest-leaning in both.
  - **Conclusion:** flips are model-specific, because the ambiguity point differs between models. The country effect is partly shared. A Qwen dataset needs its own V3 (Qwen-ambiguous descriptions) and, for exact scores, fp32.
- **Qwen3-4B transfer, BV (bf16, exploratory):** the first 60 it05 BV matrix rows (test split first) were re-scored with Qwen.
  - No row still flips, and label agreement is 0.16. Qwen sits far on the Battle side: mean s = +12.6 against −0.24 for Llama.
  - Its within-row country spread is about the same as Llama's (median SD 0.62 against 0.59), but its country ordering is mostly different (r = +0.13 across 37 countries).
  - Qwen's VAC-leaning countries are Sweden, Yemen, Australia and the UK; its Battle-leaning ones are Russia, the US, Iran and China. Only Russia and the US share a direction with Llama.
  - **Conclusion:** the BV country axis is largely specific to Llama. Qwen has country effects of similar size but organizes them differently, and on these descriptions it is far from its own boundary.

## v4.1 proposal (2026-11-07)

**Not an iteration of the inputs yet; no inputs changed.** An evidence-based construction plan is written in [../CONSTRUCTION.md](../CONSTRUCTION.md). It maps each likely objection to the evidence that answers it, and adds:
- a precision audit of the ACLED lexicon,
- frames derived from the structure of ACLED notes,
- human labels for the items and severity ratings for intensity,
- a country sampling rule,
- an intensity axis that is irrelevant to the class by definition, crossed with the class-evidence ladder.

Exploratory work so far (pilot bases only) is an order-only flip sweep and an ACLED fatality scan (CONSTRUCTION.md § 7).
