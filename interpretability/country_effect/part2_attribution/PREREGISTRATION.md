# Part 2 Pre-Registration: Where Country Names Enter the Decision

Written 21 November 2026, before any Part 2 prompt was scored. It fixes the stimuli, subsets, metrics and decision rules for the hypotheses in AGENT_SPEC.md § 2 (removed from the tree; see commit 08ed146). Later changes are logged in § 9 with their reason; the original text stays.

Throughout: model `meta-llama/Llama-3.2-3B-Instruct` at revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`, bf16, one prompt per forward pass, the Part 1 system prompts (`part1_behavioral/inputs/prompts.json`), both option orders. The score of a prompt pair is s = (d_A + d_B) / 2 with d = log P(higher letter) − log P(lower letter). A **country effect** is Δ_ic = s_ic − mean of s over the 20 Part 1 placebos of the same description, and a name's effect is the mean of Δ over descriptions (as in Part 1). Bootstrap intervals resample descriptions (2,000 draws, seed 20261121).

## 1. Splits

"Context index" is the position of a context value in the v4 value pools (`geoflip/materials.py`).

| Split | Indices | Use in Part 2 |
| --- | --- | --- |
| Pilot | 0–7 | fitting the armed-status probe and the steering direction (§ 6) |
| Study | 8–11 (the Part 1 dataset) | every Stage 1A and Stage 2 experiment |
| Held out | 12–19 | Stage 1B only, run once, not inspected before the Stage 1B rule is applied |

BV times are generic and shared across splits (as in Part 1); only places define the BV split. New pilot and held-out descriptions use the Part 1 frames, wordings, items and rendering.

- **Pilot BV contexts (8):** place i (i = 0–7) with time index (0, 3, 6, 9)[i mod 4].
- **Held-out contexts (8 per boundary):** BV place 12 + i with time index (0, 3, 6, 9)[i mod 4]; PR actors 12 + i, grievance 12 + (3i mod 8), place 12 + (5i mod 8), i = 0–7 (every value once).

## 2. Stage 0: Parity (00_verify_parity.py)

Every later script refuses to run unless `outputs/00_parity/manifest.json` reports all checks passed.

1. `pytest interpretability/geoflip/tests/test_part2_invariants.py` passes (span resolution, round trips, token-length matching, readout token, single-token answers).
2. **Part 1 reproduction:** 100 prompts (50 study-split description × name pairs drawn with seed 20261121, stratified by boundary, both orders) re-scored with the Part 2 hooked runner reproduce the Part 1 `d_A`, `d_B` to the 4 stored decimals (|difference| ≤ 5e-5).
3. **Hook transparency:** a forward pass with capture hooks on all layers gives logits identical to one without hooks; patching a prompt with its own cached activations reproduces its d exactly.
4. **`inputs_embeds` parity:** `model(inputs_embeds=embed_tokens(ids))` reproduces `model(input_ids=ids)` logits exactly.
5. A single-prompt walkthrough (rendered prompt, every span with its decoded text, d in both orders, logit-lens d by layer) is written to `outputs/00_parity/walkthrough.txt`.

## 3. Stage 1A: Token-Matched and Familiarity Controls (01, 02)

Control names are defined in `configs/stimuli_controls.json` by fixed rules, not chosen on model scores. Families:

| Family | Rule | Role |
| --- | --- | --- |
| `nonplace_1tok` | mythic or fictional places and planets whose `" " + name` is one Llama token | 1-token non-country names (the token-matched null) |
| `small_state_1tok` | sovereign states with population under 5 million whose name is one token | real but rarely mentioned countries |
| `small_state_multi` | 8 states with population under 5 million and 2–4 tokens, drawn with seed 20261121 | familiarity at placebo-like length |
| `region_3tok` | "the {Northern, Central, …} {Province, Region, …}" phrases of exactly three tokens | article + three tokens without a country (matches "the United States") |

Every control is scored on all study-split descriptions of both boundaries (1,536 BV, 1,984 PR) in both orders. Δ uses the Part 1 placebo mean of the same description.

**Statistics.** For each family and for the 32 one-token real countries: the SD across names of the name effects (BV, PR); flip rates as in Part 1; Spearman correlation of name effect with token id (BPE merge rank, a frequency proxy) over all one-token names; correlation with token count over all names.

**Decision rules (BV).**
- **H3 falsified** if `nonplace_1tok` SD ≤ 0.08 while the one-token real SD ≥ 0.18 (AGENT_SPEC § 2).
- **H3 supported** if `nonplace_1tok` SD ≥ 0.18, or the `region_3tok` mean effect is ≥ 0.30 (half the US effect) toward Battle.
- Otherwise inconclusive at the behavioural level.
- `small_state_*` families are reported descriptively: H1 predicts their effects follow region (as do the 37), H3 predicts they sit with the placebos.

## 4. Stage 1B: Held-Out Replication (03)

All held-out descriptions (24 BV items × 2 frames × 2 wordings × 8 contexts = 768; PR 31 × 2 × 2 × 8 = 992) × the 58 Part 1 names × 2 orders, scored once.

**Rule (H1 behavioural):** the correlation over the 37 real countries between study-split and held-out BV effects is > 0.90. Also reported: the same for PR, real and placebo SDs, flip rates, the US share of BV flips.

## 5. Stage 2 Panel (configs/mechanistic_panel.json)

Fixed before any Stage 2 run. Items are chosen by evidence role, not by Part 2 scores.

- **BV items (8):** clear VAC `anchor_lo`, `unarmed`, `students`; underspecified `security_guard`, `miners`, `drivers`; clear Battle `returned_fire`, `anchor_hi`.
- **PR items (8):** `anchor_lo`, `block_road`, `occupy`, `scuffle`, `barricade`, `set_off_fireworks`, `throw_stones`, `anchor_hi`.
- **Contexts:** the four diagonal study contexts (`c00`, `c11`, `c22`, `c33`: every place and time once). Both frames and wordings. 128 descriptions per boundary.
- **Countries (BV):** High-Battle `Canada`, `Norway`, `Australia` (1 token), `the United States`, `the United Kingdom` (3 tokens); High-VAC `Bangladesh`, `Nigeria`, `Sudan` (1 token). **PR:** `Ukraine`, `Australia`.
- **Baselines:** three one-token baselines drawn from `nonplace_1tok` and three three-token baselines from `region_3tok` with seed 20261121. A country is paired with baseline (description index mod 3) of its length, so pairs are token-length matched and baselines are balanced over descriptions.
- **Spans** (resolved from character offsets, `geoflip/spans.py`): `name`; `ev_group` (the target-group phrase, listed per item and wording in the panel config); `ev_rest` (the rest of the evidence sentence, whose last token, the period, is `ev_end`); `bridge` (every other token after the name: the frame remainder and the chat-template tokens before the read-out); `readout`. `bridge` is added to the four spans in the spec so that the spans cover every position the name can affect.

## 6. Stage 2 Experiments and Decision Rules

Each experiment is run for every panel pair in both orders, and its result is order-averaged.

**Logit lens (04).** d at the read-out from every layer through the final norm and unembedding; Δ^(l) = s^(l)(country) − s^(l)(baseline). Reported: the first layer at which Δ^(l) reaches half its final value, by item.

**Activation patching (05).** Denoising: run the baseline prompt and replace the residual output of layer l at span S with that of the country prompt. Spans `name`, `ev_group`, `ev_rest`, `ev_all` (both evidence spans), `bridge` at l = 0–26, `readout` at l = 0–27. Transferred fraction F_S(l) = mean over descriptions of (s_patched − s_base) / mean of (s_country − s_base), by country group (High-Battle, High-VAC) and item.

- **H1 supported** if mean over l = 18–26 of F_readout(l) ≥ 0.5 and max over l of F_ev_all(l) < 0.2, and the total Δ on `anchor_lo` is at least that on `security_guard` (same sign, |Δ_anchor_lo| ≥ |Δ_security_guard|).
- **H1 falsified** if F_readout(l) < 0.5 for every l ≤ 26 while F_ev_all(l) ≥ 0.5 for some l (patching the read-out works only once evidence tokens carry the effect), or if Stage 1A falsification of H1 holds (`nonplace_1tok` SD ≥ the one-token real SD).
- **H2 supported (patching)** if max over l of F_ev_group(l) ≥ 0.3 with a bootstrap interval excluding 0, and the underspecified items have larger F_ev_group than the clear items.

**Armed-status probe (06).** On pilot-split BV `none` prompts (both orders, averaged), the representation is the residual at layer l at the last token of the evidence sentence (its period, `ev_end`). This is the first position at which the whole evidence has been read: in many items the group phrase comes before the words that settle its armed status ("The other group carried no weapons"), so a causal model cannot encode armed status at `ev_group` itself. Items are labelled armed (`armed_residents`, `rifle`, `pistol`, `shotgun`, `grenade`, `returned_fire`, `militia`, `anchor_hi`) or unarmed (`anchor_lo`, `unarmed`, `farmers`, `students`, `fishermen`, `villagers`, `traders`, `drivers`, `miners`); the other items are not used to fit. v_armed(l) is the difference of class means (a mass-mean probe), unit-normalised. The probe layer is the layer with the best two-fold (by context) pilot accuracy, earliest on ties. *(Changed to folds by item; see § 9.)* Its scale is the pilot armed − unarmed gap g of the projection.

- Test on the study panel: for each pair, the projection shift p(country) − p(baseline) at `ev_end`, divided by g (primary). The same projection of the mean over `ev_group` is reported as secondary. **H2 mechanistic prediction:** High-Battle shift − High-VAC shift > 0 with a bootstrap interval excluding 0 and size ≥ 0.05 g. **H2 falsified** if the interval includes 0 or the size is < 0.05 g, or if evidence-span patching has near-zero effect (max F_ev_all < 0.1).

**Steering (06).**
- (a) *Armed axis:* add α · g · v_armed at `ev_end` (probe layer) to baseline prompts, α ∈ {−1, −0.5, 0.5, 1}. The slope (nats per g) times the country projection shift gives the share of Δ the armed axis can explain.
- (b) *Country direction:* u(l) = pilot mean of h_l at the read-out for High-Battle minus High-VAC one-token countries (pilot one-token baselines subtracted per description), at layers 8, 14, 18 and 22. Add α · u(l) to baseline study prompts at the read-out at layer l, α ∈ {−1, −0.5, 0.5, 1}. H1 predicts the induced shift is roughly constant across items (correlation of the induced item profile with the observed Δ item profile reported).

**Matched integrated gradients (07).** One-token countries (Canada, Norway, Australia, Bangladesh, Nigeria, Sudan; PR: Ukraine) against their one-token baseline, on panel contexts `c00` and `c22` (64 descriptions per boundary). Embeddings are interpolated E(α) = E_ctrl + α(E_real − E_ctrl), 50 midpoint steps α_k = (k + 0.5)/50; the model runs in bf16 and gradients are accumulated in float32. Attribution to residual position t at layer l is Σ_k ∇_{h_l,t} d(α_k) · (h_l,t(α_k + 1/100) − h_l,t(α_k − 1/100)), which sums over positions to d(1) − d(0) at every layer (completeness error reported). Orders are attributed separately and averaged. Reported: the share of attribution by span and layer.

**PR as null control (04, 05, 07).** The same analyses on the PR panel. Mechanisms claimed for BV should differ in size in PR in proportion to the much smaller PR country effects.

## 7. Generality (08)

`Qwen/Qwen3-4B` (revision in `geoflip/scoring.py`, float32, thinking disabled) on the BV and PR panels with all 58 Part 1 names: country effects and their correlation with Llama's, and the logit-lens layer profile of Δ for the panel countries. Patching and IG are not run on Qwen (its tokenization differs, so the token-matched pairs would have to be rebuilt).

## 8. Adjudication

The report states for each hypothesis whether its behavioural and mechanistic predictions held, using the rules above, with the numbers behind each rule. A hypothesis is called *supported* only if every rule stated for it is met, *falsified* if a falsification rule is met, and *not resolved* otherwise.

## 9. Change Log

- **2026-11-21, before any Stage 2 run on the full panel: probe folds by item, not by context.** A code smoke test (pilot contexts 0 and 4, frame f1) gave 100% two-fold accuracy at layer 1 with folds by context. With context folds every test item also occurs in training, so a detector of item wording passes; the rule would pick an early lexical layer rather than an armed-status representation. Folds are now by item (alternate items of each class list in `configs/mechanistic_panel.json`), so accuracy measures generalisation to unseen items. The smoke test also ran scripts 04–07 on three study-panel pairs to check the code; those runs were written to scratch and are not used.
- **2026-11-21: order lengths.** AGENT_SPEC § 3.2 expects the two option orders to differ in token length; for both boundaries they are equal (PR 88, BV 85 system-prompt tokens). Spans are still resolved per prompt.
- **2026-11-21, after all runs: AGENT_SPEC.md removed.** The task spec this plan cites is no longer in the tree; it is preserved in git history at commit 08ed146. Nothing in the plan changed.
