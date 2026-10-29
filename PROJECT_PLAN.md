# Study Design

Test whether controlled edits affect conflict-event classification differently across countries and whether country names alone change predictions. Assess geographic sensitivity within each model. Build on [earlier West African research](https://arxiv.org/html/2605.04177v1) and its [code](https://github.com/hoffmann-muki/llm-conflict-monitoring-eval), using an independent implementation.

## Data and Experiments

Use ACLED event notes from ten countries. Starred countries also contribute battles and violence-against-civilians events to Experiment A.

| Region | Countries |
| --- | --- |
| Northern Africa | Libya*, Morocco |
| North America | Mexico*, United States |
| South America | Brazil*, Colombia* |
| South Asia | India*, Pakistan* |
| East Asia | China, South Korea |

**A — Real events:** 590 descriptions: 20 peaceful protests and 15 violent demonstrations per country, plus 20 armed clashes and 20 civilian attacks in each starred country. Analyze protest/riot and battle/civilian-violence samples separately. Preserve original notes, sample with a fixed seed, select one representative per detected text family, and match length-band quotas and word-count targets across countries. Keep pilot, main, and reserve families disjoint; report residual length, actor, and month differences.

**B — Matched descriptions:** 120 hypothetical descriptions, 20 per [ACLED event type](https://acleddata.com/methodology/acled-codebook), each rendered under all ten countries. Use generic actors and locations with one country placeholder; remove other geographic clues. Retain source identifiers and family links. Apply identical edits across country versions to isolate country-name and country-by-edit effects.

## Edits and Review

Each original receives up to seven independent conditions: neutral control, actor identity, legitimizing framing, delegitimizing framing, equivalent action wording, intensity, and synthetic source attribution. Use deterministic recipes and versioned phrase lists; do not combine interventions.

Change only the designated cue. Keep class-defining mechanics, armed/civilian roles, casualties, weapons, claim certainty, and event boundaries fixed. Framing and intensity vary characterization, not facts; source substitutions must not imply verification. Record changed spans, applicability, and decisions. Exclude unavailable conditions from their denominators.

Use separate adversarial AI review contexts and adjudicate disagreements. Review every pilot and synthetic original/edited pair, every uncertain natural-event pair, and a stratified sample of remaining natural-event pairs. Follow the existing [input restrictions](data/accepted/report.md); input acceptance does not approve variants.

## Models and Analysis

Use two models: [Multilingual E5 Large Instruct](https://deepinfra.com/intfloat/multilingual-e5-large-instruct) through DeepInfra as the encoder, and [Llama 3.1 8B Instruct](https://openrouter.ai/meta-llama/llama-3.1-8b-instruct) through OpenRouter as the decoder. Freeze routing and settings before main inference. For E5, use a fixed task instruction and normalized cosine similarity against six frozen, country-neutral class descriptions; reject inputs exceeding its 512-token limit rather than truncate. For Llama, use the same class definitions in a zero-shot classification prompt with structured output. Exclude source labels and identifiers from model inputs. Randomize request order and retain responses, predictions, settings, errors, usage, and timing; repeat a fixed subset to measure instability.

The primary outcome is the **country-by-edit interaction**. Measure baseline accuracy against reviewed labels, edit flip rates, excess flips relative to neutral controls on common eligible pairs, directional label transitions, and country-only disagreement across all 45 country pairs. Include all accepted inputs regardless of initial model correctness; use the initially correct subset only as a secondary analysis.

Prespecify country contrasts and multiplicity adjustment. Estimate 95% bootstrap intervals by source-event families in A and base-description families in B, keeping related countries and edits together. Analyze each model separately, report invalid responses separately, and test sensitivity to qualified or disputed labels. Report countries individually and use equal country weights for regional summaries.

Natural-report comparisons remain observational. Balanced samples do not estimate operational accuracy or represent whole regions. Prediction changes measure behavioral sensitivity; nonsignificant differences do not establish invariance.

## Execution

The reviewed stimulus pool contains 28 real events and 12 synthetic descriptions under four countries. An earlier DeepSeek/Llama inference pilot passed 150 technical checks; its outputs were removed after validation. Current work stops at data preparation: validate model input formats and lengths without sending inference requests. Construct and review the main variants and 120 synthetic bases before releasing the full stimulus set. The main-study upper bound is **28,640 classifications**, excluding repeats, retries, and E5 class-description embeddings; inapplicable edits reduce it.

Deliver reproducible prediction logs, paired effect estimates with uncertainty, a country-by-edit heatmap, country-swap disagreements, and representative original/edited pairs.
