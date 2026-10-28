# Study Design

Test whether controlled edits affect LLM conflict-event classification differently across countries, whether country names alone change predictions, and whether these effects differ between open-weight and API models. Build on [earlier West African research](https://arxiv.org/html/2605.04177v1) and its [code](https://github.com/hoffmann-muki/llm-conflict-monitoring-eval), using an independent implementation.

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

Evaluate four models: two open-weight models from different families and two API models from different providers. Freeze versions after the pilot. Use one zero-shot prompt with six-class definitions and structured output, fresh requests, randomized order, and minimal sampling. Exclude source labels and identifiers from prompts. Cache responses, predictions, settings, errors, usage, and timing; repeat a fixed subset to measure instability.

The primary outcome is the **country-by-edit interaction**. Measure baseline accuracy against reviewed labels, edit flip rates, excess flips relative to neutral controls on common eligible pairs, directional label transitions, and country-only disagreement across all 45 country pairs. Include all accepted inputs regardless of initial model correctness; use the initially correct subset only as a secondary analysis.

Prespecify country contrasts and multiplicity adjustment. Estimate 95% bootstrap intervals by source-event families in A and base-description families in B, keeping related countries and edits together. Compare models on common valid pairs, report invalid responses separately, and test sensitivity to qualified or disputed labels. Report countries individually and use equal country weights for regional summaries.

Natural-report comparisons remain observational. Balanced samples do not estimate operational accuracy or represent whole regions. Prediction changes measure behavioral sensitivity; nonsignificant differences do not establish invariance.

## Execution

Run two models on the 28-event real pilot and 12 synthetic descriptions rendered under four countries. Validate edits, parsing, response stability, runtime, and cost; freeze the protocol before the main run. The upper bound is **57,280 classifications**, excluding repeats and retries; inapplicable edits reduce it.

Deliver reproducible prediction logs, paired effect estimates with uncertainty, a country-by-edit heatmap, country-swap disagreements, and representative original/edited pairs.
