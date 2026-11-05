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

**A — Real events:** 590 descriptions: 20 peaceful protests and 15 violent demonstrations per country, plus 20 armed clashes and 20 civilian attacks in each starred country. Analyze each source class separately. Preserve original notes, sample with a fixed seed, select one representative per detected text family, and match length-band quotas and word-count targets across countries. Keep pilot, main, and reserve families disjoint; report residual length, actor, and month differences.

**B — Matched descriptions:** 120 hypothetical descriptions, 20 per [ACLED event type](https://acleddata.com/methodology/acled-codebook), each rendered under all ten countries. Use generic actors and locations with one country placeholder; remove other geographic clues. Retain source identifiers and family links. Apply identical edits across country versions to isolate country-name and country-by-edit effects.

## Edits and Review

Each original receives up to seven independent conditions: neutral control, actor identity, legitimizing framing, delegitimizing framing, equivalent action wording, intensity, and synthetic source attribution. Use deterministic recipes and versioned phrase lists; do not combine interventions.

Change only the designated cue. Keep class-defining mechanics, armed/civilian roles, casualties, weapons, claim certainty, and event boundaries fixed. Framing and intensity vary characterization, not facts; source substitutions must not imply verification. Record changed spans, applicability, and decisions. Exclude unavailable conditions from their denominators.

Two separate adversarial AI contexts review every proposed real-event edit and every synthetic original/edited pair. Release only current versions accepted by both reviewers after adjudication; withhold unresolved objections and re-review corrections. Follow the existing [input restrictions](data/accepted/report.md); input acceptance does not approve variants.

## Models and Analysis

Use two models: [Multilingual E5 Large Instruct](https://deepinfra.com/intfloat/multilingual-e5-large-instruct) through DeepInfra as the encoder, and [Llama 3.1 8B Instruct](https://openrouter.ai/meta-llama/llama-3.1-8b-instruct) through OpenRouter as the decoder. The [adopted protocol](data/experiment/report.md) freezes E5's short instruction and eighteen country-neutral reference examples. Normalize each reference vector, average three per class, normalize each class mean, and classify by cosine similarity. Keep model weights frozen and reject inputs exceeding 512 tokens. Llama uses the shared class definitions in a zero-shot prompt with structured output. Exclude source labels and identifiers from event inputs. Randomize request order and retain responses, predictions, settings, errors, usage, and timing; repeat a fixed subset to measure instability.

The primary endpoint is **excess flips**: whether an edit changes the original prediction, minus whether the neutral rewrite changes it. Contrast country means within each source class in A and within matched base descriptions in B. Include all accepted inputs regardless of baseline correctness. A positive contrast means greater disruption beyond neutral rewriting, not necessarily lower accuracy.

The frozen [analysis registry](data/analysis_plan.json) specifies **438 primary contrasts per model**: 168 real (123 P, 36 B, 9 V) and 270 synthetic. Require ten independent families per country or matched pair, before and after response failures; all real Riots and actor-identity country contrasts remain descriptive. Use approximate two-sided Welch tests in A and paired t-tests in B, with Holm adjustment over all 438 planned tests within each model. Keep unavailable tests in that correction family; tests with zero contrast variance are unavailable. These exploratory tests have limited power and small-sample approximations.

Estimate nominal 95% intervals using 10,000 family-bootstrap replicates, seed 42: resample within country/class in A and source class in B, preserving related versions together. Analyze common valid original/edit/neutral triples, matched across countries in B. Report prepared, attempted and valid denominators and invalid-response rates; differential failures can bias estimates. Secondary analyses cover baseline accuracy, label transitions, country-only disagreements, originally correct baselines and exclusion of qualified A labels. Repeats measure stability without increasing sample size.

Natural-report comparisons remain observational. Balanced samples do not estimate operational accuracy or represent whole regions. Prediction changes measure behavioral sensitivity; nonsignificant differences do not establish invariance.

## Execution

The [main data release](data/variants/main/report.md) contains 10,820 stimuli, literal edit spans, eligibility denominators and a seeded request order: **21,646 classifications**, including three extra repeats per model, plus eighteen E5 reference embeddings. Use [data/experiment/protocol.json](data/experiment/protocol.json) for current settings and its updated length audit; source-release settings remain historical. The user authorized the full run on 4 November 2026 with a $3 accounting limit. Execute first attempts in saved order with at most eight concurrent requests; journal retries, missing responses, usage and timing. Retired pilot artifacts are replaced by [source-family exclusions](data/reviews/pilot_exclusions.csv) and accepted original notes.

The [compatibility check](runs/compatibility/report.md) passed. The [E5 validation](runs/e5_validation/report.md) achieved 23/30 held-out classifications with the selected method. The user accepted this performance for proceeding; the historical screening failure and V/S scores of 2/5 remain reported. Freeze the adopted method, retain all accepted main inputs, and interpret geographic effects alongside class-specific accuracy and the small validation sample.

The completed [inference run](runs/main/report.md) and [analysis](results/main/report.md) retain prediction logs, paired estimates and uncertainty, country-by-edit summaries, country-swap disagreements and label transitions. Source IDs link every result back to the reviewed original/edited texts and literal changed spans.
