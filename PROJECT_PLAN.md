# Geographic Invariance in LLM Decision-Making

## Objective

This study examines whether the features that influence LLM conflict-event classification have consistent effects across geographic contexts. It measures how controlled changes to actor identity, evaluative framing, action wording, intensity, and source attribution affect predictions, and whether those effects depend on the country.

The project extends the geographic scope of [Are LLMs Ready for Conflict Monitoring? Empirical Evidence from West Africa](https://arxiv.org/html/2605.04177v1) and its [companion repository](https://github.com/hoffmann-muki/llm-conflict-monitoring-eval). It will use an independent implementation and a protocol developed for the following questions:

1. Does the same edit produce different classification effects across countries?
2. Does changing only the country name alter the classification of an otherwise identical event?
3. How do these effects differ between open-weight models and frontier models accessed through APIs?

## Data and Geographic Scope

The five CSV files in `conflict_data/` contain **137,947 events** dated **12 October 2024 through 10 October 2025**, with a common 31-column schema. Event IDs are unique; nine groups of records have identical notes. Country, event type, and description are present for every record.

The proposed selection covers two countries in each of five regions:

| Region | Countries | Countries eligible for the additional battle/civilian-violence sample |
| --- | --- | --- |
| Northern Africa | Libya, Morocco | Libya |
| North America | Mexico, United States | Mexico |
| South America | Brazil, Colombia | Both |
| South Asia | India, Pakistan | Both |
| East Asia | China, South Korea | Neither |

Region assignments will use the `country` and `region` columns. The file named `north_african_conflict_data.csv` includes 3,541 records assigned to Caucasus and Central Asia. Event availability also varies: East Asia has no battles, and South Korea has only 29 riots. These constraints require comparisons within shared event classes rather than an identical six-class natural sample for every country.

## Experimental Design

### Experiment A: Controlled Edits to Real Events

Sample **20 protests and 20 riots per country**, yielding 400 events. Add **20 battles and 20 violence-against-civilians events** from each of the six eligible countries, yielding another 240. The target is **640 real events**, analyzed separately as protest/riot and battle/civilian-violence samples.

Sampling will use a fixed seed and no replacement. Remove duplicate notes and limit near-identical reports from the same episode. Use sub-event type, reporting month, and description length to improve comparability, and document actor composition. Reviewers will confirm that each description supports its recorded label before model evaluation. If a shared class cannot meet its quota after review, reduce that class quota consistently across the relevant countries.

Use the event notes as model inputs and retain country and label metadata for analysis. Apply each eligible edit independently to the original description. Include all accepted events in the primary analysis, regardless of whether a model initially classifies them correctly. This experiment tests sensitivity in realistic descriptions; remaining differences in reporting and event content limit causal interpretations of cross-country comparisons.

### Experiment B: Country and Edit Effects on Matched Descriptions

Construct **120 short descriptions**, 20 for each of the six event types: violence against civilians, battles, explosions/remote violence, protests, riots, and strategic developments. Derive them from diverse real examples and retain source identifiers for audit and analysis.

Each description will contain one country placeholder and generic actors and locations. Remove other geographic clues, including demonyms, named organizations, currencies, and source names. Two reviewers will approve the label, plausibility across the selected countries, and all applicable edits.

Render every accepted description under all ten countries, with the same edits in each version. The resulting **description × country × edit × model** design isolates country-name effects and tests whether country identity changes the effect of an edit. For example, a description of two organized armed groups exchanging gunfire can receive the same framing change under each country name.

These are hypothetical stimuli with reviewed labels. Their results establish sensitivity for the tested descriptions, while Experiment A assesses whether similar patterns appear in natural event reports.

## Perturbations and Validation

Each input will have an original condition and up to seven edited conditions. Edits will use deterministic rules and a fixed, reviewed phrase list; multiple interventions will not be combined.

| Condition | Intervention | Required preservation |
| --- | --- | --- |
| Original | No edit | Reference description |
| Neutral control | Equivalent wording, such as `two unarmed civilians` → `two civilians who were unarmed` | Facts, evaluation, and source status |
| Actor identity | State/non-state identity or actor designation | Armed/civilian roles, action, and target |
| Legitimizing framing | Reported justification or favorable characterization | Event mechanics and participants |
| Delegitimizing framing | Unfavorable characterization | Event mechanics and participants |
| Action wording | Equivalent verb or phrase | Action and outcome |
| Intensity | Evaluative intensity language | Force mechanism, casualties, and civilian involvement |
| Source attribution | Provenance language | Reported event facts |

All edits must preserve the event-type criteria. Changes to identity or source framing are intentional interventions; changes to casualties, civilian targeting, or whether an action occurred are outside this protocol. Review all constructed-description edits, all uncertain natural-event edits, and a stratified sample of the remaining edits. Record changed spans, applicability, and review decisions. Exclude edits with uncertain labels and treat unavailable interventions as missing conditions.

## Models and Inference

Evaluate **four models**: two open-weight models from different families compatible with available compute, and two frontier API models from different providers. Finalize model versions after the pilot and use the same accepted inputs for every model.

Develop one concise zero-shot prompt with explicit event-type definitions and structured label output. Instructions will remain constant across conditions, and dataset labels and identifiers will be excluded from prompts. Use fresh requests, randomized input order, minimal sampling where supported, and fixed settings. Document provider-specific formatting requirements.

Cache completed requests and store raw responses, parsed labels, model and prompt versions, settings, errors, usage, and timestamps. Repeat a fixed subset of identical inputs to estimate response instability. Labels are sufficient for the primary outcomes; self-reported confidence is not required.

## Analysis

| Measure | Purpose |
| --- | --- |
| Baseline accuracy and confusion matrices | Establish performance on each reviewed sample |
| Edit flip rate | Measure the proportion of predictions changed by an intervention |
| Excess flip rate | Compare treatment and neutral-control flip rates on the same eligible events |
| Directional effects | Record label transitions and percentage-point changes in accuracy and predicted-label frequencies |
| Country-only disagreement | Measure baseline disagreement across the 45 country pairs for each constructed description |
| Country-by-edit interaction | Compare the change induced by the same edit across country versions of the same description |

The interaction analysis is the primary test of geographic invariance in feature sensitivity. Prespecify country contrasts and adjust for multiple comparisons. Track battle/civilian-violence and protest/riot transitions explicitly; treat analyses with limited class support as exploratory.

Estimate 95% bootstrap intervals by resampling source events in Experiment A and base-description families in Experiment B. Keep related countries and edits together, grouping descriptions derived from the same source event. Compare models on common valid pairs and report invalid-response rates separately. An initially correct subset will provide a secondary sensitivity analysis.

Report countries individually and summarize regions with equal country weights. Balanced samples do not estimate countrywide operational accuracy, and two countries do not represent an entire region. Observable prediction changes characterize behavioral sensitivity rather than internal reasoning or legal legitimacy. Small effects with wide intervals are inconclusive; nonsignificant differences do not establish invariance.

## Implementation and Workload

A small Python pipeline will contain four scripts:

- `prepare_data.py`: inventory, deduplication, sampling, and sample manifest.
- `build_variants.py`: descriptions, country renderings, edits, and validation records.
- `run_models.py`: local/API inference, caching, retries, and prediction logs.
- `analyze.py`: paired metrics, confidence intervals, tables, and figures.

Use one experiment configuration, frozen input tables, a common prediction format, and a report directory. At eight conditions per input, the main study has the following upper bound:

| Experiment | Calculation | Classifications |
| --- | --- | ---: |
| Real events | 640 events × 8 conditions × 4 models | 20,480 |
| Matched descriptions | 120 descriptions × 10 countries × 8 conditions × 4 models | 38,400 |
| Total | Before repeated-input checks and retries | **58,880** |

Inapplicable edits reduce the workload. A pilot using 12 descriptions, four countries, two models, and a small real-event sample will establish edit validity, parsing reliability, runtime, and API cost. Use separate pilot examples and freeze the protocol before the main run. If necessary, reduce sample sizes consistently while retaining the paired design and reporting the resulting uncertainty.

## Schedule and Deliverables

| Stage | Work | Deliverable |
| --- | --- | --- |
| 1. Data and protocol | Confirm countries, quotas, label definitions, and review criteria | Sample manifest and reviewed candidate inputs |
| 2. Pilot | Validate interventions and both inference backends | Measured workload and frozen protocol |
| 3. Main experiments | Evaluate all four models | Complete prediction logs and documented exclusions |
| 4. Analysis and reporting | Estimate paired effects and review representative changes | Reproducible results and research report |

The final report will include a country-by-edit heatmap with uncertainty and sample counts, a country-swap disagreement figure, directional-transition tables, and representative original/edited descriptions.
