# Geographic Invariance in LLM Decision-Making

This project tests whether wording and event details affect LLM conflict-event classification consistently across countries. It extends [earlier conflict-monitoring research](https://github.com/hoffmann-muki/llm-conflict-monitoring-eval) through controlled edits to real reports and country substitutions in matched hypothetical descriptions. See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the study design.

## Current status

The accepted inputs are **590 main events and 28 pilot events**, following adversarial AI review and corrective action. The [pilot dataset](data/accepted/pilot.csv) is preserved byte-for-byte. These inputs are ready for variant construction; edited descriptions and model inference are not implemented.

All 600 originally allocated main records and 155 reserves have terminal AI review decisions. Independent challenges left eight main and three reserve exclusions. Seven reviewed reserves replace rejected main inputs. A shortage of suitable US riot descriptions requires an equal reduction from 16 to 15 riots per country, preserving shared length targets and yielding 590 main inputs. There are 144 reviewed, unused reserves available; CHN15848 remains quarantined for similarity to a pilot event.

The [input review report](data/accepted/report.md) records the decisions, corrections, and limitations. No human review or sign-off is required. AI review is not historical verification, and model diversity is not established. Case-specific qualifications remain binding.

## Files

| Location | Purpose |
| --- | --- |
| `conflict_data/` | Five unchanged ACLED source CSVs |
| `prepare_data.py` | Source validation, grouping, sampling, and built-in study settings |
| `data/processed/` | Current candidates, sampling pool, exclusions, family links, balance tables, quality report, and checksum manifest |
| `data/reviews/` | Pilot evidence and corrective decisions in `input_review.json`; all 755 main/reserve assessments in `sample_review.csv` |
| `finalize_inputs.py` | Review validation, corrective curation, and accepted-input export |
| `data/accepted/` | Accepted main and pilot inputs, pilot edit eligibility, cohort changes, active reserves, report, and checksum manifest |
| `tests/` | Input-integrity, sampling, review, and similarity regression tests |

Use `data/accepted/main.csv` and `pilot.csv` for experiments, and `available_reserves.csv` for replacements. The preprocessing tables preserve the original allocation and excluded records for audit; the sampling pool is unselected source inventory. No raw notes or source labels were changed.

## Reproduction

Python 3.10 or newer is sufficient; no additional packages are required. Existing output directories are never overwritten. To regenerate both stages, choose unused destinations:

```bash
python3 prepare_data.py --output-dir data/processed_new
python3 finalize_inputs.py --source-dir data/processed_new --output-dir data/accepted_new
python3 -m unittest discover -s tests
```

The default destinations are `data/processed/` and `data/accepted/`. Initial allocation settings are defined in `DEFAULT_CONFIG` in `prepare_data.py` and copied into the manifest; `--config` optionally accepts a JSON override. The review adjudications specify accepted quotas and corrections. Changed settings or source data require renewed AI review before acceptance.

Preprocessing preserves notes exactly, restricts primary comparisons to shared subtypes, matches description-length bands, and selects one representative per detected text family. A seven-day, within-country audit checks selected roles independently and triggers resampling when necessary. The accepted pilot additionally undergoes an all-date, all-country similarity screen in both directions against main candidates, active reserves, and itself. Both stages record checksums and use comparison-only text normalization.

Review applies the [ACLED codebook](https://acleddata.com/methodology/acled-codebook) to the focal event. Pilot review used notes-only, label-visible semantic, and methods contexts. Main and reserve records received label-visible adversarial review in three contexts; all proposed exclusions were independently challenged and adjudicated. The review records preserve evidence and restrictions for accepted and rejected candidates. It assesses textual support, not historical truth. Edit eligibility permits drafting only; exact original/edited pairs require AI review before inference. Report sensitivity analyses for metadata-dependent or disputed labels rather than treating review agreement as model accuracy.
