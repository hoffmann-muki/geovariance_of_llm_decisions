# Geographic Invariance in LLM Decision-Making

This project tests whether wording and event details affect LLM conflict-event classification consistently across countries. It extends [earlier conflict-monitoring research](https://github.com/hoffmann-muki/llm-conflict-monitoring-eval) through controlled edits to real reports and country substitutions in matched hypothetical descriptions. See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the study design.

## Current status

Preprocessing is complete: 600 main candidates, 28 original pilot candidates, and 155 reserves. The [accepted pilot](data/pilot/pilot.csv) contains 28 source events following adversarial AI review and nine replacements. It is ready for variant construction. The main sample remains pending review; edited descriptions and model inference are not implemented.

The [pilot report](data/pilot/report.md) records acceptance, edit eligibility, and limitations. Human review was not performed. Separate agent contexts do not establish model diversity, and AI judgments used in selection can bias later model comparisons.

## Files

| Location | Purpose |
| --- | --- |
| `conflict_data/` | Five unchanged ACLED source CSVs |
| `prepare_data.py` | Source validation, grouping, sampling, and built-in study settings |
| `data/processed/` | Current candidates, sampling pool, exclusions, family links, balance tables, quality report, and checksum manifest |
| `data/reviews/pilot_review.json` | Consolidated review evidence, adjudications, per-case edit restrictions, and accepted selections |
| `finalize_pilot.py` | Source/review validation and accepted-pilot export |
| `data/pilot/` | Accepted inputs, edit eligibility, cohort changes, active reserves, report, and checksum manifest |
| `tests/` | Input-integrity, sampling, review, and similarity regression tests |

Use `data/pilot/available_reserves.csv` for current reserve eligibility: 154 records remain available pending review. CHN15848 is quarantined because it exceeds 0.90 similarity to accepted pilot CHN16073, 48 days apart. The preprocessing reserve table records the original allocation.

## Reproduction

Python 3.10 or newer is sufficient; no additional packages are required. Existing output directories are never overwritten. To regenerate both stages, choose unused destinations:

```bash
python3 prepare_data.py --output-dir data/processed_new
python3 finalize_pilot.py --source-dir data/processed_new --output-dir data/pilot_new
python3 -m unittest discover -s tests
```

The default destinations are `data/processed/` and `data/pilot/`. Sampling settings are defined in `DEFAULT_CONFIG` in `prepare_data.py` and copied into the manifest; `--config` optionally accepts a JSON override. Changed settings or source data require a new review before pilot acceptance.

Preprocessing preserves notes exactly, restricts primary comparisons to shared subtypes, matches description-length bands, and selects one representative per detected text family. A seven-day, within-country audit checks selected roles independently and triggers resampling when necessary. The final pilot additionally undergoes an all-date, all-country similarity screen in both directions against main candidates, active reserves, and itself. Both stages record checksums and use comparison-only text normalization.

Review applies the [ACLED codebook](https://acleddata.com/methodology/acled-codebook) to the focal event. Notes-only, label-visible semantic, and methods reviewers worked in separate contexts; disagreements were adjudicated. The consolidated record preserves evidence and restrictions for accepted and rejected candidates. It assesses textual support, not historical truth. Edit eligibility permits drafting only; every exact pilot original/edited pair requires review before inference.
