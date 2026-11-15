# Repository Guidance

Use [README.md](README.md) for entry points and [interpretability/README.md](interpretability/README.md) for the project overview and commands. Follow the [study design](interpretability/country_effect/part1_behavioral/DESIGN.md) and [materials documentation](interpretability/MATERIALS.md). Proposed extensions are in [CONSTRUCTION.md](interpretability/CONSTRUCTION.md).

## Implementation

- Keep the pipeline simple: Python 3.10+, standard library where practical, and focused scripts. Add dependencies only for a concrete need.
- Keep shared code in `interpretability/geoflip/` and study-specific code in `interpretability/country_effect/`. Keep data and reports separate from code, and domain-specific logic with its owning module.
- Format changed Python files with Black targeting Python 3.10. Keep formatting separate from behavioral changes when requested.
- Verify input integrity, construction rules and experiment invariants. Run applicable tests and construction checks before committing code changes; documentation-only changes need link and diff checks.
- Fail explicitly on invalid inputs. Do not silently relax quotas, eligibility rules, or sample separation.

## Professionalism

- Treat every script as maintained research code: use clear names, focused functions, explicit failure handling and consistent UTF-8 file I/O. Avoid one-off shortcuts and unnecessary abstractions.
- Generate report facts from validated artifacts. Keep fixed study parameters explicit; keep decision history in versioned data. Avoid long escaped narratives, duplicated calculations and hardcoded run outcomes in implementation code.
- Review changes for stale assumptions, duplicated logic and failure paths. Add regression tests for material fixes and preserve completed artifacts and their historical producer hashes during refactoring.

## Data and Review

- Preserve raw CSVs and original note strings. Normalize text only for comparisons. Treat existing generated snapshots as immutable; reproduce into unused directories.
- Preserve `interpretability/inputs/` and the study's released inputs, datasets and results. Keep `conflict_data/` available for rebuilding the ACLED-derived materials.
- Carry description IDs, design factors, country conditions and provenance into derived artifacts. Follow the current study design's eligibility, review and measurement rules.
- Record the review method and evidence accurately. Do not claim human verification for AI assessments or substitute keyword checks for semantic review.
- Record justified dataset corrections and update affected design figures. Generate checksums from actual artifacts; never change hashes or judgments merely to bypass validation. Existing manifests describe the code that produced their snapshots.

## Delivery

- Keep README concise, design decisions in the study documentation, and audit details in generated reports. Link rather than repeat.
- Retain only necessary files. Use self-cleaning temporary directories and remove project scratch files without touching unrelated work. Never commit credentials.
- Keep commits scoped to the request, report validation, and verify repository status and remote synchronization after a requested push.
