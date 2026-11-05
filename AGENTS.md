# Repository Guidance

Use [README.md](README.md) for entry points and commands, and [PROJECT_PLAN.md](PROJECT_PLAN.md) for the research design.

The interpretability exploration (`interpretability/`, `src/interp_*`, `tests/test_interp_*`) follows only [interpretability/PLAN.md](interpretability/PLAN.md). PROJECT_PLAN.md's review process, analysis registry and endpoints do not apply to it. The implementation and data-handling rules below still apply.

## Implementation

- Keep the pipeline simple: Python 3.10+, standard library where practical, and focused scripts. Add dependencies only for a concrete need.
- Keep preparation, experiment, analysis and evaluation code in `src/`, with `tests/` and data directories separate. Put repository-wide helpers in `src/utils.py`; keep domain-specific logic with its owning module.
- Format changed Python files with Black targeting Python 3.10. Keep formatting separate from behavioral changes when requested.
- Test input integrity, sampling, review decisions, and experiment invariants. Run the test suite before committing code changes; documentation-only changes need link and diff checks.
- Fail explicitly on invalid inputs. Do not silently relax quotas, eligibility rules, or sample separation.

## Professionalism

- Treat every script as maintained research code: use clear names, focused functions, explicit failure handling and consistent UTF-8 file I/O. Avoid one-off shortcuts and unnecessary abstractions.
- Generate report facts from validated artifacts. Keep fixed study parameters explicit; keep decision history in versioned data. Avoid long escaped narratives, duplicated calculations and hardcoded run outcomes in implementation code.
- Review changes for stale assumptions, duplicated logic and failure paths. Add regression tests for material fixes and preserve completed artifacts and their historical producer hashes during refactoring.

## Data and Review

- Preserve raw CSVs and original note strings. Normalize text only for comparisons. Treat existing generated snapshots as immutable; reproduce into unused directories.
- Use `data/accepted/` for experimental inputs. Keep `data/reviews/`: its assessments and adjudications are pipeline inputs, not disposable logs.
- Carry source IDs, family IDs, conditions, changed spans, and case restrictions into derived artifacts. Follow the review rules in the project plan before releasing variants for inference.
- Human review has been replaced by user-authorized adversarial AI review. Use separate review contexts, cite literal evidence, and adjudicate disagreements. Do not substitute keyword rules for semantic review or claim human verification.
- Record justified dataset corrections and update affected design figures. Generate checksums from actual artifacts; never change hashes or judgments merely to bypass validation. Existing manifests describe the code that produced their snapshots.

## Delivery

- Keep README concise, design decisions in the project plan, and audit details in generated reports. Link rather than repeat.
- Retain only necessary files. Use self-cleaning temporary directories and remove project scratch files without touching unrelated work. Never commit credentials.
- Keep commits scoped to the request, report validation, and verify repository status and remote synchronization after a requested push.
