# Agent Guidance

Start from [README.md](README.md) and [interpretability/README.md](interpretability/README.md). The study design is in [part1_behavioral/DESIGN.md](interpretability/country_effect/part1_behavioral/DESIGN.md), the materials in [MATERIALS.md](interpretability/MATERIALS.md), and the Part 2 analysis plan in [part2_attribution/PREREGISTRATION.md](interpretability/country_effect/part2_attribution/PREREGISTRATION.md).

## Code

- Python 3.10+, formatted with Black (target py310). Standard library where practical; add a dependency only for a concrete need.
- Shared code goes in `interpretability/geoflip/`, study code in `interpretability/country_effect/`, one script per stage. Keep data and reports out of code.
- Fail explicitly on invalid inputs. Never silently relax eligibility rules, quotas or sample separation.
- Run the applicable tests before committing code (`python -m pytest geoflip/tests` from `interpretability/`; Part 2 scoring scripts also require the parity gate in `part2_attribution/outputs/00_parity/`). Add a regression test for any material fix.

## Measurement

- Score one prompt per forward pass with the reference scorer (`geoflip/scoring.py`). No batching or other speed changes that perturb log-probabilities.
- Model: `meta-llama/Llama-3.2-3B-Instruct` at the revision in `geoflip/scoring.py`, bf16. Read small differences in float32 (`Runner.d_fp32`); bf16 logits step by 0.125 nats.
- Before running an aggregate, show one prompt end to end and check it.

## Data and Results

- Treat `conflict_data/`, `interpretability/inputs/`, `part1_behavioral/` and every completed `outputs/` directory as read-only. Reproduce into a new directory instead of overwriting.
- Every output directory carries a `manifest.json` with input, output and script hashes. Generate hashes from the actual files; never edit them to pass a check.
- Report numbers come from `results.json` / `facts.json`, produced by code, not typed into reports.
- Record how each judgement was made. Do not claim human verification for AI assessments.

## Writing

- Write reports as a researcher would: dataset, method, result. Present design choices as deliberate; keep failure narratives and dropped analyses out of the main text.
- Keep READMEs short and link rather than repeat.

## Housekeeping

- Temporary files and caches go under `/fs/clip-scratch/pramu`, never in the repo.
- Never commit credentials (`interpretability/.env` holds `HF_TOKEN` and is gitignored).
- Commit only when asked, scoped to the request; after a requested push, confirm the remote is in sync.
