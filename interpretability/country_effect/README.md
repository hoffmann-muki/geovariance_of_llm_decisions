# Country Effects in LLM Conflict-Event Coding

This folder holds the current study. It is self-contained: the earlier exploratory work stays where it is (`../PLAN.md` v4, `../history/`, `../CONSTRUCTION.md`), and nothing here changes those files or their score caches.

## The Story

1. **Part 1: dataset and behavioural effect** ([part1_behavioral/](part1_behavioral/)).
   - Build a large, evidence-backed synthetic dataset of conflict-event descriptions.
   - Vary them systematically: sentence frame (paraphrase), wording (verbiage), evidence item (ACLED-attested acts), actors, grievances, places and times.
   - Hold each description fixed and vary only the country name: 37 real countries, 20 invented placebo names, and no country.
   - Record how Llama-3.2-3B-Instruct labels each version: Protest vs Riot (PR), and Battle vs Violence against civilians (BV).
2. **Part 2: why the country matters** ([part2_attribution/](part2_attribution/)). Analyse the mechanism behind the country effects found in Part 1, starting with token attribution. Not started.

## Reports

| Report | Contents | Status |
| --- | --- | --- |
| Preliminary behavioural report | Part 1: dataset, validity checks, country effects, placebo null | [part1_behavioral/report/preliminary_report.md](part1_behavioral/report/preliminary_report.md) |
| Final behavioural report | A larger dataset (held-out contexts; evidence steps from `../CONSTRUCTION.md`), plus token attribution from Part 2 | Not started |

## Fixed Choices for Part 1 (Do Not Change Without Logging)

- **Model:** `meta-llama/Llama-3.2-3B-Instruct`, revision `0cb88a4f76`, bf16, one prompt per forward pass, no batching.
- **Task prompt:** the *defined* version only (class definitions included), one paraphrase per boundary (PR p0, BV p3), layout `options_first`.
- **Perturbation of the task prompt:** option order only (A/B and B/A).
- **Answers:** the letters A or B, read from the first answer token.

## Layout

| Path | Contents |
| --- | --- |
| `part1_behavioral/DESIGN.md` | The Part 1 design and the analysis, specified before results were seen |
| `part1_behavioral/build.py` | Builds the design and `dataset/descriptions.csv` |
| `part1_behavioral/score.py` | Scores every description × name × order (resumable cache) |
| `part1_behavioral/analyse.py` | Per-description label tables, statistics and figures |
| `part1_behavioral/inputs/` | Contexts, names (real and placebo), task prompts, manifest of input hashes |
| `part1_behavioral/dataset/` | The released dataset: descriptions, labels per description × name, per-description summary |
| `part1_behavioral/report/` | The preliminary report and its figures |
| `part2_attribution/` | Part 2 (placeholder) |

Run everything from `interpretability/`:

```bash
export PY=/fs/clip-projects/rlab/pramu/miniconda3/envs/interp-pilot/bin/python HF_HOME=/fs/clip-scratch/rlab_models
$PY -m country_effect.part1_behavioral.build
$PY -m country_effect.part1_behavioral.score          # about 4 h on one RTX A4000; resumes
$PY -m country_effect.part1_behavioral.analyse
```

The score cache is in `/fs/clip-scratch/pramu/country_effect/work/cache/` (set `CE_WORK` to move it).
