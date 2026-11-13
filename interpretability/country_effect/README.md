# Country Effects in LLM Conflict-Event Coding

The study and its report. Shared code is in `../geoflip/`, and the materials it builds on are in `../inputs/v4/` (described in `../MATERIALS.md`).

## The Study

1. **Part 1: dataset and behavioural effect** ([part1_behavioral/](part1_behavioral/)). Done.
   - **Dataset:** 3,520 evidence-backed conflict-event descriptions. They vary systematically in evidence item (ACLED-attested acts), sentence frame, wording, actors, grievances, places and times.
   - **Manipulation:** each description is held fixed while only the country name changes, across 37 real countries, 20 invented placebo names, and no country.
   - **Measurement:** Llama-3.2-3B-Instruct labels every version, Protest vs Riot (PR) and Battle vs Violence against civilians (BV), in both option orders (408,320 prompts).
2. **Part 2: why the country matters** ([part2_attribution/](part2_attribution/)). Planned: token attribution on the descriptions identified in Part 1.

## Reports

| Report | Contents |
| --- | --- |
| [main.tex](main.tex) | Midterm project report (ACL format). References to add to `custom.bib` are in [custom_additions.bib](custom_additions.bib); figures are read from `part1_behavioral/report/figures/` |
| [part1_behavioral/report/preliminary_report.md](part1_behavioral/report/preliminary_report.md) | Longer version of the Part 1 results, with all figures |

## Fixed Choices for Part 1

- **Model:** `meta-llama/Llama-3.2-3B-Instruct`, revision `0cb88a4f76`, bf16, one prompt per forward pass, no batching.
- **Task prompt:** the *defined* version (class definitions included), one paraphrase per boundary (PR p0, BV p3), layout `options_first`. The paraphrases were selected on anchor accuracy only (`part1_behavioral/inputs/prompt_selection.json`).
- **Perturbation of the task prompt:** option order only (A/B and B/A).
- **Answers:** the letters A or B, read at the first answer token.

## Layout

| Path | Contents |
| --- | --- |
| `part1_behavioral/DESIGN.md` | Design and analysis plan, written before results were seen, with a change log |
| `part1_behavioral/build.py` | Builds the factorial design and `dataset/descriptions.csv`, and checks that only the country phrase differs between conditions |
| `part1_behavioral/score.py` | Scores every description × name × order (resumable cache); `--show ID` prints one description |
| `part1_behavioral/analyse.py` | Label tables, flip rates, country effects, figures and `report/results.json` |
| `part1_behavioral/inputs/` | Contexts, names (real and placebo), the four task prompts, prompt-selection gates, and a manifest of input and output hashes |
| `part1_behavioral/dataset/` | The released dataset (see below) |
| `part1_behavioral/report/` | `results.json` (every reported number), `figures/`, and the markdown report |
| `part2_attribution/` | Part 2 (placeholder) |

### Released Dataset

| File | One row per | Contents |
| --- | --- | --- |
| `dataset/descriptions.csv` | description (3,520) | All factor levels, ACLED evidence statistics, and the text with a `{COUNTRY}` slot |
| `dataset/labels.csv.gz` | description × name (204,160) | Score in each order, order-averaged score, letter-preference term, strict and order-averaged labels, letter mass |
| `dataset/description_summary.csv` | description | Label counts by name kind, flip indicators, the real countries on each side of a flip, and the no-country label |

## Reproducing

Run from `interpretability/`, with the packages in `../requirements.txt` and access to the gated Llama model (`HF_TOKEN` in the environment or in `interpretability/.env`, which is never committed):

```bash
python -m country_effect.part1_behavioral.build      # seconds; writes inputs/ and dataset/descriptions.csv
python -m country_effect.part1_behavioral.score      # about 4.3 h on one RTX A4000; resumes from its cache
python -m country_effect.part1_behavioral.analyse    # minutes; writes dataset/ tables, results.json, figures
```

The score cache defaults to `/fs/clip-scratch/pramu/country_effect/work/`; set `CE_WORK` to move it. `analyse.py` reads only the cache, so with the cache in place it needs no GPU.
