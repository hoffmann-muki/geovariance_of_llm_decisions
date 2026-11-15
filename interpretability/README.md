# Investigating the Influence of Geography on LLM Decision-Making in Ambiguous Contexts

Code, data and reports for the project by Hoffmann Muki and Pritika Ramu.

**Question.** When a conflict-event description is held fixed and only the name of the country where it happened changes, does a language model's event label change? The task uses two ACLED distinctions: Protest vs Riot (PR), and Battle vs Violence against civilians (BV).

**Design.**
- **Evidence:** every description is built from acts attested in at least 20 ACLED event notes.
- **Systematic variation:** descriptions vary in evidence, frame, wording, actors, grievances, places and times.
- **Manipulation:** each description is rendered with 37 real countries, 20 invented placebo names and no country.
- **Scoring:** Llama-3.2-3B-Instruct labels every rendering in both option orders.

**Main result.** For BV, the country shifts the model's judgement in a stable, geographically ordered way that placebo names do not reproduce, and it changes the label in 8.1% of descriptions. For PR, country effects are at placebo level.

## Where to Start

| You want | Go to |
| --- | --- |
| The report | [country_effect/main.tex](country_effect/main.tex) (midterm report) or [country_effect/part1_behavioral/report/preliminary_report.md](country_effect/part1_behavioral/report/preliminary_report.md) (longer version) |
| How the study was designed and analysed | [country_effect/part1_behavioral/DESIGN.md](country_effect/part1_behavioral/DESIGN.md) |
| How the materials (ACLED items, frames, names, prompts) were built | [MATERIALS.md](MATERIALS.md) |
| The dataset | [country_effect/part1_behavioral/dataset/](country_effect/part1_behavioral/dataset/) |
| Every number in the report | [country_effect/part1_behavioral/report/results.json](country_effect/part1_behavioral/report/results.json) |
| Planned extensions of the dataset construction | [CONSTRUCTION.md](CONSTRUCTION.md) |

## Layout

```
interpretability/
├── README.md               this file
├── MATERIALS.md            how inputs/v4/ was built: task, ACLED items, sentence rules, frames, names, prompts, prompt selection
├── CONSTRUCTION.md         planned evidence steps for the next dataset version (lexicon audit, frames from notes, human labels)
├── requirements.txt
├── geoflip/                shared library
│   ├── acled.py            ACLED act lexicon and P(higher | act) counts
│   ├── items.py            evidence sentences (W1, W2) from the ACLED acts
│   ├── materials.py        frames, context value lists, pilot contexts
│   ├── placebo.py          placebo country names
│   ├── render.py           renders a description with a country phrase
│   ├── scoring.py          chat-template rendering and first-token A/B log-probabilities
│   ├── study.py            V0 rules, task-prompt rendering, cached scorer, prompt-selection pilot
│   └── utils.py
├── inputs/v4/              materials: ACLED act table, items, frames, names, task prompts, pilot contexts
└── country_effect/         the study (see country_effect/README.md)
    ├── main.tex            midterm report
    ├── part1_behavioral/   build, score, analyse; inputs, dataset, results, figures, DESIGN.md
    └── part2_attribution/  planned
```

## Reproducing

```bash
cd interpretability
pip install -r requirements.txt
export HF_TOKEN=...                                  # gated model; or put it in interpretability/.env
python -m country_effect.part1_behavioral.build      # dataset/descriptions.csv
python -m country_effect.part1_behavioral.score      # 408,320 prompts, about 4.3 h on one RTX A4000
python -m country_effect.part1_behavioral.analyse    # tables, results.json, figures
```

Rebuilding the materials themselves from the raw ACLED exports in `../conflict_data/` is described in [MATERIALS.md § 6](MATERIALS.md#6-rebuilding-the-materials).
