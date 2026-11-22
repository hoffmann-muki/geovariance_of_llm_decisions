# Geography in LLM Decision-Making

Research by Hoffmann Muki and Pritika Ramu on whether, and how, a country name changes a language model's label for a fixed conflict-event description.

- **Part 1 (behavioural):** with the description held fixed, swapping the country shifts Llama-3.2-3B-Instruct's Battle vs Violence-against-civilians judgement in a stable, geographically ordered way that invented placebo names do not reproduce.
- **Part 2 (mechanistic):** traces that effect through the model: when the country becomes an answer preference, which layers and token positions carry it, and in what form it reaches the answer.

| You want | Go to |
| --- | --- |
| Project overview, layout and commands | [interpretability/README.md](interpretability/README.md) |
| The report | [interpretability/country_effect/main.tex](interpretability/country_effect/main.tex) |
| Part 1 design and results | [part1_behavioral/](interpretability/country_effect/part1_behavioral/) |
| Part 2 paper draft and results | [part2_attribution/](interpretability/country_effect/part2_attribution/) |
| How the materials were built from ACLED | [interpretability/MATERIALS.md](interpretability/MATERIALS.md) |
| Guidance for coding agents | [AGENTS.md](AGENTS.md) |

`conflict_data/` holds the raw ACLED exports the materials are rebuilt from. Install dependencies with `python3 -m pip install -r interpretability/requirements.txt` and run commands from `interpretability/`.
