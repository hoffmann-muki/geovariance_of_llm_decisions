# Geography in LLM Decision-Making

Code, data and logs for the project by Hoffmann Muki and Pritika Ramu on whether, and how, a country name changes a language model's label for a fixed conflict-event description.

- **Behavioural effect:** with the description held fixed, swapping the country shifts Llama-3.2-3B-Instruct's Battle vs Violence-against-civilians judgement in a stable, geographically ordered way that invented placebo names do not reproduce.
- **Mechanism:** the effect is traced through the model: which layers and token positions carry it, and in what form it reaches the answer.

| You want | Go to |
| --- | --- |
| Project overview, layout and commands | [interpretability/README.md](interpretability/README.md) |
| Behavioural study: design, dataset, results | [part1_behavioral/](interpretability/country_effect/part1_behavioral/) |
| Mechanistic analysis: scripts, outputs, figures | [part2_attribution/](interpretability/country_effect/part2_attribution/) |
| How the materials were built from ACLED | [interpretability/MATERIALS.md](interpretability/MATERIALS.md) |

`conflict_data/` holds the raw ACLED exports the materials are rebuilt from. Install dependencies with `python3 -m pip install -r interpretability/requirements.txt` and run commands from `interpretability/`.
