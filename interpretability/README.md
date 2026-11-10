# Geographic Flips in LLM Event Coding

This folder is the main repository for the interpretability project. It builds a dataset of event descriptions whose class, as coded by Llama 3.2 3B, changes when only the country named in the description changes. Version 4 (in progress) scores a fully crossed population of descriptions built from ACLED-attested acts, with single-token A/B answers and one pre-specified filtering rule against a placebo-name null. The earlier exploratory v3 dataset is in `history/v3/`.

- **[PLAN.md](PLAN.md):** the full method (task, materials, scoring, verification stages, results, limitations). Start here.
- **[HANDOFF.md](HANDOFF.md):** current status and next steps.
- **[CONSTRUCTION.md](CONSTRUCTION.md):** proposed evidence-based construction of the inputs (v4.1), with the evidence ledger and the intensity axis.

## Layout

| Path | Contents |
| --- | --- |
| `geoflip/` | The code. v4: `acled.py` (ACLED act counts), `items.py` (item sentences), `materials.py` (frames and bases), `placebo.py` (placebo names), `study.py` (V0 rules, A/B prompts, first-token scorer, pilot); `scoring.py` (prompt rendering, `score_first_token`). v3: `pipeline.py`, `dataset.py` |
| `inputs/v4/` | v4 inputs: ACLED act table, items, descriptions, task prompts, names, bases. `inputs/` itself holds the v3 inputs used by `pipeline.py` |
| `scripts/run_loop.sh` | Runs the stages for one boundary on one GPU |
| `tests/` | Unit tests (`GEOFLIP_TOKENIZER_TESTS=1` adds checks against the real tokenizers) |
| `history/` | Frozen record of the earlier iterations and of version 1 ([history/README.md](history/README.md)) |
| `.env` | `HF_TOKEN` for the gated Llama weights. Not in git; never copy or commit it |

## Quick Start (v4)

```bash
cd interpretability
export PY=/fs/clip-projects/rlab/pramu/miniconda3/envs/interp-pilot/bin/python HF_HOME=/fs/clip-scratch/rlab_models
$PY -m unittest discover -s tests -t .        # tests
$PY -m geoflip.study check                    # V0 rules on inputs/v4
CUDA_VISIBLE_DEVICES=0 $PY -m geoflip.study pilot   # pilot (resumes from the score cache)
```

See [HANDOFF.md](HANDOFF.md) for how to resume the interrupted pilot. The v3 pipeline (`scripts/run_loop.sh`, `geoflip.pipeline`, `geoflip.dataset`) still runs on the v3 inputs in `inputs/`. New experiments go in new input directories and work directories, so existing results and their hashes stay valid.
