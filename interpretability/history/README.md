# History

Everything here is a frozen record of how the dataset came to be. Nothing in `geoflip/` reads from this folder, and the files are not meant to be edited or rerun as they stand.

| Path | Contents |
| --- | --- |
| [ITERATIONS.md](ITERATIONS.md) | Decision log for iterations it01–it05: what changed, why, and what the verifier loop showed |
| `iterations/` | The exact inputs of every iteration. `geo/` holds it01–it03; `geo/it04/` and `geo/it05/` hold their own copies; `bases_pr.csv` and `bases_bv.csv` are the original 24-base pools. Relative paths inside each `splits.json` still resolve |
| `reports/` | Generated reports (`itNN_<B>.md`), flip lists and country matrices of each iteration |
| `stages/` | Stage summaries (`check`, `ambiguity`, `screen`, `confirm`, `test`, `matrix`) of it03–it05. Their `meta.digests` are the sha256 of the inputs in `iterations/` |
| `transfer_qwen3-4b_bf16/` | Exploratory re-scoring of the Llama matrix rows with Qwen3-4B in bf16 |
| `code/` | `interp_geo.py` and its tests, the pipeline version that produced it03–it05 (the `producer_sha256` in the stage summaries). It imported `src/interp_score.py` and `src/utils.py` from the parent repository |
| `v1/` | Version 1, the ambiguity-ladder design that was abandoned: inputs (`data/`), the Stage A run on Llama 3.2 (`runs/`), its code (`code/interp_score.py`, `code/interp_stimuli.py`) and the debugging notebooks (`notebooks/`). The notebooks import the old `src.` modules and no longer run from here |
| `HANDOFF_v2_2026-11-06.md` | The handoff written when the dataset was finished |

## Old Paths

Documents in this folder refer to paths from before the reorganisation on 6 November 2026:

| Old path | Now |
| --- | --- |
| `interpretability/data/geo/` | `history/iterations/geo/` |
| `interpretability/data/{bases_pr,bases_bv}.csv` | `history/iterations/` |
| `interpretability/data/{ladders,prompts}.json`, `data/stimuli/` | `history/v1/data/` |
| `interpretability/results/geo/*.md`, `*_flips.jsonl`, `*_matrix_*.csv` | `history/reports/` |
| `interpretability/results/geo/stages/` | `history/stages/` |
| `interpretability/runs/llama3.2-3b/stage_A/` | `history/v1/runs/llama3.2-3b/stage_A/` |
| `interpretability/notebooks/` | `history/v1/notebooks/` |
| `src/interp_geo.py`, `tests/test_interp_geo.py` | `history/code/` (superseded by `geoflip/pipeline.py`) |
| `src/interp_score.py`, `src/interp_stimuli.py` and their tests | `history/v1/code/` (scoring superseded by `geoflip/scoring.py`) |

## How the Final Inputs Map to it05

`inputs/` is it05 with the base splits written out as explicit files:
- `bases/<B>_explore.csv`: the 8 explore bases of `geo/it05/splits.json`, in its order.
- `bases/<B>_confirm.csv`: the other 16 bases of `bases_<b>.csv`, in file order, followed by the 8 bases of `geo/bases_<b>_test.csv` (the it03 test bases).
- `bases/<B>_test.csv`: `geo/it05/bases_<b>_test2.csv`.

`task_prompts.json` and `countries.json` are byte-for-byte copies of it05. `descriptions.json` has the same content, with only its `version` field changed. Protest vs Riot was run as it04 and Battle vs Violence against civilians as it05; the two iterations have identical Protest vs Riot inputs.
