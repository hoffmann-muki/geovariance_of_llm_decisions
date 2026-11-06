# Geo Flip Dataset

302 English event descriptions with a `[COUNTRY]` slot, each scored by `meta-llama/Llama-3.2-3B-Instruct` (bf16, revision `0cb88a4f76`) under 40 country conditions. Every description has at least one pair of countries for which the model's robust label differs, from a country pair verified as systematic on held-out bases. How the dataset was built is in [../PLAN.md](../PLAN.md); `manifest.json` records the sha256 of each file, the inputs and the code.

**Items were selected because they flip.** Counts and effect sizes are not prevalence estimates.

## Files

| File | One row per | Rows |
| --- | --- | --- |
| `descriptions.csv` | description | 302 (PR 68, BV 234) |
| `cells.csv` | description × country condition | 12,080 |
| `matrix_<B>_labels.csv`, `matrix_<B>_s.csv` | description, with one column per country condition (wide form of `cells.csv`) | 68 (PR), 234 (BV) |
| `system_prompts.csv` | task prompt (boundary × version × paraphrase × order) | 32 |
| `pairs.csv` | country pair tested in V5 (confirm) or V6 (test), per item | 520 |
| `flips.jsonl` | verified flip: one description and one systematic country pair | 1,554 (PR 335, BV 1,219) |
| `manifest.json` | – | – |

## Columns

**`descriptions.csv`**

| Column | Meaning |
| --- | --- |
| `row_id` | `PR-001`…, `BV-001`…; rows are ordered test split first, then by flip size |
| `boundary` | `PR` (Protest vs Riot) or `BV` (Violence against civilians vs Battle) |
| `lower_class`, `higher_class` | The two classes; s > 0 favours `higher_class` |
| `item` | `<boundary>:<frame>:<evidence>`, for example `BV:f1:watchman` |
| `frame`, `evidence` | Frame id and evidence id in `../inputs/descriptions.json` |
| `version` | `defined` (task prompts include class definitions) or `undefined` |
| `split` | `confirm` or `test` (held-out bases, scored once) |
| `base_id`, `wording` | Base in `../inputs/bases/<B>_<split>.csv`; evidence wording 1 or 2 |
| `evidence_sentence` | The evidence sentence as used |
| `template` | The description with ` in [COUNTRY]` in the country slot |

**`cells.csv`**

| Column | Meaning |
| --- | --- |
| `row_id` | Key into `descriptions.csv` |
| `country`, `country_kind` | One of 37 real countries (`real`), Veloria or Tarvania (`fictional`), or `none` (no country phrase) |
| `text` | The exact user-turn text given to the model |
| `dbar_p0` … `dbar_p3` | For each task paraphrase: log P(higher) − log P(lower), averaged over the two class orders |
| `s` | Mean of the four `dbar` values |
| `label` | The class if \|s\| ≥ 0.5 and at least 3 of the 4 `dbar` share the sign of s; otherwise `unstable` |

The task prompts for a cell are the 8 rows of `system_prompts.csv` with the description's boundary and version. Scoring one cell again: render each system prompt and the cell's `text` with the model's chat template, sum the log-probabilities of each class name's tokens after the assistant header, and average as above (PLAN.md § 4).

**`pairs.csv`**: `share_high_above_low` is the share of the split's descriptions with s(high) > s(low); `mean_gap` is the mean of s(high) − s(low); `wording_reference` is the item's median \|s(W1) − s(W2)\| with no country; `systematic` is `share ≥ 0.75` and `mean_gap > wording_reference`; `flips` counts descriptions with a robust higher label for `country_high` and a robust lower label for `country_low`.

**`flips.jsonl`**: per flip, the template and both rendered texts, both countries' s and per-paraphrase d̄, the no-country score and label, the fictional countries' scores and labels, and the smaller answer mass of the two countries.
