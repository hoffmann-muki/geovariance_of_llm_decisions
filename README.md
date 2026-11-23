# Geography in LLM Decision-Making

Code, data and logs for the project by Hoffmann Muki and Pritika Ramu.

**Question.** When a conflict-event description is held fixed and only the name of the country where it happened changes, does a language model's event label change, and how does the country reach the decision? The task uses two ACLED distinctions: Protest vs Riot (PR) and Battle vs Violence against civilians (BV).

**Design.**
- **Evidence:** 3,520 descriptions built from acts attested in at least 20 ACLED event notes. They vary in evidence item, sentence frame, wording, actors, grievances, places and times.
- **Manipulation:** each description is rendered with 37 real countries, 20 invented placebo names and no country.
- **Scoring:** `meta-llama/Llama-3.2-3B-Instruct` (revision `0cb88a4f76`, bf16, one prompt per forward pass) labels every rendering in both option orders (408,320 prompts). The score is the log-probability difference between the two answer letters, averaged over orders.
- **Mechanism:** on 128 BV descriptions, each country prompt is paired with a baseline prompt whose name has the same token length. The pair difference is traced with activation patching (with matched-norm random controls), integrated gradients, a logit lens, a probe and steering. Qwen3-4B is used as a second model.

**Results.**
- For BV, the country shifts the judgement in a stable, geographically ordered way that placebo names do not reproduce. It changes the label for 8.1% of descriptions. For PR, country effects are at placebo level.
- The country's influence travels from the name through the chat-template tokens to the final position by layer 14. There a single direction carries it, and adding that direction reproduces the item-by-item pattern of the behavioural effect.

## Layout

```
conflict_data/                      raw ACLED exports (five regions)
interpretability/
├── requirements.txt
├── geoflip/                        shared library
│   ├── acled.py                    ACLED act lexicon and P(higher | act) counts
│   ├── items.py                    evidence sentences from the ACLED acts
│   ├── materials.py                frames, context values, pilot contexts
│   ├── placebo.py                  placebo country names
│   ├── render.py                   renders a description with a country phrase
│   ├── scoring.py                  chat-template rendering and first-token A/B log-probabilities
│   ├── study.py                    sentence rules, task prompts, cached scorer, prompt-selection pilot
│   ├── spans.py, hooks.py          token spans; residual-stream capture, patching, steering
│   ├── attribution.py              layer-wise integrated gradients
│   └── tests/
├── inputs/v4/                      materials: ACLED act table, items, frames, names, task prompts
└── country_effect/
    ├── part1_behavioral/           behavioural study
    │   ├── build.py, score.py, analyse.py
    │   ├── inputs/                 contexts, names, task prompts, prompt selection, manifest
    │   ├── dataset/                the released dataset (below)
    │   └── report/                 results.json (every behavioural number) and figures
    └── part2_attribution/          mechanistic analysis
        ├── common.py, configs/, scripts/
        ├── outputs/NN_stage/       one directory per stage: results, logs, manifest.json
        └── report/                 facts.json (every mechanistic number) and figures
```

## Dataset

In `interpretability/country_effect/part1_behavioral/dataset/`:

| File | One row per | Contents |
| --- | --- | --- |
| `descriptions.csv` | description (3,520) | All factor levels, ACLED evidence statistics, and the text with a `{COUNTRY}` slot |
| `labels.csv.gz` | description × name (204,160) | Score in each order, order-averaged score, letter-preference term, strict and order-averaged labels, letter mass |
| `description_summary.csv` | description | Label counts by name kind, flip indicators, the real countries on each side of a flip, and the no-country label |

## Reproducing

Run everything from `interpretability/`. The Llama model is gated, so set `HF_TOKEN` in the environment or in `interpretability/.env` (gitignored).

```bash
cd interpretability
pip install -r requirements.txt
```

**Materials** (from the raw ACLED exports in `../conflict_data/`):

```bash
python -m geoflip.acled --out inputs/v4/acled_acts.csv
python -m geoflip.items --acts inputs/v4/acled_acts.csv --out inputs/v4/items.json
python -m geoflip.materials --items inputs/v4/items.json --out inputs/v4
python -m geoflip.placebo --countries inputs/v4/names.json --out country_effect/part1_behavioral/inputs/placebos.json
python -m geoflip.study check          # sentence rules: prints "V0: no findings"
python -m geoflip.study pilot          # prompt-selection pilot (GPU)
```

**Behavioural study:**

```bash
python -m country_effect.part1_behavioral.build      # seconds; dataset/descriptions.csv
python -m country_effect.part1_behavioral.score      # about 4.3 h on one RTX A4000; resumes from its cache
python -m country_effect.part1_behavioral.analyse    # minutes; dataset tables, results.json, figures
```

The score cache defaults to `/fs/clip-scratch/pramu/country_effect/work/` (set `CE_WORK` to move it). `analyse.py` reads only the cache.

**Mechanistic analysis** (GPU, `HF_HOME` set), one stage at a time as `python -m country_effect.part2_attribution.scripts.<script>`:

| Script | Does | Output |
| --- | --- | --- |
| `00_verify_parity` | reproduces the behavioural scores; checks hooks leave logits unchanged; runs `geoflip/tests` | `outputs/00_parity/` |
| `01_build_controls` | control names and token-length-matched baseline names | `outputs/01_controls/` |
| `02_run_behavioral_controls` | control-name families, both boundaries | `outputs/02_controls/` |
| `03_run_heldout_replication` | replication on held-out context values | `outputs/03_heldout/` |
| `04_run_logit_lens` | final-position logit lens by layer | `outputs/04_logit_lens/` |
| `05_run_activation_patching` | span × layer patching | `outputs/05_patching/` |
| `05b_run_span_patching_control` | split intermediate spans, matched-norm controls | `outputs/05b_span_control/` |
| `06_run_probes_and_steering` | armed-status probe; steering | `outputs/06_probes_steering/` |
| `07_run_matched_ig` | layer-wise integrated gradients | `outputs/07_ig/` |
| `08_run_generality_check` | Qwen3-4B on BV and PR | `outputs/08_generality/` |
| `09_compile_report_artifacts` | figures and `report/facts.json` | `report/` |

Scoring scripts refuse to run unless `outputs/00_parity/manifest.json` records a passing gate. Every output directory has a `manifest.json` with input, output and script hashes and the model revision. Result stores are append-only JSONL, so interrupted runs resume; caches go to `$CE2_WORK` (default `/fs/clip-scratch/pramu/country_effect/part2/work`).

Tests: `python -m pytest geoflip/tests` from `interpretability/`.
