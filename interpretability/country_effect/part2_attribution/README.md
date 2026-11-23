# Part 2: How the Country Reaches the Decision

Mechanistic analysis of the Part 1 country effect in Llama-3.2-3B-Instruct on the Battle vs Violence-against-civilians (BV) boundary. Each country prompt is paired with a baseline prompt whose name has the same token length, and the pair difference is traced through the model:

- **Logit lens:** when the country becomes readable as an answer preference at the final position.
- **Activation patching** (with a matched-norm random control): which layers and token spans carry it.
- **Layer-wise integrated gradients:** an independent attribution of the same route.
- **Steering:** whether one direction at the final position carries the effect.
- **Qwen3-4B:** whether the behavioural ordering and the late arrival recur in a second model.

## Files

| File | Contents |
| --- | --- |
| [report/facts.json](report/facts.json) | Every reported number, compiled from the stage outputs |
| `report/figures/` | Figures 1–6 |
| [PREREGISTRATION.md](PREREGISTRATION.md) | Analysis plan: panel, subsets, metrics and decision rules, with a change log |

## Stages

Run from `interpretability/` as `python -m country_effect.part2_attribution.scripts.<script>`, with `HF_HOME` set and a GPU.

| Script | Does | Output |
| --- | --- | --- |
| `00_verify_parity` | reproduces Part 1 scores; checks hooks leave logits unchanged; runs `geoflip/tests` | `outputs/00_parity/` |
| `01_build_controls` | token-matched baseline names | `outputs/01_controls/` |
| `02_run_behavioral_controls` | control-name families, both boundaries | `outputs/02_controls/` |
| `03_run_heldout_replication` | held-out-context replication of Part 1 | `outputs/03_heldout/` |
| `04_run_logit_lens` | final-position lens by layer | `outputs/04_logit_lens/` |
| `05_run_activation_patching` | span × layer patching, full panel | `outputs/05_patching/` |
| `05b_run_span_patching_control` | split intermediate spans, matched-norm controls | `outputs/05b_span_control/` |
| `06_run_probes_and_steering` | armed-status probe; steering with u_ℓ | `outputs/06_probes_steering/` |
| `07_run_matched_ig` | layer-wise integrated gradients | `outputs/07_ig/` |
| `08_run_generality_check` | Qwen3-4B on BV and PR | `outputs/08_generality/` |
| `09_compile_report_artifacts` | figures and `report/facts.json` | `report/` |

## Notes

- Scoring scripts refuse to run unless `outputs/00_parity/manifest.json` records a passing gate.
- Every output directory has a `manifest.json` with input and output hashes, the script hash and the model revision.
- Result stores are append-only JSONL, so interrupted runs resume. Score caches go to `$CE2_WORK` (default `/fs/clip-scratch/pramu/country_effect/part2/work`).
- Part 1 files are read only. Shared code is in `../../geoflip/` (`spans.py`, `hooks.py`, `attribution.py`).
