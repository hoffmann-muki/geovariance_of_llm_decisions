# Part 2: Why the Country Matters

Mechanistic analysis of the Part 1 country effect in Llama-3.2-3B-Instruct (BV boundary): when the country becomes readable at the answer position (logit lens), where its influence is carried (activation patching with a matched-norm random control, layer-wise integrated gradients), and whether a single direction at the final position carries it (steering), with a Qwen3-4B comparison.

- Paper: `report/part2_report.md`; LaTeX section `report/part2_section.tex` for `../main.tex` (BibTeX entries in its trailing comments).
- Numbers: `report/facts.json` and `outputs/*/results.json`; figures in `report/figures/`.
- `PREREGISTRATION.md` fixed the panel and analyses in advance. Stages 02 (control names), 03 (held-out replication) and the armed-status probe in 06 were run and their outputs kept, but they are not part of the paper.

## Layout

| Path | Contents |
| --- | --- |
| `PREREGISTRATION.md` | splits, controls, panel, decision rules, change log |
| `configs/stimuli_controls.json` | control-name families (Stage 1A) |
| `configs/mechanistic_panel.json` | Stage 2 panel: items, countries, contexts, spans, probe classes |
| `common.py` | shared loaders, prompt resolution, panel pairs, manifests |
| `scripts/00`–`09`, `05b` | one script per stage (see the paper, Appendix A) |
| `outputs/NN_*/` | results and `manifest.json` (input/output hashes, script hash, model revision) |
| `../../geoflip/spans.py`, `hooks.py`, `attribution.py` | span resolution, residual capture and patching, layer-wise IG |
| `../../geoflip/tests/test_part2_invariants.py` | invariant tests; run by `00_verify_parity` |

## Running

From `interpretability/`, with `HF_HOME` set and a GPU:

```
python -m country_effect.part2_attribution.scripts.00_verify_parity
python -m country_effect.part2_attribution.scripts.01_build_controls
python -m country_effect.part2_attribution.scripts.02_run_behavioral_controls --score --boundary BV   # and PR, then --analyse
...
python -m country_effect.part2_attribution.scripts.09_compile_report_artifacts
```

Every scoring script refuses to run unless `outputs/00_parity/manifest.json` records a passing parity gate. Result stores are append-only JSONL, so interrupted runs resume where they stopped. Score caches go to `$CE2_WORK`, which defaults to `/fs/clip-scratch/pramu/country_effect/part2/work`. Part 1 files are read only.
