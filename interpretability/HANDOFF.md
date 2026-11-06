# Handoff

Status as of 6 November 2026, written when GPU access ended. [PLAN.md](PLAN.md) (v4) is the governing document; `PROJECT_PLAN.md` at the repository root belongs to the earlier behavioral study and does not apply here. The approved v4 plan also lives at `/nfshomes/pramu/.claude/plans/joyful-finding-kazoo.md`, and the step log at `/fs/clip-scratch/pramu/interp_v4/progress.md`.

## Where Things Stand

| Step (PLAN.md § 7) | State |
| --- | --- |
| 1. Archive v3 | Done: `history/v3/` (dataset, final results, PLAN_v3.md, inputs) |
| 2. Items | Done: ACLED act table `inputs/v4/acled_acts.csv` (lexicon v1), items `inputs/v4/items.json`: 29 PR, 22 BV, plus anchors |
| 3. Frames, V0 rules, bases | Done: `inputs/v4/descriptions.json`, `inputs/v4/bases/{PR,BV}_{pilot,main}.csv`; `python -m geoflip.study check` reports no findings |
| 4. Task prompts and scorer | Done: `inputs/v4/task_prompts.json` (A/B answers, 2 candidate layouts), `scoring.score_first_token`, `study.Scorer`. Tests: `tests/test_study.py` (11 pass with `GEOFLIP_TOKENIZER_TESTS=1`) |
| **5. Pilot** | **In progress, interrupted.** Scored at about 31 prompts/s. Layout `options_last` was being scored on GPU 0 (PR done, BV partly); layout `options_first` on GPU 1 into a separate work directory. No gate has been evaluated |
| 6. Placebos, pre-registration, main run | Not started. `geoflip/placebo.py` is written but has not been run |
| 7–8. Filter, null, release, second review | Not started |

## How to Resume the Pilot

Scores are cached line by line (flushed every 2,000 prompts), so a rerun only scores what is missing.

```bash
cd interpretability
PY=/fs/clip-projects/rlab/pramu/miniconda3/envs/interp-pilot/bin/python
export HF_HOME=/fs/clip-scratch/rlab_models
# 1. Merge the second layout's cache (from the parallel run) into the main cache.
cat /fs/clip-scratch/pramu/geoflip_work/v4_par/cache/*_first_token.jsonl \
  >> /fs/clip-scratch/pramu/geoflip_work/v4/cache/llama3.2-3b_bfloat16_0cb88a4f76_first_token.jsonl
# 2. Finish and summarise the pilot (writes geoflip_work/v4/pilot.json and pilot_rows.jsonl).
CUDA_VISIBLE_DEVICES=0 $PY -m geoflip.study pilot
# 3. fp32 audit on the chosen layout.
CUDA_VISIBLE_DEVICES=0 $PY -m geoflip.study fp32_audit --layout <chosen_layout from pilot.json>
```

Duplicate cache lines are harmless (the cache is keyed). The helper used for the parallel run is `/fs/clip-scratch/pramu/interp_v4/pilot_layout2.py`.

## Next Steps, in Order

1. **Finish Step 5.** Read `pilot.json`:
   - If `chosen_layout` is null, a gate failed; the `gates` list says which prompt and frame. Revise the frame or layout, log the change in `history/ITERATIONS.md`, and rerun the pilot.
   - Record the fp32 audit result (bf16 is kept if max |Δd| ≤ 0.05).
   - Fix the number of main bases from the measured throughput.
2. **Step 6.**
   - Run `python -m geoflip.placebo --out inputs/v4/placebos.json` and add the 20 placebos to `inputs/v4/names.json` as `fictional`.
   - Write `geoflip/prereg.py`, so that `prereg.json` records the population, the thresholds (m = 0.5, K = 4 of 8) and the input hashes, and the pipeline refuses to run on a hash mismatch.
   - Add the `score` stage and run the main population: about 1.4M PR and 1.1M BV prompts, about 12 h and 10 h on two GPUs.
3. **Step 7.** The `filter` stage: the flip rule within each version, the placebo null, the K curve and split-half replication.
4. **Step 8.** The `export` stage to `dataset/{defined,undefined}/`, PLAN.md results, and a second adversarial review before release.

## Decisions Already Made (Do Not Reopen Without the User)

- A flip is a change in the model's **actual answer** on a concrete prompt.
- Answers are **neutral A/B letters**, counterbalanced with order. Only the labels must be single tokens; country and placebo names can be any length, with token count recorded as a covariate. No activation patching is planned.
- No dates in the frames; the template's date header stays fixed at "26 Jul 2024".
- Items come from the class definitions plus ACLED-attested acts (at least 20 notes); they are frozen.
- Separate datasets with and without definitions.

## Facts Already Verified

- **Llama 3.2 template.** It always writes `Cutting Knowledge Date: December 2023 / Today Date: …`, so the date can only be fixed, not removed, without leaving the trained format. Encoding the rendered text reproduces the template's token ids.
- **Letters are single tokens.** "A" and "B" are each one token at the answer position, which comes right after `<|end_header_id|>\n\n` (tested).
- **No batching.** Padded bf16 batches moved log-probabilities by up to 0.6 nats in v3, so every prompt is scored alone.
- **The v3 order bias,** confirmed from its cache: BV prompts preferred the second-listed class by about 10 nats. The v4 pilot measures order bias per layout before anything else.

## Files to Leave Alone

- `Notes.md` in this folder is the user's own notes.
- `interpretability/.env` holds `HF_TOKEN`. It is not in git; never copy or commit it.
- At the repository root, `2605.04177v1.pdf`, `src/run_inference.py` and `tests/test_run_inference.py` are unrelated.
