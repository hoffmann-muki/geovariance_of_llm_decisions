# Handoff: Ambiguity-Ladder Pilot

Status as of 5 November 2026. The design is in [PLAN.md](PLAN.md), which is the only governing document for this exploration. `PROJECT_PLAN.md` at the repository root does not apply.

## Where Things Stand

| Item | State |
| --- | --- |
| Research question, H1, task and dataset design | Done: [PLAN.md](PLAN.md) |
| Ladders and bases (`data/ladders.json`, `data/bases_*.csv`) | v1, approved for calibration 2026-11-05 |
| System-prompt parts (`data/prompts.json`) | v1, approved 2026-11-05 |
| Stimulus snapshot (`data/stimuli/`) | Built and verified: 5,280 stimuli, 8 system prompts, 21,120 scheduled prompts |
| Stimulus builder `src/interp_stimuli.py` + tests | Done; 12 tests pass |
| Scorer `src/interp_score.py` | Written; logic and tokenizer tiers pass; **model tier not yet run** |
| Stage A scoring | Not started |
| Calibration analysis (Stage A gate) | **Not written** |
| Stage B scoring and H1 analysis | Not started; blocked on the Stage A gate |

**The code is not committed.** `interpretability/`, `src/interp_*.py`, `tests/test_interp_*.py` and the `AGENTS.md` scoping note exist only in the local working tree. To get them onto the GPU server, either commit and push them (`origin` is GitHub) or copy them across. Never copy or commit `interpretability/.env`.

## Facts Already Verified

Verified with transformers 5.22.1 and the real tokenizers. Do not re-derive these; the tokenizer tests enforce them.

- **Answer position:** the first token after the assistant header.
  - Qwen3.5: `<|im_start|>assistant\n<think>\n\n</think>\n\n`, rendered with `enable_thinking=False`, which inserts an empty reasoning block.
  - Llama 3.2: `<|start_header_id|>assistant<|end_header_id|>\n\n`, with `date_string="26 Jul 2024"`.
- **Answer tokens are multi-token** in both models: `Pro`+`test`, `R`+`iot`, `Battle`, `Viol`+`ence`+` against`+` civilians`. Each answer is scored as the teacher-forced sum of all its token log-probabilities, never the first-token logit. They are identical across all tasks and country conditions, and distinct from the space-prefixed and lowercase variants.
- **Tokenization:** encoding the rendered template text with `add_special_tokens=False` reproduces `apply_chat_template(tokenize=True)` exactly. In transformers 5, the tokenized form returns a `BatchEncoding`, which `as_ids` handles.
- **Loading:** the Qwen3.5 checkpoint is `Qwen3_5ForConditionalGeneration` (vision-language). The scorer loads it with `AutoModelForImageTextToText` rather than `AutoModelForCausalLM`, because a text-only class may silently leave weights randomly initialized. `load_model` stops if any weight is missing or has the wrong shape.

## Next Steps on the GPU Server

### 1. Environment

- Python 3.10+ and a CUDA build of PyTorch, as provided by any recent PyTorch container image.
- A single 16–24 GB GPU is enough. An RTX 4090 or L4 (24 GB) also covers Stage B.
- Install the pinned packages:
  ```bash
  pip install -r interpretability/requirements-gpu.txt   # transformers 5.22.1, huggingface_hub 2.4.0, jinja2
  pip install -r requirements.txt                         # numpy/scipy for later analysis
  ```
- Set `HF_TOKEN` in the environment, or as `HF_TOKEN=<token>` in `interpretability/.env`. Its account must have accepted the Llama 3.2 license. **The token previously in that file was exposed in a session transcript and must be revoked; use a new one.**
- Qwen3.5 uses hybrid linear-attention layers. Without the optional fast kernels, transformers falls back to a slower PyTorch path. Check the load log, and install those kernels only if throughput is a problem.

### 2. Run the Test Tiers, in Order

```bash
python3 -m unittest discover -s tests                                  # logic tier + rest of repo
INTERP_TOKENIZER_TESTS=1 python3 -m unittest tests.test_interp_score   # real tokenizers
INTERP_MODEL_TESTS=1 INTERP_MODEL=qwen3.5-4b  python3 -m unittest tests.test_interp_score.RealModelTests
INTERP_MODEL_TESTS=1 INTERP_MODEL=llama3.2-3b python3 -m unittest tests.test_interp_score.RealModelTests
```

The model tier, run on the 16 anchor prompts (levels 1 and 5, no country), checks that:
- the scorer's first-answer-token log-probability equals the model's own first generation step (within 0.05);
- greedy output begins with exactly one answer, and its per-token log-probabilities match the teacher-forced scores (within 0.1);
- the median combined probability of the two answers is at least 0.9.

If a model test fails, find the cause. Do not widen tolerances or change expected tokens to make it pass. In particular, a failure in the greedy-output test means the answer position or format differs from what the scorer assumes. This tier has never run, so treat its first run as the real verification, especially generation through Qwen's vision-language class with text-only input.

### 3. Stage A Scoring (No-Country Calibration)

Run this only after all test tiers pass.

```bash
python3 -m src.interp_score --model qwen3.5-4b  --stage A   # → interpretability/runs/qwen3.5-4b/stage_A/
python3 -m src.interp_score --model llama3.2-3b --stage A
```

- Each stage has 1,920 prompts per model, plus 20 repeats for the determinism check. Each prompt takes two forward passes, one per answer.
- Interrupted runs resume from `scores.jsonl`. A changed setting, such as a different model revision or snapshot, stops the run instead of mixing settings.
- `manifest.json` appears only when a run is complete.
- Copy `interpretability/runs/` back as soon as a stage ends.

### 4. Write the Calibration Analysis

A module is still needed, for example `src/interp_calibration.py` with tests. It should read the Stage A runs and apply the pass criteria in PLAN.md § Procedure, separately for each model × boundary × task version:
- anchor accuracy ≥ 90%;
- Spearman ρ > 0 in ≥ 80% of base × wording ladders;
- a middle level with renormalized probability between 0.2 and 0.8 for at least half of the bases;
- median combined answer probability ≥ 0.9;
- repeats within 0.01.

Use d = (d_A + d_B) / 2 as defined in PLAN.md § Analysis. The report should go to `interpretability/results/`, with its facts generated from the run artifacts.

### 5. Stage B and H1

Only for combinations whose ladder passed:
1. Freeze the phrase lists and record their hashes.
2. Run `--stage B`: 19,200 prompts per model.
3. Write the H1 analysis exactly as PLAN.md § Analysis of H1 specifies: Friedman per combination, Holm across the eight, bootstrap intervals with seed 42.

## Rules That Protect the Result

- **Never score Stage B before the Stage A gate passes and the ladders are frozen.** Calibration must not see country data.
- If a ladder fails, revise the phrase list from Stage A results only, record the revision as a new ladder version, rebuild the snapshot into a new, unused directory, and rerun Stage A. Existing snapshots and runs are immutable.
- Every output goes to an unused directory. Snapshots and runs carry manifests with producer and source hashes; never edit hashes to get past validation.
- Format changed Python with Black (`--target-version py310`), and run the full test suite before committing.

## Unrelated Files in the Working Tree

These are not part of this exploration; leave them alone:
- `2605.04177v1.pdf`, `src/run_inference.py` and `tests/test_run_inference.py` were untracked before this work began.
- `interpretability/temp.md` is a draft of the user's own.
