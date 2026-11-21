# AGENT_SPEC.md — Part 2: Mechanistic Interpretability of Country Effects in Conflict Event Coding

## 1. Mission & Scope
Execute Part 2 of the `country_effect` study to determine **where** and **how** country names alter `meta-llama/Llama-3.2-3B-Instruct` decisions between `Battle` and `Violence against civilians` (BV), using `Protest` vs. `Riot` (PR) as a within-model null control.

- **Primary Deliverable:** Reproducible code, manifests, statistical summaries, figures, and a publication-ready report in `interpretability/country_effect/part2_attribution/`.
- **Immutable Boundary:** NEVER modify, overwrite, or delete any file under `interpretability/country_effect/part1_behavioral/`.
- **Context Reference:** Read `interpretability/country_effect/part1_behavioral/` (`dataset/`, `inputs/`, `report/results.json`) and `interpretability/geoflip/` (`render.py`, `scoring.py`) before writing any code.

---

## 2. Competing Hypotheses & Pre-Registered Falsification Criteria

Part 1 established that BV country shifts are positive across **every** item for the US/UK and largest on the clearest VAC evidence (`anchor_lo` unarmed farmers: US $+0.97$, UK $+1.22$, within-description SD $0.56$), whereas ambiguous items (`security_guard`, `miners`, `drivers`) flip only because $s_{\text{none}}$ lies near $0$ within $\pm 0.5$ nats. Part 2 adjudicates three hypotheses:

| Hypothesis | Proposed Mechanism | Behavioral Prediction (Stage 1) | Mechanistic Prediction (Stage 2) | Falsified If |
| :--- | :--- | :--- | :--- | :--- |
| **$H_1$: Additive Geopolitical Prior** | Country tokens inject a description-independent class vector into the residual stream toward the readout position. | 1-token real countries retain wide spread vs. 1-token controls; held-out contexts (`12–19`) replicate with $r > 0.90$. | Patching late residual states (L18–L27) **at `pos_readout`** transfers $\Delta$ across items; effect does **not** route through evidence tokens and is $\ge$ as large on `anchor_lo` as on `security_guard`. | Patching `pos_readout` fails unless evidence tokens are patched, or 1-token controls collapse real-country variance. |
| **$H_2$: Evidence Modulation ("Can Fight Back")** | Country modifies how the model encodes the target group's armed capacity during the evidence sentence. | Country shift concentrates on items where armed status is underspecified (`security_guard`, `miners`, `drivers`). | Causal mediation routes **through `span_ev_group`**; projecting `span_ev_group` onto an armed-status probe (`v_armed`, fit on `none` prompts) shifts along the armed axis for High-Battle vs. High-VAC countries. | Evidence-span patching has near-zero causal effect on $\Delta$, or `v_armed` projection does not shift with country. |
| **$H_3$: Tokenization / Familiarity Artifact** | Shift is driven by 1-token vs. multi-token geometry or pretraining frequency rather than country semantics. | 1-token non-country proper nouns / obscure places reproduce the spread of 1-token countries; 3-token phrases match US/UK. | Residual shifts correlate with token count / unigram frequency rather than geopolitical region, and replicate in PR. | 1-token controls show near-zero spread ($\text{SD} \le 0.08$) while 1-token real countries remain wide ($\text{SD} \ge 0.18$). |

---

## 3. Hard Invariants & Prompt-Alignment Contract

Every script in Part 2 MUST enforce these invariants via automated assertions and `pytest` before producing outputs.

### 3.1 Model, Precision, and Scoring Invariants
1. **Exact Model & Revision:** `meta-llama/Llama-3.2-3B-Instruct`, revision `0cb88a4f764b7a12671c53f0838cd831a0843b95` (28 layers, hidden size 3072, RMSNorm).
2. **Precision:** `torch.bfloat16` for all forward, logit-lens, patching, and steering runs. For Integrated Gradients (IG), accumulate gradients in `torch.float32` (log exact dtype mode in `manifest.json`).
3. **No Batching (`batch_size = 1`):** Always execute **one prompt per forward pass**. Batched `bf16` shifts log-probabilities by up to 0.6 nats and is strictly forbidden.
4. **Exact Chat Template & Readout:**
   - Fixed date header: `Today Date: 26 Jul 2024`.
   - Readout token: the final prompt token (`"\n\n"`) immediately following `<|start_header_id|>assistant<|end_header_id|>`.
   - Assert `"A"` and `"B"` resolve to single token IDs via `geoflip.scoring.answer_token_ids`.
5. **Order Averaging:**
   - Score every prompt in **both** Order A and Order B.
   - Always compute $d = \log P(\text{higher}) - \log P(\text{lower})$ (positive = `Battle` in BV, `Riot` in PR) and average $s = (d_A + d_B) / 2$ so additive letter bias ($\approx 1.1\text{–}1.2$ nats) cancels cleanly.
6. **Sample Split Hygiene:**
   - `context_id` indices `0–7`: Pilot split (allowed for fitting linear probes).
   - `context_id` indices `8–11`: Study split (used for all Part 2 mechanistic experiments).
   - `context_id` indices `12–19`: **Held-out confirmatory split**. Run **once** in Stage 1B; NEVER inspect or tune on `12–19`.

### 3.2 Prompt-Formatting, Tokenization & Span Alignment Rules (`geoflip/spans.py`)
Do **not** hardcode token indices anywhere in the codebase. Every `(description_id, name, order)` prompt has different token offsets because:
- Order A vs. Order B changes the system prompt token length (`Violence against civilians` definition is longer than `Battle`), shifting every user-turn token index.
- Frames `f1` vs. `f2` place `{COUNTRY}` at different character offsets.
- Wordings `W1` vs. `W2` swap subjects and verbs inside `evidence_sentence`.
- Llama-3 BPE tokenization is whitespace-sensitive (`" Kenya"` != `"Kenya"`).

Therefore, `interpretability/geoflip/spans.py` MUST implement deterministic character-to-token resolution on the **full rendered chat string**:
1. Render `user_text` via `geoflip.render` and `full_prompt` via `geoflip.scoring.render_prompt(system_prompt, user_text)`.
2. Tokenize with `add_special_tokens=False` (since `render_prompt` already emits `<|begin_of_text|>`) and `return_offsets_mapping=True`. Assert `input_ids` matches `geoflip.scoring` token-for-token.
3. Locate `user_char_start = full_prompt.index(user_text)` and resolve four non-overlapping half-open token slices `[start, end)` from exact character offsets:
   - `span_name`: maps the substring `" " + rendered_name` (including the leading space so the BPE word-boundary prefix is captured cleanly). Assert `len(range(*span_name)) == expected_token_count` (`1` for 1-token names, `3` for `"the United States"` / `"the United Kingdom"`). For `name == "none"`, `span_name = None`.
   - `span_ev_group`: maps the exact target-group noun phrase inside `evidence_sentence` (e.g., `"farmers"`, `"a night watchman"`), defined explicitly for every `(item, wording)` pair in `configs/mechanistic_panel.json`.
   - `span_ev_rest`: all remaining tokens of `evidence_sentence` (including its terminal period) excluding `span_ev_group`.
   - `pos_readout`: single index `len(input_ids) - 1`, asserted to decode to `"\n\n"` right after `<|start_header_id|>assistant<|end_header_id|>`.
4. **Round-Trip Invariant:** For every resolved non-empty span `[start, end)`, assert `tokenizer.decode(input_ids[start:end]) == exact_expected_substring`.

### 3.3 Exact PyTorch Hook & Attribution Contract (`geoflip/hooks.py`, `geoflip/attribution.py`)
1. **Residual Stream Definition:** `h_l` (`l` in `0..27`) is the primary hidden-state tensor returned as `output[0]` by `model.model.layers[l]` (post-attention + post-MLP residual stream, shape `[1, seq_len, 3072]`). `h_{-1}` is the output of `model.model.embed_tokens`.
2. **RMSNorm-Calibrated Logit Lens:** When projecting intermediate layer `l` at token index `t`, you MUST pass `h_{l, t}` through `model.model.norm` before `model.lm_head`:
   $$\text{logits}^{(l)}_t = \text{lm\_head}(\text{model.model.norm}(h_{l, t}))$$
3. **Token-Length-Matched Causal Patching:**
   - Only patch between prompts with **identical `seq_len`** (same `description_id`, same `order`, and token-length-matched `name_base` vs. `name_target`). Assert `len(ids_base) == len(ids_target)` and identical token IDs at every position outside `span_name`.
   - Respect causal masking: patching `span_S != pos_readout` at layer `l` can only affect `pos_readout` via layers `l+1..27`. Sweep `l` in `0..26` for `span_name`, `span_ev_group`, and `span_ev_rest`, and `l` in `0..27` for `pos_readout`.
4. **`inputs_embeds` Parity & Matched-Length IG:**
   - Before running IG, assert that `model(inputs_embeds=model.model.embed_tokens(input_ids))` reproduces `model(input_ids=input_ids)` logits identically.
   - Run IG **only** on 1-token real countries paired against a 1-token control baseline (`len(ids_real) == len(ids_ctrl)`), interpolating $E(\alpha) = E_{\text{ctrl}} + \alpha (E_{\text{real}} - E_{\text{ctrl}})$ over 50 midpoint steps $\alpha_k = (k + 0.5)/50$, computing attributions separately for Order A and Order B against scalar $d_{\text{order}}$, then averaging.

---

## 4. Repository Layout & Manifest Standard

```text
interpretability/
├── geoflip/
│   ├── render.py                     # Existing (keep compatible)
│   ├── scoring.py                    # Existing (keep compatible)
│   ├── spans.py                      # [NEW] Exact char-to-token span resolver
│   ├── hooks.py                      # [NEW] Residual cache, RMSNorm logit lens, span patching
│   ├── attribution.py                # [NEW] Token-matched IG & residual steering
│   └── tests/
│       └── test_part2_invariants.py  # [NEW] Mandatory unit & parity suite
└── country_effect/
    ├── part1_behavioral/             # READ-ONLY (immutable)
    └── part2_attribution/            # [NEW]
        ├── AGENT_SPEC.md
        ├── PREREGISTRATION.md
        ├── configs/
        │   ├── stimuli_controls.json
        │   └── mechanistic_panel.json
        ├── scripts/
        │   ├── 00_verify_parity.py
        │   ├── 01_build_controls.py
        │   ├── 02_run_behavioral_controls.py
        │   ├── 03_run_heldout_replication.py
        │   ├── 04_run_logit_lens.py
        │   ├── 05_run_activation_patching.py
        │   ├── 06_run_probes_and_steering.py
        │   ├── 07_run_matched_ig.py
        │   ├── 08_run_generality_check.py
        │   └── 09_compile_report_artifacts.py
        ├── outputs/                  # Each subdirectory writes manifest.json
        └── report/
            ├── part2_report.md
            ├── part2_section.tex
            └── figures/
