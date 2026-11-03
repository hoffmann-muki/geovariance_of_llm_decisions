# Adopted Experiment Protocol

E5 uses the validation-selected short instruction and eighteen reviewed, country-neutral examples: three per class, averaged after normalization and normalized again. Its weights remain frozen. Llama retains its existing prompt and CoreWeave bf16 route.

The user selected this approach on 3 November 2026 after its 23/30 held-out result. The original screening guardrail remains failed; V and S each scored 2/5. This is a documented decision after validation, not a retroactive screening pass.

All 10,820 stimuli and the 21,646-classification request order are reused from the immutable [source release](../variants/main/report.md). The eighteen reference embeddings replace the original six-definition embeddings. Updated E5 inputs reach at most 254 of 512 tokens; truncation is disabled.

The [analysis registry](../analysis_plan.json) retains its endpoints and 438 contrasts per model. The protocol records the encoder amendment and preserves the registry's historical binding. Main inference has not run; the CLI only prepares inputs.
