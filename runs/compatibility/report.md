# Provider Compatibility Check

Technical gate: **passed**. Six held-out events; no main-study texts submitted.

| Model | Valid classifications | Original labels correct | Repeat labels unchanged |
| --- | --- | --- | --- |
| intfloat/multilingual-e5-large-instruct | 9/9 | 3/6 | 3/3 |
| meta-llama/llama-3.1-8b-instruct | 9/9 | 6/6 | 3/3 |

19 POST attempts, including the six-prototype E5 batch. Provider-reported charges: $0.000795; other usage/reserves estimated: $0.000019.

E5 vectors are checked for dimension, finite nonzero norms, index order and exact input-token accounting. Llama uses strict JSON labels, CoreWeave bf16 routing without fallbacks, and disabled OpenRouter response caching. Fresh generation IDs are checked for uniqueness.

This small check establishes API compatibility, not classification validity or geographic invariance. Six source labels provide descriptive diagnostics only; prompts and main stimuli were not tuned from these predictions. Repeat requests do not add independent events.

The manifest freezes settings, selection and code hashes. The journal retains raw responses, usage, timing and failures. Providers do not expose an immutable served-weight revision; a later main run must recheck availability and routing.
