# Main Stimulus Preparation

Status: AI-reviewed data release; inference not performed.

- Real: 2320 rows, including all 590 unchanged accepted baselines.
- Synthetic: 8500 rows from 120 source-inspired bases (20/class), each under ten countries.
- Review units: 2651; accepted: 2580. Two independent AI contexts review every proposed edit/base; original real inputs retain their existing input review.
- Maximum E5 input: 348/512 tokens, including instruction and special tokens. No truncation.

`stimuli.csv` contains exact texts, baseline/family links and case restrictions. `changes.csv` records literal spans; `omissions.csv` and `coverage.csv` define eligible denominators. `protocol.json` freezes the shared class definitions and model-visible formatting. Source qualifications remain attached to real inputs; do not filter on later model correctness.
`request_order.csv` freezes the randomized two-model schedule and three fresh repeat texts per model. Expected labels, conditions, identifiers and review metadata must not be sent to models. Reviewers saw intended labels and construction rationales; these judgments are not independent model-accuracy measurements.

Synthetic notes and identity/framing/intensity/attribution edits are counterfactuals, not historical findings. Synthetic source families shared with Experiment A are listed in the manifest; keep these links when resampling. Pilot/reserve families are excluded. AI review is not human verification and contexts may share systematic errors. Provider compatibility and classification behavior have not been tested for the selected models; no main inference is authorized by this data release.
