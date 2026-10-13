# Geographic Invariance in LLM Decision-Making

This project tests whether the same event details and wording influence LLM conflict-event classification consistently across countries. It compares open-weight models with frontier models accessed through APIs.

The study has two experiments:

1. Apply controlled edits to real event descriptions, changing actor identity, framing, action wording, intensity, or source attribution while preserving event-type criteria.
2. Change the country name in matched descriptions and test whether geography alters predictions or the effects of those edits.

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the research design and proposed workflow. The five ACLED CSV files in [conflict_data/](conflict_data/) contain the input events; use their country and region columns to identify geographic coverage.

The project extends [earlier work on conflict monitoring in West Africa](https://github.com/hoffmann-muki/llm-conflict-monitoring-eval). This repository currently contains the plan and data; the experimental pipeline is planned.
