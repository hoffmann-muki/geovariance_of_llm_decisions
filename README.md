# Geographic Invariance in LLM Decision-Making

[Study design](PROJECT_PLAN.md) · [Input review](data/accepted/report.md) · [Pilot stimuli](data/variants/pilot/report.md) · [Data audit](data/processed/quality_report.md)

Reviewed inputs: [main](data/accepted/main.csv), [pilot](data/accepted/pilot.csv), and [reserves](data/accepted/available_reserves.csv). Source data is in [conflict_data/](conflict_data/).

Python 3.10+; standard library only. Reproduce into unused directories:

```bash
python3 prepare_data.py --output-dir data/processed_new
python3 finalize_inputs.py --source-dir data/processed_new --output-dir data/accepted_new
python3 build_variants.py --input-dir data/accepted_new --output-dir data/variants_new
python3 -m unittest discover -s tests
```

Existing outputs are never overwritten. Initial settings are in `prepare_data.py`; accepted quotas and review decisions are in [data/reviews/](data/reviews/).
