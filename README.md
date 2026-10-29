# Geographic Invariance in LLM Decision-Making

[Study design](PROJECT_PLAN.md) · [Input review](data/accepted/report.md) · [Pilot stimuli](data/variants/pilot/report.md) · [Data audit](data/processed/quality_report.md)

Reviewed inputs: [main](data/accepted/main.csv), [pilot](data/accepted/pilot.csv), and [reserves](data/accepted/available_reserves.csv). Source data is in [conflict_data/](conflict_data/).

Python 3.10+; install `requirements.txt` for token-length checks. Reproduce into unused directories:

```bash
python3 -m pip install -r requirements.txt
python3 prepare_data.py --output-dir data/processed_new
python3 finalize_inputs.py --source-dir data/processed_new --output-dir data/accepted_new
python3 build_variants.py --input-dir data/accepted_new --output-dir data/variants_new
python3 prepare_model_inputs.py --output-dir data/model_inputs_new
python3 -m unittest discover -s tests
```

Preparation performs no inference and needs no API keys. [Model input formats and length audit](data/model_inputs/manifest.json) cover existing baselines and pilot stimuli; main variants still require construction and review. Completed snapshots are preserved. Review decisions are in [data/reviews/](data/reviews/).
