# Geographic Invariance in LLM Decision-Making

[Study design](PROJECT_PLAN.md) · [Main stimuli](data/variants/main/report.md) · [Input review](data/accepted/report.md) · [Data audit](data/processed/quality_report.md)

Source data is in [conflict_data/](conflict_data/); reviewed source inputs are in [data/accepted/](data/accepted/).

`src/` holds preparation, experiment, analysis and evaluation code; tests and data remain separate.

Python 3.10+; install `requirements.txt` for token-length checks. Reproduce into unused directories:

```bash
python3 -m pip install -r requirements.txt
python3 -m src.prepare_data --output-dir data/processed_new
python3 -m src.finalize_inputs --source-dir data/processed_new --output-dir data/accepted_new
python3 -m src.build_main_variants --input-dir data/accepted_new --output-dir data/variants/main_new
python3 -m unittest discover -s tests
```

Preparation performs no inference and needs no API keys. Completed snapshots are preserved; recipes and review decisions are in [data/reviews/](data/reviews/).
