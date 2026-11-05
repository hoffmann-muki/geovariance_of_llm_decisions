# Geographic Invariance in LLM Decision-Making

[Study design](PROJECT_PLAN.md) · [Main stimuli](data/variants/main/report.md) · [Adopted protocol](data/experiment/report.md) · [Results](results/main/report.md)

Source data is in [conflict_data/](conflict_data/); reviewed source inputs are in [data/accepted/](data/accepted/).

`src/` holds preparation, experiment, analysis and evaluation code; tests and data remain separate.

Python 3.10+; install `requirements.txt` for token-length checks. Reproduce into unused directories:

```bash
python3 -m pip install -r requirements.txt
python3 -m src.prepare_data --output-dir data/processed_new
python3 -m src.finalize_inputs --source-dir data/processed_new --output-dir data/accepted_new
python3 -m src.build_main_variants --input-dir data/accepted_new --output-dir data/variants/main_new
python3 -m src.analysis --output-file data/analysis_plan_new.json
python3 -m src.experiment --output-dir data/experiment_new
python3 -m unittest discover -s tests
```

Preparation performs no inference and needs no API keys. Completed snapshots are preserved; recipes and review decisions are in [data/reviews/](data/reviews/).

Archived model checks retain producer-code hashes; replay them from their producing revision (`52beeb9`).

Current model settings and lengths are in [data/experiment/](data/experiment/); it reuses the main stimuli and request order.

Configure `OPENROUTER_API_KEY` and `DEEPINFRA_API_KEY` in the ignored `.env` file. Execute the frozen schedule, then analyze the completed run:

```bash
python3 -m src.run_experiment --output-dir runs/main --budget 3 --workers 8
python3 -m src.evaluate --run-dir runs/main --output-dir results/main
```

The runner resumes unfinished work and verifies completed runs without resending requests. Raw responses and accounting are retained in its SQLite journal. [Provider checks](runs/compatibility/report.md) and [E5 validation](runs/e5_validation/report.md) preserve the earlier evidence.
