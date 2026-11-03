"""Network-free checks of validation separation, selection and interrupted batches."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import e5_validation as validation
from src import inference, model_inputs
from src.utils import digest, file_digest, write_csv, write_json


def fixture():
    rows = validation.reference_rows()
    for index, label in enumerate(validation.CLASSES):
        for n in range(10):
            text = f"Validation fixture text {index * 10 + n}."
            rows.append(
                {
                    "input_id": f"event-{index}-{n}",
                    "kind": "event",
                    "split": "development" if n < 5 else "test",
                    "label": label,
                    "text": text,
                    "text_sha256": digest(text),
                    "source_family_id": f"family-{index}-{n}",
                }
            )
    original = json.loads((inference.MAIN / "protocol.json").read_text())
    config = {
        "scope": "e5_classifier_validation_only",
        "selection": "Highest development macro-F1, accuracy, then method-list order",
        "class_order": validation.CLASSES,
        "methods": validation.METHODS,
        "batch_size": validation.BATCH_SIZE,
        "budget_usd": validation.BUDGET,
        "development_n": 30,
        "test_n": 30,
        "readiness_guardrail": {
            "test_correct_at_least": 24,
            "test_correct_per_class_at_least": 3,
            "improvement_over_baseline_at_least": 3,
        },
        "instructions": {
            "baseline": original["encoder_instruction"],
            "short": validation.SHORT_INSTRUCTION,
        },
        "models": {
            model_inputs.ENCODER: {
                "api": "deepinfra",
                "model": model_inputs.ENCODER,
                "dimension": 1024,
                "input_price_usd": 1e-8,
                "output_price_usd": 0,
            }
        },
        "inputs": {
            "validation_code_sha256": file_digest(Path(validation.__file__)),
            "inference_code_sha256": file_digest(Path(inference.__file__)),
            "utilities_sha256": file_digest(validation.ROOT / "src/utils.py"),
            "main_manifest_sha256": file_digest(inference.MAIN / "manifest.json"),
            "main_protocol_sha256": file_digest(inference.MAIN / "protocol.json"),
        },
    }
    tokens, vectors = {}, {}
    for row in rows:
        texts = (
            [row["text"]]
            if row["split"] == "reference"
            else [
                validation.format_input(row["text"], instruction)
                for instruction in config["instructions"].values()
            ]
        )
        for text in texts:
            tokens[digest(text)] = 10
            # Example centroids win: definitions collapse onto B; short queries match their source class.
            label = (
                row["label"]
                if row["kind"] == "example_centroids"
                or (
                    row["kind"] == "event"
                    and text.startswith("Instruct: " + validation.SHORT_INSTRUCTION)
                )
                else "B"
            )
            vector = [0.0] * 1024
            vector[validation.CLASSES.index(label)] = 1.0
            vectors[text] = vector
    config["tokens"] = tokens
    return rows, config, vectors


class E5ValidationTests(unittest.TestCase):
    def test_execution_respects_frozen_batch_budget_and_threshold_values(self):
        rows, config, _ = fixture()
        config["batch_size"] = 7
        batches = validation.batch_requests(rows, config, "development")
        self.assertEqual(len(batches), 13)
        self.assertTrue(all(len(batch) <= 7 for batch in batches))
        config["budget_usd"] = 1e-12
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(inference, "send") as sender,
        ):
            journal = Path(directory) / "attempts.jsonl"
            with self.assertRaisesRegex(ValueError, "budget reached"):
                validation.fetch_batch(
                    journal, "development:0", batches[0], config, "test-key"
                )
            sender.assert_not_called()
            self.assertFalse(journal.exists())
        candidate = {
            "correct": 24,
            "by_class": {label: {"correct": 4} for label in validation.CLASSES},
        }
        baseline = {"correct": 11}
        thresholds = config["readiness_guardrail"]
        self.assertTrue(validation.guardrail_passed(candidate, baseline, thresholds))
        for change in (
            {"test_correct_at_least": 25},
            {"test_correct_per_class_at_least": 5},
            {"improvement_over_baseline_at_least": 14},
        ):
            self.assertFalse(
                validation.guardrail_passed(
                    candidate, baseline, {**thresholds, **change}
                )
            )

    def test_programming_errors_are_not_reclassified_as_provider_failures(self):
        rows, config, _ = fixture()
        batch = validation.batch_requests(rows, config, "development")[0]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                inference,
                "send",
                side_effect=RuntimeError("unexpected implementation failure"),
            ),
        ):
            journal = Path(directory) / "attempts.jsonl"
            with self.assertRaisesRegex(RuntimeError, "implementation failure"):
                validation.fetch_batch(
                    journal, "development:0", batch, config, "test-key"
                )
            with self.assertRaisesRegex(ValueError, "Unfinished"):
                inference.journal_state(journal)

    def test_method_selection_uses_macro_f1_then_accuracy_then_simplicity(self):
        scores = {
            method: {"macro_f1": 0.5, "accuracy": 0.5} for method in validation.METHODS
        }
        self.assertEqual(validation.select_method(scores), "baseline")
        scores["short_definitions"]["accuracy"] = 0.6
        self.assertEqual(validation.select_method(scores), "short_definitions")
        scores["example_centroids"]["macro_f1"] = 0.6
        self.assertEqual(validation.select_method(scores), "example_centroids")
        for label in validation.CLASSES:
            self.assertEqual(len(validation.EXAMPLES[label]), 3)

    def test_batching_does_not_submit_test_during_development(self):
        rows, config, _ = fixture()
        development = [
            text
            for batch in validation.batch_requests(rows, config, "development")
            for _, text in batch
        ]
        test = [
            text
            for batch in validation.batch_requests(
                rows, config, "test", "example_centroids"
            )
            for _, text in batch
        ]
        self.assertEqual(len(development), 90)
        self.assertEqual(len(test), 60)
        self.assertFalse(set(development) & set(test))
        for row in rows:
            if row["split"] == "test":
                self.assertFalse(any(row["text"] in text for text in development))
        self.assertTrue(
            all(
                len(batch) <= 24
                for batch in validation.batch_requests(rows, config, "development")
            )
        )

    def test_collapse_is_visible_in_macro_f1_and_class_denominators(self):
        scores = validation.metrics(
            [{"label": label, "prediction": "V"} for label in validation.CLASSES]
        )
        self.assertEqual(scores["correct"], 1)
        self.assertAlmostEqual(scores["macro_f1"], 1 / 21)
        self.assertEqual(scores["by_class"]["B"], {"correct": 0, "n": 1})

    def test_reviews_need_two_matching_contexts_and_literal_evidence(self):
        row = {
            "input_id": "test",
            "label": "B",
            "text": "Two organized armed groups fought.",
            "text_sha256": digest("Two organized armed groups fought."),
        }
        assessment = {
            "input_id": "test",
            "text_sha256": row["text_sha256"],
            "predicted_label": "B",
            "decision": "accept",
            "evidence": "armed groups fought",
        }
        reviews = {
            "reviews": [
                {"reviewer": ident, "assessments": [dict(assessment)]}
                for ident in ("A", "B")
            ],
            "adjudications": [dict(assessment)],
        }
        validation.validate_reviews([row], reviews)
        reviews["reviews"][1]["assessments"][0]["decision"] = "withhold"
        with self.assertRaisesRegex(ValueError, "concordant"):
            validation.validate_reviews([row], reviews)
        reviews["reviews"][1]["assessments"][0] = {
            **assessment,
            "evidence": "invented evidence",
        }
        with self.assertRaisesRegex(ValueError, "literal evidence"):
            validation.validate_reviews([row], reviews)

    def test_main_family_is_rejected_before_any_provider_call(self):
        from src.utils import read_csv

        source = read_csv(inference.MAIN / "stimuli.csv")[0]
        row = {**source, "input_id": source["source_event_id"], "kind": "event"}
        with patch.object(inference, "send") as send:
            with self.assertRaisesRegex(ValueError, "overlaps allocated families"):
                validation.validate_sources([row])
            send.assert_not_called()

    def test_development_and_test_resume_never_resend_completed_batches(self):
        rows, config, vectors = fixture()
        requests = []

        def sender(model, body, key):
            requests.append(body)
            return (
                json.dumps(
                    {
                        "model": model["model"],
                        "data": [
                            {"index": i, "embedding": vectors[text]}
                            for i, text in enumerate(body["input"])
                        ],
                        "usage": {"prompt_tokens": len(body["input"]) * 10},
                    }
                ),
                None,
            )

        with tempfile.TemporaryDirectory() as directory:
            inputs, outputs = Path(directory) / "inputs", Path(directory) / "outputs"
            inputs.mkdir()
            write_csv(inputs / "inputs.csv", validation.FIELDS, rows)
            write_json(inputs / "reviews.json", {})
            config["outputs"] = {
                path.name: {"sha256": file_digest(path)} for path in inputs.iterdir()
            }
            write_json(inputs / "manifest.json", config)
            original_fetch = validation.fetch_batch

            def interrupted(path, ident, *args):
                if ident.endswith(":1"):
                    raise RuntimeError("Interrupted between completed batches")
                return original_fetch(path, ident, *args)

            with (
                patch.object(
                    validation,
                    "metadata",
                    return_value=config["models"][model_inputs.ENCODER],
                ),
                patch.object(validation, "validate_sources"),
                patch.object(validation, "validate_reviews"),
                patch.object(validation, "load_secret", return_value="TEST-CREDENTIAL"),
                patch.object(inference, "send", side_effect=sender),
            ):
                for phase in ("development", "test"):
                    with patch.object(
                        validation, "fetch_batch", side_effect=interrupted
                    ):
                        with self.assertRaisesRegex(RuntimeError, "Interrupted"):
                            validation.run_phase(inputs, outputs, phase)
                    result = validation.run_phase(inputs, outputs, phase)
                    self.assertTrue(result[phase + "_complete"])
                self.assertEqual(result["selected_method"], "example_centroids")
                self.assertTrue(result["guardrail_passed"])
                self.assertEqual(len(requests), 7)
                validation.run_phase(inputs, outputs, "test")
                self.assertEqual(len(requests), 7)


if __name__ == "__main__":
    unittest.main()
