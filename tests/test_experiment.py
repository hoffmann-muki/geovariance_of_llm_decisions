"""Offline checks of the adopted encoder method and unchanged main inputs."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import e5_validation, experiment, inference
from src.utils import digest, file_digest, read_csv, verified_snapshot


class ExperimentTests(unittest.TestCase):
    def test_report_uses_recorded_decision_and_computed_metrics(self):
        protocol = json.loads(
            (experiment.ROOT / "data/experiment/protocol.json").read_text(
                encoding="utf-8"
            )
        )
        protocol["decision"]["date"] = "2030-02-04"
        protocol["decision"]["original_screening_guardrail_passed"] = True
        metrics = {
            "correct": 31,
            "n": 42,
            "by_class": {
                label: {"correct": 1, "n": 7} for label in protocol["class_order"]
            },
        }
        manifest = {
            "encoder_reference_embeddings": 18,
            "stimuli": 7,
            "planned_classifications": 14,
            "maximum_encoder_tokens": 123,
        }
        report = experiment.render_report(
            protocol,
            manifest,
            metrics,
            {"primary_contrasts": [None] * 9},
            experiment.ROOT / "data/reproduced",
        )
        self.assertIn("Adopted on 2030-02-04", report)
        self.assertIn("31/42", report)
        self.assertIn("screening: passed", report)
        self.assertIn("123/512", report)
        self.assertIn("9 contrasts", report)
        self.assertNotIn("23/30", report)

    def test_reference_centroids_reproduce_validation_predictions(self):
        config = json.loads((experiment.VALIDATION / "manifest.json").read_text())
        rows = read_csv(experiment.VALIDATION / "inputs.csv")
        _, complete = inference.journal_state(experiment.RESULTS / "attempts.jsonl")
        vectors = {}
        for phase in ("development", "test"):
            for index, batch in enumerate(
                e5_validation.batch_requests(rows, config, phase, "example_centroids")
            ):
                response = json.loads(complete[f"{phase}:{index}"]["raw_response"])
                for (sha, _), result in zip(
                    batch, sorted(response["data"], key=lambda r: r["index"])
                ):
                    vectors[sha] = result["embedding"]
        protocol = {"class_order": config["class_order"]}
        references = experiment.reviewed_references(protocol)
        protocol["encoder_reference_inputs"] = references
        prototypes = experiment.encoder_prototypes(
            [vectors[r["text_sha256"]] for r in references], protocol
        )
        correct = 0
        for row in rows:
            if row["split"] != "test":
                continue
            text = e5_validation.format_input(
                row["text"], config["instructions"]["short"]
            )
            vector = inference.normalise(vectors[digest(text)])
            scores = [
                sum(a * b for a, b in zip(vector, prototype))
                for prototype in prototypes
            ]
            label = protocol["class_order"][max(range(6), key=scores.__getitem__)]
            correct += label == row["label"]
        self.assertEqual(correct, 23)

    def test_preparation_preserves_stimuli_schedule_and_historical_failure(self):
        original = json.loads((experiment.MAIN / "protocol.json").read_text())
        stimuli = read_csv(experiment.MAIN / "stimuli.csv")
        old_lengths = {
            r["input_id"]: int(r["encoder_tokens"])
            for r in read_csv(experiment.MAIN / "length_audit.csv")
        }
        short = json.loads((experiment.VALIDATION / "manifest.json").read_text())[
            "instructions"
        ]["short"]
        counts = {
            original["encoder_input_template"].format(
                instruction=short, text=r["text"]
            ): old_lengths[r["stimulus_id"]]
            for r in stimuli
        }
        for row in experiment.reviewed_references(original):
            counts[row["text"]] = 20
        before = {
            name: file_digest(experiment.MAIN / name)
            for name in (
                "stimuli.csv",
                "request_order.csv",
                "protocol.json",
                "manifest.json",
            )
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                inference,
                "send",
                side_effect=AssertionError("No model requests allowed"),
            ),
        ):
            output = Path(directory) / "experiment"
            result = experiment.prepare(
                output, token_counter=(counts.__getitem__, original["tokenizer"])
            )
            verified_snapshot(output)
            adopted = json.loads((output / "protocol.json").read_text())
            self.assertEqual(result["encoder_reference_embeddings"], 18)
            self.assertEqual(result["planned_classifications"], 21646)
            self.assertFalse(result["main_inference_performed"])
            self.assertFalse(adopted["decision"]["original_screening_guardrail_passed"])
            self.assertEqual(adopted["decision"]["held_out_correct"], 23)
            self.assertEqual(
                adopted["decoder_system_prompt"], original["decoder_system_prompt"]
            )
            self.assertEqual(
                adopted["decoder_parameters"], original["decoder_parameters"]
            )
            self.assertEqual(
                adopted["repeat_stimulus_ids"], original["repeat_stimulus_ids"]
            )
            with self.assertRaisesRegex(ValueError, "unused"):
                experiment.prepare(output)
        self.assertEqual(
            before, {name: file_digest(experiment.MAIN / name) for name in before}
        )

    def test_invalid_reference_vectors_are_rejected(self):
        protocol = {"class_order": list(e5_validation.CLASSES)}
        references = experiment.reviewed_references(protocol)
        protocol["encoder_reference_inputs"] = references
        vector = [1.0] + [0.0] * 1023
        with self.assertRaisesRegex(ValueError, "eighteen"):
            experiment.encoder_prototypes([vector] * 17, protocol)
        with self.assertRaisesRegex(ValueError, "Zero"):
            experiment.encoder_prototypes([[0.0] * 1024] + [vector] * 17, protocol)
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            experiment.encoder_prototypes(
                [[float("nan")] * 1024] + [vector] * 17, protocol
            )


if __name__ == "__main__":
    unittest.main()
