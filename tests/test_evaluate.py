"""Frozen estimands, paired-family resampling and multiplicity safeguards."""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy import stats

from src import evaluate
from src.utils import digest, file_digest, write_json


class EvaluationTests(unittest.TestCase):
    def test_source_snapshot_cannot_substitute_a_different_prespecified_input(self):
        root = evaluate.ROOT
        run = {
            "complete": True,
            "outputs": {},
            "inputs": {
                "experiment_manifest_sha256": file_digest(
                    root / "data/experiment/manifest.json"
                ),
                "protocol_sha256": file_digest(root / "data/experiment/protocol.json"),
                "analysis_plan_sha256": file_digest(root / "data/analysis_plan.json"),
            },
        }
        original = file_digest

        def substituted(path):
            return (
                "substituted"
                if path == root / "data/variants/main/stimuli.csv"
                else original(path)
            )

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            write_json(run_dir / "manifest.json", run)
            with patch.object(evaluate, "file_digest", side_effect=substituted):
                with self.assertRaisesRegex(
                    ValueError, "Prespecified source binding changed: stimuli.csv"
                ):
                    evaluate.evaluate(run_dir, Path(directory) / "analysis")

    def test_prediction_identity_and_provenance_must_match_the_schedule(self):
        stimulus = {
            "stimulus_id": "input",
            "label": "B",
            "source_family_id": "family",
            "cohort": "real",
            "base_id": "base",
            "country": "A",
            "condition": "original",
            "text_sha256": digest("text"),
        }
        task = {
            "request_id": "request",
            "model": "model",
            "stimulus_id": "input",
            "replicate": "0",
        }
        prediction = {
            **stimulus,
            **task,
            "expected_label": "B",
            "status": "valid",
            "prediction": "B",
        }
        evaluate.validate_prediction_inventory([stimulus], [prediction], [task])
        for predictions in (
            [],
            [prediction, prediction],
            [{**prediction, "request_id": "other"}],
        ):
            with self.assertRaisesRegex(ValueError, "identities"):
                evaluate.validate_prediction_inventory([stimulus], predictions, [task])
        for key in ("source_family_id", "text_sha256", "expected_label"):
            with self.assertRaisesRegex(ValueError, "provenance"):
                evaluate.validate_prediction_inventory(
                    [stimulus], [{**prediction, key: "wrong"}], [task]
                )

    def test_originally_correct_sensitivity_matches_both_countries(self):
        rows = []
        for country in ("A", "B"):
            for i in range(20):
                rows.append(
                    {
                        "population": "synthetic:all_eligible_classes",
                        "condition": "action_wording",
                        "country": country,
                        "source_family_id": str(i),
                        "valid": True,
                        "originally_correct": i != (0 if country == "A" else 1),
                        "qualified": False,
                        "cohort": "synthetic",
                        "label": "B",
                        "d": 0,
                    }
                )
        units = evaluate.unit_maps(rows, "originally_correct")
        contrast = {
            "population": "synthetic:all_eligible_classes",
            "condition": "action_wording",
            "country_a": "A",
            "country_b": "B",
            "test": "paired_t",
        }
        plan = {
            "minimum_independent_families": 10,
            "uncertainty": {"replicates": 100, "confidence_level": 0.95},
        }
        result = evaluate.estimate(
            contrast, units, plan, np.random.default_rng(42), "originally_correct"
        )
        self.assertEqual(result["valid_n_a"], 18)
        self.assertEqual(result["valid_n_b"], 18)

    def test_welch_and_paired_tests_match_reference_implementations(self):
        a = np.array([-1, 0, 0, 1, 1] * 3, dtype=float)
        b = np.array([0, 1, 1, 1, 1] * 3, dtype=float)
        for paired in (False, True):
            effect, p, status = evaluate.mean_test(a, b, paired)
            expected = (
                stats.ttest_rel(a, b)
                if paired
                else stats.ttest_ind(a, b, equal_var=False)
            )
            self.assertAlmostEqual(effect, a.mean() - b.mean())
            self.assertAlmostEqual(p, expected.pvalue)
            self.assertEqual(status, "tested")
        _, p, status = evaluate.mean_test(np.zeros(15), b, False)
        self.assertIsNotNone(p)
        self.assertEqual(status, "tested")

    def test_constant_contrasts_have_no_manufactured_significance(self):
        for paired in (False, True):
            effect, p, status = evaluate.mean_test(np.ones(12), np.zeros(12), paired)
            self.assertEqual(effect, 1)
            self.assertIsNone(p)
            self.assertEqual(status, "zero_contrast_variance")

    def test_holm_keeps_unavailable_planned_tests_in_the_family(self):
        adjusted = evaluate.holm([0.01, 0.04, None, 0.001])
        for actual, expected in zip(adjusted, [0.03, 0.08, None, 0.004]):
            if expected is None:
                self.assertIsNone(actual)
            else:
                self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(evaluate.holm([0.001] + [None] * 437)[0], 0.438)

    def test_triplets_require_common_valid_original_neutral_and_edit(self):
        rows = [
            {
                "stimulus_id": condition,
                "cohort": "real",
                "base_id": "source",
                "source_family_id": "family",
                "country": "A",
                "label": "B",
                "label_support": "supported",
                "condition": condition,
            }
            for condition in ("original", "neutral_control", "action_wording")
        ]
        predictions = [
            {
                "stimulus_id": condition,
                "model": "model",
                "replicate": "0",
                "status": "valid",
                "prediction": label,
            }
            for condition, label in (
                ("original", "B"),
                ("neutral_control", "P"),
                ("action_wording", "B"),
            )
        ]
        record = next(
            r
            for r in evaluate.triple_records(rows, predictions, "model")
            if r["condition"] == "action_wording"
        )
        self.assertEqual(record["d"], -1)
        predictions[1].update(status="invalid", prediction="")
        record = next(
            r
            for r in evaluate.triple_records(rows, predictions, "model")
            if r["condition"] == "action_wording"
        )
        self.assertFalse(record["valid"])
        self.assertIsNone(record["d"])

    def test_matching_and_bootstrap_preserve_joint_family_differences(self):
        contrast = {
            "id": "test",
            "population": "synthetic:all_eligible_classes",
            "condition": "action_wording",
            "country_a": "A",
            "country_b": "B",
            "test": "paired_t",
        }
        plan = {
            "minimum_independent_families": 10,
            "uncertainty": {"replicates": 2000, "confidence_level": 0.95},
        }
        a = {
            str(i): {"d": (-1) ** i, "label": "B" if i < 10 else "P"} for i in range(20)
        }
        b = {key: dict(value) for key, value in reversed(list(a.items()))}
        units = {
            (contrast["population"], contrast["condition"], "A"): a,
            (contrast["population"], contrast["condition"], "B"): b,
        }
        result = evaluate.estimate(
            contrast, units, plan, np.random.default_rng(42), "primary"
        )
        self.assertEqual(result["effect"], 0)
        self.assertEqual(result["ci_low"], 0)
        self.assertEqual(result["ci_high"], 0)
        self.assertIsNone(result["p_value"])
        # Ten valid families in each country do not imply ten common paired families.
        units[(contrast["population"], contrast["condition"], "A")] = dict(
            list(a.items())[:10]
        )
        units[(contrast["population"], contrast["condition"], "B")] = dict(
            list(b.items())[:10]
        )
        result = evaluate.estimate(
            contrast, units, plan, np.random.default_rng(42), "primary"
        )
        self.assertEqual(result["valid_n_a"], 0)
        self.assertEqual(result["test_status"], "insufficient_valid_families")

    def test_empirical_mean_bootstrap_has_expected_first_two_moments(self):
        values = np.array([-1, -1, 0, 1, 1], dtype=float)
        draws = evaluate.bootstrap_means(values, np.random.default_rng(42), 50000)
        self.assertAlmostEqual(draws.mean(), values.mean(), delta=0.01)
        self.assertAlmostEqual(draws.var(), values.var() / len(values), delta=0.01)


if __name__ == "__main__":
    unittest.main()
