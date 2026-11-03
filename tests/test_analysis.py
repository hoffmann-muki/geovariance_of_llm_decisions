"""Prespecified contrast eligibility must not depend on model responses."""

import unittest
from collections import Counter

from src import analysis
from src.utils import read_csv


class AnalysisPlanTests(unittest.TestCase):
    def test_frozen_eligibility_and_sparse_real_identity(self):
        contrasts = analysis.planned_contrasts(read_csv(analysis.MAIN / "stimuli.csv"))
        real = [c for c in contrasts if c["population"].startswith("real:")]
        synthetic = [c for c in contrasts if c["population"].startswith("synthetic:")]
        self.assertEqual(len(real), 168)
        self.assertEqual(len(synthetic), 270)
        self.assertEqual(len({c["id"] for c in contrasts}), 438)
        self.assertNotIn("actor_identity", {c["condition"] for c in real})
        self.assertEqual(set(Counter(c["condition"] for c in synthetic).values()), {45})
        self.assertTrue(
            all(min(c["prepared_n_a"], c["prepared_n_b"]) >= 10 for c in contrasts)
        )

    def test_missing_neutral_and_unmatched_bases_do_not_qualify(self):
        def row(base, country, condition):
            return {
                "cohort": "synthetic",
                "base_id": base,
                "source_family_id": base,
                "country": country,
                "condition": condition,
                "label": "B",
            }

        rows = [
            row(str(i), country, condition)
            for i in range(10)
            for country in ("A", "B")
            for condition in ("original", "neutral_control", "action_wording")
        ]
        self.assertEqual(len(analysis.planned_contrasts(rows)), 1)
        missing = [
            r
            for r in rows
            if not (
                r["base_id"] == "0"
                and r["country"] == "A"
                and r["condition"] == "neutral_control"
            )
        ]
        self.assertEqual(analysis.planned_contrasts(missing), [])
        for record in rows:
            if record["country"] == "B":
                record["base_id"] = record["source_family_id"] = (
                    "different-" + record["base_id"]
                )
        self.assertEqual(analysis.planned_contrasts(rows), [])


if __name__ == "__main__":
    unittest.main()
