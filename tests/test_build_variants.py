"""Checks for exact edit provenance and independent, current pair reviews."""

import copy
import tempfile
import unittest
from pathlib import Path

import build_variants as build


class VariantTests(unittest.TestCase):
    def test_edits_use_original_offsets_and_preserve_untouched_text(self):
        original = "  Alpha moved. Alpha stopped.\n"
        edited, spans = build.apply_edits(
            original,
            [
                {"old": "Alpha", "new": "The unit", "occurrence": 1},
                {"old": "Alpha", "new": "It", "occurrence": 2},
            ],
        )
        self.assertEqual(edited, "  The unit moved. It stopped.\n")
        for span in spans:
            self.assertEqual(
                original[span["original_start"] : span["original_end"]],
                span["original_span"],
            )
            self.assertEqual(
                edited[span["edited_start"] : span["edited_end"]], span["edited_span"]
            )

    def test_ambiguous_overlapping_missing_and_noop_edits_fail(self):
        examples = [
            ("one one", [{"old": "one", "new": "two"}]),
            (
                "one two",
                [{"old": "one two", "new": "three"}, {"old": "two", "new": "four"}],
            ),
            ("one", [{"old": "absent", "new": "two"}]),
            ("one", [{"old": "one", "new": "one"}]),
        ]
        for original, edits in examples:
            with (
                self.subTest(original=original, edits=edits),
                self.assertRaises(ValueError),
            ):
                build.apply_edits(original, edits)

    def test_neutral_control_changes_date_format_only(self):
        recipe = {
            "conditions": {
                "neutral_control": {
                    "edits": [{"old": "1 May 2025", "new": "2025-05-01"}],
                    "rationale": "Date format.",
                }
            },
            "omissions": {
                f: "Not used." for f in build.FAMILIES if f != "neutral_control"
            },
        }
        build.validate_conditions(recipe)
        recipe["conditions"]["neutral_control"]["edits"][0]["new"] = "2025-05-02"
        with self.assertRaisesRegex(ValueError, "changes the date"):
            build.validate_conditions(recipe)

    def test_country_rendering_offsets_and_single_placeholder(self):
        source = {
            "event_id_cnty": "source",
            "event_group_id": "family",
            "source_file": "events.csv",
            "source_record": "1",
        }
        case = build.render_case(
            "synthetic",
            "base",
            source,
            "P",
            "In {country}, workers gathered.",
            "action_wording",
            {
                "edits": [{"old": "gathered", "new": "assembled"}],
                "rationale": "Equivalent wording.",
            },
            ["China", "Mexico"],
            [],
            "Hypothetical.",
        )
        for row in case["renderings"]:
            self.assertEqual(row["edited"], f"In {row['country']}, workers assembled.")
            span = row["changes"][0]
            self.assertEqual(
                row["original"][span["original_start"] : span["original_end"]],
                "gathered",
            )
        with self.assertRaisesRegex(ValueError, "placeholder"):
            build.render_case(
                "synthetic",
                "base",
                source,
                "P",
                "In {country}, workers gathered.",
                "actor_identity",
                {
                    "edits": [{"old": "{country}", "new": "China"}],
                    "rationale": "Invalid.",
                },
                ["China"],
                [],
                "",
            )

    def review_fixture(self):
        case = {
            "case_id": "real:event:action_wording",
            "case_sha256": "current",
            "cohort": "real",
            "condition": "action_wording",
            "renderings": [
                {"original": "Workers gathered.", "edited": "Workers assembled."}
            ],
        }
        recipes = {
            "design_contexts": {"real": "designer"},
            "reviewer_contexts": {
                "a": {"kind": "AI", "context_id": "critic_a"},
                "b": {"kind": "AI", "context_id": "critic_b"},
            },
            "reviews": [
                {
                    "case_id": case["case_id"],
                    "case_sha256": "current",
                    "reviewer_id": reviewer,
                    "decision": "accept",
                    "evidence_span": "Workers assembled.",
                    "rationale": "Equivalent conduct.",
                }
                for reviewer in ("a", "b")
            ],
            "adjudications": [
                {
                    "case_id": case["case_id"],
                    "case_sha256": "current",
                    "decision": "accept",
                    "reason": "Both independent reviewers agree.",
                }
            ],
        }
        return case, recipes

    def test_stale_self_duplicated_and_unresolved_reviews_cannot_release_pairs(self):
        case, baseline = self.review_fixture()
        build.adjudicated_cases(baseline, [case])
        mutations = [
            lambda r: r["reviews"][0].update(case_sha256="obsolete"),
            lambda r: r["reviews"][0].update(evidence_span="invented evidence"),
            lambda r: r["reviews"][0].update(decision="reject"),
            lambda r: r["reviewer_contexts"]["b"].update(context_id="critic_a"),
            lambda r: r["reviewer_contexts"]["a"].update(context_id="designer"),
            lambda r: r["reviews"].pop(),
        ]
        for mutation in mutations:
            recipes = copy.deepcopy(baseline)
            mutation(recipes)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                build.adjudicated_cases(recipes, [case])

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            marker = output / "review.txt"
            marker.write_text("preserve")
            with self.assertRaisesRegex(ValueError, "already exists"):
                build.build(output)
            self.assertEqual(marker.read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
