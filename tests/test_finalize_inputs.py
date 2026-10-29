"""Acceptance guards for AI-only input review and directional leakage screening."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from src import finalize_inputs as final


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.row = {
            "event_id_cnty": "example",
            "label": "B",
            "notes": "Two armed groups exchanged fire.",
        }
        self.blind = {
            "predicted_label": "B",
            "low_ambiguity_suitability": "yes",
            "evidence_span": "exchanged fire",
        }
        self.semantic = {
            "source_label": "B",
            "decision": "retain",
            "verbatim_evidence": ["Two armed groups"],
            "allowed_edit_families_for_construction": [
                f for f in final.FAMILIES if f != "intensity"
            ],
            "excluded_edit_families_for_initial_batch": ["intensity"],
        }

    def test_unresolved_reviews_cannot_be_accepted(self):
        for kind, key, value in (
            ("blind", "predicted_label", "uncertain"),
            ("blind", "low_ambiguity_suitability", "conditional"),
            ("semantic", "decision", "exclude"),
            ("semantic", "source_label", "P"),
        ):
            with self.subTest(kind=kind, key=key):
                blind, semantic = copy.deepcopy(self.blind), copy.deepcopy(
                    self.semantic
                )
                {"blind": blind, "semantic": semantic}[kind][key] = value
                with self.assertRaises(ValueError):
                    final.accepted_review(self.row, blind, semantic)

    def test_invented_evidence_cannot_be_accepted(self):
        blind = dict(
            self.blind, evidence_span="This statement does not appear in the source."
        )
        with self.assertRaisesRegex(ValueError, "Nonverbatim"):
            final.accepted_review(self.row, blind, self.semantic)

    def test_conflicting_family_decisions_cannot_be_accepted(self):
        semantic = copy.deepcopy(self.semantic)
        semantic["allowed_edit_families_for_construction"].append("intensity")
        with self.assertRaisesRegex(ValueError, "family policy"):
            final.accepted_review(self.row, self.blind, semantic)

    def test_reverse_direction_and_cross_country_date_are_screened(self):
        # Forward=.25, reverse=.50; a forward-only comparison misses this pair.
        pilot = [
            {
                "event_id_cnty": "a",
                "notes": "tide",
                "country": "A",
                "event_date": "2020-01-01",
            }
        ]
        other = [
            {
                "event_id_cnty": "b",
                "notes": "diet",
                "country": "B",
                "event_date": "2026-01-01",
            }
        ]
        checked, hits = final.similarity_hits(pilot, other, threshold=0.45)
        self.assertEqual(checked, 1)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["reverse_similarity"], 0.5)

    def test_long_note_similarity_disables_autojunk(self):
        pilot = [{"event_id_cnty": "a", "notes": "x" + "abcde " * 60}]
        other = [{"event_id_cnty": "b", "notes": "y" + "abcde " * 60}]
        _, hits = final.similarity_hits(pilot, other)
        self.assertEqual(len(hits), 1)
        self.assertGreater(hits[0]["forward_similarity"], 0.99)

    def test_changed_decisions_cannot_inherit_previous_signoff(self):
        review = {
            "review_mode": "ai_only_adversarial_user_authorized",
            "human_review_performed": False,
            "methods_review": {
                "decision": "approved_from_methods_perspective_for_input_preparation",
                "blocking_findings": [],
            },
            "cases": [],
            "selection": [],
            "family_policy": {},
            "reserve_quarantine": [],
        }
        review["methods_review"]["reviewed_decisions_sha256"] = final.decision_digest(
            review
        )
        review["selection"].append({"event_id": "different"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.json"
            path.write_text(json.dumps(review))
            with self.assertRaisesRegex(ValueError, "changed after sign-off"):
                final.load_review(path)

    def test_existing_review_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            marker = output / "review.txt"
            marker.write_text("preserve")
            with self.assertRaisesRegex(ValueError, "already exists"):
                final.finalize(output)
            self.assertEqual(marker.read_text(), "preserve")

    def test_sample_review_rejects_unresolved_or_misattributed_decisions(self):
        source = {
            "event_id_cnty": "example",
            "sample_role": "main",
            "notes": self.row["notes"],
        }
        review = {
            "event_id": "example",
            "sample_role": "main",
            "review_type": "AI",
            "human_review_required": "false",
            "decision": "accept",
            "label_support": "supported",
            "notes_sufficiency": "sufficient",
            "notes_sha256": final.prep.digest(source["notes"]),
            "evidence_span": "exchanged fire",
            "rationale": "Organized armed parties exchange fire.",
            "reviewer": "AI reviewer",
            "initial_decision": "accept",
        }
        final.validate_sample_review([review], [source])
        for key, value in (
            ("decision", "pending"),
            ("human_review_required", "true"),
            ("notes_sha256", "changed"),
            ("evidence_span", "invented evidence"),
            ("label_support", "unsupported"),
            ("sample_role", "reserve"),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                final.validate_sample_review([{**review, key: value}], [source])
        with self.assertRaisesRegex(ValueError, "exactly once"):
            final.validate_sample_review([review, review], [source])

    def test_exclusions_require_peer_challenge_and_adjudication(self):
        source = {
            "event_id_cnty": "example",
            "sample_role": "main",
            "notes": self.row["notes"],
        }
        review = {
            "event_id": "example",
            "sample_role": "main",
            "review_type": "AI",
            "human_review_required": "false",
            "decision": "exclude",
            "label_support": "unsupported",
            "notes_sufficiency": "insufficient",
            "notes_sha256": final.prep.digest(source["notes"]),
            "evidence_span": "exchanged fire",
            "rationale": "Review objection.",
            "reviewer": "AI reviewer",
            "initial_decision": "exclude",
            "peer_decision": "",
            "peer_rationale": "",
            "adjudication_reason": "",
        }
        with self.assertRaisesRegex(ValueError, "Missing exclusion adjudication"):
            final.validate_sample_review([review], [source])

    def test_corrective_curation_preserves_shared_targets_and_uses_reviewed_reserves(
        self,
    ):
        def record(event_id, country, target, words=60):
            return {
                "event_id_cnty": event_id,
                "country": country,
                "event_type": "Riots",
                "label": "R",
                "sub_event_type": "Violent demonstration",
                "length_bin": "41-80",
                "word_count": str(words),
                "target_word_count": str(target),
                "selection_rank": "1",
                "study_group": "example",
                "actor_group": "rioters",
                "actor1": "Demonstrators",
                "year_month": "2025-01",
            }

        original = [
            record("a50", "A", 50),
            record("a60", "A", 60),
            record("b50", "B", 50),
            record("b60", "B", 60),
        ]
        reserve = record("replacement", "A", "", 70)
        assessments = {
            r["event_id_cnty"]: {
                "decision": "accept",
                "notes_sufficiency": "sufficient",
                "flags": "",
            }
            for r in original + [reserve]
        }
        assessments["a50"]["decision"] = assessments["a60"]["decision"] = "exclude"
        policy = {
            "replacements": [
                {
                    "original_event_id": "a60",
                    "accepted_event_id": "replacement",
                    "reserve_rank": "1",
                    "reason": "Replace unsupported input.",
                }
            ],
            "omit_for_balance": ["a50", "b50"],
            "removed_length_bin": "41-80",
            "removed_target_word_count": "50",
            "main_per_event_type": {"Riots": 1},
            "rationale": "Remove a shared target in every country.",
        }
        selected, changes, promoted = final.curate_main(
            original, [reserve], assessments, policy
        )
        self.assertEqual({r["event_id_cnty"] for r in selected}, {"replacement", "b60"})
        self.assertEqual({r["target_word_count"] for r in selected}, {"60"})
        self.assertEqual(
            next(r for r in selected if r["event_id_cnty"] == "replacement")[
                "word_difference_from_target"
            ],
            10,
        )
        self.assertEqual(promoted, {"replacement"})
        self.assertEqual(len(changes), 3)
        with self.assertRaisesRegex(ValueError, "every country"):
            final.curate_main(
                original,
                [reserve],
                assessments,
                {**policy, "omit_for_balance": ["a50"]},
            )
        assessments["replacement"]["decision"] = "exclude"
        with self.assertRaisesRegex(ValueError, "Ineligible reserve"):
            final.curate_main(original, [reserve], assessments, policy)


if __name__ == "__main__":
    unittest.main()
