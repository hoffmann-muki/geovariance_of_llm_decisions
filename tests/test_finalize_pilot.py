"""Acceptance guards for AI-only input review and directional leakage screening."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import finalize_pilot as final


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.row = {"event_id_cnty": "example", "label": "B", "notes": "Two armed groups exchanged fire."}
        self.blind = {"predicted_label": "B", "low_ambiguity_suitability": "yes", "evidence_span": "exchanged fire"}
        self.semantic = {"source_label": "B", "decision": "retain", "verbatim_evidence": ["Two armed groups"],
                         "allowed_edit_families_for_construction": [f for f in final.FAMILIES if f != "intensity"],
                         "excluded_edit_families_for_initial_batch": ["intensity"]}

    def test_unresolved_reviews_cannot_be_accepted(self):
        for kind, key, value in (("blind", "predicted_label", "uncertain"),
                                 ("blind", "low_ambiguity_suitability", "conditional"),
                                 ("semantic", "decision", "exclude"),
                                 ("semantic", "source_label", "P")):
            with self.subTest(kind=kind, key=key):
                blind, semantic = copy.deepcopy(self.blind), copy.deepcopy(self.semantic)
                {"blind": blind, "semantic": semantic}[kind][key] = value
                with self.assertRaises(ValueError):
                    final.accepted_review(self.row, blind, semantic)

    def test_invented_evidence_cannot_be_accepted(self):
        blind = dict(self.blind, evidence_span="This statement does not appear in the source.")
        with self.assertRaisesRegex(ValueError, "Nonverbatim"):
            final.accepted_review(self.row, blind, self.semantic)

    def test_conflicting_family_decisions_cannot_be_accepted(self):
        semantic = copy.deepcopy(self.semantic)
        semantic["allowed_edit_families_for_construction"].append("intensity")
        with self.assertRaisesRegex(ValueError, "family policy"):
            final.accepted_review(self.row, self.blind, semantic)

    def test_reverse_direction_and_cross_country_date_are_screened(self):
        # Forward=.25, reverse=.50; a forward-only comparison misses this pair.
        pilot = [{"event_id_cnty": "a", "notes": "tide", "country": "A", "event_date": "2020-01-01"}]
        other = [{"event_id_cnty": "b", "notes": "diet", "country": "B", "event_date": "2026-01-01"}]
        checked, hits = final.similarity_hits(pilot, other, threshold=.45)
        self.assertEqual(checked, 1)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["reverse_similarity"], .5)

    def test_long_note_similarity_disables_autojunk(self):
        pilot = [{"event_id_cnty": "a", "notes": "x" + "abcde " * 60}]
        other = [{"event_id_cnty": "b", "notes": "y" + "abcde " * 60}]
        _, hits = final.similarity_hits(pilot, other)
        self.assertEqual(len(hits), 1)
        self.assertGreater(hits[0]["forward_similarity"], .99)

    def test_changed_decisions_cannot_inherit_previous_signoff(self):
        review = {"review_mode": "ai_only_adversarial_user_authorized", "human_review_performed": False,
                  "methods_review": {"decision": "approved_from_methods_perspective_for_input_preparation",
                                     "blocking_findings": []},
                  "cases": [], "selection": [], "family_policy": {}, "reserve_quarantine": []}
        review["methods_review"]["reviewed_decisions_sha256"] = final.decision_digest(review)
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


if __name__ == "__main__":
    unittest.main()
