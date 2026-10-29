"""Checks for exact edit provenance and independent, current pair reviews."""

import copy
import tempfile
import unittest
from pathlib import Path

import build_main_variants as build
from utils import read_csv


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

    def test_retired_pilot_exclusions_cannot_be_dropped_or_reassigned(self):
        rows = read_csv(build.ROOT / "data/reviews/pilot_exclusions.csv")
        pilot = read_csv(build.ROOT / "data/accepted/pilot.csv")
        family = {}
        for pair in read_csv(build.ROOT / "data/processed/related_event_pairs.csv"):
            for key in ("event_id_1", "event_id_2"):
                family[pair[key]] = pair["event_group_id"]
        build.validate_pilot_exclusions(rows, pilot, family)
        mutations = [
            lambda r: r.pop(),
            lambda r: r[0].update(source_event_id="replacement"),
            lambda r: r[0].update(source_family_id="different_family"),
            lambda r: r[0].update(
                source_file=r[1]["source_file"], source_record=r[1]["source_record"]
            ),
        ]
        for mutation in mutations:
            changed = copy.deepcopy(rows)
            mutation(changed)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                build.validate_pilot_exclusions(changed, pilot, family)


class MainDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json

        cls.recipes = json.loads(build.RECIPES.read_text())
        cls.baselines, cls.cases, cls.omissions, cls.source_audit, _ = (
            build.draft_cases(cls.recipes)
        )

    def test_all_main_originals_survive_byte_for_byte_with_input_qualifications(self):
        main = {
            r["event_id_cnty"]: r
            for r in read_csv(build.ROOT / "data/accepted/main.csv")
        }
        self.assertEqual({r["base_id"] for r in self.baselines}, set(main))
        self.assertEqual(len(self.baselines), 590)
        for case in self.baselines:
            self.assertEqual(
                case["renderings"][0]["original"], main[case["base_id"]]["notes"]
            )
            self.assertTrue(case["constraints"])

    def test_synthetic_classes_countries_and_source_families_are_complete(self):
        from collections import Counter

        originals = [
            c
            for c in self.cases
            if c["cohort"] == "synthetic" and c["condition"] == "original"
        ]
        self.assertEqual(
            Counter(c["label"] for c in originals),
            Counter({label: 20 for label in build.model_inputs.CLASSES}),
        )
        self.assertEqual(len({c["source_family_id"] for c in originals}), 120)
        excluded = {
            r["source_family_id"]
            for r in read_csv(build.ROOT / "data/reviews/pilot_exclusions.csv")
        }
        self.assertFalse(excluded & {c["source_family_id"] for c in originals})
        self.assertEqual(self.source_audit["conflicts"], [])
        for case in originals:
            self.assertEqual(
                {r["country"] for r in case["renderings"]},
                set(self.recipes["countries"]),
            )
            self.assertTrue(case["derivation_note"])

    def test_every_changed_span_reconstructs_text_and_country_substitution_commutes(
        self,
    ):
        for case in self.cases:
            self.assertEqual(
                case["case_sha256"],
                build.fingerprint(
                    {k: v for k, v in case.items() if k != "case_sha256"}
                ),
            )
            templates = set()
            for rendering in case["renderings"]:
                text = rendering["original"]
                for change in reversed(rendering["changes"]):
                    self.assertEqual(
                        text[change["original_start"] : change["original_end"]],
                        change["original_span"],
                    )
                    text = (
                        text[: change["original_start"]]
                        + change["edited_span"]
                        + text[change["original_end"] :]
                    )
                    self.assertEqual(
                        rendering["edited"][
                            change["edited_start"] : change["edited_end"]
                        ],
                        change["edited_span"],
                    )
                self.assertEqual(text, rendering["edited"])
                if case["cohort"] == "synthetic":
                    self.assertEqual(rendering["edited"].count(rendering["country"]), 1)
                    templates.add(
                        rendering["edited"].replace(rendering["country"], "{country}")
                    )
            if case["cohort"] == "synthetic":
                self.assertEqual(len(templates), 1)

    def test_current_complete_independent_reviews_are_required_for_release(self):
        changed = copy.deepcopy(self.recipes)
        changed["reviews"] = []
        changed["adjudications"] = []
        with self.assertRaisesRegex(ValueError, "terminal AI adjudication"):
            build.adjudicated_cases(changed, self.cases)

    def test_released_schedule_covers_each_text_and_exactly_three_fresh_repeats(self):
        import json
        from collections import Counter

        directory = build.ROOT / "data/variants/main"
        manifest = build.verified_snapshot(directory)
        stimuli = {r["stimulus_id"]: r for r in read_csv(directory / "stimuli.csv")}
        schedule = read_csv(directory / "request_order.csv")
        protocol = json.loads((directory / "protocol.json").read_text())
        self.assertTrue(manifest["data_preparation_complete"])
        self.assertFalse(manifest["inference_performed"])
        self.assertFalse(manifest["main_inference_ready"])
        self.assertEqual(len({r["request_id"] for r in schedule}), len(schedule))
        self.assertEqual(
            [int(r["sequence"]) for r in schedule], list(range(1, len(schedule) + 1))
        )
        selected = []
        for model_id in [
            "intfloat/multilingual-e5-large-instruct",
            "meta-llama/llama-3.1-8b-instruct",
        ]:
            rows = [r for r in schedule if r["model"] == model_id]
            core = Counter(r["stimulus_id"] for r in rows if r["replicate"] == "0")
            repeats = [r["stimulus_id"] for r in rows if r["replicate"] == "1"]
            self.assertEqual(core, Counter({ident: 1 for ident in stimuli}))
            self.assertEqual(len(repeats), 3)
            self.assertEqual(len(set(repeats)), 3)
            self.assertTrue(
                all(stimuli[ident]["condition"] == "original" for ident in repeats)
            )
            selected.append(set(repeats))
        self.assertEqual(selected[0], selected[1])
        self.assertEqual(selected[0], set(protocol["repeat_stimulus_ids"]))
        self.assertEqual(protocol["class_order"], ["B", "P", "R", "E", "V", "S"])
        self.assertEqual(len(schedule), manifest["planned_classifications_two_models"])


if __name__ == "__main__":
    unittest.main()
