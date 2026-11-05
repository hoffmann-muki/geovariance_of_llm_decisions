"""Ambiguity-ladder construction must enforce the plan's isolation and design rules."""

import csv
import json
import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from src import interp_stimuli
from src.utils import read_csv, read_json, verified_snapshot, write_json

SOURCES = ["ladders.json", "prompts.json", "bases_pr.csv", "bases_bv.csv"]


class StimulusBuildTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.data = self.root / "data"
        self.data.mkdir()
        for name in SOURCES:
            shutil.copy(interp_stimuli.DATA / name, self.data / name)

    def tearDown(self):
        self.directory.cleanup()

    def build(self):
        return interp_stimuli.build(self.root / "out", self.data)

    def edit_json(self, name, change):
        value = read_json(self.data / name)
        change(value)
        write_json(self.data / name, value)

    def edit_bases(self, boundary, change):
        path = self.data / f"bases_{boundary.lower()}.csv"
        rows = read_csv(path)
        rows = change(rows)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)

    def assert_rejected(self, message):
        with self.assertRaisesRegex(ValueError, message):
            self.build()
        self.assertFalse((self.root / "out" / "manifest.json").exists())

    def test_reviewed_inputs_build_the_planned_design(self):
        checks = self.build()
        out = self.root / "out"
        verified_snapshot(out)
        stimuli = read_csv(out / "stimuli.csv")
        schedule = read_csv(out / "schedule.csv")
        self.assertEqual(checks["stimuli"]["stimuli"], 5280)
        self.assertEqual(
            Counter((row["boundary"], row["stage"]) for row in schedule),
            {("PR", "A"): 960, ("PR", "B"): 9600, ("BV", "A"): 960, ("BV", "B"): 9600},
        )
        no_country = [row for row in stimuli if row["country"] == "none"]
        self.assertTrue(all(row["stage"] == "A" for row in no_country))
        us = next(row for row in stimuli if row["country"] == "United States")
        self.assertIn(" in the United States", us["text"])

    def test_order_b_reverses_classes_and_keeps_definitions_with_them(self):
        self.build()
        prompts = {
            row["task_id"]: row["system_prompt"]
            for row in read_csv(self.root / "out" / "system_prompts.csv")
        }
        self.assertEqual(len(prompts), 8)
        defined_b = prompts["PR:defined:B"]
        self.assertLess(defined_b.index("Riot:"), defined_b.index("Protest:"))
        self.assertIn("Riot: a demonstration in which participants engage", defined_b)
        self.assertTrue(defined_b.endswith("Answer with one word: Riot or Protest."))
        self.assertNotIn("Battle:", prompts["BV:undefined:A"])
        self.assertTrue(
            prompts["BV:undefined:B"].endswith(
                "Answer with exactly one of: Violence against civilians, Battle."
            )
        )

    def test_existing_output_is_not_overwritten(self):
        (self.root / "out").mkdir()
        (self.root / "out" / "keep.txt").write_text("x", encoding="utf-8")
        self.assert_rejected("not empty")

    def test_unapproved_ladders_are_rejected(self):
        self.edit_json("ladders.json", lambda value: value.update(status="draft"))
        self.assert_rejected("not approved")

    def test_incomplete_crossing_is_rejected(self):
        self.edit_bases("BV", lambda rows: rows[:-1])
        self.assert_rejected("fully crossed")

    def test_unbalanced_place_is_rejected(self):
        def change(rows):
            rows[0]["place"] = rows[1]["place"]
            return rows

        self.edit_bases("PR", change)
        self.assert_rejected("place is")

    def test_evidence_repeated_across_levels_is_rejected(self):
        def change(value):
            levels = value["boundaries"]["PR"]["levels"]
            levels[2]["wordings"][0] = levels[1]["wordings"][0]

        self.edit_json("ladders.json", change)
        self.assert_rejected("repeat across levels")

    def test_country_cue_outside_the_slot_is_rejected(self):
        def change(value):
            spec = value["boundaries"]["BV"]
            spec["frame"] = spec["frame"].replace(
                "armed group", "armed group from Mexico"
            )

        self.edit_json("ladders.json", change)
        self.assert_rejected("Country")

    def test_excluded_terms_are_rejected(self):
        def change(value):
            value["boundaries"]["PR"]["levels"][2]["wordings"][
                1
            ] = "Police said a car parked nearby was damaged."

        self.edit_json("ladders.json", change)
        self.assert_rejected("Excluded term")

    def test_mismatched_prompt_classes_are_rejected(self):
        def change(value):
            value["boundaries"]["PR"]["order_a"] = ["Protest", "Battle"]

        self.edit_json("prompts.json", change)
        self.assert_rejected("classes")

    def test_prompts_may_not_mention_country(self):
        def change(value):
            value["intro_template"] += " Consider the country."

        self.edit_json("prompts.json", change)
        self.assert_rejected("mentions country")

    def test_stimulus_ids_carry_design_factors(self):
        self.build()
        row = read_csv(self.root / "out" / "stimuli.csv")[0]
        boundary, base, level, wording, country = row["stimulus_id"].split(":")
        self.assertEqual(boundary, row["boundary"])
        self.assertEqual(base, row["base_id"])
        self.assertEqual(level, f"L{row['level']}")
        self.assertEqual(wording, f"W{row['wording']}")
        self.assertEqual(json.loads(row["base_factors"]).keys() & {"country"}, set())
        self.assertIn(country, {"none", "libya"})


if __name__ == "__main__":
    unittest.main()
