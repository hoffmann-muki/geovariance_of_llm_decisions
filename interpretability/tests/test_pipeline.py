import copy
import unittest

from geoflip import pipeline as G


class InputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = G.load_inputs()

    def test_committed_inputs_pass_static_checks(self):
        self.assertEqual(G.static_findings(self.inputs), [])

    def test_leaked_class_name_is_found(self):
        inputs = copy.deepcopy(self.inputs)
        frames = inputs["descriptions"]["boundaries"]["PR"]["frames"]
        frames["f1"] = frames["f1"].replace("to demonstrate against", "to protest")
        self.assertTrue(any("'protest'" in f for f in G.static_findings(inputs)))

    def test_leak_in_evidence_and_hedge_in_task_are_found(self):
        inputs = copy.deepcopy(self.inputs)
        evidence = inputs["descriptions"]["boundaries"]["BV"]["evidence"]
        evidence["air"]["wordings"][0] = "The other group was attacked."
        para = inputs["task_prompts"]["boundaries"]["PR"]["paraphrases"][1]
        para["intro"] += " If unclear, choose Protest."
        findings = G.static_findings(inputs)
        self.assertTrue(any("BV/air/W1" in f for f in findings))
        self.assertTrue(any("task PR/p1" in f and "unclear" in f for f in findings))

    def test_country_slot_is_the_only_difference(self):
        spec = self.inputs["descriptions"]["boundaries"]["BV"]
        base = self.inputs["bases"]["BV"]["explore"][0]
        evidence = spec["evidence"]["selfdef"]["wordings"][0]
        plain = G.render(spec["frames"]["f1"], base, evidence, "")
        text = G.render(spec["frames"]["f1"], base, evidence, " in the United States")
        self.assertEqual(text.replace(" in the United States", ""), plain)
        broken = copy.deepcopy(self.inputs)
        broken["descriptions"]["boundaries"]["BV"]["frames"][
            "f1"
        ] = "In {place} {time}{country}: {evidence}"
        self.assertTrue(any("follow {place}" in f for f in G.static_findings(broken)))

    def test_splits_are_disjoint_and_sized(self):
        for boundary in G.BOUNDARIES:
            splits = self.inputs["bases"][boundary]
            ids = [row["base_id"] for rows in splits.values() for row in rows]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(len(splits["explore"]), 8)
            self.assertEqual(len(splits["confirm"]), 24)
            self.assertEqual(len(splits["test"]), 16)

    def test_tasks_reverse_classes_and_keep_definitions_with_class(self):
        tasks = {
            t["task_id"]: t
            for t in G.render_tasks(
                self.inputs["task_prompts"], self.inputs["descriptions"]
            )
        }
        self.assertEqual(len(tasks), 2 * 2 * 4 * 2)
        a, b = (
            tasks["PR:defined:p2:A"]["system_prompt"],
            tasks["PR:defined:p2:B"]["system_prompt"],
        )
        self.assertLess(a.index("Protest:"), a.index("Riot:"))
        self.assertLess(b.index("Riot:"), b.index("Protest:"))
        self.assertTrue(a.endswith("Protest or Riot."))
        self.assertTrue(b.endswith("Riot or Protest."))
        self.assertNotIn("Riot:", tasks["PR:undefined:p2:A"]["system_prompt"])

    def test_overlapping_splits_are_found(self):
        inputs = copy.deepcopy(self.inputs)
        inputs["bases"]["PR"]["test"].append(inputs["bases"]["PR"]["confirm"][0])
        self.assertTrue(
            any(
                "confirm and test bases overlap" in f for f in G.static_findings(inputs)
            )
        )

    def test_reference_scores_match_the_inputs(self):
        rows = G.read_jsonl(G.GEO / "scorer_reference.jsonl")
        self.assertEqual(len(rows), 8)
        for row in rows:
            spec = G.S.MODELS[row["model_key"]]
            self.assertEqual(
                (row["dtype"], row["revision"]), (spec["dtype"], spec["revision"])
            )
            self.assertEqual(set(row["logprob"]), {row["lower"], row["higher"]})


class SelectionTests(unittest.TestCase):
    def test_robust_label_needs_margin_and_agreement(self):
        self.assertEqual(G.robust_label([1.0, 0.8, 0.6, -0.1]), 1)
        self.assertEqual(G.robust_label([-1.0, -0.8, -0.6, -0.2]), -1)
        self.assertEqual(G.robust_label([0.3, 0.2, 0.4, 0.1]), 0)  # below margin
        self.assertEqual(G.robust_label([3.0, 1.0, -0.5, -0.5]), 0)  # only 2 agree

    def test_collapse_averages_orders_then_paraphrases(self):
        rows = []
        for para, (d_a, d_b) in enumerate(
            [(2.0, 0.0), (1.0, 1.0), (3.0, -1.0), (0.0, 2.0)]
        ):
            for order, d in (("A", d_a), ("B", d_b)):
                rows.append(
                    dict(
                        boundary="PR",
                        frame="f1",
                        evidence="x",
                        base_id="b",
                        wording=1,
                        country="none",
                        version="defined",
                        paraphrase=para,
                        order=order,
                        d=d,
                        mass=0.95,
                    )
                )
        (cell,) = G.collapse(rows, 4).values()
        self.assertEqual(cell["dbar"], [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(cell["s"], 1.0)
        self.assertEqual(cell["label"], 1)
        self.assertEqual(cell["mass_min"], 0.95)
        with self.assertRaises(ValueError):
            G.collapse(rows[:-1], 4)  # a missing order fails loudly
        with self.assertRaises(ValueError):
            G.collapse(rows, 5)  # a missing paraphrase fails loudly

    def test_screen_countries_needs_consistent_sign(self):
        shifts = {
            "A": [1.0, 0.5, 0.2, -0.1],
            "B": [2.0, -1.0, -1.0, 3.0],
            "C": [-1.0, -0.5, -2.0, 0.1],
            "D": [0.1, 0.1, 0.1, 0.1],
        }
        up, down = G.screen_countries(shifts, consistency=3, top=4)
        self.assertEqual(up, ["A", "D"])
        self.assertEqual(down, ["C"])

    def test_pair_stats_uses_shared_cells(self):
        stats = G.pair_stats(
            {(1, 1): 1.0, (1, 2): 0.0, (2, 1): 2.0}, {(1, 1): 0.0, (1, 2): 1.0}
        )
        self.assertEqual(stats["n"], 2)
        self.assertEqual(stats["share_positive"], 0.5)
        self.assertEqual(stats["mean_gap"], 0.0)


if __name__ == "__main__":
    unittest.main()
