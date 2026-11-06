import copy
import unittest

from geoflip import study as T
from geoflip import scoring as S
from geoflip import items as I
from geoflip import materials as M
from tests.test_scoring import FAKE_SPEC, FakeTokenizer


class FakeModel:
    """Returns fixed logits: token 'A' gets 2.0, 'B' gets 1.0, everything else 0."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids):
        import torch

        logits = torch.zeros(1, input_ids.shape[1], len(self.tokenizer.vocab))
        logits[0, -1, self.tokenizer.ids["A"]] = 2.0
        logits[0, -1, self.tokenizer.ids["B"]] = 1.0

        class Out:
            pass

        out = Out()
        out.logits = logits
        return out


class FirstTokenTests(unittest.TestCase):
    def test_reads_last_position_and_single_token_letters(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        tok = FakeTokenizer()
        result = S.score_first_token(
            FakeModel(tok), tok, FAKE_SPEC, "Code it.", "Event.", ("A", "B"), "cpu"
        )
        self.assertAlmostEqual(result["lp"]["A"] - result["lp"]["B"], 1.0, places=5)
        self.assertEqual(result["top"][0][0], "A")

    def test_multi_token_letter_is_rejected(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        tok = FakeTokenizer()
        with self.assertRaisesRegex(ValueError, "not one"):
            S.score_first_token(
                FakeModel(tok),
                tok,
                FAKE_SPEC,
                "Code it.",
                "Event.",
                ("A", "Riot"),
                "cpu",
            )


class StaticRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = T.load_inputs()

    def test_inputs_are_clean(self):
        self.assertEqual(T.static_findings(self.inputs), [])

    def _with_evidence(self, text):
        inputs = copy.deepcopy(self.inputs)
        inputs["descriptions"]["boundaries"]["PR"]["evidence"]["throw_eggs"][
            "wordings"
        ][0] = text
        return T.static_findings(inputs)

    def test_each_injected_violation_is_found(self):
        cases = {
            "hedge": "Some participants reportedly threw eggs at a nearby building.",
            "definition": "Some participants damaged a nearby building with eggs.",
            "capital": "Some participants threw eggs at the Parliament building.",
            "outcome": "Some participants threw eggs and injured a guard nearby.",
            "length": "Participants threw eggs.",
            "sentences": "Some participants threw eggs. Others watched from the side.",
            "identity": "Some participants threw eggs at a police van nearby.",
        }
        for name, text in cases.items():
            with self.subTest(rule=name):
                self.assertTrue(self._with_evidence(text), name)

    def test_task_prompt_mentioning_location_is_found(self):
        inputs = copy.deepcopy(self.inputs)
        inputs["task_prompts"]["boundaries"]["BV"]["paraphrases"][0][
            "intro"
        ] += " Consider the country."
        self.assertTrue(any("country" in f for f in T.static_findings(inputs)))


class TaskTests(unittest.TestCase):
    def test_orders_swap_letters_and_definitions(self):
        tasks = {
            t["task_id"]: t
            for t in T.render_tasks(T.load_inputs()["task_prompts"], "options_last")
        }
        self.assertEqual(len(tasks), 32)
        a, b = tasks["PR:defined:p1:A"], tasks["PR:defined:p1:B"]
        self.assertTrue(a["system_prompt"].endswith("A. Protest\nB. Riot"))
        self.assertTrue(b["system_prompt"].endswith("A. Riot\nB. Protest"))
        self.assertEqual((a["letter_higher"], b["letter_higher"]), ("B", "A"))
        self.assertLess(
            b["system_prompt"].index("Riot:"), b["system_prompt"].index("Protest:")
        )
        self.assertNotIn("Protest:", tasks["PR:undefined:p1:A"]["system_prompt"])


class PilotRuleTests(unittest.TestCase):
    def _rows(self, d_lo, d_hi, mass=0.99, top1="A"):
        rows = []
        for role, d in (("anchor_lower", d_lo), ("anchor_higher", d_hi)):
            for order, shift in (("A", 1.0), ("B", -1.0)):
                rows.append(
                    dict(
                        boundary="PR",
                        frame="f1",
                        evidence=role,
                        role=role,
                        base_id="b",
                        wording=1,
                        version="defined",
                        paraphrase=0,
                        order=order,
                        layout="x",
                        d=d + shift,
                        mass=mass,
                        top1=top1,
                    )
                )
        return rows

    def test_gates_and_choice(self):
        good = T.pilot_summary(self._rows(-3, 3))
        self.assertTrue(good["layouts"]["x"]["all_pass"])
        self.assertEqual(good["chosen_layout"], "x")
        self.assertEqual(good["layouts"]["x"]["mean_abs_order_effect"], 2.0)
        self.assertIsNone(
            T.pilot_summary(self._rows(3, 3))["chosen_layout"]
        )  # lower anchor wrong
        self.assertIsNone(T.pilot_summary(self._rows(-3, 3, mass=0.5))["chosen_layout"])
        self.assertIsNone(
            T.pilot_summary(self._rows(-3, 3, top1="The"))["chosen_layout"]
        )


class MaterialTests(unittest.TestCase):
    def test_items_follow_the_inclusion_rule(self):
        items = I.build(T.INPUTS / "acled_acts.csv")
        self.assertEqual(sum(e["role"] == "item" for e in items["PR"].values()), 29)
        self.assertEqual(sum(e["role"] == "item" for e in items["BV"].values()), 22)

    def test_bases_are_fresh_and_balanced(self):
        M.check_fresh()
        for boundary in ("PR", "BV"):
            bases = M.build_bases(boundary)
            for split, rows in bases.items():
                for field in rows[0]:
                    self.assertEqual(
                        len({r[field] for r in rows}),
                        len(rows),
                        (boundary, split, field),
                    )


if __name__ == "__main__":
    unittest.main()


import os  # noqa: E402
import re  # noqa: E402


@unittest.skipUnless(os.environ.get("GEOFLIP_TOKENIZER_TESTS"), "tokenizer tests")
class RealTokenizerV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        spec = S.MODELS["llama3.2-3b"]
        cls.spec = spec
        cls.tok = AutoTokenizer.from_pretrained(
            spec["hf_id"], revision=spec["revision"]
        )
        inputs = T.load_inputs()
        cls.tasks = T.render_tasks(
            inputs["task_prompts"], "options_last"
        ) + T.render_tasks(inputs["task_prompts"], "options_first")
        cls.stims = [
            s
            for b in T.BOUNDARIES
            for s in T.stimuli(inputs, b, "main", ("none", "real"))[:200]
        ]

    def test_letters_are_single_tokens_at_the_answer_position(self):
        for task in self.tasks:
            text, ids = S.render_prompt(
                self.tok, self.spec, task["system_prompt"], self.stims[0]["text"]
            )
            for letter in T.LETTERS:
                self.assertEqual(
                    len(S.answer_token_ids(self.tok, text, ids, letter)), 1
                )

    def test_no_date_other_than_the_fixed_header(self):
        months = r"(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        for task in self.tasks[:4]:
            for stim in self.stims:
                text, _ = S.render_prompt(
                    self.tok, self.spec, task["system_prompt"], stim["text"]
                )
                rest = text.replace(
                    "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n",
                    "",
                )
                self.assertIsNone(
                    re.search(months + r"|\b(19|20)\d\d\b", rest), stim["text"]
                )
