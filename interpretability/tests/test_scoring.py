"""Answer scoring must read the right logit rows for the right vocabulary tokens.

Two tiers:
- Logic tests use a fake tokenizer and always run.
- Tokenizer tests (GEOFLIP_TOKENIZER_TESTS=1) check each model's real chat
  template and answer tokens; they need `transformers` and Hub access.
"""

import os
import unittest

from geoflip import pipeline as P
from geoflip import scoring as S

ANSWERS = ["Protest", "Riot", "Battle", "Violence against civilians"]
EXPECTED_PIECES = {
    "Protest": ["Pro", "test"],
    "Riot": ["R", "iot"],
    "Battle": ["Battle"],
    "Violence against civilians": ["Viol", "ence", " against", " civilians"],
}
FAKE_SPEC = {
    "template_kwargs": {},
    "assistant_suffix": "\nASSISTANT:",
    "required_fragments": [],
}


class FakeTokenizer:
    """Greedy longest-match tokenizer with a minimal chat template."""

    def __init__(self, extra_pieces=()):
        pieces = [chr(code) for code in range(32, 127)] + ["\n"]
        pieces += ["<s>", "SYS:", "USER:", "ASSISTANT:", "Pro", "test", "R", "iot"]
        self.vocab = pieces + list(extra_pieces)
        self.ids = {piece: index for index, piece in enumerate(self.vocab)}

    def encode(self, text, add_special_tokens=False):
        tokens, position = [], 0
        while position < len(text):
            piece = max(
                (p for p in self.vocab if text.startswith(p, position)), key=len
            )
            tokens.append(self.ids[piece])
            position += len(piece)
        return tokens

    def decode(self, tokens):
        return "".join(self.vocab[token] for token in tokens)

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        system, user = (message["content"] for message in messages)
        text = f"<s>SYS:{system}\nUSER:{user}"
        text += "\nASSISTANT:" if add_generation_prompt else ""
        return {"input_ids": self.encode(text)} if tokenize else text


class PositionLogicTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.text, self.ids = S.render_prompt(
            self.tokenizer, FAKE_SPEC, "Code it.", "Event."
        )

    def test_logit_row_before_each_answer_token_scores_it(self):
        # Row i of the logits predicts token i + 1, so a 2-token answer after
        # a 5-token prompt is scored from rows 4 and 5.
        self.assertEqual(S.scoring_positions(5, 2), [4, 5])
        self.assertEqual(S.model_input([1, 2, 3], [7, 8, 9]), [1, 2, 3, 7, 8])

    def test_answer_logprobs_gather_predicting_rows(self):
        answer = [2, 0]
        table = [[-9.0] * 3 for _ in range(4)]
        table[1][2], table[2][0] = -0.1, -0.2  # rows that predict the answer
        table[2][2], table[3][0] = -5.0, -6.0  # off-by-one rows
        self.assertEqual(S.answer_logprobs(table, 2, answer), [-0.1, -0.2])

    def test_scored_model_input_covers_every_answer_row(self):
        answer = S.answer_token_ids(self.tokenizer, self.text, self.ids, "Protest")
        rows = S.scoring_positions(len(self.ids), len(answer))
        length = len(S.model_input(self.ids, answer))
        self.assertEqual(rows[0], len(self.ids) - 1)
        self.assertEqual(rows[-1], length - 1)

    def test_multitoken_answers_resolve_to_in_context_pieces(self):
        answers = S.resolve_answers(
            self.tokenizer, self.text, self.ids, ["Protest", "Riot"]
        )
        decoded = {
            name: [self.tokenizer.decode([t]) for t in ids]
            for name, ids in answers.items()
        }
        self.assertEqual(decoded, {"Protest": ["Pro", "test"], "Riot": ["R", "iot"]})

    def test_tokens_merging_across_the_answer_boundary_are_rejected(self):
        tokenizer = FakeTokenizer(extra_pieces=["ASSISTANT:Pro"])
        text = tokenizer.apply_chat_template(
            [{"content": "S"}, {"content": "U"}], False, True
        )
        ids = tokenizer.encode(text)
        with self.assertRaisesRegex(ValueError, "changes the prompt"):
            S.answer_token_ids(tokenizer, text, ids, "Protest")

    def test_prefix_answers_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "prefix"):
            S.resolve_answers(self.tokenizer, self.text, self.ids, ["Riot", "Riots"])

    def test_prompt_must_end_at_the_assistant_position(self):
        spec = {**FAKE_SPEC, "assistant_suffix": "\nASSISTANT: "}
        with self.assertRaisesRegex(ValueError, "answer position"):
            S.render_prompt(self.tokenizer, spec, "Code it.", "Event.")

    def test_required_template_fragments_are_enforced(self):
        spec = {**FAKE_SPEC, "required_fragments": ["Today Date: 26 Jul 2024\n"]}
        with self.assertRaisesRegex(ValueError, "lacks"):
            S.render_prompt(self.tokenizer, spec, "Code it.", "Event.")


def input_samples():
    """Every task prompt with three country conditions of one borderline item."""
    inputs = P.load_inputs()
    tasks = P.render_tasks(inputs["task_prompts"], inputs["descriptions"])
    countries = [
        c
        for c in P.country_conditions(inputs["countries"])
        if c[0] in ("none", "United States", "Veloria")
    ]
    samples = []
    for boundary, evidence in (("PR", "shoes"), ("BV", "watchman")):
        stimuli = P.build_stimuli(
            inputs,
            boundary,
            [evidence],
            inputs["bases"][boundary]["explore"][:1],
            (1,),
            countries,
            ["f1"],
        )
        samples += [
            (task, stimulus)
            for task in tasks
            for stimulus in stimuli
            if task["boundary"] == boundary
        ]
    return samples


@unittest.skipUnless(os.environ.get("GEOFLIP_TOKENIZER_TESTS"), "tokenizer tests")
class RealTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        if not os.environ.get("HF_TOKEN"):
            os.environ["HF_TOKEN"] = S.load_secret("HF_TOKEN", S.ENV_FILE)
        cls.tokenizers = {
            key: AutoTokenizer.from_pretrained(spec["hf_id"], revision=spec["revision"])
            for key, spec in S.MODELS.items()
        }
        cls.samples = input_samples()

    def test_every_sample_prompt_ends_at_the_assistant_position(self):
        for key, tokenizer in self.tokenizers.items():
            spec = S.MODELS[key]
            suffix = tokenizer.encode(
                spec["assistant_suffix"], add_special_tokens=False
            )
            for task, stimulus in self.samples:
                with self.subTest(
                    model=key, task=task["task_id"], country=stimulus["country"]
                ):
                    text, ids = S.render_prompt(
                        tokenizer, spec, task["system_prompt"], stimulus["text"]
                    )
                    self.assertEqual(ids[-len(suffix) :], suffix)
                    self.assertTrue(text.endswith(spec["assistant_suffix"]))

    def test_qwen_template_has_an_empty_reasoning_block(self):
        tokenizer = self.tokenizers["qwen3-4b"]
        spec = S.MODELS["qwen3-4b"]
        text, _ = S.render_prompt(tokenizer, spec, "System.", "User.")
        self.assertEqual(text.count("<think>"), 1)
        self.assertTrue(text.endswith("<think>\n\n</think>\n\n"))

    def test_llama_template_date_is_fixed(self):
        tokenizer = self.tokenizers["llama3.2-3b"]
        spec = S.MODELS["llama3.2-3b"]
        text, _ = S.render_prompt(tokenizer, spec, "System.", "User.")
        self.assertIn("Today Date: 26 Jul 2024\n", text)
        self.assertEqual(text.count("<|begin_of_text|>"), 1)

    def test_answer_tokens_are_known_and_context_independent(self):
        for key, tokenizer in self.tokenizers.items():
            spec = S.MODELS[key]
            seen = {}
            for task, stimulus in self.samples:
                text, ids = S.render_prompt(
                    tokenizer, spec, task["system_prompt"], stimulus["text"]
                )
                classes = [task["lower"], task["higher"]]
                for name, tokens in S.resolve_answers(
                    tokenizer, text, ids, classes
                ).items():
                    pieces = [tokenizer.decode([token]) for token in tokens]
                    with self.subTest(model=key, answer=name):
                        self.assertEqual(pieces, EXPECTED_PIECES[name])
                        self.assertEqual(seen.setdefault(name, tokens), tokens)
            self.assertEqual(set(seen), set(ANSWERS))

    def test_first_answer_tokens_differ_from_space_and_case_variants(self):
        for key, tokenizer in self.tokenizers.items():
            spec = S.MODELS[key]
            text, ids = S.render_prompt(tokenizer, spec, "System.", "User.")
            for name in ANSWERS:
                first = S.answer_token_ids(tokenizer, text, ids, name)[0]
                for variant in (" " + name, name.lower()):
                    other = tokenizer.encode(variant, add_special_tokens=False)[0]
                    with self.subTest(model=key, variant=variant):
                        self.assertNotEqual(first, other)


if __name__ == "__main__":
    unittest.main()
