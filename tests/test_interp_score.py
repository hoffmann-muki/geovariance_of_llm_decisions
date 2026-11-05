"""Answer scoring must read the right logit rows for the right vocabulary tokens.

Three tiers:
- Logic tests use a fake tokenizer and always run.
- Tokenizer tests (INTERP_TOKENIZER_TESTS=1) check each model's real chat
  template and answer tokens; they need `transformers` and Hub access.
- Model tests (INTERP_MODEL_TESTS=1, optional INTERP_MODEL and INTERP_DEVICE)
  compare scores with the model's own greedy generation; they need weights.
"""

import os
import statistics
import unittest

from src import interp_score
from src.utils import read_csv

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
        self.text, self.ids = interp_score.render_prompt(
            self.tokenizer, FAKE_SPEC, "Code it.", "Event."
        )

    def test_logit_row_before_each_answer_token_scores_it(self):
        # Row i of the logits predicts token i + 1, so a 2-token answer after
        # a 5-token prompt is scored from rows 4 and 5.
        self.assertEqual(interp_score.scoring_positions(5, 2), [4, 5])
        self.assertEqual(
            interp_score.model_input([1, 2, 3], [7, 8, 9]), [1, 2, 3, 7, 8]
        )

    def test_answer_logprobs_gather_predicting_rows(self):
        answer = [2, 0]
        table = [[-9.0] * 3 for _ in range(4)]
        table[1][2], table[2][0] = -0.1, -0.2  # rows that predict the answer
        table[2][2], table[3][0] = -5.0, -6.0  # off-by-one rows
        self.assertEqual(interp_score.answer_logprobs(table, 2, answer), [-0.1, -0.2])

    def test_scored_model_input_covers_every_answer_row(self):
        answer = interp_score.answer_token_ids(
            self.tokenizer, self.text, self.ids, "Protest"
        )
        rows = interp_score.scoring_positions(len(self.ids), len(answer))
        length = len(interp_score.model_input(self.ids, answer))
        self.assertEqual(rows[0], len(self.ids) - 1)
        self.assertEqual(rows[-1], length - 1)

    def test_multitoken_answers_resolve_to_in_context_pieces(self):
        answers = interp_score.resolve_answers(
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
            interp_score.answer_token_ids(tokenizer, text, ids, "Protest")

    def test_prefix_answers_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "prefix"):
            interp_score.resolve_answers(
                self.tokenizer, self.text, self.ids, ["Riot", "Riots"]
            )

    def test_prompt_must_end_at_the_assistant_position(self):
        spec = {**FAKE_SPEC, "assistant_suffix": "\nASSISTANT: "}
        with self.assertRaisesRegex(ValueError, "answer position"):
            interp_score.render_prompt(self.tokenizer, spec, "Code it.", "Event.")

    def test_required_template_fragments_are_enforced(self):
        spec = {**FAKE_SPEC, "required_fragments": ["Today Date: 26 Jul 2024\n"]}
        with self.assertRaisesRegex(ValueError, "lacks"):
            interp_score.render_prompt(self.tokenizer, spec, "Code it.", "Event.")


def snapshot_samples():
    """One prompt per task and country condition at level 3, wording 1."""
    snapshot = interp_score.SNAPSHOT
    stimuli = [
        row
        for row in read_csv(snapshot / "stimuli.csv")
        if row["level"] == "3" and row["wording"] == "1" and row["base_id"][-2:] == "01"
    ]
    tasks = read_csv(snapshot / "system_prompts.csv")
    return [
        (task, stimulus)
        for task in tasks
        for stimulus in stimuli
        if stimulus["boundary"] == task["boundary"]
    ]


@unittest.skipUnless(os.environ.get("INTERP_TOKENIZER_TESTS"), "tokenizer tests")
class RealTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        if not os.environ.get("HF_TOKEN"):
            os.environ["HF_TOKEN"] = interp_score.load_secret(
                "HF_TOKEN", interp_score.ENV_FILE
            )
        cls.tokenizers = {
            key: AutoTokenizer.from_pretrained(spec["hf_id"])
            for key, spec in interp_score.MODELS.items()
        }
        cls.samples = snapshot_samples()

    def test_every_snapshot_prompt_ends_at_the_assistant_position(self):
        for key, tokenizer in self.tokenizers.items():
            spec = interp_score.MODELS[key]
            suffix = tokenizer.encode(
                spec["assistant_suffix"], add_special_tokens=False
            )
            for task, stimulus in self.samples:
                with self.subTest(
                    model=key, task=task["task_id"], id=stimulus["stimulus_id"]
                ):
                    text, ids = interp_score.render_prompt(
                        tokenizer, spec, task["system_prompt"], stimulus["text"]
                    )
                    self.assertEqual(ids[-len(suffix) :], suffix)
                    self.assertTrue(text.endswith(spec["assistant_suffix"]))

    def test_qwen_template_has_an_empty_reasoning_block(self):
        tokenizer = self.tokenizers["qwen3.5-4b"]
        spec = interp_score.MODELS["qwen3.5-4b"]
        text, _ = interp_score.render_prompt(tokenizer, spec, "System.", "User.")
        self.assertEqual(text.count("<think>"), 1)
        self.assertTrue(text.endswith("<think>\n\n</think>\n\n"))

    def test_llama_template_date_is_fixed(self):
        tokenizer = self.tokenizers["llama3.2-3b"]
        spec = interp_score.MODELS["llama3.2-3b"]
        text, _ = interp_score.render_prompt(tokenizer, spec, "System.", "User.")
        self.assertIn("Today Date: 26 Jul 2024\n", text)
        self.assertEqual(text.count("<|begin_of_text|>"), 1)

    def test_answer_tokens_are_known_and_context_independent(self):
        for key, tokenizer in self.tokenizers.items():
            spec = interp_score.MODELS[key]
            seen = {}
            for task, stimulus in self.samples:
                text, ids = interp_score.render_prompt(
                    tokenizer, spec, task["system_prompt"], stimulus["text"]
                )
                classes = [task["lower_class"], task["higher_class"]]
                for name, tokens in interp_score.resolve_answers(
                    tokenizer, text, ids, classes
                ).items():
                    pieces = [tokenizer.decode([token]) for token in tokens]
                    with self.subTest(model=key, answer=name):
                        self.assertEqual(pieces, EXPECTED_PIECES[name])
                        self.assertEqual(seen.setdefault(name, tokens), tokens)
            self.assertEqual(set(seen), set(ANSWERS))

    def test_first_answer_tokens_differ_from_space_and_case_variants(self):
        for key, tokenizer in self.tokenizers.items():
            spec = interp_score.MODELS[key]
            text, ids = interp_score.render_prompt(tokenizer, spec, "System.", "User.")
            for name in ANSWERS:
                first = interp_score.answer_token_ids(tokenizer, text, ids, name)[0]
                for variant in (" " + name, name.lower()):
                    other = tokenizer.encode(variant, add_special_tokens=False)[0]
                    with self.subTest(model=key, variant=variant):
                        self.assertNotEqual(first, other)


@unittest.skipUnless(os.environ.get("INTERP_MODEL_TESTS"), "model tests need weights")
class RealModelTests(unittest.TestCase):
    """Scores must agree with what the model itself generates at the answer."""

    @classmethod
    def setUpClass(cls):
        cls.key = os.environ.get("INTERP_MODEL", "llama3.2-3b")
        cls.device = os.environ.get("INTERP_DEVICE", "cuda")
        cls.spec = interp_score.MODELS[cls.key]
        cls.tokenizer, cls.model, _ = interp_score.load_model(
            cls.spec, "main", cls.device
        )
        anchors = [
            row
            for row in read_csv(interp_score.SNAPSHOT / "stimuli.csv")
            if row["country"] == "none"
            and row["level"] in {"1", "5"}
            and row["wording"] == "1"
            and row["base_id"][-2:] == "01"
        ]
        tasks = read_csv(interp_score.SNAPSHOT / "system_prompts.csv")
        cls.prompts = [
            (task, row)
            for task in tasks
            for row in anchors
            if row["boundary"] == task["boundary"]
        ]

    def generate(self, ids, steps):
        import torch

        with torch.inference_mode():
            output = self.model.generate(
                input_ids=torch.tensor([ids], device=self.device),
                attention_mask=torch.ones(
                    1, len(ids), dtype=torch.long, device=self.device
                ),
                max_new_tokens=steps,
                do_sample=False,
                output_logits=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        tokens = output.sequences[0, len(ids) :].tolist()
        log_probs = [torch.log_softmax(step[0].float(), -1) for step in output.logits]
        return tokens, log_probs

    def score(self, task, row):
        classes = [task["lower_class"], task["higher_class"]]
        record = interp_score.score_prompt(
            self.model,
            self.tokenizer,
            self.spec,
            task["system_prompt"],
            row["text"],
            classes,
            self.device,
        )
        _, ids = interp_score.render_prompt(
            self.tokenizer, self.spec, task["system_prompt"], row["text"]
        )
        return record, ids

    def test_first_answer_token_matches_generation_first_step(self):
        for task, row in self.prompts:
            record, ids = self.score(task, row)
            _, steps = self.generate(ids, 1)
            for name, entry in record["answers"].items():
                with self.subTest(
                    task=task["task_id"], id=row["stimulus_id"], answer=name
                ):
                    generated = float(steps[0][entry["token_ids"][0]])
                    self.assertAlmostEqual(
                        entry["token_logprobs"][0], generated, delta=0.05
                    )

    def test_greedy_output_is_one_answer_and_matches_teacher_forcing(self):
        for task, row in self.prompts:
            record, ids = self.score(task, row)
            longest = max(len(e["token_ids"]) for e in record["answers"].values())
            tokens, steps = self.generate(ids, longest)
            matches = [
                (name, entry)
                for name, entry in record["answers"].items()
                if tokens[: len(entry["token_ids"])] == entry["token_ids"]
            ]
            with self.subTest(task=task["task_id"], id=row["stimulus_id"]):
                self.assertEqual(len(matches), 1, self.tokenizer.decode(tokens))
                _, entry = matches[0]
                for position, (token, forced) in enumerate(
                    zip(entry["token_ids"], entry["token_logprobs"])
                ):
                    self.assertAlmostEqual(
                        forced, float(steps[position][token]), delta=0.1
                    )

    def test_answers_hold_nearly_all_probability(self):
        masses = [self.score(task, row)[0]["answer_mass"] for task, row in self.prompts]
        self.assertGreaterEqual(statistics.median(masses), 0.9, masses)


if __name__ == "__main__":
    unittest.main()
