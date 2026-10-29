"""Model-visible formatting and exact token-length checks without inference."""

import json
import unittest
from unittest.mock import patch

import model_inputs as model
import prepare_data as prep
from utils import read_csv


def source_fixture():
    return [
        {
            "dataset": "main",
            "input_id": row["event_id_cnty"],
            "source_event_id": row["event_id_cnty"],
            "source_family_id": row["event_group_id"],
            "country": row["country"],
            "condition": "original",
            "source_file": row["source_file"],
            "source_record": row["source_record"],
            "original_id": row["event_id_cnty"],
            "case_id": "",
            "case_sha256": "",
            "text": row["notes"],
        }
        for row in read_csv(model.ROOT / "data/accepted/main.csv")
    ]


class ModelInputTests(unittest.TestCase):
    def test_model_inputs_exclude_labels_ids_and_conditions(self):
        rows = source_fixture()
        row = {
            **rows[0],
            "source_event_id": "SECRET_ID",
            "country": "SECRET_METADATA",
            "label": "SECRET_LABEL",
            "condition": "SECRET_CONDITION",
        }
        visible = json.dumps(model.decoder_messages(row["text"])) + model.encoder_input(
            row["text"]
        )
        for value in (
            "SECRET_ID",
            "SECRET_METADATA",
            "SECRET_LABEL",
            "SECRET_CONDITION",
        ):
            self.assertNotIn(value, visible)
        for label, definition in model.CLASSES.items():
            self.assertIn(f"{label} — {definition}", model.DECODER_PROMPT)
        self.assertIn(model.EVENT_RULES, model.ENCODER_INSTRUCTION)
        self.assertIn(model.EVENT_RULES, model.DECODER_PROMPT)

    def test_length_bound_includes_instruction_and_rejects_without_truncation(self):
        row = source_fixture()[0]
        seen = []

        def counter(text):
            seen.append(text)
            return 512

        result = model.audit_lengths([row], counter)
        self.assertEqual(seen, [model.encoder_input(row["text"])])
        self.assertEqual(result[0]["encoder_tokens"], 512)
        self.assertEqual(result[0]["text_sha256"], prep.digest(row["text"]))
        with self.assertRaisesRegex(ValueError, "exceeds 512"):
            model.audit_lengths([row], lambda _: 513)

    def test_downloaded_tokenizer_disables_truncation_and_counts_special_tokens(self):
        tokenizer = unittest.mock.MagicMock()
        tokenizer.encode.return_value.ids = [0, 10, 2]
        tokenizer_type = unittest.mock.MagicMock()
        tokenizer_type.from_str.return_value = tokenizer
        response = unittest.mock.MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = b'{"test":"tokenizer"}'
        with (
            patch.dict(
                "sys.modules",
                {"tokenizers": unittest.mock.MagicMock(Tokenizer=tokenizer_type)},
            ),
            patch.object(
                model.importlib.metadata,
                "version",
                return_value=model.TOKENIZERS_VERSION,
            ),
            patch.object(
                model.urllib.request, "urlopen", return_value=response
            ) as download,
        ):
            count, provenance = model.load_token_counter()
        self.assertEqual(count("event"), 3)
        tokenizer.no_truncation.assert_called_once()
        tokenizer.no_padding.assert_called_once()
        tokenizer.encode.assert_called_once_with("event", add_special_tokens=True)
        self.assertIn(model.TOKENIZER_REVISION, download.call_args.args[0])
        self.assertTrue(provenance["special_tokens_included"])
        self.assertFalse(provenance["truncation"])


if __name__ == "__main__":
    unittest.main()
