"""Offline checks for model-visible inputs, lengths, provenance, and release scope."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import prepare_model_inputs as model
import prepare_data as prep
from finalize_inputs import read_csv, verified_snapshot


class ModelInputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_inputs_preserve_notes_and_keep_pilot_separate(self):
        rows = model.source_records(
            model.ROOT / "data/accepted", model.ROOT / "data/variants/pilot"
        )
        main = [row for row in rows if row["dataset"] == "main_baselines"]
        pilot = [row for row in rows if row["dataset"] == "reviewed_pilot"]
        self.assertEqual(len(main), 590)
        self.assertEqual(len(pilot), 531)
        self.assertFalse(
            {r["source_family_id"] for r in main}
            & {r["source_family_id"] for r in pilot}
        )
        originals = {
            row["event_id_cnty"]: row["notes"]
            for row in read_csv(model.ROOT / "data/accepted/main.csv")
        }
        for row in main:
            self.assertEqual(row["text"], originals[row["source_event_id"]])
            self.assertEqual(
                model.decoder_messages(row["text"])[1],
                {"role": "user", "content": row["text"]},
            )
            self.assertEqual(
                model.encoder_input(row["text"]),
                f"Instruct: {model.ENCODER_INSTRUCTION}\nQuery: {row['text']}",
            )
        pilot_sources = {
            r["stimulus_id"]: r
            for r in read_csv(model.ROOT / "data/variants/pilot/stimuli.csv")
        }
        audit = model.audit_lengths(pilot, lambda _: 1)
        for row in audit:
            source = pilot_sources[row["input_id"]]
            for field in (
                "case_id",
                "case_sha256",
                "original_id",
                "source_file",
                "source_record",
            ):
                self.assertEqual(row[field], source[field])

    def test_model_inputs_exclude_labels_ids_and_conditions(self):
        rows = model.source_records(
            model.ROOT / "data/accepted", model.ROOT / "data/variants/pilot"
        )
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
        row = model.source_records(
            model.ROOT / "data/accepted", model.ROOT / "data/variants/pilot"
        )[0]
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

    def test_prepare_is_offline_immutable_and_does_not_release_main_variants(self):
        destination = self.root / "inputs"
        counter = (lambda text: len(text.split()) + 2, {"test_tokenizer": True})
        with patch.object(
            model.urllib.request,
            "urlopen",
            side_effect=AssertionError("No network calls with an injected tokenizer"),
        ):
            manifest = model.prepare(destination, token_counter=counter)
        self.assertEqual(
            manifest["models"]["encoder"]["id"],
            "intfloat/multilingual-e5-large-instruct",
        )
        self.assertEqual(
            manifest["models"]["decoder"]["id"], "meta-llama/llama-3.1-8b-instruct"
        )
        self.assertFalse(manifest["inference_performed"])
        self.assertFalse(manifest["main_inference_ready"])
        self.assertEqual(
            manifest["audited_inputs"], {"main_baselines": 590, "reviewed_pilot": 531}
        )
        self.assertEqual(
            set(p.name for p in destination.iterdir()),
            {"manifest.json", "length_audit.csv"},
        )
        verified_snapshot(destination)
        frozen = (destination / "manifest.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            model.prepare(destination, token_counter=counter)
        self.assertEqual(frozen, (destination / "manifest.json").read_bytes())

    def test_invalid_inputs_leave_no_partial_output(self):
        destination = self.root / "bad"
        with self.assertRaisesRegex(ValueError, "Class definition exceeds"):
            model.prepare(destination, token_counter=(lambda _: 513, {}))
        self.assertFalse(destination.exists())
        row = model.source_records(
            model.ROOT / "data/accepted", model.ROOT / "data/variants/pilot"
        )[0]
        empty = {**row, "text": " "}
        with self.assertRaisesRegex(ValueError, "Empty input"):
            model.audit_lengths([empty], lambda _: 1)
        for count in (0, -1, 3.5, True):
            with (
                self.subTest(count=count),
                self.assertRaisesRegex(ValueError, "Invalid encoder"),
            ):
                model.audit_lengths([row], lambda _: count)


if __name__ == "__main__":
    unittest.main()
