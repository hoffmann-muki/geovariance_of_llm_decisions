"""Network-free checks of scoring, provider safeguards and billing-safe resumption."""

import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import Mock

from src import inference as runner
from src import model_inputs
from src.utils import append_jsonl, load_secret

PROTOCOL = json.loads((runner.MAIN / "protocol.json").read_text())
ENCODER = {
    "api": "deepinfra",
    "model": model_inputs.ENCODER,
    "dimension": 2,
    "input_price_usd": 1e-8,
    "output_price_usd": 0,
}
DECODER = {
    "api": "openrouter",
    "model": model_inputs.DECODER,
    "canonical_slug": "canonical",
    "provider": runner.PROVIDER,
    "provider_name": "CoreWeave",
    "input_price_usd": 2.2e-7,
    "output_price_usd": 2.2e-7,
}


def embedding_response(count=1, tokens=7):
    return {
        "model": ENCODER["model"],
        "data": [{"index": i, "embedding": [i + 1, 1]} for i in reversed(range(count))],
        "usage": {"prompt_tokens": tokens},
    }


def decoder_response(content='{"label":"B"}'):
    return {
        "id": "gen-test",
        "model": DECODER["model"],
        "provider": "CoreWeave",
        "choices": [{"finish_reason": "stop", "message": {"content": content}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 5, "cost": 0.0000231},
    }


class InferenceTests(unittest.TestCase):
    def test_malformed_http_success_retains_terminal_raw_response(self):
        malformed = [
            None,
            [],
            {"usage": []},
            {"usage": "invalid"},
            {**decoder_response(), "choices": [None]},
            {
                **decoder_response(),
                "choices": [{"message": None, "finish_reason": "stop"}],
            },
        ]
        for response in malformed:
            with (
                self.subTest(response=response),
                tempfile.TemporaryDirectory() as directory,
            ):
                journal = Path(directory) / "attempts.jsonl"
                raw = json.dumps(response)
                result = runner.request_once(
                    journal,
                    "request",
                    DECODER,
                    runner.payload(DECODER, {"text": "event"}, PROTOCOL),
                    "secret",
                    PROTOCOL,
                    {},
                    None,
                    0.1,
                    sender=Mock(return_value=(raw, None)),
                )
                self.assertEqual(result["status"], "fatal")
                self.assertEqual(result["raw_response"], raw)
                self.assertIn("request", runner.journal_state(journal)[1])

    def test_embeddings_sort_indices_validate_tokens_and_normalize(self):
        response = embedding_response(2)
        vectors = runner.embeddings(response, ENCODER, 2, 7)
        self.assertAlmostEqual(vectors[0][0], 2**-0.5)
        self.assertGreater(vectors[1][0], vectors[0][0])
        with self.assertRaisesRegex(ValueError, "token count"):
            runner.embeddings(response, ENCODER, 2, 8)
        response["data"][0]["index"] = 0
        with self.assertRaisesRegex(ValueError, "indices"):
            runner.embeddings(response, ENCODER, 2, 7)
        for vector in ([0, 0], [float("nan"), 1], [float("inf"), 1], [True, 1], [1]):
            with self.assertRaises(ValueError):
                runner.normalise(vector, 2)

    def test_cosine_ties_follow_explicit_class_order(self):
        protocol = {**PROTOCOL, "class_order": ["P", "B"]}
        result = runner.interpret(
            embedding_response(),
            ENCODER,
            protocol,
            {"encoder_tokens": 7},
            [[1, 0], [1, 0]],
            None,
        )
        self.assertEqual(result["label"], "P")
        self.assertEqual(result["margin"], 0)

    def test_strict_decoder_schema_and_provider(self):
        self.assertEqual(
            runner.decoder_result(
                decoder_response(), DECODER, PROTOCOL["class_order"], None
            )["label"],
            "B",
        )
        for content in (
            '{"label":"B","other":1}',
            '{"label":"B","label":"P"}',
            '{"label":"unknown"}',
            '```json\n{"label":"B"}\n```',
        ):
            with self.assertRaises(ValueError):
                runner.decoder_result(
                    decoder_response(content), DECODER, PROTOCOL["class_order"], None
                )
        response = decoder_response()
        response["provider"] = "Other"
        with self.assertRaisesRegex(ValueError, "provider drift"):
            runner.decoder_result(response, DECODER, PROTOCOL["class_order"], None)
        with self.assertRaisesRegex(ValueError, "cache hit"):
            runner.decoder_result(
                decoder_response(), DECODER, PROTOCOL["class_order"], "HIT"
            )

    def test_payload_excludes_metadata_and_pins_route(self):
        row = {
            "text": "event",
            "expected_label": "SECRET_LABEL",
            "source_event_id": "SECRET_ID",
            "country": "SECRET_COUNTRY",
        }
        for model in (ENCODER, DECODER):
            body = runner.payload(model, row, PROTOCOL)
            self.assertNotIn("SECRET", json.dumps(body))
        self.assertEqual(
            body["provider"],
            {
                "only": [runner.PROVIDER],
                "allow_fallbacks": False,
                "require_parameters": True,
            },
        )
        self.assertEqual(body["response_format"]["type"], "json_schema")

    def test_success_is_not_sent_again_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "attempts.jsonl"
            send = Mock(return_value=(json.dumps(decoder_response()), None))
            arguments = (
                journal,
                "request",
                DECODER,
                runner.payload(DECODER, {"text": "event"}, PROTOCOL),
                "secret-test-key",
                PROTOCOL,
                {},
                None,
                0.1,
            )
            first = runner.request_once(*arguments, sender=send)
            second = runner.request_once(*arguments, sender=send)
            self.assertEqual(first, second)
            send.assert_called_once()
            self.assertNotIn("secret-test-key", journal.read_text())

    def test_retries_only_explicit_server_failure_and_stops_at_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "attempts.jsonl"
            body = runner.payload(DECODER, {"text": "event"}, PROTOCOL)
            sender = Mock(
                side_effect=[
                    urllib.error.HTTPError("url", 503, "failure", {}, None),
                    (json.dumps(decoder_response()), None),
                ]
            )
            result = runner.request_once(
                journal,
                "request",
                DECODER,
                body,
                "secret",
                PROTOCOL,
                {},
                None,
                0.1,
                sender=sender,
                pause=lambda _: None,
            )
            self.assertEqual(result["status"], "valid")
            self.assertEqual(sender.call_count, 2)
            with self.assertRaisesRegex(ValueError, "budget"):
                runner.request_once(
                    journal,
                    "another",
                    DECODER,
                    body,
                    "secret",
                    PROTOCOL,
                    {},
                    None,
                    1e-9,
                    sender=sender,
                )
            self.assertEqual(sender.call_count, 2)

    def test_unknown_transport_and_unfinished_attempt_never_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "attempts.jsonl"
            sender = Mock(side_effect=TimeoutError())
            args = (
                journal,
                "request",
                DECODER,
                runner.payload(DECODER, {"text": "event"}, PROTOCOL),
                "secret",
                PROTOCOL,
                {},
                None,
                0.1,
            )
            self.assertEqual(
                runner.request_once(*args, sender=sender)["status"], "fatal"
            )
            runner.request_once(*args, sender=sender)
            sender.assert_called_once()
            append_jsonl(
                journal, {"event": "started", "request_id": "interrupted", "attempt": 1}
            )
            with self.assertRaisesRegex(ValueError, "Unfinished"):
                runner.journal_state(journal)

    def test_heldout_inputs_are_six_classes_outside_main(self):
        rows = runner.heldout_records()
        self.assertEqual({r["expected_label"] for r in rows}, set(model_inputs.CLASSES))
        self.assertEqual(len(rows), 6)

    def test_env_parser_does_not_execute_shell_syntax(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("TEST_REPO_CREDENTIAL='$(touch /tmp/never-run-this)'\n")
            self.assertEqual(
                load_secret("TEST_REPO_CREDENTIAL", path),
                "$(touch /tmp/never-run-this)",
            )


if __name__ == "__main__":
    unittest.main()
