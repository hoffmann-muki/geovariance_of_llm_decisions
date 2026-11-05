"""Network-free main-run safeguards and faithful reference-to-label execution."""

import copy
import json
import tempfile
import threading
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from src import inference, model_inputs, run_experiment as runner
from src.utils import read_csv, read_json


def fixture():
    protocol = read_json(runner.INPUTS / "protocol.json")
    rows = read_csv(runner.ROOT / protocol["stimuli"])
    selected = {}
    for row in rows:
        if row["condition"] == "original":
            selected.setdefault(row["label"], copy.deepcopy(row))
    stimuli = {row["stimulus_id"]: row for row in selected.values()}
    for row in stimuli.values():
        row["encoder_tokens"] = 40
    schedule = []
    for model in protocol["provider_metadata"]:
        for index, row in enumerate(stimuli.values()):
            schedule.append(
                {
                    "sequence": str(len(schedule) + 1),
                    "request_id": f"{model}:{index}",
                    "model": model,
                    "stimulus_id": row["stimulus_id"],
                    "replicate": "0",
                }
            )
    texts = {row["text"]: row["label"] for row in stimuli.values()}
    references = {row["text"]: row for row in protocol["encoder_reference_inputs"]}
    call_lock = threading.Lock()
    calls = []

    def vector(label):
        result = [0.0] * 1024
        result[protocol["class_order"].index(label)] = 1.0
        return result

    def sender(model, body, key):
        with call_lock:
            calls.append(body)
            generation = len(calls)
        if model["api"] == "deepinfra":
            reference_batch = len(body["input"]) == 18
            labels = (
                [references[text]["label"] for text in body["input"]]
                if reference_batch
                else [texts[body["input"][0].split("\nQuery: ", 1)[1]]]
            )
            tokens = (
                sum(references[text]["encoder_tokens"] for text in body["input"])
                if reference_batch
                else 40
            )
            response = {
                "model": model["model"],
                "usage": {"prompt_tokens": tokens},
                "data": [
                    {"index": index, "embedding": vector(label)}
                    for index, label in enumerate(labels)
                ],
            }
        else:
            response = {
                "model": model["model"],
                "provider": model["provider_name"],
                "id": f"gen-{generation}",
                "usage": {
                    "prompt_tokens": 400,
                    "completion_tokens": 7,
                    "cost": 0.00008954,
                },
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {"label": texts[body["messages"][1]["content"]]}
                            )
                        },
                    }
                ],
            }
        return json.dumps(response), None

    return protocol, stimuli, schedule, sender, calls


class MainRunTests(unittest.TestCase):
    def test_partial_checkpoint_resume_survives_a_second_interruption(self):
        protocol, stimuli, schedule, sender, calls = fixture()
        original = runner.execute

        def pause_after_one(journal, tasks, *args, **kwargs):
            if len(tasks) == 1:
                return original(journal, tasks, *args, **kwargs)
            original(journal, tasks[:1], *args, **kwargs)
            return "test checkpoint"

        def interrupt_after_next(journal, tasks, *args, **kwargs):
            if len(tasks) == 1:
                return original(journal, tasks, *args, **kwargs)
            original(journal, tasks[1:2], *args, **kwargs)
            raise RuntimeError("Interrupted after durable completion")

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                runner, "load_inputs", return_value=(protocol, stimuli, schedule)
            ),
            patch.object(
                inference, "model_metadata", return_value=protocol["provider_metadata"]
            ),
            patch.object(inference, "get_json", return_value={"data": {}}),
            patch.object(runner, "load_secret", return_value="TEST-SECRET"),
            patch.object(inference, "send", side_effect=sender),
        ):
            output = Path(directory) / "main"
            with patch.object(runner, "execute", side_effect=pause_after_one):
                self.assertFalse(runner.run(output, workers=3)["complete"])
            with patch.object(runner, "execute", side_effect=interrupt_after_next):
                with self.assertRaisesRegex(RuntimeError, "Interrupted"):
                    runner.run(output, workers=3)
            self.assertTrue(runner.run(output, workers=3)["complete"])
            self.assertEqual(len(calls), 13)

    def test_zero_usage_cache_hit_stops_before_usage_validation(self):
        protocol, stimuli, schedule, _, _ = fixture()
        task = next(r for r in schedule if r["model"] == protocol["decoder_model"])
        model = protocol["provider_metadata"][task["model"]]
        response = {
            "status": "received",
            "raw_response": json.dumps(
                {
                    "model": model["model"],
                    "provider": model["provider_name"],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                }
            ),
            "cache_status": "HIT",
            "cost_usd": 0.01,
            "cost_basis": "conservative_reserve",
            "elapsed_seconds": 0,
        }
        result = runner.interpret(task, response, protocol, stimuli, None)
        self.assertEqual(result["status"], "fatal")
        self.assertIn("cache drift", result["error"])

    def test_frozen_inputs_validate_without_network(self):
        protocol, stimuli, schedule = runner.load_inputs()
        self.assertEqual(len(stimuli), 10820)
        self.assertEqual(len(schedule), 21646)
        self.assertEqual(len(protocol["encoder_reference_inputs"]), 18)

    def test_full_mocked_execution_and_completed_resume(self):
        protocol, stimuli, schedule, sender, calls = fixture()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                runner, "load_inputs", return_value=(protocol, stimuli, schedule)
            ),
            patch.object(
                inference, "model_metadata", return_value=protocol["provider_metadata"]
            ),
            patch.object(inference, "get_json", return_value={"data": {}}),
            patch.object(runner, "load_secret", return_value="TEST-SECRET"),
            patch.object(inference, "send", side_effect=sender),
        ):
            output = Path(directory) / "main"
            result = runner.run(output, workers=3)
            self.assertTrue(result["complete"])
            self.assertEqual(len(calls), 13)
            self.assertEqual(len(calls[0]["input"]), 18)
            rows = read_csv(output / "predictions.csv")
            self.assertEqual(len(rows), 12)
            self.assertTrue(
                all(row["prediction"] == row["expected_label"] for row in rows)
            )
            self.assertTrue(all("source_family_id" in row for row in rows))
            runner.run(output, workers=3)
            self.assertEqual(len(calls), 13)
            journal = runner.Journal(output / "requests.sqlite3")
            order = [
                row[0]
                for row in journal.connection.execute(
                    "SELECT request_id FROM attempts ORDER BY rowid"
                )
            ]
            self.assertEqual(
                order, [runner.REFERENCES] + [row["request_id"] for row in schedule]
            )
            self.assertAlmostEqual(journal.accounting(), result["cost_usd"])
            journal.close()

    def test_unknown_delivery_is_terminal_missing_and_not_resent(self):
        protocol, stimuli, schedule, _, _ = fixture()
        task = schedule[0]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(inference, "send", side_effect=TimeoutError()) as sender,
        ):
            journal = runner.Journal(Path(directory) / "requests.sqlite3")
            runner.execute(
                journal,
                [task],
                protocol,
                stimuli,
                {"deepinfra": "secret", "openrouter": "secret"},
                3,
                1,
            )
            self.assertEqual(journal.status(task["request_id"]), "transport_unknown")
            runner.execute(
                journal,
                [task],
                protocol,
                stimuli,
                {"deepinfra": "secret", "openrouter": "secret"},
                3,
                1,
            )
            sender.assert_called_once()
            journal.close()

    def test_interrupted_attempt_recovery_accounts_reserve(self):
        protocol, stimuli, schedule, _, _ = fixture()
        task = schedule[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.sqlite3"
            journal = runner.Journal(path)
            journal.register(
                task, inference.object_hash(runner.body_for(task, protocol, stimuli))
            )
            journal.start(task["request_id"], 0.01)
            journal.close()
            journal = runner.Journal(path)
            journal.recover()
            self.assertEqual(journal.status(task["request_id"]), "transport_unknown")
            self.assertAlmostEqual(journal.accounting(), 0.01)
            with self.assertRaisesRegex(ValueError, "eligible"):
                journal.start(task["request_id"], 0.01)
            journal.close()

    def test_blocked_request_requires_reconciliation_and_budget_prevents_post(self):
        protocol, stimuli, schedule, _, _ = fixture()
        task = schedule[0]
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                inference,
                "send",
                side_effect=urllib.error.HTTPError("url", 401, "rejected", {}, None),
            ) as sender,
        ):
            journal = runner.Journal(Path(directory) / "requests.sqlite3")
            runner.execute(
                journal,
                [task],
                protocol,
                stimuli,
                {"deepinfra": "secret", "openrouter": "secret"},
                3,
                1,
            )
            self.assertEqual(journal.status(task["request_id"]), "blocked")
            with self.assertRaisesRegex(ValueError, "reconciliation"):
                runner.execute(
                    journal,
                    [task],
                    protocol,
                    stimuli,
                    {"deepinfra": "secret", "openrouter": "secret"},
                    3,
                    1,
                )
            sender.assert_called_once()
            journal.close()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(inference, "send") as sender,
        ):
            journal = runner.Journal(Path(directory) / "requests.sqlite3")
            reason = runner.execute(journal, [task], protocol, stimuli, {}, 1e-12, 1)
            self.assertEqual(reason, "Accounting budget reached")
            sender.assert_not_called()
            journal.close()


if __name__ == "__main__":
    unittest.main()
