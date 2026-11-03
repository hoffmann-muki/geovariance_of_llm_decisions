"""Portable artifact serialization must preserve text and reject invalid JSON."""

import tempfile
import unittest
from pathlib import Path

from src.utils import read_json, write_json


class ArtifactIOTests(unittest.TestCase):
    def test_json_round_trip_preserves_utf8_text(self):
        record = {"location": "Bogotá — São Paulo", "label": "P"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            write_json(path, record)
            self.assertIn(record["location"].encode("utf-8"), path.read_bytes())
            self.assertEqual(read_json(path), record)

    def test_nonfinite_values_do_not_create_an_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            for value in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    write_json(path, {"score": value})
                self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
