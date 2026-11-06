"""File I/O, hashing and validation helpers."""

import csv
import hashlib
import json
import os
import shlex


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_secret(name, path):
    """Read a named credential without expanding shell syntax or logging values."""
    value = os.environ.get(name, "").strip()
    if not value and path.exists():
        matches = []
        for line in path.read_text(encoding="utf-8").splitlines():
            key, separator, raw = line.partition("=")
            if separator and key.strip().removeprefix("export ").strip() == name:
                parts = shlex.split(raw, comments=True)
                require(len(parts) == 1, f"Invalid {name} entry in .env")
                matches.append(parts[0])
        require(len(matches) == 1, f"Define {name} exactly once in .env")
        value = matches[0]
    require(bool(value), f"Configure {name} in the environment or local .env")
    return value
