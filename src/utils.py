"""Repository-wide file I/O, hashing, validation and immutable snapshot checks."""

import csv
import hashlib
import json
import os
import shlex


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_digest(path):
    with path.open("rb") as stream:
        return (
            hashlib.file_digest(stream, "sha256").hexdigest()
            if hasattr(hashlib, "file_digest")
            else digest_bytes(stream)
        )


def digest_bytes(stream):
    result = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        result.update(chunk)
    return result.hexdigest()


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


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verified_snapshot(path):
    manifest = read_json(path / "manifest.json")
    for name, info in manifest["outputs"].items():
        require(
            file_digest(path / name) == info["sha256"],
            f"Prepared output changed: {name}",
        )
    return manifest


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


def append_jsonl(path, record):
    """Persist a journal event before proceeding to the next operation."""
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
