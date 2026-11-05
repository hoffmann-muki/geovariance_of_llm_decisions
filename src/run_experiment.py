"""Execute the frozen main schedule with bounded concurrency and durable accounting."""

import argparse
import concurrent.futures
import heapq
import http.client
import json
import math
import sqlite3
import time
import urllib.error
import zlib
from collections import Counter, deque
from pathlib import Path

from . import experiment, inference, model_inputs
from .utils import (
    digest,
    file_digest,
    load_secret,
    read_csv,
    read_json,
    require,
    verified_snapshot,
    write_csv,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "data/experiment"
REFERENCES = "encoder:references"
TERMINAL = {"valid", "invalid", "transport_unknown", "http_exhausted"}
MAX_ATTEMPTS = 3
POLICY = {
    "first_attempt_order": "Frozen request_order.csv; asynchronous completion",
    "retry": "HTTP 429/5xx only, at most three attempts; delays 2 and 4 seconds",
    "unknown_delivery": "Record missing and never automatically resend",
    "invalid_output": "Record missing without retrying for a preferred answer",
    "response_caching": False,
}


def load_inputs(directory=INPUTS):
    manifest = verified_snapshot(directory)
    verified_snapshot(experiment.MAIN)
    protocol = read_json(directory / "protocol.json")
    require(
        manifest["scope"] == "adopted_main_protocol_preparation"
        and protocol["encoder_method"] == "example_centroids",
        "Unexpected experiment protocol",
    )
    for name, info in manifest["inputs"].items():
        if not name.startswith(
            "src/"
        ):  # These source hashes describe the historical producer.
            require(
                file_digest(ROOT / name) == info["sha256"],
                f"Frozen input changed: {name}",
            )
    rows = read_csv(ROOT / protocol["stimuli"])
    stimuli = {r["stimulus_id"]: r for r in rows}
    audit = {r["input_id"]: r for r in read_csv(directory / "length_audit.csv")}
    require(
        len(rows) == len(stimuli) == len(audit) == manifest["stimuli"],
        "Duplicate/missing stimuli",
    )
    for row in rows:
        check = audit[row["stimulus_id"]]
        require(
            digest(row["text"]) == check["text_sha256"] == row["text_sha256"],
            "Stimulus text changed",
        )
        require(
            digest(experiment.encoder_input(row["text"], protocol))
            == check["encoder_input_sha256"]
            and 0 < int(check["encoder_tokens"]) <= model_inputs.ENCODER_LIMIT,
            "Model-visible input changed",
        )
        row["encoder_tokens"] = int(check["encoder_tokens"])
    schedule = read_csv(ROOT / protocol["request_order"])
    require(
        len(schedule)
        == len({r["request_id"] for r in schedule})
        == manifest["planned_classifications"],
        "Duplicate/missing scheduled requests",
    )
    expected = Counter(
        (model, ident, "0")
        for model in protocol["provider_metadata"]
        for ident in stimuli
    )
    expected.update(
        (model, ident, "1")
        for model in protocol["provider_metadata"]
        for ident in protocol["repeat_stimulus_ids"]
    )
    require(
        Counter((r["model"], r["stimulus_id"], r["replicate"]) for r in schedule)
        == expected,
        "Request schedule changed",
    )
    require(
        [int(r["sequence"]) for r in schedule] == list(range(1, len(schedule) + 1)),
        "Schedule sequence changed",
    )
    require(
        protocol["encoder_reference_embeddings"]
        == len(protocol["encoder_reference_inputs"])
        == 18,
        "Reference count changed",
    )
    for row in protocol["encoder_reference_inputs"]:
        require(
            digest(row["text"]) == row["text_sha256"]
            and 0 < row["encoder_tokens"] <= 512,
            "Reference input changed",
        )
    require(
        experiment.reviewed_references(protocol)
        == [
            {k: v for k, v in r.items() if k != "encoder_tokens"}
            for r in protocol["encoder_reference_inputs"]
        ],
        "References differ from accepted reviews",
    )
    registry = read_json(ROOT / protocol["analysis_amendment"]["registry"])
    require(
        file_digest(ROOT / protocol["analysis_amendment"]["registry"])
        == protocol["analysis_amendment"]["registry_sha256"]
        and len(registry["primary_contrasts"]) == 438,
        "Analysis plan changed",
    )
    return protocol, stimuli, schedule


class Journal:
    """One SQLite writer; compressed raw responses avoid a second response archive."""

    def __init__(self, path):
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS requests (
                request_id TEXT PRIMARY KEY, model TEXT NOT NULL,
                sequence INTEGER NOT NULL, stimulus_id TEXT,
                replicate INTEGER NOT NULL, payload_sha256 TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending', result TEXT,
                generation_id TEXT UNIQUE
            );
            CREATE TABLE IF NOT EXISTS attempts (
                request_id TEXT NOT NULL REFERENCES requests(request_id),
                attempt INTEGER NOT NULL, started_at TEXT NOT NULL,
                finished_at TEXT, elapsed_seconds REAL, reserve_usd REAL NOT NULL,
                cost_usd REAL NOT NULL DEFAULT 0, cost_basis TEXT,
                status TEXT NOT NULL DEFAULT 'started', http_status INTEGER,
                cache_status TEXT, error TEXT, raw_response BLOB,
                PRIMARY KEY (request_id, attempt)
            );
        """)
        require(
            self.connection.execute("PRAGMA quick_check").fetchone()[0] == "ok",
            "Journal integrity check failed",
        )
        self.spent = self.connection.execute(
            "SELECT COALESCE(SUM(cost_usd),0) FROM attempts WHERE finished_at IS NOT NULL"
        ).fetchone()[0]

    def register(self, task, payload_sha256, commit=True):
        ident = task["request_id"]
        existing = self.connection.execute(
            "SELECT * FROM requests WHERE request_id=?", (ident,)
        ).fetchone()
        if existing:
            require(
                existing["payload_sha256"] == payload_sha256
                and existing["model"] == task["model"]
                and existing["sequence"] == int(task["sequence"]),
                "Saved request binding changed",
            )
        else:
            self.connection.execute(
                "INSERT INTO requests(request_id,model,sequence,stimulus_id,replicate,payload_sha256) VALUES(?,?,?,?,?,?)",
                (
                    ident,
                    task["model"],
                    int(task["sequence"]),
                    task.get("stimulus_id"),
                    int(task.get("replicate", 0)),
                    payload_sha256,
                ),
            )
            if commit:
                self.connection.commit()

    def recover(self):
        with self.connection:
            self.connection.execute(
                "UPDATE requests SET status='transport_unknown' WHERE request_id IN (SELECT request_id FROM attempts WHERE finished_at IS NULL)"
            )
            self.connection.execute(
                "UPDATE attempts SET status='transport_unknown', finished_at=?, cost_usd=reserve_usd, cost_basis='conservative_reserve', error='Interrupted delivery; not automatically resent' WHERE finished_at IS NULL",
                (inference.now(),),
            )
        self.spent = self.connection.execute(
            "SELECT COALESCE(SUM(cost_usd),0) FROM attempts WHERE finished_at IS NOT NULL"
        ).fetchone()[0]

    def status(self, ident):
        return self.connection.execute(
            "SELECT status FROM requests WHERE request_id=?", (ident,)
        ).fetchone()[0]

    def start(self, ident, reserve):
        require(
            self.status(ident) in {"pending", "retryable"},
            "Request is not eligible for dispatch",
        )
        attempt = (
            self.connection.execute(
                "SELECT COUNT(*) FROM attempts WHERE request_id=?", (ident,)
            ).fetchone()[0]
            + 1
        )
        require(attempt <= MAX_ATTEMPTS, "Attempt limit reached")
        with self.connection:
            self.connection.execute(
                "INSERT INTO attempts(request_id,attempt,started_at,reserve_usd) VALUES(?,?,?,?)",
                (ident, attempt, inference.now(), reserve),
            )
        return attempt

    def finish(self, ident, attempt, response):
        existing = self.connection.execute(
            "SELECT status FROM attempts WHERE request_id=? AND attempt=?",
            (ident, attempt),
        ).fetchone()
        require(
            existing is not None and existing[0] == "started",
            "Attempt already finished or unknown",
        )
        result = response.get("result")
        generation = result.get("generation_id") if result else None
        if (
            generation
            and self.connection.execute(
                "SELECT 1 FROM requests WHERE generation_id=? AND request_id!=?",
                (generation, ident),
            ).fetchone()
        ):
            response.update(
                status="fatal", error="Duplicate decoder generation ID", result=None
            )
            generation = None
        raw = response.get("raw_response")
        with self.connection:
            self.connection.execute(
                "UPDATE attempts SET finished_at=?, elapsed_seconds=?, cost_usd=?, cost_basis=?, status=?, http_status=?, cache_status=?, error=?, raw_response=? WHERE request_id=? AND attempt=?",
                (
                    inference.now(),
                    response["elapsed_seconds"],
                    response["cost_usd"],
                    response["cost_basis"],
                    response["status"],
                    response.get("http_status"),
                    response.get("cache_status"),
                    response.get("error"),
                    zlib.compress(raw.encode("utf-8")) if raw is not None else None,
                    ident,
                    attempt,
                ),
            )
            self.connection.execute(
                "UPDATE requests SET status=?,result=?,generation_id=? WHERE request_id=?",
                (
                    response["status"],
                    (
                        json.dumps(response["result"], allow_nan=False)
                        if response.get("result")
                        else None
                    ),
                    generation,
                    ident,
                ),
            )
        self.spent += response["cost_usd"]

    def raw_response(self, ident):
        row = self.connection.execute(
            "SELECT raw_response FROM attempts WHERE request_id=? AND status='valid' ORDER BY attempt DESC LIMIT 1",
            (ident,),
        ).fetchone()
        require(row is not None, "Reference response unavailable")
        return json.loads(zlib.decompress(row[0]).decode("utf-8"))

    def accounting(self):
        return self.spent

    def counts(self):
        return {
            row[0]: row[1]
            for row in self.connection.execute(
                "SELECT status,COUNT(*) FROM requests WHERE request_id!=? GROUP BY status",
                (REFERENCES,),
            )
        }

    def close(self):
        self.connection.close()


def perform(model, body, key, reserve):
    """Make exactly one POST; all retries are scheduled and journaled by the controller."""
    started = time.monotonic()
    result = {
        "status": "received",
        "cost_usd": reserve,
        "cost_basis": "conservative_reserve",
        "result": None,
    }
    try:
        raw, cache = inference.send(model, body, key)
        result.update(
            raw_response=raw.replace(key, "[REDACTED]"),
            cache_status=cache,
            http_status=200,
        )
    except urllib.error.HTTPError as error:
        result.update(
            status=(
                "retryable"
                if error.code == 429 or 500 <= error.code < 600
                else "blocked"
            ),
            http_status=error.code,
            error="HTTP request rejected",
        )
        if error.code in {400, 401, 402, 403, 404, 413, 422, 429}:
            result.update(cost_usd=0, cost_basis="rejected_request")
        error.close()
    except (
        urllib.error.URLError,
        TimeoutError,
        OSError,
        http.client.HTTPException,
    ) as error:
        result.update(
            status="transport_unknown",
            error=f"{type(error).__name__}: delivery outcome unknown; not resent",
        )
    except UnicodeError:
        result.update(status="invalid", error="Response is not UTF-8")
    result["elapsed_seconds"] = time.monotonic() - started
    return result


def interpret(task, response, protocol, stimuli, prototypes):
    if response["status"] != "received":
        return response
    model = protocol["provider_metadata"][task["model"]]
    try:
        parsed = json.loads(response["raw_response"])
        if model["api"] == "openrouter":
            require(
                str(response.get("cache_status")).upper() != "HIT",
                "Decoder response cache drift",
            )
            require(isinstance(parsed, dict), "Invalid response object")
            require(
                parsed.get("model") in {model["model"], model["canonical_slug"]}
                and parsed.get("provider") == model["provider_name"],
                "Decoder routing/model drift",
            )
        cost, basis = inference.response_cost(parsed, model)
        response.update(cost_usd=cost, cost_basis=basis)
        if model["api"] == "deepinfra":
            count = 18 if task["request_id"] == REFERENCES else 1
            tokens = (
                sum(r["encoder_tokens"] for r in protocol["encoder_reference_inputs"])
                if count == 18
                else stimuli[task["stimulus_id"]]["encoder_tokens"]
            )
            vectors = inference.embeddings(parsed, model, count, tokens)
            if count == 18:
                experiment.encoder_prototypes(vectors, protocol)
                result = {"reference_embeddings": 18}
            else:
                scores = [
                    math.fsum(a * b for a, b in zip(vectors[0], prototype))
                    for prototype in prototypes
                ]
                winner = max(range(len(scores)), key=scores.__getitem__)
                ordered = sorted(scores, reverse=True)
                result = {
                    "label": protocol["class_order"][winner],
                    "scores": dict(zip(protocol["class_order"], scores)),
                    "margin": ordered[0] - ordered[1],
                }
        else:
            result = inference.decoder_result(
                parsed, model, protocol["class_order"], response.get("cache_status")
            )
        response.update(status="valid", result=result)
    except (ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
        fatal = (
            model["api"] == "deepinfra"
            or "drift" in str(error)
            or "cache hit" in str(error)
        )
        response.update(
            status="fatal" if fatal else "invalid", error=str(error), result=None
        )
    return response


def body_for(task, protocol, stimuli):
    model = protocol["provider_metadata"][task["model"]]
    if task["request_id"] == REFERENCES:
        return {
            "model": model["model"],
            "input": [r["text"] for r in protocol["encoder_reference_inputs"]],
            **protocol["encoder_parameters"],
        }
    return inference.payload(model, stimuli[task["stimulus_id"]], protocol)


def execute(
    journal,
    tasks,
    protocol,
    stimuli,
    keys,
    budget,
    workers,
    prototypes=None,
    progress=None,
):
    todo, delayed, active = deque(), [], {}
    for task in tasks:
        body = body_for(task, protocol, stimuli)
        journal.register(task, inference.object_hash(body), commit=False)
        status = journal.status(task["request_id"])
        require(
            status not in {"fatal", "blocked"},
            "A previous fatal or blocked response requires reconciliation",
        )
        if status not in TERMINAL:
            todo.append(task)
    journal.connection.commit()
    stop, completed, serial = None, 0, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        while todo or delayed or active:
            while delayed and delayed[0][0] <= time.monotonic():
                _, _, task = heapq.heappop(delayed)
                todo.appendleft(task)
            while todo and len(active) < workers and stop is None:
                task = todo[0]
                model = protocol["provider_metadata"][task["model"]]
                body = body_for(task, protocol, stimuli)
                reserve = inference.cost_bound(model, body)
                if (
                    journal.accounting()
                    + sum(item[2] for item in active.values())
                    + reserve
                    > budget
                ):
                    if active:
                        break
                    stop = "Accounting budget reached"
                    break
                todo.popleft()
                attempt = journal.start(task["request_id"], reserve)
                future = executor.submit(
                    perform, model, body, keys[model["api"]], reserve
                )
                active[future] = (task, attempt, reserve)
            if stop and not active:
                break
            if not active:
                time.sleep(0.1)
                continue
            finished, _ = concurrent.futures.wait(
                active, timeout=1, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in finished:
                task, attempt, reserve = active.pop(future)
                response = interpret(
                    task, future.result(), protocol, stimuli, prototypes
                )
                if response["status"] == "retryable":
                    if attempt < MAX_ATTEMPTS:
                        serial += 1
                        heapq.heappush(
                            delayed, (time.monotonic() + 2**attempt, serial, task)
                        )
                    else:
                        response["status"] = "http_exhausted"
                journal.finish(task["request_id"], attempt, response)
                if response["status"] in {"fatal", "blocked"}:
                    stop = response.get("error", "Provider configuration error")
                if response["cost_usd"] > reserve + 1e-9:
                    stop = "Provider charge exceeded the conservative request reserve"
                completed += response["status"] in TERMINAL
                if progress and completed and completed % 250 == 0:
                    progress()
    return stop


def export(journal, output_dir, protocol, stimuli, schedule, run):
    results = {
        row["request_id"]: row
        for row in journal.connection.execute("SELECT * FROM requests")
    }
    predictions = []
    for task in schedule:
        saved = results.get(task["request_id"])
        if not saved or saved["status"] not in TERMINAL | {"fatal", "blocked"}:
            continue
        row = stimuli[task["stimulus_id"]]
        result = json.loads(saved["result"]) if saved["result"] else {}
        predictions.append(
            {
                **task,
                **{key: row[key] for key in model_inputs.IDENTITY_FIELDS if key in row},
                "source_event_id": row["source_event_id"],
                "source_family_id": row["source_family_id"],
                "cohort": row["cohort"],
                "base_id": row["base_id"],
                "country": row["country"],
                "condition": row["condition"],
                "expected_label": row["label"],
                "text_sha256": row["text_sha256"],
                "status": saved["status"],
                "prediction": result.get("label", ""),
                "generation_id": result.get("generation_id", ""),
                "scores": json.dumps(result.get("scores", {}), sort_keys=True),
                "margin": result.get("margin", ""),
                "label_support": row.get("label_support", ""),
                "notes_sufficiency": row.get("notes_sufficiency", ""),
                "review_flags": row.get("review_flags", ""),
                "case_constraints": row.get("case_constraints", ""),
                "review_status": row.get("review_status", ""),
            }
        )
    fields = [
        "sequence",
        "request_id",
        "model",
        "stimulus_id",
        "replicate",
        "cohort",
        "base_id",
        "source_event_id",
        "source_family_id",
        "source_file",
        "source_record",
        "country",
        "condition",
        "original_id",
        "case_id",
        "case_sha256",
        "text_sha256",
        "label_support",
        "notes_sufficiency",
        "review_flags",
        "case_constraints",
        "review_status",
        "expected_label",
        "status",
        "prediction",
        "generation_id",
        "scores",
        "margin",
    ]
    write_csv(output_dir / "predictions.csv", fields, predictions)
    accounting = {
        row[0]: row[1]
        for row in journal.connection.execute(
            "SELECT cost_basis,SUM(cost_usd) FROM attempts GROUP BY cost_basis"
        )
    }
    attempted = sum(
        row["status"] in TERMINAL | {"fatal", "blocked"}
        for row in results.values()
        if row["request_id"] != REFERENCES
    )
    run.update(
        completed_classifications=attempted,
        complete=all(
            r["request_id"] in results
            and results[r["request_id"]]["status"] in TERMINAL
            for r in schedule
        ),
        status_counts=journal.counts(),
        cost_usd=journal.accounting(),
        accounting=accounting,
        updated_at=inference.now(),
    )
    lines = [
        "# Main Inference Run",
        "",
        f"Completed schedule: **{run['complete']}**. Attempted classifications: {attempted:,}/{len(schedule):,}.",
        "",
        "| Model | Valid | Invalid or unavailable |",
        "| --- | --- | --- |",
    ]
    for model in protocol["provider_metadata"]:
        subset = [r for r in predictions if r["model"] == model]
        valid = sum(r["status"] == "valid" for r in subset)
        lines.append(f"| {model} | {valid:,} | {len(subset) - valid:,} |")
    lines += [
        "",
        f"Total accounted cost: ${run['cost_usd']:.6f}. Accounting by basis: `{json.dumps(accounting, sort_keys=True)}`.",
        "",
        "The journal retains every dispatch and response, including rejected and unknown deliveries. Raw responses are UTF-8 JSON compressed with zlib in SQLite. Missing responses are not labels. Repeat requests do not increase the independent sample size.",
        "",
        "The frozen protocol, schedule and analysis registry are bound in the run manifest. This report describes execution; geographic effect estimates require the prespecified analysis.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(output_dir, budget=3.0, workers=8, input_dir=INPUTS):
    require(
        math.isfinite(budget)
        and 0 < budget <= 3
        and type(workers) is int
        and 1 <= workers <= 8,
        "Use at most $3 and eight workers",
    )
    protocol, stimuli, schedule = load_inputs(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = output_dir / ".active"
    lock.mkdir()
    journal = None
    try:
        manifest_path = output_dir / "manifest.json"
        source_files = [
            Path(__file__),
            Path(inference.__file__),
            Path(experiment.__file__),
            Path(model_inputs.__file__),
            ROOT / "src/utils.py",
        ]
        binding = {
            "experiment_manifest_sha256": file_digest(input_dir / "manifest.json"),
            "protocol_sha256": file_digest(input_dir / "protocol.json"),
            "schedule_sha256": file_digest(ROOT / protocol["request_order"]),
            "analysis_plan_sha256": protocol["analysis_amendment"]["registry_sha256"],
            "producer_sha256": {
                str(path.relative_to(ROOT)): file_digest(path) for path in source_files
            },
        }
        if manifest_path.exists():
            metadata = read_json(manifest_path)
            require(
                metadata["inputs"] == binding
                and metadata["budget_usd"] == budget
                and metadata["workers"] == workers,
                "Run inputs/code/settings changed",
            )
            if metadata.get("complete"):
                verified_snapshot(output_dir)
                return metadata
            if metadata.get("outputs"):
                verified_snapshot(output_dir)
        else:
            require(
                not any(path != lock for path in output_dir.iterdir()),
                "Use an empty run directory",
            )
            metadata = {
                "inputs": binding,
                "budget_usd": budget,
                "workers": workers,
                "policy": POLICY,
                "started_at": inference.now(),
                "complete": False,
                "main_inference_performed": False,
            }
            write_json(manifest_path, metadata)
        require(
            inference.model_metadata() == protocol["provider_metadata"],
            "Provider availability/routing/pricing differs from the frozen protocol",
        )
        keys = {
            "deepinfra": load_secret("DEEPINFRA_API_KEY", ROOT / ".env"),
            "openrouter": load_secret("OPENROUTER_API_KEY", ROOT / ".env"),
        }
        key_info = inference.get_json(
            inference.OPENROUTER + "/key", keys["openrouter"]
        )["data"]
        require(
            not key_info.get("is_management_key")
            and not key_info.get("is_provisioning_key"),
            "Use an OpenRouter inference key",
        )
        if metadata.get("outputs"):
            metadata.setdefault("resumes", []).append(
                {"at": inference.now(), "previous_outputs": metadata.pop("outputs")}
            )
            write_json(manifest_path, metadata)
        journal = Journal(output_dir / "requests.sqlite3")
        journal.recover()
        reference_task = {
            "request_id": REFERENCES,
            "model": protocol["encoder_model"],
            "sequence": 0,
        }
        metadata.update(
            main_inference_performed=True,
            provider_metadata=protocol["provider_metadata"],
            preflight_at=inference.now(),
        )
        write_json(manifest_path, metadata)
        stop = execute(journal, [reference_task], protocol, stimuli, keys, budget, 1)
        if journal.status(REFERENCES) == "valid":
            raw = journal.raw_response(REFERENCES)
            reference_vectors = inference.embeddings(
                raw,
                protocol["provider_metadata"][protocol["encoder_model"]],
                18,
                sum(r["encoder_tokens"] for r in protocol["encoder_reference_inputs"]),
            )
            prototypes = experiment.encoder_prototypes(reference_vectors, protocol)

            def progress():
                metadata.update(
                    status_counts=journal.counts(),
                    cost_usd=journal.accounting(),
                    updated_at=inference.now(),
                )
                write_json(manifest_path, metadata)
                print(
                    json.dumps(
                        {
                            "completed": sum(
                                n
                                for status, n in metadata["status_counts"].items()
                                if status in TERMINAL
                            ),
                            "counts": metadata["status_counts"],
                            "cost_usd": round(metadata["cost_usd"], 6),
                        }
                    ),
                    flush=True,
                )

            if stop is None:
                stop = execute(
                    journal,
                    schedule,
                    protocol,
                    stimuli,
                    keys,
                    budget,
                    workers,
                    prototypes,
                    progress,
                )
        else:
            stop = stop or "Class reference embeddings unavailable"
        metadata["stop_reason"] = stop
        export(journal, output_dir, protocol, stimuli, schedule, metadata)
        journal.close()
        journal = None
        metadata["outputs"] = {
            path.name: {"sha256": file_digest(path)}
            for path in output_dir.iterdir()
            if path.is_file() and path.name != "manifest.json"
        }
        write_json(manifest_path, metadata)
        return metadata
    finally:
        if journal is not None:
            journal.close()
        lock.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/main")
    parser.add_argument("--budget", type=float, default=3.0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    result = run(args.output_dir, args.budget, args.workers)
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "complete",
                    "completed_classifications",
                    "cost_usd",
                    "stop_reason",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
